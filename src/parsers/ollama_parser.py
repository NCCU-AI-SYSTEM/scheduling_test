"""Condition parser on local Ollama (JSON-schema constrained output).

Same output contract as the other parsers (`ConditionResult`), plus the
topical remainder `need` so the dense leg can embed the topic without the
condition words ("週三下午", "英文授課") that only add noise to a semantic vector.

Results are cached per (model, query) in a JSON file: parsing is the slow,
non-deterministic step and experiments re-run many times over the same queries.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.parsers.base import ConditionResult

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed" / "parse_cache"

_PROMPT = """把學生的選課查詢拆成兩部分，輸出 JSON：
- need：學生想學的「主題/內容」，去掉時間、語言、學分、系所、課別等條件後剩下的部分；只有條件沒有主題時為空字串。
- must / must_not：結構化條件。「不要/避開/除了/不想」放 must_not。只能用以下欄位：
  course_lang: 如 ["英文"]、["中文"]、["日文"]
  weekday: 1=週一 … 7=週日，如 [3]；「週一到週三」= [1,2,3]
  hour_range: 早上[8,12] 中午[12,13] 下午[13,18] 晚上[18,24]；「早八」= [8,9]；明確時段如「兩點到五點」= [14,17]
  point: 學分，如 3.0
  kind: ["必修"]/["選修"]/["通識"]/["體育"]
  lmt_kind: ["社會通識"]/["人文通識"]/["自然通識"]/["外文通識"]/["中文通識"]/["資訊通識"]/["書院通識"]
  unit: 開課系所簡稱，如 ["企管"]、["資科"]、["外文中心"]
沒提到的欄位不要輸出。

範例：
查詢：週三下午英文授課的經濟學，不要早八
{{"need": "經濟學", "must": {{"weekday": [3], "hour_range": [13, 18], "course_lang": ["英文"]}}, "must_not": {{"hour_range": [8, 9]}}}}
查詢：想找有趣的社會通識
{{"need": "有趣的課", "must": {{"lmt_kind": ["社會通識"]}}, "must_not": {{}}}}

查詢：{query}"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "need": {"type": "string"},
        "must": {"type": "object"},
        "must_not": {"type": "object"},
    },
    "required": ["need", "must", "must_not"],
}


class OllamaParser:
    def __init__(self, model: str = "gemma4:e4b"):
        self.model = model
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{model.replace(':', '_').replace('/', '_')}.json"
        self.cache: dict[str, dict] = (
            json.loads(self.path.read_text()) if self.path.exists() else {}
        )
        self._lock = threading.Lock()

    def _call(self, query: str) -> dict:
        import ollama

        resp = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": _PROMPT.format(query=query)}],
            format=_SCHEMA,
            think=False,
            options={"temperature": 0},
            keep_alive="10m",
        )
        try:
            return json.loads(resp["message"]["content"])
        except json.JSONDecodeError:
            return {"need": query, "must": {}, "must_not": {}}

    def parse_raw(self, query: str) -> dict:
        if query not in self.cache:
            out = self._call(query)
            with self._lock:
                self.cache[query] = out
        return self.cache[query]

    def parse_many(self, queries: list[str], workers: int = 4) -> list[dict]:
        todo = [q for q in dict.fromkeys(queries) if q not in self.cache]
        if todo:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(self.parse_raw, todo))
            self.path.write_text(json.dumps(self.cache, ensure_ascii=False))
        return [self.cache[q] for q in queries]

    @staticmethod
    def to_condition(raw: dict) -> ConditionResult:
        return ConditionResult(must=sanitize(raw.get("must")), must_not=sanitize(raw.get("must_not")))


_WEEKDAY_ZH = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "日": 7, "天": 7}


def _as_list(v) -> list:
    return v if isinstance(v, list) else [v]


def _weekday(v) -> int | None:
    if isinstance(v, int) and 1 <= v <= 7:
        return v
    s = str(v)
    if s.isdigit() and 1 <= int(s) <= 7:
        return int(s)
    return next((n for zh, n in _WEEKDAY_ZH.items() if zh in s), None)


def sanitize(d) -> dict:
    """Coerce LLM output to the ConditionResult value types; drop what can't be coerced."""
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    for dim in ("course_lang", "kind", "lmt_kind", "unit"):
        vals = [str(x).strip() for x in _as_list(d.get(dim) or []) if str(x).strip()]
        if vals:
            out[dim] = vals
    days = [w for w in (_weekday(x) for x in _as_list(d.get("weekday") or [])) if w]
    if days:
        out["weekday"] = sorted(set(days))
    hr = d.get("hour_range")
    if isinstance(hr, list) and len(hr) == 2:
        try:
            lo, hi = int(hr[0]), int(hr[1])
            if 0 <= lo < hi <= 24:
                out["hour_range"] = [lo, hi]
        except (TypeError, ValueError):
            pass
    pt = d.get("point")
    pt = pt[0] if isinstance(pt, list) and pt else pt
    try:
        if pt not in (None, ""):
            out["point"] = float(pt)
    except (TypeError, ValueError):
        pass
    return out
