"""Build eval v4 R-set: realistic forum-style queries with reference labels.

Stages (outputs under data/v4/eval/):
  instantiate  fill {老師}/{課} placeholders with real 114-2 (teacher, course) pairs
               that have past student evaluations, so review lookups are testable
               and named-course queries get an exact target   -> r_queries.jsonl
  reference    reference decomposition by an independent model family
               (gpt-oss:120b on Ollama Cloud, neither Gemma nor Qwen)  -> r_queries.jsonl

  python -m scripts.v4.build_eval_v4 instantiate
  python -m scripts.v4.build_eval_v4 reference
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from src.llm.ollama_http import chat
from src.loader import load_courses
from src.parsers.ollama_parser import sanitize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "v4" / "rset"
REF_MODEL = "gpt-oss:120b-cloud"


def _norm(name: str) -> str:
    return re.sub(r"[（(].*?[)）]|\s", "", name or "")


def course_families() -> dict[tuple[str, str], dict]:
    """(normalised name, teacher) for 114-2 -> {ids, n_reviews} using past RATE rows."""
    con = sqlite3.connect(ROOT / "data/1142.db")
    past: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cid, name, teacher in con.execute(
            "SELECT DISTINCT substr(id,1,13), name, teacher FROM COURSE WHERE NOT (y='114' AND s='2')"):
        past[(_norm(name), teacher)].add(cid)
    n_rate: dict[str, int] = defaultdict(int)
    for (cid,) in con.execute("SELECT courseId FROM RATE"):
        n_rate[cid[:13]] += 1
    fam: dict[tuple[str, str], dict] = {}
    for c in load_courses():
        key = (_norm(c.name), c.teacher)
        f = fam.setdefault(key, {"name": c.name, "teacher": c.teacher, "ids": [], "n_reviews": 0})
        f["ids"].append(c.course_id)
    for key, f in fam.items():
        f["n_reviews"] = sum(n_rate.get(p, 0) for p in past.get(key, ()))
    return fam


def stage_instantiate(seed: int) -> None:
    rng = random.Random(seed)
    fam = course_families()
    # single-teacher courses with enough history; skip generic language/PE sections
    pool = [f for f in fam.values() if f["n_reviews"] >= 10 and "、" not in f["teacher"]
            and not re.search(r"體育|大學英文|英文（|服務學習", f["name"])]
    rng.shuffle(pool)
    out, k = [], 0
    for i, line in enumerate((ROOT / "data/v4/query_bank_forum.jsonl").open(encoding="utf-8")):
        r = json.loads(line)
        q = r["query"]
        target = None
        if "{" in q:
            f, g = pool[k % len(pool)], pool[(k + 1) % len(pool)]
            k += 2
            q = (q.replace("{老師A}", f["teacher"]).replace("{老師B}", g["teacher"])
                  .replace("{老師}", f["teacher"]).replace("{課}", f["name"]))
            target = {"name": f["name"], "teacher": f["teacher"] if "老師" in r["query"] else None,
                      "ids": f["ids"], "n_reviews": f["n_reviews"]}
        out.append({"qid": f"r{i:03d}", "query": q, "template": r["query"], "need_tags": r["need"],
                    "school": r["school"], "target": target})
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "r_queries.jsonl", out)
    print(len(out), "queries;", sum(o["target"] is not None for o in out), "with a named target")


_REF_PROMPT = """你是選課查詢分析專家。分析下面這句政大學生的查詢，輸出 JSON：
- need：想學的主題或內容（去掉條件、評價、老師名、課名後剩下的主題）；沒有主題就是空字串。
- named_course：查詢中明確提到的課名，沒有就 null。
- named_teacher：查詢中明確提到的老師姓名，沒有就 null。
- must / must_not：硬條件，只能用 course_lang（授課語言，想學某語言不算）、weekday（1–7）、hour_range（[開始,結束] 時）、point、kind（必修/選修/通識/體育）、lmt_kind（通識類別）、unit（系所簡稱）。
- prefs：軟性偏好列表，每項 {"aspect": 下列之一, "want": "low" 或 "high"}；aspect 可為 workload（作業報告考試等負擔）、grading（給分甜度）、difficulty（難度）、attendance（點名）、group_work（分組）、teaching（老師教學與人格）、usefulness（實用性、對未來幫助）、popularity（熱門程度、選不選得上）。
- intents：此查詢要系統做的事，可多選：search（找課）、lookup（查某門課或某老師的資訊）、review（想知道評價）、eligibility（能不能修）、enroll_chance（選不選得上、志願序）、compare（比較多門課）、out_of_scope（與本學期課程查詢無關，例如加簽流程、考試日期）。

查詢：{query}"""

_REF_SCHEMA = {"type": "object", "properties": {
    "need": {"type": "string"}, "named_course": {"type": ["string", "null"]},
    "named_teacher": {"type": ["string", "null"]}, "must": {"type": "object"}, "must_not": {"type": "object"},
    "prefs": {"type": "array", "items": {"type": "object", "properties": {
        "aspect": {"type": "string"}, "want": {"type": "string"}}, "required": ["aspect", "want"]}},
    "intents": {"type": "array", "items": {"type": "string"}}},
    "required": ["need", "named_course", "named_teacher", "must", "must_not", "prefs", "intents"]}


def stage_reference() -> None:
    rows = [json.loads(x) for x in (OUT / "r_queries.jsonl").open(encoding="utf-8")]
    for r in rows:
        if r.get("ref"):
            continue
        res = chat(REF_MODEL, [{"role": "user", "content": _REF_PROMPT.replace("{query}", r["query"])}],
                   fmt=_REF_SCHEMA, think=False, timeout=300)
        try:
            d = json.loads(res.content)
        except json.JSONDecodeError:
            print("bad json", r["qid"], res.error, res.content[:120])
            continue
        d["must"], d["must_not"] = sanitize(d.get("must")), sanitize(d.get("must_not"))
        r["ref"] = d
        _write(OUT / "r_queries.jsonl", rows)  # checkpoint: cloud calls are slow and rate-limited
    print("referenced", sum(bool(r.get("ref")) for r in rows), "/", len(rows))


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["instantiate", "reference"])
    p.add_argument("--seed", type=int, default=20261018)
    a = p.parse_args()
    if a.stage == "instantiate":
        stage_instantiate(a.seed)
    else:
        stage_reference()


if __name__ == "__main__":
    main()
