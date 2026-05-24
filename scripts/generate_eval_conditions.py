"""
生成 eval_conditions_v1.jsonl（1500 筆）

從 DB 隨機挑 gold 課程 → 決定 must/must_not 條件 → 用 LLM 生成 zh+en query。
每筆格式：
  {qid, queries:{zh,en}, gold_course_id, must, must_not, should,
   condition_meta, n_must, n_must_not, n_should, has_negation, complexity, source}

使用：
  uv run python scripts/generate_eval_conditions.py
  uv run python scripts/generate_eval_conditions.py --dry-run     # 只印前5筆，不打API
  uv run python scripts/generate_eval_conditions.py --resume      # 從上次中斷繼續
  uv run python scripts/generate_eval_conditions.py --workers 10  # 並行數（預設10）
  uv run python scripts/generate_eval_conditions.py --model claude-haiku-4-5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sqlite3
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.loader.time_parser import parse_time_str

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Trend Micro endpoint ──────────────────────────────────────────────────────
def _make_client() -> openai.OpenAI:
    auth_path = Path.home() / ".hermes" / "auth.json"
    with open(auth_path) as f:
        auth = json.load(f)
    creds = auth["credential_pool"]["custom:api.rdsec.trendmicro.com"]
    # priority=1 (model_config)
    cred = next(c for c in creds if c["priority"] == 1)
    return openai.OpenAI(base_url=cred["base_url"], api_key=cred["access_token"])

MODEL = "gpt-5.4-mini"  # 快又聰明

# ── DB helpers ────────────────────────────────────────────────────────────────
WEEKDAY_LABEL = {1:"一",2:"二",3:"三",4:"四",5:"五",6:"六",7:"日"}
KIND_LABEL     = {1:"必修",2:"選修",3:"通識",4:"體育",0:"其他"}

def load_courses(db_path: Path) -> list[dict]:
    con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM COURSE WHERE y='114' AND s='2'").fetchall()
    courses = []
    for r in rows:
        sessions = parse_time_str(r["time"] or "")
        weekdays = sorted(set(s.weekday for s in sessions))
        hours    = [(s.start_hour, s.end_hour) for s in sessions]
        courses.append({
            "id":       r["id"],
            "name":     r["name"] or "",
            "lang":     r["lang"] or "",
            "point":    float(r["point"]) if r["point"] else 0.0,
            "kind":     r["kind"] or 0,
            "lmt_kind": r["lmtKind"] or "",
            "unit":     r["unit"] or "",
            "time_raw": r["time"] or "",
            "weekdays": weekdays,
            "hours":    hours,   # [(start,end),...]
            "has_morning":   any(s < 12 for s,_ in hours),
            "has_noon":      any(s >= 12 and s < 13 for s,_ in hours),
            "has_afternoon": any(s >= 13 and s < 18 for s,_ in hours),
            "has_evening":   any(s >= 18 for s,_ in hours),
        })
    con.close()
    return courses

# ── 分布設計表 ────────────────────────────────────────────────────────────────
# 每個 slot = (must_dims, must_not_dims, count)
# dims = list of dimension names
DISTRIBUTION: list[tuple[list[str], list[str], int]] = [
    # ── must 單維度 (350) ──
    (["course_lang"],  [],                  60),
    (["weekday"],      [],                  60),
    (["hour_range"],   [],                  60),
    (["point"],        [],                  50),
    (["kind"],         [],                  50),
    (["lmt_kind"],     [],                  30),
    (["unit"],         [],                  40),
    # ── must 兩維度 (350) ──
    (["course_lang","weekday"],      [],    50),
    (["course_lang","hour_range"],   [],    50),
    (["course_lang","point"],        [],    40),
    (["course_lang","kind"],         [],    40),
    (["course_lang","unit"],         [],    40),
    (["weekday","hour_range"],       [],    50),
    (["point","kind"],               [],    40),
    (["unit","point"],               [],    40),
    # ── must 三維度以上 (200) ──
    (["course_lang","weekday","hour_range"], [],  60),
    (["course_lang","unit","point"],         [],  50),
    (["course_lang","kind","lmt_kind"],      [],  40),
    (["weekday","hour_range","course_lang"], [],  50),
    # ── must_not 單維度 (150) ──
    ([],  ["course_lang"],   30),
    ([],  ["weekday"],       30),
    ([],  ["hour_range"],    30),
    ([],  ["point"],         20),
    ([],  ["kind"],          20),
    ([],  ["lmt_kind"],      20),
    # ── must + must_not 混合 (350) ──
    (["course_lang"],          ["weekday"],      40),
    (["course_lang"],          ["hour_range"],   40),
    (["unit"],                 ["course_lang"],  30),
    (["weekday"],              ["hour_range"],   30),
    (["kind"],                 ["course_lang"],  30),
    (["course_lang","kind"],   ["weekday"],      30),
    (["course_lang","weekday"],["hour_range"],   30),
    (["unit","point"],         ["kind"],         25),
    (["course_lang","unit"],   ["weekday","kind"], 25),
    (["weekday","course_lang"],["hour_range","kind"], 20),
    (["course_lang","hour_range","kind"], ["weekday"], 20),
    (["unit","course_lang","weekday"],    ["hour_range"], 20),
    (["course_lang","weekday","kind"],    ["hour_range","lmt_kind"], 10),
]

TOTAL_PLANNED = sum(c for _,_,c in DISTRIBUTION)

# ── 挑 gold 課程（符合條件）─────────────────────────────────────────────────
SLOT_MAP = {"morning":[8,12], "noon":[12,13], "afternoon":[13,18], "evening":[18,24]}

def _courses_matching(courses: list[dict], must: list[str], must_not: list[str]) -> list[dict]:
    """找出可當 gold 的課（有對應的欄位值）。"""
    pool = courses[:]
    # must 維度：課程在該維度必須有值
    for dim in must:
        if dim == "course_lang":
            pool = [c for c in pool if c["lang"]]
        elif dim == "weekday":
            pool = [c for c in pool if c["weekdays"]]
        elif dim == "hour_range":
            pool = [c for c in pool if c["hours"]]
        elif dim == "point":
            pool = [c for c in pool if c["point"] in (1.0,2.0,3.0,4.0)]
        elif dim == "kind":
            pool = [c for c in pool if c["kind"] in (1,2,3,4)]
        elif dim == "lmt_kind":
            pool = [c for c in pool if c["lmt_kind"]]
        elif dim == "unit":
            pool = [c for c in pool if c["unit"]]
    # must_not 維度：課程在該維度也要有值（才能構成有意義的排除）
    for dim in must_not:
        if dim == "course_lang":
            pool = [c for c in pool if c["lang"]]
        elif dim == "weekday":
            pool = [c for c in pool if c["weekdays"]]
        elif dim == "hour_range":
            pool = [c for c in pool if c["hours"]]
        elif dim == "point":
            pool = [c for c in pool if c["point"] in (1.0,2.0,3.0,4.0)]
        elif dim == "kind":
            pool = [c for c in pool if c["kind"] in (1,2,3,4)]
        elif dim == "lmt_kind":
            pool = [c for c in pool if c["lmt_kind"]]
        elif dim == "unit":
            pool = [c for c in pool if c["unit"]]
    return pool

# ── 從 gold 課程的欄位值填 conditions ────────────────────────────────────────
def _build_conditions(course: dict, must_dims: list[str], must_not_dims: list[str]) -> dict:
    """
    從 gold 課程直接取欄位值填入 must / must_not。
    must 的值 = gold 課程的實際值（這樣 gold 一定符合 filter）。
    must_not 的值 = 跟 gold 不同的值（這樣 filter 不會誤殺 gold）。
    """
    must: dict[str,Any]     = {}
    must_not: dict[str,Any] = {}
    meta: dict[str,str]     = {}

    for dim in must_dims:
        meta[dim] = "explicit"
        if dim == "course_lang":
            must["course_lang"] = [course["lang"]]
        elif dim == "weekday":
            # 隨機選 gold 有的 weekday 子集（1到全部）
            wds = course["weekdays"]
            n   = random.randint(1, max(1, len(wds)))
            must["weekday"] = sorted(random.sample(wds, n))
        elif dim == "hour_range":
            # 選 gold 有課的時段
            slots = []
            if course["has_morning"]:   slots.append("morning")
            if course["has_noon"]:      slots.append("noon")
            if course["has_afternoon"]: slots.append("afternoon")
            if course["has_evening"]:   slots.append("evening")
            chosen = random.choice(slots) if slots else "afternoon"
            must["hour_range"] = SLOT_MAP[chosen]
        elif dim == "point":
            must["point"] = course["point"]
        elif dim == "kind":
            must["kind"] = [KIND_LABEL.get(course["kind"], "選修")]
        elif dim == "lmt_kind":
            must["lmt_kind"] = [course["lmt_kind"]]
        elif dim == "unit":
            must["unit"] = [course["unit"]]

    for dim in must_not_dims:
        meta[dim] = meta.get(dim, "explicit")
        if dim == "course_lang":
            all_langs = ["中文","英文","日文","韓文","法文","德文"]
            others = [l for l in all_langs if l != course["lang"]]
            must_not["course_lang"] = [random.choice(others)] if others else ["英文"]
        elif dim == "weekday":
            all_wds = [1,2,3,4,5,6,7]
            others  = [w for w in all_wds if w not in course["weekdays"]]
            if not others:
                others = course["weekdays"][:1]  # 邊界：所有星期都有課，就挑一個當排除
            n = random.randint(1, min(2, len(others)))
            must_not["weekday"] = sorted(random.sample(others, n))
        elif dim == "hour_range":
            # 選一個 gold 沒有的時段排除
            all_slots = {"morning":[8,12],"noon":[12,13],"afternoon":[13,18],"evening":[18,24]}
            gold_slots = set()
            if course["has_morning"]:   gold_slots.add("morning")
            if course["has_noon"]:      gold_slots.add("noon")
            if course["has_afternoon"]: gold_slots.add("afternoon")
            if course["has_evening"]:   gold_slots.add("evening")
            others = [s for s in all_slots if s not in gold_slots]
            chosen = random.choice(others) if others else "morning"
            must_not["hour_range"] = all_slots[chosen]
        elif dim == "point":
            all_pts = [1.0,2.0,3.0,4.0]
            others  = [p for p in all_pts if p != course["point"]]
            must_not["point"] = random.choice(others) if others else 1.0
        elif dim == "kind":
            all_kinds = ["必修","選修","通識","體育"]
            gold_kind = KIND_LABEL.get(course["kind"],"選修")
            others    = [k for k in all_kinds if k != gold_kind]
            must_not["kind"] = [random.choice(others)] if others else ["必修"]
        elif dim == "lmt_kind":
            all_lmts = ["外文通識","社會通識","人文通識","自然通識","資訊通識","書院通識","中文通識"]
            others   = [l for l in all_lmts if l != course["lmt_kind"]]
            must_not["lmt_kind"] = [random.choice(others)] if others else ["外文通識"]
        elif dim == "unit":
            pass  # unit 負向不做

    return {"must": must, "must_not": must_not, "meta": meta}

# ── LLM prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一個選課系統的測試資料生成器。
根據給定的課程條件，生成一個自然的繁體中文查詢句子（zh）和英文查詢句子（en）。

規則：
- 查詢要像真實學生在問，口語自然，不要像在列清單
- must 條件要在 zh/en 中以正向方式表達
- must_not 條件要在 zh/en 中以否定方式表達（不要、非、排除等）
- 不要提到課程名稱（管理學、統計學等），那是答案
- 長度 10-40 個中文字
- 直接輸出 JSON：{"zh":"...","en":"..."}"""

def _cond_to_text(must: dict, must_not: dict, course: dict) -> str:
    lines = [f"gold課程：{course['name']}（{course['unit']}）"]
    if must:
        parts = []
        if "course_lang" in must:
            parts.append(f"授課語言={must['course_lang']}")
        if "weekday" in must:
            wds = [f"星期{WEEKDAY_LABEL[w]}" for w in must["weekday"]]
            parts.append(f"星期={'、'.join(wds)}")
        if "hour_range" in must:
            hr = must["hour_range"]
            slot = next((k for k,v in SLOT_MAP.items() if v==hr), f"{hr[0]}-{hr[1]}點")
            parts.append(f"時段={slot}（{hr[0]}:00-{hr[1]}:00）")
        if "point" in must:
            parts.append(f"學分={must['point']}")
        if "kind" in must:
            parts.append(f"課程類型={must['kind']}")
        if "lmt_kind" in must:
            parts.append(f"通識類型={must['lmt_kind']}")
        if "unit" in must:
            parts.append(f"系所={must['unit']}")
        lines.append("must（必須）：" + "，".join(parts))
    if must_not:
        parts = []
        if "course_lang" in must_not:
            parts.append(f"不要授課語言={must_not['course_lang']}")
        if "weekday" in must_not:
            wds = [f"星期{WEEKDAY_LABEL[w]}" for w in must_not["weekday"]]
            parts.append(f"不要={'、'.join(wds)}")
        if "hour_range" in must_not:
            hr = must_not["hour_range"]
            slot = next((k for k,v in SLOT_MAP.items() if v==hr), f"{hr[0]}-{hr[1]}點")
            parts.append(f"不要時段={slot}（{hr[0]}:00-{hr[1]}:00）")
        if "point" in must_not:
            parts.append(f"不要學分={must_not['point']}")
        if "kind" in must_not:
            parts.append(f"不要課程類型={must_not['kind']}")
        if "lmt_kind" in must_not:
            parts.append(f"不要通識類型={must_not['lmt_kind']}")
        lines.append("must_not（排除）：" + "，".join(parts))
    return "\n".join(lines)

def _call_llm(client: openai.OpenAI, prompt: str, model: str = MODEL, retry: int = 3) -> dict | None:
    for attempt in range(retry):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=300,
            )
            raw = r.choices[0].message.content or ""
            m   = re.search(r'\{.*?\}', raw, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                if obj.get("zh") and obj.get("en"):
                    return obj
        except Exception as e:
            log.warning(f"LLM error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None

# ── 主程式 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="不打 API，只印前 5 筆結構")
    parser.add_argument("--resume",   action="store_true", help="從上次中斷繼續")
    parser.add_argument("--workers",  type=int, default=10, help="並行 thread 數（預設 10）")
    parser.add_argument("--model",    default=MODEL, help=f"LLM model（預設 {MODEL}）")
    parser.add_argument("--out", default=str(ROOT/"data/raw/eval_conditions_v1.jsonl"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 已生成的 qid
    done_qids: set[str] = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_qids.add(json.loads(line)["qid"])
                except Exception:
                    pass
        log.info(f"Resume: {len(done_qids)} already done")

    courses = load_courses(ROOT / "data/1142.db")
    log.info(f"Loaded {len(courses)} courses from DB")

    client = None if args.dry_run else _make_client()
    model  = args.model

    rng = random.Random(42)

    # ── 先展開所有要生成的任務清單 ─────────────────────────────────────────────
    tasks: list[dict] = []
    for must_dims, must_not_dims, count in DISTRIBUTION:
        pool = _courses_matching(courses, must_dims, must_not_dims)
        if not pool:
            log.warning(f"No courses for must={must_dims} must_not={must_not_dims}, skip")
            continue
        generated = 0
        attempts  = 0
        while generated < count and attempts < count * 5:
            attempts += 1
            course = rng.choice(pool)
            cond   = _build_conditions(course, must_dims, must_not_dims)
            must, must_not, meta = cond["must"], cond["must_not"], cond["meta"]
            key = f"{course['id']}-{'+'.join(sorted(must))}-{'+'.join(sorted(must_not))}"
            qid = "cond-" + hashlib.md5(key.encode()).hexdigest()[:8]
            if qid in done_qids:
                continue
            if any(t["qid"] == qid for t in tasks):
                continue
            tasks.append({
                "qid": qid, "course": course,
                "must": must, "must_not": must_not, "meta": meta,
            })
            generated += 1

    log.info(f"Tasks to generate: {len(tasks)} (already done: {len(done_qids)})")

    if args.dry_run:
        for t in tasks[:5]:
            prompt = _cond_to_text(t["must"], t["must_not"], t["course"])
            print(f"\n[{t['qid']}]")
            print(f"  gold: {t['course']['name']} ({t['course']['unit']})")
            print(f"  must: {t['must']}")
            print(f"  must_not: {t['must_not']}")
            print(f"  prompt:\n    " + prompt.replace("\n", "\n    "))
        return

    # ── 並行生成 ─────────────────────────────────────────────────────────────
    write_lock = threading.Lock()
    counter    = {"written": 0, "failed": 0}

    def process_task(t: dict) -> dict | None:
        prompt  = _cond_to_text(t["must"], t["must_not"], t["course"])
        queries = _call_llm(client, prompt, model=model)  # type: ignore[arg-type]
        if not queries:
            return None

        must, must_not = t["must"], t["must_not"]
        n_must     = len(must)
        n_must_not = len(must_not)
        complexity = ("single" if (n_must + n_must_not) == 1
                      else "multi" if (n_must + n_must_not) <= 3
                      else "complex")
        return {
            "qid":            t["qid"],
            "queries":        {"zh": queries["zh"], "en": queries["en"]},
            "gold_course_id": t["course"]["id"],
            "must":           must,
            "must_not":       must_not,
            "should":         {},
            "condition_meta": t["meta"],
            "n_must":         n_must,
            "n_must_not":     n_must_not,
            "n_should":       0,
            "has_negation":   bool(must_not),
            "complexity":     complexity,
            "source":         "generated-v2",
        }

    mode = "a" if args.resume else "w"
    with open(out_path, mode) as fout:
        # 分批 submit，避免一次 submit 過多 request 被 rate limit throttle
        BATCH = 50
        for batch_start in range(0, len(tasks), BATCH):
            batch = tasks[batch_start: batch_start + BATCH]
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_task, t): t for t in batch}
                for future in as_completed(futures):
                    result = future.result()
                    if result is None:
                        with write_lock:
                            counter["failed"] += 1
                        log.warning(f"Failed: {futures[future]['qid']}")
                        continue
                    with write_lock:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                        counter["written"] += 1
                        n = counter["written"]
                        if n % 50 == 0:
                            log.info(f"Progress: {n}/{len(tasks)} written, "
                                     f"{counter['failed']} failed")

    total = counter["written"]
    log.info(f"Done: {total} written, {counter['failed']} failed")
    log.info(f"Output: {out_path}")

if __name__ == "__main__":
    main()
