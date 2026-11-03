"""Can rules or small LLMs decide course eligibility from free-text restriction notes?

Cases: 114-2 courses whose note mentions a grade minimum, graduate-only, department-only,
prerequisite or no-repeat rule, crossed with synthetic student profiles. Reference
labels (yes / no / unsure) come from gpt-oss:120b, a model family the system does not
use. Systems compared: the regex rule engine (src/v4/eligibility.py) and LLMs reading
the note directly.

  python -m scripts.v4.analysis.eligibility_bench build --n-courses 60
  python -m scripts.v4.analysis.eligibility_bench score --models rules,gemma4:e4b,qwen3.5:9b,gemma4:31b-cloud
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.llm.ollama_http import chat, parse_json_object
from src.v4.catalog import load_catalog
from src.v4.eligibility import check
from src.v4.profile import StudentProfile

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "v4" / "eligibility"
REF = "gpt-oss:120b-cloud"

PROFILES = {
    "cs1": StudentProfile(major="資訊科學系", dept_abbr="資科", year=1, taken_names={"計算機程式設計"}),
    "cs3": StudentProfile(major="資訊科學系", dept_abbr="資科", year=3,
                          taken_names={"計算機程式設計", "資料結構", "微積分", "線性代數", "離散數學", "經濟學"}),
    "ba2": StudentProfile(major="企業管理學系", dept_abbr="企管", year=2,
                          taken_names={"初級會計學（一）", "初級會計學（二）", "經濟學", "微積分", "管理學", "統計學（一）"}),
    "law4": StudentProfile(major="法律學系", dept_abbr="法律", year=4,
                           taken_names={"民法總則", "刑法總則", "憲法", "債編總論", "行政法"}),
    "eng_m1": StudentProfile(major="英國語文學系碩士班", dept_abbr="英文", degree="碩士", year=1,
                             taken_names={"英語語言學概論", "英國文學史"}),
}


def describe(p: StudentProfile) -> str:
    return (f"{p.major}{p.degree}班{p.year}年級，已修過：{'、'.join(sorted(p.taken_names)) or '無'}")


PROMPT = """判斷這位政大學生能不能選這門課。只根據課程的開課對象與備註，以及學生資料判斷。
注意：開課對象（班級）只表示主要開給哪些學生，不代表其他學生不能選；只有備註明寫的限制才算。
- yes：備註沒有限制，或學生符合限制
- no：學生明確不符合限制（年級不夠、限研究生、僅限其他系所、未修過備註寫的先修／擋修課、已經修過同一門課）
- unsure：備註寫的限制不明確，或寫「優先」「建議」這種非強制條件，或需要問開課單位
輸出 JSON：{"answer": "yes|no|unsure", "reason": "20 字內"}

學生：{student}
課程：{name}
開課對象（班級）：{unit}
備註：{note}"""
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
                                           "reason": {"type": "string"}}, "required": ["answer", "reason"]}


def ask(model: str, case: dict) -> dict:
    text = PROMPT
    for key in ("student", "name", "unit", "note"):
        text = text.replace("{" + key + "}", str(case[key]))
    res = chat(model, [{"role": "user", "content": text}], fmt=SCHEMA, timeout=300)
    d = parse_json_object(res.content) or {"answer": "unsure", "reason": "parse_error"}
    return {"answer": d.get("answer", "unsure"), "reason": d.get("reason", ""), "seconds": round(res.seconds, 2),
            "error": res.error}


def stage_build(n_courses: int, seed: int) -> None:
    rng = random.Random(seed)
    pat = re.compile(r"限大[一二三四]|年級以上|限碩|限博士生|先修|擋修|須修過|(僅限|只限|限).{0,20}(系|所|學院|學程)|不得重複")
    pool = [c for c in load_catalog(with_reviews=False) if pat.search(c.course.note or "")]
    courses = rng.sample(pool, min(n_courses, len(pool)))
    cases = []
    for c in courses:
        for pid in rng.sample(sorted(PROFILES), 3):
            cases.append({"cid": c.course_id, "pid": pid, "student": describe(PROFILES[pid]),
                          "name": c.course.name, "unit": c.course.unit, "note": (c.course.note or "")[:600]})
    OUT.mkdir(parents=True, exist_ok=True)

    def ref(case):
        return case | {"ref": ask(REF, case)}

    with ThreadPoolExecutor(2) as ex:
        rows = list(tqdm(ex.map(ref, cases), total=len(cases)))
    with (OUT / "cases.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), Counter(r["ref"]["answer"] for r in rows))


def rules_answer(case: dict, cat: dict) -> str:
    flags = check(cat[case["cid"]], PROFILES[case["pid"]])
    if any(f.level == "block" for f in flags):
        return "no"
    if any(f.level == "warn" for f in flags):
        return "unsure"
    return "yes"


def stage_eval(models: list[str]) -> None:
    rows = [json.loads(x) for x in (OUT / "cases.jsonl").open(encoding="utf-8")]
    cat = {c.course_id: c for c in load_catalog(with_reviews=False)}
    summary = {}
    for m in models:
        if m.startswith("hybrid:"):
            # rules decide what they can; the LLM's saved answer fills in the cases rules mark unsure
            llm = [json.loads(x) for x in
                   (OUT / f"pred_{re.sub(r'[^A-Za-z0-9.]+', '_', m.split(':', 1)[1])}.jsonl").open(encoding="utf-8")]
            preds = []
            for r, lp in zip(rows, llm):
                ra = rules_answer(r, cat)
                preds.append({"answer": lp["answer"] if ra == "unsure" else ra,
                              "seconds": lp["seconds"] if ra == "unsure" else 0.0})
        elif m in ("rules", "rules_strict"):
            preds = [{"answer": rules_answer(r, cat), "seconds": 0.0} for r in rows]
            if m == "rules_strict":  # treat every warning as a block
                preds = [{"answer": "no" if p["answer"] == "unsure" else p["answer"], "seconds": 0.0}
                         for p in preds]
        else:
            with ThreadPoolExecutor(1 if not m.endswith("-cloud") else 2) as ex:
                preds = list(tqdm(ex.map(lambda r: ask(m, r), rows), total=len(rows), desc=m))
        gold = [r["ref"]["answer"] for r in rows]
        pred = [p["answer"] for p in preds]
        acc = sum(g == p for g, p in zip(gold, pred)) / len(rows)
        no_idx = [i for i, g in enumerate(gold) if g == "no"]
        yes_idx = [i for i, g in enumerate(gold) if g == "yes"]
        s = {"n": len(rows), "acc": round(acc, 3),
             "no_recall": round(sum(pred[i] == "no" for i in no_idx) / max(1, len(no_idx)), 3),
             "false_block": round(sum(pred[i] == "no" for i in yes_idx) / max(1, len(yes_idx)), 3),
             "unsure_rate": round(pred.count("unsure") / len(rows), 3),
             "p50_s": sorted(p["seconds"] for p in preds)[len(preds) // 2]}
        summary[m] = s
        print(m, json.dumps(s))
        with (OUT / f"pred_{re.sub(r'[^A-Za-z0-9.]+', '_', m)}.jsonl").open("w", encoding="utf-8") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"cid": r["cid"], "pid": r["pid"], "gold": r["ref"]["answer"]} | p,
                                   ensure_ascii=False) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "score"])
    p.add_argument("--n-courses", type=int, default=60)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--models", default="rules")
    a = p.parse_args()
    if a.stage == "build":
        stage_build(a.n_courses, a.seed)
    else:
        stage_eval(a.models.split(","))


if __name__ == "__main__":
    main()
