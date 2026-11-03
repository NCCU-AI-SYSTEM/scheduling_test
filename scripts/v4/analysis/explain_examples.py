"""Walk real queries through each v4 stage and print what happens (for the write-up).

  python -m scripts.v4.analysis.explain_examples --planner gemma4:12b
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.eval.conditions import check as check_cond
from src.parsers.base import ConditionResult
from src.rerankers.models import load as load_reranker
from src.retrievers.encoders import SPECS, Encoder
from src.v4.access import StudentIdentity, judge
from src.v4.pipeline import Config, V4Search
from src.v4.planner import plan
from src.v4.present import row
from src.v4.struct_store import expand_aliases, load_struct

CONDITION_QUERIES = ["週三下午的英文授課通識，不要早八", "禮拜五晚上有什麼 2 學分的選修"]
PLANNER_QUERIES = ["想找跟資料分析有關、不用分組報告的課", "黃瀚萱的自然語言處理還有名額嗎", "會計系大二可以修的財務管理"]
NAMED_QUERIES = ["個經哪個老師比較好", "初會(一)有晚上的班嗎", "大英要選哪一班"]


def show_plan(q: str, p) -> None:
    print(f"\n  問句：{q}")
    print(f"  規劃：need={p.need!r} named_course={p.named_course!r} named_teacher={p.named_teacher!r}")
    print(f"        must={json.dumps(p.cond.must, ensure_ascii=False)} must_not={json.dumps(p.cond.must_not, ensure_ascii=False)}")
    print(f"        prefs={p.prefs} assess_must={p.assess_must} assess_not={p.assess_not}  ({p.seconds:.2f}s)")


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--planner", default="gemma4:12b")
    args = a.parse_args()
    b = load_struct(embedding_models=("e5-large-instruct",))
    s = V4Search(b.catalog, Encoder(SPECS["e5-large-instruct"]), load_reranker("qwen3-rerank-0.6b"), b.aspects,
                 offerings=b.offerings, doc_emb=b.embeddings["e5-large-instruct"], aliases=b.aliases, assess=b.assess)
    cmap = {c.course_id: c for c in b.catalog}
    cfg = Config(merge_offerings=True, named_skips_conditions=True)

    print("=== 1. 條件交給資料庫：每加一個條件剩幾門 ===")
    for q in CONDITION_QUERIES:
        p = plan(q, args.planner)
        show_plan(q, p)
        must, must_not = {}, {}
        print(f"    全部 {len(b.catalog)} 門")
        for side, d in (("must", p.cond.must), ("must_not", p.cond.must_not)):
            for dim, val in d.items():
                (must if side == "must" else must_not)[dim] = val
                c = ConditionResult(must=dict(must), must_not=dict(must_not))
                n = sum(all(check_cond(x.meta, c).values()) for x in b.catalog)
                print(f"    + {side} {dim}={val} -> {n} 門")
        hits = s.search(p, q, 5, cfg)
        print(f"    放寬：{hits[0].relaxed if hits else []}")
        for h in hits[:5]:
            r = row(h, cmap)
            print(f"    {r['課名']} | {r['老師']} | {r['時間']} | {r['學分']:g} 學分 | {cmap[h.course_id].course.lang}")

    print("\n=== 2. 規劃器 ===")
    for q in PLANNER_QUERIES:
        show_plan(q, plan(q, args.planner))

    print("\n=== 3. 指名比對與縮寫 ===")
    for q in NAMED_QUERIES:
        p = plan(q, args.planner)
        show_plan(q, p)
        print(f"    縮寫展開：{expand_aliases(p.named_course or q, b.aliases)[1]}  比對形式：{s._named_forms(p.named_course) if p.named_course else []}")
        for h in s.search(p, q, 5, cfg)[:5]:
            r = row(h, cmap)
            print(f"    {r['課名']} | {r['老師']} | {r['時間']} | 課號 {r['課號']}")

    print("\n=== 4. 同一堂課合併 ===")
    for name in ("分散式系統", "自然語言處理"):
        off = next(o for o in b.offerings if o.key[0] == name and len(o.channels) > 1)
        print(f"  {name} {off.key[1]} {off.key[2]}：{len(off.channels)} 個課號")
        for ch in off.channels:
            print(f"    {ch.sub_num} 對象 {ch.unit} 學制 {ch.level} 選課方式 {ch.enroll_mode}")
        for label, ident in (("資科學三", StudentIdentity("703", "資訊", "學士", 3)),
                             ("資碩一", StudentIdentity("753", "資訊", "碩士", 1)),
                             ("企管學一", StudentIdentity("304", "商", "學士", 1))):
            acc = judge(off, ident)
            print(f"    {label} -> 採用 {acc.channel.sub_num}，{'排除' if acc.excluded else '顯示'} {[f.detail for f in acc.flags]}")
    p = plan("想修分散式系統", args.planner)
    for merge in (False, True):
        hits = s.search(p, "想修分散式系統", 6, Config(merge_offerings=merge, named_skips_conditions=True))
        print(f"  合併={merge}: {[(cmap[h.course_id].course.name, cmap[h.course_id].course.sub_num) for h in hits]}")

    print("\n=== 5. 身分篩選：大學英文（二）===")
    eng = [o for o in b.offerings if o.key[0].startswith("大學英文")]
    for label, ident in (("資科學三", StudentIdentity("703", "資訊", "學士", 3)),
                         ("企管學一", StudentIdentity("304", "商", "學士", 1)),
                         ("僑生（法律學二）", StudentIdentity("601", "法", "學士", 2, frozenset({"僑生"})))):
        shown = [(o, judge(o, ident)) for o in eng]
        ok = [(o, acc) for o, acc in shown if not acc.excluded]
        print(f"  {label}: 顯示 {len(ok)}/{len(eng)} 班，時段 {sorted({o.key[2] for o, _ in ok})}")
    _ = np


if __name__ == "__main__":
    main()
