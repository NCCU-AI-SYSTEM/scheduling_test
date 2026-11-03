"""Analyze E6 (RRF + Rerank + SmartFilter + BM25 expansion) results.

Compares E6 vs E5 (no expansion) on:
  - overall R@10
  - intent slice (topic / colloquial / constraint)
  - constraint subset cleaned (using eval_constraint_clean.jsonl)
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
RUNS_DIR   = ROOT / "results" / "runs"
CLEAN_PATH = ROOT / "data" / "raw" / "eval_constraint_clean.jsonl"

E5_RUN = RUNS_DIR / "dv2_rrf_rerank_smart_struct_synth.jsonl"
E6_RUN = RUNS_DIR / "dv2_rrf_rerank_smartfilter_expand_synth.jsonl"


def load_run(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def hit(r, k=10):
    return bool(set(r["gold"]) & set(r["retrieved"][:k]))


def main():
    if not E6_RUN.exists():
        print(f"❌ E6 not found: {E6_RUN}")
        return

    e5 = load_run(E5_RUN)
    e6 = load_run(E6_RUN)
    print(f"E5: {len(e5)} queries")
    print(f"E6: {len(e6)} queries")

    # overall + intent slice
    print("\n=== Overall + Intent Slice ===")
    print(f"{'metric':<25s} {'E5':>10s} {'E6':>10s} {'Δ':>8s}")
    for qtype in (None, "topic", "constraint", "colloquial"):
        sub5 = [r for r in e5.values() if qtype is None or r.get("qtype") == qtype]
        sub6 = [r for r in e6.values() if qtype is None or r.get("qtype") == qtype]
        h5 = sum(1 for r in sub5 if hit(r))
        h6 = sum(1 for r in sub6 if hit(r))
        n5 = len(sub5); n6 = len(sub6)
        r5 = h5 / n5 if n5 else 0
        r6 = h6 / n6 if n6 else 0
        label = qtype or "OVERALL"
        print(f"  {label:<23s} {r5:>10.4f} {r6:>10.4f} {(r6-r5)*100:>+7.1f}pp")

    # constraint clean subset
    clean_qids = set()
    with open(CLEAN_PATH) as f:
        for line in f:
            clean_qids.add(json.loads(line)["qid"])
    print(f"\n=== Constraint CLEAN subset (n={len(clean_qids)}) ===")
    sub5 = [r for r in e5.values() if r.get("qtype") == "constraint" and r["qid"] in clean_qids]
    sub6 = [r for r in e6.values() if r.get("qtype") == "constraint" and r["qid"] in clean_qids]
    h5 = sum(1 for r in sub5 if hit(r))
    h6 = sum(1 for r in sub6 if hit(r))
    print(f"  E5 R@10: {h5}/{len(sub5)} = {h5/len(sub5):.4f}")
    print(f"  E6 R@10: {h6}/{len(sub6)} = {h6/len(sub6):.4f}")
    print(f"  Δ: {(h6/len(sub6) - h5/len(sub5))*100:+.1f}pp")

    # rescued / broken on constraint clean
    rescued = []
    broken  = []
    common = set(r["qid"] for r in sub5) & set(r["qid"] for r in sub6)
    for qid in common:
        r5 = next(r for r in sub5 if r["qid"] == qid)
        r6 = next(r for r in sub6 if r["qid"] == qid)
        h5_q = hit(r5); h6_q = hit(r6)
        if h6_q and not h5_q:
            rescued.append(r6)
        elif h5_q and not h6_q:
            broken.append(r5)

    print(f"\n  Rescued (E5 miss → E6 hit): {len(rescued)}")
    for r in rescued[:5]:
        print(f"    Q: {r['query']}")
    print(f"\n  Broken (E5 hit → E6 miss):  {len(broken)}")
    for r in broken[:5]:
        print(f"    Q: {r['query']}")


if __name__ == "__main__":
    main()
