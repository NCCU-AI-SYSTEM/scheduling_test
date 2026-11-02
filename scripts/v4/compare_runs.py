"""How far do two run files agree before any judging? Top-10 overlap and identical top-1.

  python -m scripts.v4.compare_runs v4s_g12b_merge_namedfix pg_whitespace_g12b pg_ngram_g12b ...

The first name is the reference; latency columns come from each run's meta file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "v4" / "e2e"


def _runs(name: str) -> dict[str, list[str]]:
    return {json.loads(x)["qid"]: json.loads(x)["ranked"] for x in (OUT / f"{name}.runs.jsonl").open(encoding="utf-8")}


def main() -> None:
    ref_name, others = sys.argv[1], sys.argv[2:]
    ref = _runs(ref_name)
    print(f"{'run':36s} {'top10 overlap':>13s} {'same top1':>9s} {'identical':>9s} {'filter ms':>9s} "
          f"{'retrieve ms':>11s} {'search ms':>9s}")
    for name in [ref_name] + others:
        run = _runs(name)
        qs = [q for q in ref if q in run]
        ov = sum(len(set(ref[q][:10]) & set(run[q][:10])) / max(1, len(set(ref[q][:10]) | set(run[q][:10])))
                 for q in qs) / len(qs)
        top1 = sum(bool(ref[q]) and bool(run[q]) and ref[q][0] == run[q][0] for q in qs) / len(qs)
        ident = sum(ref[q][:10] == run[q][:10] for q in qs) / len(qs)
        meta = json.loads((OUT / f"{name}.meta.json").read_text())
        print(f"{name:36s} {ov:13.3f} {top1:9.3f} {ident:9.3f} {meta.get('filter_ms_p50', float('nan')):9} "
              f"{meta.get('retrieve_ms_p50', float('nan')):11} {meta.get('search_ms_p50', float('nan')):9}")


if __name__ == "__main__":
    main()
