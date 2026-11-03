"""Print the retrieval component benchmark (latest entry per run name).

If rescore_v3 has judged the runs, its per-run JSON (results/v4/retrieval/<name>.json)
holds nDCG after judging; that is shown next to the pre-judging numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "v4" / "retrieval"


def main() -> None:
    latest: dict[str, dict] = {}
    for line in (RES / "summary.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        latest[d["name"]] = d
    print(f"{'run':58s} {'nDCG':>6s} {'R@50':>6s} {'T':>6s} {'M':>6s} {'judg':>5s} {'nDCG*':>6s} {'cost':>22s}")
    for name, d in latest.items():
        rescored = RES / f"{name}.json"
        nd2 = json.loads(rescored.read_text()).get("ALL", {}).get("ndcg") if rescored.exists() else None
        cost = " ".join(f"{k}={d[k]}" for k in ("ms_per_q", "gpu_ms_per_q", "cpu_ms_per_q", "doc_encode_s") if k in d)
        print(f"{name[:58]:58s} {d['ALL']['ndcg10']:6.3f} {d['ALL']['recall50'] or 0:6.3f} {d['T']['ndcg10']:6.3f} "
              f"{d['M']['ndcg10']:6.3f} {d['ALL']['judged10']:5.2f} {nd2 if nd2 is not None else float('nan'):6.3f} {cost:>22s}")


if __name__ == "__main__":
    main()
