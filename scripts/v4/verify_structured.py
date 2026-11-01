"""Check data/v4/course_struct.db against data/1142.db: coverage, precision, surprises, aliases.

  python -m scripts.v4.verify_structured            # coverage + surprises + precision from judgements
  python -m scripts.v4.verify_structured --samples  # also (re)write the fixed-seed precision samples

Writes data/v4/structured_audit/report.json and prints the tables used in
docs/v4/07_structured_data.md.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from src.v4.structured import aliaseval, audit, verify
from src.v4.structured.aliases import Alias, Resolver, load_catalogue
from src.v4.structured.build import OUT_DB, ROOT, SRC_DB

AUDIT = ROOT / "data" / "v4" / "structured_audit"
GE_ORDER = ["外文", "社會", "人文", "中文", "自然", "資訊", "跨領域", "書院", "核心通識", "非通識"]


def pct(hit: int, n: int) -> str:
    return f"{hit / n:.1%} ({hit}/{n})" if n else "-"


def _resolver(out: sqlite3.Connection, src: sqlite3.Connection) -> Resolver:
    cols = ["alias", "canonical", "members", "kind", "source", "evidence", "confidence", "in_1142", "note"]
    rows = [dict(zip(cols, r)) for r in out.execute(f"SELECT {', '.join(cols)} FROM alias")]
    aliases = [Alias(**{**r, "members": tuple(json.loads(r["members"])), "in_1142": bool(r["in_1142"])})
               for r in rows]
    return Resolver(aliases, load_catalogue(src))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="store_true")
    args = ap.parse_args()
    src = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    out = sqlite3.connect(f"file:{OUT_DB}?mode=ro", uri=True)
    raw, st = verify.load(src, out)
    cov = verify.coverage(raw, st)
    print("| 欄位 | 覆蓋率 | 說明 |\n|---|---|---|")
    for name, c in cov.items():
        print(f"| {name} | {pct(*c['all'])} | {c['note']} |")
    print("\n| 欄位 | " + " | ".join(GE_ORDER) + " |\n" + "|---" * (len(GE_ORDER) + 1) + "|")
    for name, c in cov.items():
        print(f"| {name} | " + " | ".join(pct(*c["ge"].get(g, (0, 0))) for g in GE_ORDER) + " |")
    colleges = sorted({k for c in cov.values() for k in c["college"]})
    print("\n| 欄位 | " + " | ".join(colleges) + " |\n" + "|---" * (len(colleges) + 1) + "|")
    for name, c in cov.items():
        print(f"| {name} | " + " | ".join(pct(*c["college"].get(k, (0, 0))) for k in colleges) + " |")

    sur = audit.surprises(out, raw)
    print("\nsurprises:", {k: len(v) for k, v in sur.items()})
    if args.samples:
        audit.write_samples(audit.samples(out, raw), AUDIT)
        print(f"samples written to {AUDIT}")
    prec = audit.precision(AUDIT)
    for k, v in prec.items():
        print(f"precision {k}: {pct(v['ok'], v['n'])}")
    al = aliaseval.evaluate(_resolver(out, src), out)
    for k, v in al["summary"].items():
        print(f"alias {k}: resolved {pct(v['resolved'], v['n'])}, in 114-2 {v['in_1142']}; unresolved {v['unresolved'][:12]}")
    aliaseval.save(al, AUDIT / "alias_eval.json")
    report = {"coverage": {k: {kk: vv for kk, vv in v.items() if kk != "misses"} | {"misses": v["misses"][:40]}
                           for k, v in cov.items()},
              "surprises": {k: v[:60] if isinstance(v, list) else v for k, v in sur.items()},
              "surprise_counts": {k: len(v) for k, v in sur.items()},
              "precision": prec, "alias": al["summary"]}
    (AUDIT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
