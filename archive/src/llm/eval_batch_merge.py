"""Merge OpenAI + Anthropic batch results into eval_synth.jsonl.

Output format:
  {"qid": "...", "query": "...", "type": "topic|constraint|colloquial",
   "source": "openai|anthropic", "gold_course_id": "1142..."}

Dedupes near-identical queries via normalised string match per gold_course_id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "batches"
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_PATH = BATCH_DIR / "openai_eval.results.jsonl"
ANTHROPIC_PATH = BATCH_DIR / "anthropic_eval.results.jsonl"
OUT_PATH = DATA_DIR / "eval_synth.jsonl"


def _norm(q: str) -> str:
    return re.sub(r"\s+", "", q).lower()


def _qid(course_id: str, source: str, qtype: str, query: str) -> str:
    h = hashlib.md5(f"{source}|{qtype}|{query}".encode()).hexdigest()[:8]
    return f"{course_id}-{source[:1]}{qtype[:1]}-{h}"


def _parse_openai(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        print(f"[merge] skip openai: {path} missing")
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        rec = json.loads(line)
        course_id = rec["custom_id"]
        try:
            content = rec["response"]["body"]["choices"][0]["message"]["content"]
            obj = json.loads(content)
        except (KeyError, json.JSONDecodeError):
            continue
        for qtype in ("topic", "constraint", "colloquial"):
            q = (obj.get(qtype) or "").strip()
            if not q:
                continue
            out.append(
                {
                    "qid": _qid(course_id, "openai", qtype, q),
                    "query": q,
                    "type": qtype,
                    "source": "openai",
                    "gold_course_id": course_id,
                }
            )
    return out


def _parse_anthropic(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        print(f"[merge] skip anthropic: {path} missing")
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        rec = json.loads(line)
        course_id = rec["custom_id"]
        try:
            res = rec["result"]
            if res["type"] != "succeeded":
                continue
            blocks = res["message"]["content"]
            text = "".join(b["text"] for b in blocks if b["type"] == "text")
            obj = json.loads(text)
        except (KeyError, json.JSONDecodeError):
            continue
        for qtype in ("topic", "constraint", "colloquial"):
            q = (obj.get(qtype) or "").strip()
            if not q:
                continue
            out.append(
                {
                    "qid": _qid(course_id, "anthropic", qtype, q),
                    "query": q,
                    "type": qtype,
                    "source": "anthropic",
                    "gold_course_id": course_id,
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args(argv)

    rows = _parse_openai(OPENAI_PATH) + _parse_anthropic(ANTHROPIC_PATH)
    print(f"[merge] raw rows = {len(rows)}")

    # dedupe per (gold_course_id, normalised query)
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in rows:
        key = (r["gold_course_id"], _norm(r["query"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    type_dist = Counter(r["type"] for r in deduped)
    src_dist = Counter(r["source"] for r in deduped)
    course_cov = len({r["gold_course_id"] for r in deduped})
    avg_q = len(deduped) / course_cov if course_cov else 0
    print(f"[merge] deduped = {len(deduped)}  ({len(rows)-len(deduped)} duplicates)")
    print(f"[merge] type_dist = {dict(type_dist)}")
    print(f"[merge] source_dist = {dict(src_dist)}")
    print(f"[merge] courses_covered = {course_cov}  avg_q_per_course = {avg_q:.2f}")
    print(f"[merge] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
