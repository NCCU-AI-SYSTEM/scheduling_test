"""Quick eval-set sanity dataset.

Until the LLM batch results land, we synthesise a tiny in-memory eval set
straight from `Course.objective`: pull the first sentence and use it as the
query whose gold answer is the course itself. This is cheap and lets us
verify the harness end-to-end. Also reads `data/raw/eval_synth.jsonl` if
present (real LLM-generated set from W3 batches).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from src.loader import Course

ROOT = Path(__file__).resolve().parents[2]
EVAL_SYNTH = ROOT / "data" / "raw" / "eval_synth.jsonl"


@dataclass(slots=True)
class EvalQuery:
    qid: str
    query: str
    gold: set[str]
    qtype: str = "topic"
    source: str = "synthetic"


_SENT_BREAK = re.compile(r"[。！？!?\n]+")


def _first_sentence(text: str, min_len: int = 8, max_len: int = 80) -> str | None:
    if not text:
        return None
    for sent in _SENT_BREAK.split(text):
        sent = sent.strip()
        if min_len <= len(sent) <= max_len:
            return sent
    return None


def from_objective(courses: list[Course], n: int = 100, seed: int = 42) -> list[EvalQuery]:
    """Cheap pseudo-eval: first sentence of objective -> query, gold = that course."""
    rng = random.Random(seed)
    pool = [c for c in courses if c.objective]
    rng.shuffle(pool)
    out: list[EvalQuery] = []
    for c in pool:
        s = _first_sentence(c.objective)
        if not s:
            continue
        out.append(
            EvalQuery(
                qid=f"smoke-{c.course_id}",
                query=s,
                gold={c.course_id},
                qtype="topic",
                source="objective_smoke",
            )
        )
        if len(out) >= n:
            break
    return out


def from_jsonl(path: Path = EVAL_SYNTH) -> list[EvalQuery]:
    if not path.exists():
        return []
    out: list[EvalQuery] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gold = set(r.get("gold") or [r.get("gold_course_id")])
            out.append(
                EvalQuery(
                    qid=r["qid"],
                    query=r["query"],
                    gold=gold,
                    qtype=r.get("type", "topic"),
                    source=r.get("source", "synth"),
                )
            )
    return out
