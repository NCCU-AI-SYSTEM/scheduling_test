"""Soft constraint scoring: turn per-clause pass/fail into a ranking multiplier.

Hard pre-filtering deletes every course that fails any parsed clause. That is
exact when the parse is right, and unrecoverable when it is wrong (the LLM
parser's exact-match rate is ~68%). Soft scoring keeps every candidate and
multiplies its relevance score by a factor that shrinks with each violated
clause, so a mis-parsed clause demotes the gold course instead of deleting it.

Inputs come from `src.eval.conditions.check(meta, cond)`, e.g.
    {"must:course_lang": True, "must:weekday": False, "not:hour_range": True}
"""

from __future__ import annotations

# How much one violated clause costs, per dimension. Dimensions the parser gets
# right almost always (weekday/hour/point F1 >= 0.99) can be near-hard; `unit`
# (F1 0.89, messy class-name strings) and `kind` (0.95) deserve more slack.
PARSER_F1 = {
    "course_lang": 0.985, "weekday": 0.999, "hour_range": 0.993, "point": 0.999,
    "kind": 0.955, "lmt_kind": 0.986, "unit": 0.893,
}


def soft_factor(verdicts: dict[str, bool]) -> float:
    """Return a multiplier in (0, 1] for a course given its clause verdicts.

    TODO(user): decide the penalty shape. Trade-offs:
      - constant epsilon per violation (e.g. 0.05 ** n_violations): simple,
        close to a hard filter, still lets an all-violating list fill top-k
      - per-dimension slack from PARSER_F1: trusts reliable dims, forgives
        noisy ones (unit); a violated weekday should hurt more than a violated unit
      - must vs not: a violated negation ("不要早八" but the course is at 8:00)
        is what students notice first — should it cost more than a missed must?
    The score being multiplied is a reranker probability in [0, 1].
    """
    raise NotImplementedError
