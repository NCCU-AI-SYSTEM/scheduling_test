"""In-memory evaluation of a ConditionResult against course metadata.

Mirrors the SQL semantics in `src/db/filter.py` so the same conditions can be
used three ways without PostgreSQL:
  1. eval  — constraint-satisfaction metrics (CSR@k) computed from the DB, no labels
  2. hard pre-filter — boolean mask over the corpus before ranking
  3. soft scoring — per-dimension pass/fail vector for fuzzy re-scoring

`meta` is the dict produced by `doc_builders._course_metadata`.
"""

from __future__ import annotations

from src.parsers.base import ConditionResult

# COURSE.kind codes, checked against 114-2 data (2026/10/18): 1 必修 (PE sections are
# kind 1 too), 2 選修, 3 群修, 4 通識, 0 interdisciplinary 通識 (lmtKind set). The
# earlier table {通識: 3, 體育: 4} was wrong and silently mapped 通識 to 群修.
def course_categories(meta: dict) -> set[str]:
    kind = meta.get("kind")
    cats = {{1: "必修", 2: "選修", 3: "群修", 4: "通識"}.get(kind, "")} - {""}
    if meta.get("lmt_kind") or kind == 0:
        cats.add("通識")
    if "體育" in (meta.get("name") or ""):
        cats.add("體育")
        cats.discard("必修")
    return cats

DIMENSIONS = ("course_lang", "weekday", "hour_range", "point", "kind", "lmt_kind", "unit")


def _overlaps(sessions: list[dict], lo: int, hi: int) -> bool:
    return any(s["start"] < hi and s["end"] > lo for s in sessions)


def _dim_match(meta: dict, dim: str, value) -> bool | None:
    """True if the course has `value` on `dim`; None if the condition is empty."""
    if value is None or (isinstance(value, list) and not value):
        return None
    if dim == "course_lang":
        return meta.get("lang") in value
    if dim == "weekday":
        days = {s["weekday"] for s in meta.get("sessions", [])}
        return bool(days & set(value))
    if dim == "hour_range":
        lo, hi = value[0], value[1]
        return _overlaps(meta.get("sessions", []), lo, hi)
    if dim == "point":
        return float(meta.get("point") or 0) == float(value)
    if dim == "kind":
        wanted = {k for k in value if k in ("必修", "選修", "群修", "通識", "體育")}
        return bool(course_categories(meta) & wanted) if wanted else None
    if dim == "lmt_kind":
        return (meta.get("lmt_kind") or "") in value
    if dim == "unit":
        unit = meta.get("unit") or ""
        return any(u and u in unit for u in value)
    raise ValueError(f"unknown dimension {dim}")


def check(meta: dict, cond: ConditionResult) -> dict[str, bool]:
    """Per-clause verdicts, keyed `must:<dim>` / `not:<dim>`. Empty clauses are skipped."""
    out: dict[str, bool] = {}
    for dim, value in cond.must.items():
        m = _dim_match(meta, dim, value)
        if m is not None:
            out[f"must:{dim}"] = m
    for dim, value in cond.must_not.items():
        m = _dim_match(meta, dim, value)
        if m is not None:
            out[f"not:{dim}"] = not m
    return out


def satisfies(meta: dict, cond: ConditionResult) -> bool:
    return all(check(meta, cond).values())


def n_clauses(cond: ConditionResult) -> int:
    def _count(d: dict) -> int:
        return sum(1 for v in d.values() if v is not None and v != [])

    return _count(cond.must) + _count(cond.must_not)
