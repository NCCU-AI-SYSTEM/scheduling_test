"""Apply structured constraints from QueryConstraints to filter retrieval docs.

Filtering happens BEFORE BM25/dense retrieval (pre-filter, narrowing the index)
or AFTER (post-filter, ranking-aware). We support both modes; the runner picks
post-filter to keep BM25 stats stable across configurations.
"""

from __future__ import annotations

from src.doc_builders import RetrievalDoc
from src.query_rewriters import QueryConstraints


def _session_overlaps(sessions: list[dict], hour_min: int | None, hour_max: int | None) -> bool:
    if hour_min is None and hour_max is None:
        return True
    for s in sessions:
        a, b = s["start"], s["end"]
        if hour_min is not None and b <= hour_min:
            continue
        if hour_max is not None and a >= hour_max:
            continue
        return True
    return False


def doc_passes(doc: RetrievalDoc, c: QueryConstraints) -> bool:
    md = doc.metadata
    sessions = md.get("sessions") or []

    # weekday include: must have at least one session on an included weekday
    if c.weekday_include:
        if not any(s["weekday"] in c.weekday_include for s in sessions):
            return False
    if c.weekday_exclude:
        if any(s["weekday"] in c.weekday_exclude for s in sessions):
            return False

    if c.hour_min is not None or c.hour_max is not None:
        if sessions and not _session_overlaps(sessions, c.hour_min, c.hour_max):
            return False

    if c.lang_include and md.get("lang") not in c.lang_include:
        return False
    if c.lang_exclude and md.get("lang") in c.lang_exclude:
        return False

    if c.kind_include and md.get("kind") not in c.kind_include:
        return False
    if c.kind_exclude and md.get("kind") in c.kind_exclude:
        return False

    pt = md.get("point") or 0.0
    if c.point_min is not None and pt < c.point_min:
        return False
    if c.point_max is not None and pt > c.point_max:
        return False

    if c.unit_include:
        unit = md.get("unit") or ""
        if not any(kw in unit for kw in c.unit_include):
            return False
    if c.unit_exclude:
        unit = md.get("unit") or ""
        if any(kw in unit for kw in c.unit_exclude):
            return False

    return True


def filter_hits(
    hits: list[tuple[RetrievalDoc, float]],
    c: QueryConstraints,
) -> list[tuple[RetrievalDoc, float]]:
    return [(d, s) for d, s in hits if doc_passes(d, c)]


__all__ = ["doc_passes", "filter_hits"]
