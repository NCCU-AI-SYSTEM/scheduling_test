"""Does the struct-DB loader give the search the same courses as the 1142.db loader?

Compares src.v4.struct_store.load_struct() with src.v4.catalog.load_catalog() course by
course on every field the search filters on, the offering/channel grouping, and the
assessment verdicts (expected to differ: eval_item categories replace keyword scans).

  python -m scripts.v4.check_struct_catalog
"""

from __future__ import annotations

from collections import Counter

from src.v4.access import StudentIdentity, judge
from src.v4.assessment import CONDITION_ITEMS, profile_from, verdict
from src.v4.catalog import load_catalog
from src.v4.offering import group_offerings
from src.v4.struct_store import load_struct

FIELDS = ["name", "teacher", "kind", "lmt_kind", "core", "point", "lang", "unit", "time_raw", "sessions",
          "dept_code"]
IDENTITIES = [StudentIdentity("703", "資訊", "學士", 3), StudentIdentity("304", "商", "學士", 1),
              StudentIdentity("753", "資訊", "碩士", 1)]


def main() -> None:
    old = {c.course_id: c for c in load_catalog(with_reviews=False)}
    b = load_struct()
    new = {c.course_id: c for c in b.catalog}
    print(f"courses old {len(old)} new {len(new)}; only old {len(old.keys() - new.keys())}, "
          f"only new {len(new.keys() - old.keys())}")
    diff = Counter()
    examples: dict[str, list] = {}
    for cid in old.keys() & new.keys():
        o, n = old[cid], new[cid]
        for f in FIELDS:
            if o.meta.get(f) != n.meta.get(f):
                diff[f] += 1
                examples.setdefault(f, []).append((cid, o.meta.get(f), n.meta.get(f)))
        if o.categories != n.categories:
            diff["categories"] += 1
            examples.setdefault("categories", []).append((cid, o.categories, n.categories))
        if len(o.topics) != len(n.topics):
            diff["topics_len"] += 1
    print("field mismatches:", dict(diff) or "none")
    for f, ex in examples.items():
        print(f"  {f}: {ex[:3]}")

    old_off = {frozenset(o.course_ids) for o in group_offerings(list(old.values()))}
    new_off = {frozenset(o.course_ids) for o in b.offerings}
    print(f"offerings old {len(old_off)} new {len(new_off)} identical {len(old_off & new_off)}")
    by_key = {frozenset(o.course_ids): o for o in group_offerings(list(old.values()))}
    access_diff = 0
    for o in b.offerings:
        oo = by_key.get(frozenset(o.course_ids))
        if oo is None:
            continue
        for s in IDENTITIES:
            a, aa = judge(o, s), judge(oo, s)
            if (a.excluded, a.channel.course_id) != (aa.excluded, aa.channel.course_id):
                access_diff += 1
    print(f"access decisions that differ (3 identities): {access_diff}")

    print("assessment verdicts, old keyword scan -> new eval_item (不要 X):")
    for item in CONDITION_ITEMS:
        moves = Counter()
        for cid in old.keys() & new.keys():
            c = old[cid]
            p_old = profile_from([lbl for lbl, _ in c.grading.get("items", [])], c.course.evaluation)
            moves[(verdict(p_old, (), (item,)), verdict(b.assess[cid], (), (item,)))] += 1
        unknown_old = sum(v for (a, _), v in moves.items() if a == "unknown")
        unknown_new = sum(v for (_, z), v in moves.items() if z == "unknown")
        flips = {f"{a}->{z}": v for (a, z), v in moves.items() if a != z}
        print(f"  {item}: unknown {unknown_old} -> {unknown_new}; changes {flips}")
    print(f"aliases (rate/seed): {len(b.aliases)}; depts {len(b.depts)}; aspects {len(b.aspects)}; "
          f"embedding models {list(b.embeddings)}")


if __name__ == "__main__":
    main()
