"""Offerings and identity filtering on the real 114-2 catalogue.

Prints: how many classes the 2,655 course codes collapse into, how channel facts are
covered, and for a few synthetic identities how many classes are excluded or annotated,
with the two user-raised cases (資科 學碩合開, 大學英文 by college) shown explicitly.

  python -m scripts.v4.offering_report
"""

from __future__ import annotations

from collections import Counter

from src.v4.access import StudentIdentity, judge
from src.v4.catalog import load_catalog
from src.v4.offering import dept_registry, group_offerings

IDENTITIES = {
    "資科學士三年級": StudentIdentity("703", "資訊", "學士", 3),
    "資科碩士一年級": StudentIdentity("753", "資訊", "碩士", 1),
    "企管學士一年級": StudentIdentity("304", "商", "學士", 1),
    "法律學士二年級": StudentIdentity("601", "法", "學士", 2),
}


def main() -> None:
    cat = load_catalog(with_reviews=False)
    depts = dept_registry()
    offs = group_offerings(cat, depts)
    multi = [o for o in offs if len(o.channels) > 1]
    print(f"course codes {len(cat)} -> classes {len(offs)}; classes with >1 code {len(multi)}")
    chans = [ch for o in offs for ch in o.channels]
    cov = Counter()
    for ch in chans:
        cov["enroll_mode=" + (ch.enroll_mode or "空")] += 1
        cov["has_level"] += bool(ch.level)
        cov["has_targets"] += bool(ch.targets)
        cov["audience_level"] += bool(ch.audience_level)
        cov["colleges_only"] += bool(ch.colleges_only)
        cov["status_only"] += bool(ch.status_only)
        cov["special_class"] += bool(ch.special_class)
        cov["has_url"] += bool(ch.url)
    print("channel facts:", dict(sorted(cov.items())))
    print(f"dept codes in registry: {len(depts)}; course dept codes without entry: "
          f"{len({ch.dept_code for ch in chans} - set(depts))}")

    for name, s in IDENTITIES.items():
        acc = [judge(o, s) for o in offs]
        excl = [a for a in acc if a.excluded]
        warns = Counter(f.kind for a in acc if not a.excluded for f in a.flags if f.level == "warn")
        excl_kinds = Counter(next(f.kind for f in a.flags if f.level == "exclude") for a in excl)
        print(f"\n[{name}] excluded {len(excl)}/{len(offs)}  by rule {dict(excl_kinds)}")
        print(f"  shown with a warning {sum(any(f.level == 'warn' for f in a.flags) for a in acc if not a.excluded)}"
              f"  by rule {dict(warns)}")

    print("\n— 資科 學碩合開 —")
    for o in offs:
        if o.key[0] in ("分散式系統", "自然語言處理") and len(o.channels) > 1:
            for name in ("資科學士三年級", "資科碩士一年級", "企管學士一年級"):
                a = judge(o, IDENTITIES[name])
                print(f"  {o.key[0]} {name}: 課號 {a.channel.sub_num}（{a.channel.unit}） "
                      f"{'排除' if a.excluded else '顯示'} {[f.detail for f in a.flags]}")

    print("\n— 大學英文（二）—")
    eng = [o for o in offs if o.key[0].startswith("大學英文")]
    for name in ("資科學士三年級", "企管學士一年級", "法律學士二年級"):
        shown = [(o, judge(o, IDENTITIES[name])) for o in eng]
        ok = [(o, a) for o, a in shown if not a.excluded]
        times = Counter(o.key[2] for o, _ in ok)
        print(f"  {name}: 顯示 {len(ok)}/{len(eng)} 班，時段 {dict(times)}；"
              f"加註 {sorted({f.detail for _, a in ok for f in a.flags if f.level == 'warn'})}")


if __name__ == "__main__":
    main()
