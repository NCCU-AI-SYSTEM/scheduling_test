"""How often is one physical class listed under several course codes in 114-2?

Groups courses by (normalised name, teacher, time). Inside a group, codes usually
differ in who may enrol: undergraduate vs master's (學碩合開), several departments, or
an English-taught twin. Also summarises college-restricted sections such as 大學英文.

  python -m scripts.v4.analysis.analyze_duplicates
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from src.v4.catalog import load_catalog


def norm(name: str) -> str:
    return re.sub(r"[（(].*?[)）]|\s", "", name)


def level(unit: str) -> str:
    if re.search(r"博", unit):
        return "博"
    if re.search(r"碩", unit):
        return "碩"
    return "學士"


def main() -> None:
    cat = load_catalog(with_reviews=False)
    groups: dict[tuple, list] = defaultdict(list)
    for c in cat:
        x = c.course
        groups[(norm(x.name), x.teacher, x.time_raw)].append(c)
    multi = {k: v for k, v in groups.items() if len(v) > 1 and k[2]}
    n_courses = sum(len(v) for v in multi.values())
    print(f"courses: {len(cat)}  same name+teacher+time groups with >1 code: {len(multi)} "
          f"covering {n_courses} codes ({n_courses / len(cat):.1%})")
    kinds = Counter()
    for v in multi.values():
        levels = tuple(sorted({level(c.course.unit) for c in v}))
        kinds[levels] += 1
    print("level mix per group:", dict(kinds.most_common()))
    joint = sum(1 for v in multi.values() if any("合開" in (c.course.note or "") for c in v))
    print(f"groups whose note says 合開: {joint}")
    by_dept = Counter(c.dept_code for v in multi.values() for c in v)
    print("top departments (subNum[:3]) in such groups:", by_dept.most_common(10))
    for k, v in list(multi.items())[:12]:
        print(" ", k[0], k[1], k[2], "|", " / ".join(f"{c.course_id[-9:]}:{c.course.unit[:10]}" for c in v))

    college = Counter()
    for c in cat:
        m = re.search(r"([^\s:：]{2,40}?)學院修讀", c.course.note or "")
        if m:
            college[(norm(c.course.name), m.group(1))] += 1
    names = Counter(k[0] for k in college)
    print(f"\nsections with an X學院修讀 restriction: {sum(college.values())} across {len(names)} course names")
    for n, cnt in names.most_common(8):
        print(" ", n, cnt, "college groups;", sum(v for (nn, _), v in college.items() if nn == n), "sections")


if __name__ == "__main__":
    main()
