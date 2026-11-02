"""Search once and print the result table (Markdown), on a GPU or CPU-only machine.

  python -m scripts.v4.demo_search "週三下午的英文授課通識" --deploy cpu
  python -m scripts.v4.demo_search "想修自然語言處理" --profile data/exportStudentData.json
  python -m scripts.v4.demo_search "大英要選哪一班" --dept 資訊科學系 --year 1
  STUDENT_ID=112703001 python -m scripts.v4.demo_search "想修自然語言處理"

Identity, best source first (src/v4/identity.py): --profile (全人 export) > --dept/--year
(form input) > STUDENT_ID. The student number is read from the environment so it stays
out of shell history and is never printed; only derived fields are shown.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.v4.deploy import build, preset
from src.v4.identity import FormInput, UnitTable, resolve
from src.v4.planner import plan
from src.v4.present import row, to_markdown
from src.v4.profile import load_profile
from src.v4.semester import parse as parse_semester


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="+")
    p.add_argument("--deploy", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--planner", default=None, help="override the preset's planner model")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--profile", type=Path, default=None, help="全人 export JSON")
    p.add_argument("--dept", default=None, help="department code or name (form input)")
    p.add_argument("--level", default="學士")
    p.add_argument("--year", type=int, default=1)
    p.add_argument("--status", action="append", default=[], help="僑生 / 外籍生 / 交換生 ...")
    p.add_argument("--english-exempt", action="store_true")
    a = p.parse_args()
    pre = preset(a.deploy, a.planner, a.threads)
    search, b, cfg = build(pre)
    prof = load_profile(a.profile) if a.profile else None
    form = FormInput(a.dept, a.level, a.year, tuple(a.status), english_exempt=a.english_exempt) if a.dept else None
    if form is None and a.status:
        form = FormInput("", statuses=tuple(a.status))
    ident = resolve(b.units or UnitTable.load(), b.depts, profile=prof, form=form if form and form.dept else None,
                    student_id=os.environ.get("STUDENT_ID"),
                    academic_year=parse_semester(b.meta.get("semester", "114-2")).academic_year)
    if ident:
        print(f"身分（來源：{ident.source}）：{ident.college}學院 {ident.level} {ident.year} 年級，系所代碼 "
              f"{ident.dept_code}；其他系所 {sorted(ident.other_depts) or '無'}；身分 {sorted(ident.statuses) or '一般'}；"
              f"大英免修 {'是' if ident.english_exempt else '否'}")
    print(f"部署：{pre.name}，規劃器 {pre.planner}，重排 {pre.reranker or '關閉'}")
    cmap = {c.course_id: c for c in b.catalog}
    for q in a.query:
        pl = plan(q, pre.planner, pre.planner_options)
        hits = search.search(pl, q, a.k, cfg, profile=prof, identity=ident)
        print(f"\n### {q}\n規劃：need={pl.need!r} 課名={pl.named_course} 老師={pl.named_teacher} "
              f"must={pl.cond.must} must_not={pl.cond.must_not} prefs={pl.prefs}"
              f"{'　放寬：' + '、'.join(hits[0].relaxed) if hits and hits[0].relaxed else ''}\n")
        print(to_markdown([row(h, cmap) for h in hits]))
        print(f"\n（規劃 {pl.seconds:.2f}s，搜尋 {search.timing['total'] * 1000:.0f}ms）")


if __name__ == "__main__":
    main()
