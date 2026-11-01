"""How many course-name mentions written by students or teachers resolve to a course.

Mention sets:
  user     the abbreviations the user listed (個經 總經 初會 中會 五管 大英) and the two
           note phrasings 擋修初會(一) / 先修五管
  forum    course references in data/v4/query_bank_forum.jsonl that are not the exact
           catalogue name, picked by reading the 248 queries (listed below with the query)
  note     every prerequisite mention extracted from 114-2 notes (prereq table, kind=course);
           the subset that is not an exact catalogue name is reported on its own
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.v4.structured.aliases import Resolver

USER = ["個經", "總經", "初會", "中會", "五管", "大英", "初會(一)", "五管"]
# (mention, query it came from)
FORUM = [
    ("個經", "沒修過經濟學原理可以直接修個經嗎"),
    ("個經", "個經哪個老師比較不硬"),
    ("總經", "總經老師推薦？三個班不知道怎麼選"),
    ("中級會計", "外院可以修會計系的中級會計嗎？還是限本系"),
    ("日文二", "修完日文二大概有 N 幾的程度"),
    ("計程", "程式設計跟計算機概論內容重複嗎？選哪個（notes: PTT 程式設計 or 計程）"),
    ("大學英文", "大學英文哪個老師比較好？想要不太點名的"),
    ("微積分", "微積分兩個老師，一個比較涼一個分數好拿，怎麼選"),
    ("統計學", "統計學 哪個老師"),
    ("經濟學原理", "沒修過經濟學原理可以直接修個經嗎"),
    ("民法總則", "民法總則好過嗎？聽說很容易被當"),
    ("政治學概論", "非本院可以選 EMI 的政治學概論嗎"),
    ("國際金融", "不想修英文授課的，有中文班的國際金融嗎"),
    ("計量經濟學", "計量經濟學很硬嗎？數學要多好"),
]


def _row(r: Resolver, mention: str, origin: str) -> dict:
    x = r.resolve(mention)
    return {"mention": mention, "origin": origin, "base": x.base, "method": x.method,
            "resolved": x.method != "unresolved", "in_1142": x.in_1142, "names": list(x.names[:4])}


def evaluate(r: Resolver, out: sqlite3.Connection) -> dict:
    user = [_row(r, m, "user") for m in USER]
    forum = [_row(r, m, q) for m, q in FORUM]
    note = []
    for mention, base, method, in_1142, span in out.execute(
            "SELECT p.mention, p.resolved_base, p.method, p.in_1142, s.span FROM prereq p "
            "JOIN restriction s ON s.id = p.restriction_id WHERE p.kind = 'course'"):
        resolved = not method.endswith("unresolved")
        note.append({"mention": mention, "origin": span, "base": base, "method": method,
                     "resolved": resolved, "in_1142": bool(in_1142)})
    abbr = [n for n in note if not n["method"].endswith(("/exact", "/base"))]
    summary = {}
    for name, rows in (("user", user), ("forum", forum), ("note", note), ("note_non_exact", abbr)):
        summary[name] = {"n": len(rows), "resolved": sum(x["resolved"] for x in rows),
                         "in_1142": sum(x["resolved"] and x["in_1142"] for x in rows),
                         "unresolved": sorted({x["mention"] for x in rows if not x["resolved"]})}
    return {"summary": summary, "user": user, "forum": forum, "note": note}


def save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
