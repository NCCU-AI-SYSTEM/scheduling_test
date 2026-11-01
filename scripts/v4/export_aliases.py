"""Export the alias table of data/v4/course_struct.db for review.

  docs/v4/aliases.md       the aliases the search uses (reviews + hand seeds, course runs in 114-2)
  data/v4/aliases_all.tsv  every row, including jieba-generated ones the search ignores

  python -m scripts.v4.export_aliases
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.v4.struct_store import STRUCT_DB, TRUSTED_ALIAS_SOURCES

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ZH = {"rate": "評鑑留言", "seed": "人工整理", "rule": "jieba 自動產生"}


def main() -> None:
    con = sqlite3.connect(STRUCT_DB)
    rows = con.execute("SELECT alias, canonical, members, kind, source, evidence, confidence, in_1142, note "
                       "FROM alias ORDER BY source, evidence DESC, alias").fetchall()
    used = [r for r in rows if r[4] in TRUSTED_ALIAS_SOURCES and r[7]]
    with (ROOT / "data" / "v4" / "aliases_all.tsv").open("w", encoding="utf-8") as f:
        f.write("alias\tcanonical\tmembers\tkind\tsource\tevidence\tconfidence\tin_1142\tused_by_search\tnote\n")
        for r in rows:
            use = r[4] in TRUSTED_ALIAS_SOURCES and bool(r[7])
            f.write("\t".join(str(x if x is not None else "") for x in (*r[:8], int(use), r[8])) + "\n")
    lines = ["# 課名縮寫對照表", "",
             f"資料來源：`data/v4/course_struct.db` 的 `alias` 表，共 {len(rows)} 列。"
             f"搜尋只用評鑑留言抽出與人工整理、且課程在 114-2 有開的 {len(used)} 個（使用者 2026/11/01 決定），"
             "jieba 自動產生的不用（抽查 27/33 正確）。全部列表在 `data/v4/aliases_all.tsv`。", "",
             "產生方式：`src/v4/structured/aliases.py`；評鑑留言的證據是提到該縮寫的留言中，屬於這門課的比例。"
             "搜尋時的展開：`src/v4/struct_store.py` 的 `expand_aliases()`。", "",
             "| 縮寫 | 對應課名 | 來源 | 證據 | 說明 |", "|---|---|---|---|---|"]
    for alias, canonical, members, _kind, source, evidence, _conf, _in, note in used:
        names = "、".join(json.loads(members or "[]")) or canonical
        lines.append(f"| {alias} | {names} | {SOURCE_ZH.get(source, source)} | {evidence or ''} | {note or ''} |")
    (ROOT / "docs" / "v4" / "aliases.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    by = {}
    for r in rows:
        by[r[4]] = by.get(r[4], 0) + 1
    print(f"rows {len(rows)} by source {by}; used {len(used)}")


if __name__ == "__main__":
    main()
