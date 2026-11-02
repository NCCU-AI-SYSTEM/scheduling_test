"""search_doc / search_session: what build_runtime writes so CourseDataPrep's load_v4 can build the
PostgreSQL search tables without importing v4 code."""

import json
from types import SimpleNamespace

from scripts.v4.build_runtime import search_rows
from src.retrievers.sparse import _char12


def _course(cid, name, meta, cats):
    return SimpleNamespace(course_id=cid, course=SimpleNamespace(name=name), meta=meta, categories=set(cats))


CATALOG = [
    _course("1142703834001", "自然語言處理", {"lang": "中文", "lmt_kind": "", "point": 3, "unit": "資訊科學系",
                                         "sessions": [{"weekday": 4, "start": 9, "end": 12}]}, {"選修"}),
    _course("1142000123001", "大學英文", {"lang": "英文", "lmt_kind": None, "point": None, "unit": None,
                                      "sessions": [{"weekday": 1, "start": 8, "end": 10},
                                                   {"weekday": 3, "start": 13, "end": 15}]}, {"必修", "通識"}),
]
TEXTS = ["自然語言處理 NLP 課程", "大學英文 English"]


def test_doc_rows_follow_catalogue_order_and_tokens():
    docs, _ = search_rows(CATALOG, TEXTS)
    assert [d[0] for d in docs] == ["1142703834001", "1142000123001"]
    assert [d[1] for d in docs] == [0, 1]
    assert docs[0][8] == TEXTS[0]
    assert docs[0][9] == " ".join(_char12(TEXTS[0]))


def test_doc_rows_normalise_empty_condition_fields():
    docs, _ = search_rows(CATALOG, TEXTS)
    cid, pos, name, lang, cats, lmt_kind, point, unit, text, toks = docs[1]
    assert (name, lang) == ("大學英文", "英文")
    assert json.loads(cats) == ["必修", "通識"]      # sorted, JSON array
    assert (lmt_kind, point, unit) == ("", 0.0, "")


def test_session_rows_use_integer_hours():
    _, ses = search_rows(CATALOG, TEXTS)
    assert ses == [("1142703834001", 4, 9, 12), ("1142000123001", 1, 8, 10), ("1142000123001", 3, 13, 15)]
