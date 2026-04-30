from src.doc_builders import RetrievalDoc
from src.filters import doc_passes
from src.query_rewriters import parse_constraints


def _doc(**md):
    md.setdefault("sessions", [])
    return RetrievalDoc(course_id="x", text="", metadata=md)


def test_filter_weekday():
    doc_mon = _doc(sessions=[{"weekday": 1, "start": 13, "end": 15}])
    doc_thu = _doc(sessions=[{"weekday": 4, "start": 13, "end": 15}])
    c = parse_constraints("週四下午")
    assert not doc_passes(doc_mon, c)
    assert doc_passes(doc_thu, c)


def test_filter_lang_exclude():
    en = _doc(lang="英文", sessions=[])
    zh = _doc(lang="中文", sessions=[])
    c = parse_constraints("不要英文授課的課")
    assert not doc_passes(en, c)
    assert doc_passes(zh, c)


def test_filter_kind():
    req = _doc(kind=1, sessions=[])
    elec = _doc(kind=2, sessions=[])
    c = parse_constraints("選修課")
    assert doc_passes(elec, c)
    assert not doc_passes(req, c)


def test_filter_points_min():
    c2 = _doc(point=2.0, sessions=[])
    c3 = _doc(point=3.0, sessions=[])
    c = parse_constraints("3 學分以上")
    assert not doc_passes(c2, c)
    assert doc_passes(c3, c)


def test_filter_no_sessions_skip_time():
    """A course with no time slot should not be rejected by hour-range filter."""
    d = _doc(sessions=[])
    c = parse_constraints("早上的課")
    assert doc_passes(d, c)
