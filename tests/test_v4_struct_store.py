import pytest

from src.v4.struct_store import STRUCT_DB, _targets, expand_aliases

ALIASES = {"個經": ("個體經濟學",), "初會": ("初級會計學",), "五管": ("營運與供應鏈管理", "行銷管理"),
           "大英": ("大學英文",)}


def test_expand_aliases():
    assert expand_aliases("想修個經", ALIASES) == ("想修個經 個體經濟學", ["個體經濟學"])
    assert expand_aliases("擋修初會(一)", ALIASES)[1] == ["初級會計學"]
    assert expand_aliases("先修五管", ALIASES)[1] == ["營運與供應鏈管理", "行銷管理"]
    assert expand_aliases("個體經濟學哪班好", ALIASES) == ("個體經濟學哪班好", [])   # already written out
    assert expand_aliases("統計學", ALIASES) == ("統計學", [])


def test_targets():
    assert _targets('["學士3", "學士4"]') == {("學士", 3), ("學士", 4)}
    assert _targets(None) == frozenset()


def test_load_struct_matches_search_needs():
    if not STRUCT_DB.exists():
        pytest.skip("needs data/v4/course_struct.db")
    from src.v4.struct_store import load_struct

    b = load_struct()
    assert len(b.catalog) == 2655 and not any(c.cancelled for c in b.catalog)
    assert sum(len(o.channels) for o in b.offerings) == 2655
    assert "個經" in b.aliases and "國法" not in b.aliases       # jieba-generated aliases are excluded
    nlp = next(c for c in b.catalog if c.course.sub_num == "703834001")
    assert nlp.meta["sessions"] == [{"weekday": 4, "start": 9, "end": 12}]


def test_by_cat_uses_legacy_keys():
    if not STRUCT_DB.exists():
        pytest.skip("needs data/v4/course_struct.db")
    from src.v4.aspects import syllabus_aspects
    from src.v4.catalog import load_catalog
    from src.v4.struct_store import load_struct

    new = {c.course_id: c for c in load_struct().catalog}
    old = {c.course_id: c for c in load_catalog(with_reviews=False)}
    keys = {k for c in new.values() for k in c.grading["by_cat"]}
    assert keys <= {"exam", "report", "presentation", "homework", "project", "participation", "group"}
    # eval_item marks more syllabi reliable than the old regex, so compare where both give a value
    pairs = [(syllabus_aspects(new[cid]).get("attendance"), syllabus_aspects(old[cid]).get("attendance"))
             for cid in old]
    both = [(a, b) for a, b in pairs if a is not None and b is not None]
    assert len(both) > 1000
    assert sum(a == b for a, b in both) / len(both) > 0.8
    assert sum(a is not None for a, _ in pairs) >= sum(b is not None for _, b in pairs)
