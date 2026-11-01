from src.eval.conditions import check, n_clauses, satisfies
from src.parsers.base import ConditionResult
from src.parsers.ollama_parser import sanitize

# Wed 14-17, English, 3 credits, elective, accounting
META = {
    "lang": "英文", "kind": 2, "lmt_kind": "", "point": 3.0, "unit": "會計系",
    "sessions": [{"weekday": 3, "start": 14, "end": 17}],
}


def test_must_all_match():
    c = ConditionResult(must={"course_lang": ["英文"], "weekday": [3], "hour_range": [13, 18],
                              "point": 3.0, "kind": ["選修"], "unit": ["會計"]})
    assert satisfies(META, c)


def test_must_weekday_miss():
    c = ConditionResult(must={"weekday": [1, 2]})
    assert check(META, c) == {"must:weekday": False}


def test_hour_range_is_overlap_not_containment():
    assert satisfies(META, ConditionResult(must={"hour_range": [16, 18]}))
    assert not satisfies(META, ConditionResult(must={"hour_range": [8, 12]}))


def test_must_not_morning_passes_afternoon_course():
    assert satisfies(META, ConditionResult(must_not={"hour_range": [8, 12]}))


def test_must_not_violated():
    c = ConditionResult(must_not={"course_lang": ["英文"]})
    assert check(META, c) == {"not:course_lang": False}


def test_empty_clauses_ignored():
    c = ConditionResult(must={"weekday": [], "hour_range": None}, must_not={})
    assert check(META, c) == {}
    assert n_clauses(c) == 0


def test_sanitize_coerces_llm_types():
    out = sanitize({"weekday": ["週三", "5"], "point": ["3"], "hour_range": ["13", "18"],
                    "course_lang": "英文", "bogus": 1})
    assert out == {"weekday": [3, 5], "point": 3.0, "hour_range": [13, 18], "course_lang": ["英文"]}


def test_sanitize_drops_bad_values():
    assert sanitize({"hour_range": [18, 8], "point": "三學分", "weekday": ["x"]}) == {}
    assert sanitize(None) == {}


def test_kind_codes_follow_catalogue():
    pe = {"kind": 1, "name": "體育—羽球初級", "lmt_kind": "", "sessions": []}
    ge = {"kind": 4, "name": "社會學與生活", "lmt_kind": "社會通識", "sessions": []}
    group = {"kind": 3, "name": "貨幣銀行學", "lmt_kind": "", "sessions": []}
    assert satisfies(pe, ConditionResult(must={"kind": ["體育"]}))
    assert not satisfies(pe, ConditionResult(must={"kind": ["必修"]}))
    assert satisfies(ge, ConditionResult(must={"kind": ["通識"]}))
    assert not satisfies(group, ConditionResult(must={"kind": ["通識"]}))
    assert satisfies(group, ConditionResult(must={"kind": ["群修"]}))
