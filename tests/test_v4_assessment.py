from src.v4.assessment import profile_from, verdict
from src.v4.planner import Plan


def test_profile_items():
    p = profile_from(["期中考", "期末考", "作業"], "期中考:30%期末考:40%作業:30%")
    assert {"期中考", "期末考", "作業"} <= p.items and not p.generic_exam


def test_no_midterm():
    has_mid = profile_from(["期中考"], "期中考 50% 期末考 50%")
    report_only = profile_from(["期末報告"], "期末報告 60% 出席 40%")
    silent = profile_from([], "")
    generic = profile_from(["考試"], "考試 40% 作業 60%")
    want = ((), ("期中考",))
    assert verdict(has_mid, *want) == "violate"
    assert verdict(report_only, *want) == "ok"
    assert verdict(silent, *want) == "unknown"
    assert verdict(generic, *want) == "unknown"   # 考試 without type: could be a midterm


def test_must_have_quiz():
    quiz = profile_from(["小考"], "小考 20% 期末考 40%")
    no_quiz = profile_from(["期末考"], "期末考 60% 作業 40%")
    want = (("小考",), ())
    assert verdict(quiz, *want) == "ok"
    assert verdict(no_quiz, *want) == "violate"
    assert verdict(profile_from([], ""), *want) == "unknown"


def test_plan_extracts_assessment_before_sanitize():
    p = Plan.from_dict({"need": "統計", "must": {"assessment": ["小考", "亂寫"], "weekday": [3]},
                        "must_not": {"assessment": "期中考"}, "prefs": [], "intents": ["search"]})
    assert p.assess_must == ("小考",) and p.assess_not == ("期中考",)
    assert p.cond.must == {"weekday": [3]} and "assessment" not in p.cond.must_not


def test_assessment_counts_as_exam():
    from src.v4.assessment import parse_query

    p = profile_from(["期中評量", "期末評量"], "期中評量 40% 期末評量 40% 出席 20%")
    assert {"期中考", "期末考"} <= p.items
    assert verdict(p, (), ("期中考",)) == "violate"
    assert parse_query("不要期中評量的課") == ((), ("期中考",))


def test_parse_query():
    from src.v4.assessment import parse_query

    cases = {
        "不要期中考的通識": ((), ("期中考",)),
        "有小考的統計課": (("小考",), ()),
        "沒有期末考，用報告取代的課": ((), ("期末考",)),
        "週三下午不用考試的課": ((), ("小考", "期中考", "期末考")),
        "想修不要上台報告的管理課": ((), ("口頭報告",)),
        "要有期中期末考的經濟學": (("期中考", "期末考"), ()),
        "不要分組的英文授課課程": ((), ()),
        "作業少一點的程式課": ((), ()),
        "作業不要太多的課": ((), ()),
        "有沒有期中考": ((), ()),
        "有趣的哲學課": ((), ()),
        "不想寫期末報告": ((), ("書面報告",)),
    }
    for q, want in cases.items():
        assert parse_query(q) == want, q
