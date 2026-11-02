import pytest

from src.v4.semester import Semester, current, parse


def test_parse_forms():
    assert parse("114-2") == Semester("114", "2")
    assert parse("1142") == Semester("114", "2")
    assert parse(" 115 1 ") == Semester("115", "1")
    s = parse("115-1")
    assert (s.ys, s.label, s.academic_year) == ("1151", "115-1", 115)
    for bad in ("114-4", "abc", "", "11422"):
        with pytest.raises(ValueError):
            parse(bad)


def test_current_from_env(monkeypatch):
    monkeypatch.setenv("V4_SEMESTER", "115-1")
    assert current() == Semester("115", "1")
    monkeypatch.delenv("V4_SEMESTER")
    assert current() == Semester("114", "2")


def test_gate_flags_low_coverage():
    from scripts.v4.prepare_semester import gate

    report = {"coverage": {"sessions": {"all": [99, 100]}, "eval_item": {"all": [80, 100]},
                           "tiny": {"all": [1, 10]}}}
    assert gate(report, 0.90) == ["抽取 eval_item 80/100 = 80.0%"]   # fields under 50 courses are not gated


def test_gate_catches_missing_source_and_shrink():
    from scripts.v4.prepare_semester import gate

    ok = {"coverage": {"sessions": {"all": [100, 100]}}}
    pre = {"courses": 1000, "filled": {"syllabus": 990, "evaluation": 0}}
    fails = gate(ok, 0.9, pre=pre, new_counts={"eval_item": 0, "weekly": 9000},
                 live_counts={"eval_item": 9000, "weekly": 9000})
    assert any("原始資料 evaluation" in f for f in fails) and any("eval_item" in f for f in fails)
    assert not any("weekly" in f for f in fails) and not any("syllabus" in f for f in fails)
