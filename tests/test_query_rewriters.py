from src.query_rewriters import parse_constraints


def test_weekday_include():
    c = parse_constraints("週二下午的通識")
    assert c.weekday_include == {2}
    assert 3 in c.kind_include


def test_weekday_exclude():
    c = parse_constraints("不要週一的英文授課課")
    assert c.weekday_exclude == {1}
    assert "英文" in c.lang_include


def test_time_bucket_morning():
    c = parse_constraints("找早上的程式設計課")
    assert c.hour_min == 8 and c.hour_max == 12


def test_time_bucket_afternoon_and_lang():
    c = parse_constraints("週四下午英文授課的行銷")
    assert 4 in c.weekday_include
    assert c.hour_min == 13 and c.hour_max == 18
    assert "英文" in c.lang_include


def test_kind_required():
    c = parse_constraints("企管系大二必修")
    assert 1 in c.kind_include
    assert "企管" in c.unit_include


def test_kind_neg():
    c = parse_constraints("不要必修，想找選修")
    assert 1 in c.kind_exclude
    assert 2 in c.kind_include


def test_points_threshold():
    c = parse_constraints("3 學分以上的法律課")
    assert c.point_min == 3.0
    assert "法律" in c.unit_include


def test_lang_keyword_only_when_collocated():
    # bare 中文 is too generic, should not trigger lang filter
    c = parse_constraints("中文系的課")
    assert "中文" not in c.lang_include


def test_residual():
    c = parse_constraints("不要週一早八的英文授課行銷課")
    assert "行銷" in c.semantic_residual
    assert "英文" not in c.semantic_residual
