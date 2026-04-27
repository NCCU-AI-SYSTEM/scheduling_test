from src.loader import load_courses, parse_time_str


def test_parse_time_basic():
    s = parse_time_str("一23")
    assert len(s) == 1
    assert s[0].weekday == 1
    # period 2,3 -> indices 3,4 -> start=9, end=11
    assert s[0].start_hour == 9
    assert s[0].end_hour == 11


def test_parse_time_multi_day():
    s = parse_time_str("一23四56")
    assert [x.weekday for x in s] == [1, 4]


def test_parse_time_undefined():
    assert parse_time_str("未定或彈性") == []
    assert parse_time_str("") == []


def test_parse_time_dc56_management_course():
    # From real example: 五D56 -> 13:00-16:00 (D=13, 5=14, 6=15)
    s = parse_time_str("五D56")
    assert len(s) == 1
    assert s[0].weekday == 5
    # index D=7 -> start 13, index 6=9 -> end 16
    assert s[0].start_hour == 13
    assert s[0].end_hour == 16


def test_load_courses_smoke():
    courses = load_courses(year="114", semester="2")
    assert len(courses) > 2000
    assert all(c.course_id.startswith("1142") for c in courses)
    # dedupe sanity
    ids = {c.course_id for c in courses}
    assert len(ids) == len(courses)
