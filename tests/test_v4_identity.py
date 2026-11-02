import pytest

from src.v4.access import StudentIdentity, check_channel, judge
from src.v4.identity import FormInput, UnitTable, from_form, from_profile, resolve
from src.v4.offering import Channel, Dept, Offering, unit_targets
from src.v4.profile import StudentProfile, parse_exempt, parse_statuses


@pytest.fixture(scope="module")
def units():
    return UnitTable.load()


def test_unit_tree_level_decides_code(units):
    assert units.code_for("資訊科學系", "學士") == "703"
    assert units.code_for("資訊科學系", "碩士") == "753"
    assert units.code_for("資訊科學系碩士班", "碩士") == "753"
    assert units.code_for("企業管理學系(305)", "學士") == "305"     # 全人 doubleMajor carries the code
    assert units.code_for("704", "學士") == "704"                   # codes pass through
    assert units.dept("753", "碩士") == Dept("753", "資訊", "碩士")


def test_parse_statuses_and_exempt():
    assert parse_statuses("一般生") == frozenset()
    assert parse_statuses("僑生（港澳）") == {"僑生"}
    assert parse_statuses("外籍生") == {"外籍生"}
    assert parse_exempt("免修") and not parse_exempt("未免修") and not parse_exempt("") and not parse_exempt("否")


def test_from_profile(units):
    p = StudentProfile(major="資訊科學系", degree="學士", year=3, other_majors=("企業管理學系(305)",),
                       statuses=frozenset({"僑生"}), english_exempt=True)
    s = from_profile(p, units)
    assert (s.dept_code, s.college, s.level, s.year, s.source) == ("703", "資訊", "學士", 3, "全人")
    assert s.other_depts == {"305"} and s.statuses == {"僑生"} and s.english_exempt
    with pytest.raises(ValueError):
        from_profile(StudentProfile(major="不存在學系", year=1), units)


def test_from_form(units):
    s = from_form(FormInput(dept="資訊科學系", level="碩士", year=1), units)
    assert (s.dept_code, s.level, s.source) == ("753", "碩士", "自行輸入")
    with pytest.raises(ValueError):
        from_form(FormInput(dept="資訊科學系", level="大學"), units)


def test_resolve_priority(units):
    depts = {"703": Dept("703", "資訊", "學士"), "305": Dept("305", "商", "學士")}
    prof = StudentProfile(major="企業管理學系", degree="學士", year=2)
    form = FormInput(dept="資訊科學系", year=4, statuses=("外籍生",))
    assert resolve(units, depts, profile=prof, form=form, student_id="112703001").dept_code == "305"
    assert resolve(units, depts, profile=prof, form=form).statuses == {"外籍生"}   # form statuses added
    assert resolve(units, depts, form=form, student_id="112703001").source == "自行輸入"
    assert resolve(units, depts, student_id="112703001").source == "學號"
    assert resolve(units, depts) is None


def test_unit_table_round_trips_through_rows(units):
    again = UnitTable.from_rows(units.rows())
    for name, level in [("資訊科學系", "學士"), ("資訊科學系", "碩士"), ("企業管理學系", "學士"), ("法律學系", "碩士")]:
        assert again.code_for(name, level) == units.code_for(name, level)
    assert again.names == units.names
    assert again.code_for("資訊科學系碩士在職專班", "碩士") == "971"


def test_struct_bundle_carries_unit_table():
    from src.v4.struct_store import STRUCT_DB, load_struct

    if not STRUCT_DB.exists():
        pytest.skip("needs data/v4/course_struct.db")
    b = load_struct()
    if b.units is None:
        pytest.skip("course_struct.db built before the unit table existed")
    assert b.units.code_for("資訊科學系", "碩士") == "753"
    assert b.units.names.get("971") == "資訊科學系碩士在職專班"


def _ch(**kw) -> Channel:
    base = dict(course_id="1142305001001", sub_num="305001001", unit="企管三", dept_code="305", college="商",
                level="學士", targets=unit_targets("企管三"), enroll_mode="一般", audience_level="",
                colleges_only=frozenset(), status_only=frozenset(), special_class="", url="", note="")
    return Channel(**(base | kw))


def test_double_major_passes_dept_or_minor():
    cs = StudentIdentity("703", "資訊", "學士", 3)
    cs_ba = StudentIdentity("703", "資訊", "學士", 3, other_depts=frozenset({"305"}))
    ch = _ch(note="限本系及輔系雙主修生")
    assert "dept_or_minor" in {f.kind for f in check_channel(ch, cs)}
    assert check_channel(ch, cs_ba) == ()


def test_english_exempt_excludes_college_english():
    ch = _ch(unit="外文中", targets=frozenset(), dept_code="032", college="", enroll_mode="必修灌檔",
             colleges_only=frozenset({"資訊"}))
    off = Offering(key=("大學英文", "某老師", "三D56"), course_ids=(ch.course_id,), channels=(ch,))
    assert not judge(off, StudentIdentity("703", "資訊", "學士", 1)).excluded
    assert judge(off, StudentIdentity("703", "資訊", "學士", 1, english_exempt=True)).excluded
