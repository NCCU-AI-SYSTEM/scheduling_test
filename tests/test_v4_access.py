from dataclasses import replace

import pytest

from src.v4.access import StudentIdentity, check_channel, from_student_id, judge
from src.v4.offering import (Channel, Dept, Offering, audience_level, colleges_only, special_class,
                             status_only, unit_targets)

CS3 = StudentIdentity(dept_code="703", college="資訊", level="學士", year=3)
CS_M1 = StudentIdentity(dept_code="753", college="資訊", level="碩士", year=1)
BA1 = StudentIdentity(dept_code="304", college="商", level="學士", year=1)


def _ch(**kw) -> Channel:
    base = dict(course_id="1142703059001", sub_num="703059001", unit="資訊三資訊四", dept_code="703",
                college="資訊", level="學士", targets=unit_targets("資訊三資訊四"), enroll_mode="一般",
                audience_level="", colleges_only=frozenset(), status_only=frozenset(), special_class="",
                url="", note="")
    return Channel(**(base | kw))


def _levels(flags):
    return {f.kind: f.level for f in flags}


# ── parsing ──────────────────────────────────────────────────────────────────
def test_unit_targets():
    assert unit_targets("資訊三資訊四") == {("學士", 3), ("學士", 4)}
    assert unit_targets("資碩工一資碩計一") == {("碩士", 1)}
    assert unit_targets("資訊博一資碩工一") == {("博士", 1), ("碩士", 1)}
    assert unit_targets("外文中") == frozenset()


def test_colleges_only():
    assert colleges_only("文、理、教育、資訊學院修讀") == {"文", "理", "教育", "資訊"}
    assert colleges_only("社會科學、國際事務學院修讀") == {"社會科學", "國際事務"}
    assert colleges_only("僑生（含港澳生）及外國學生修讀（商學院）") == {"商"}
    assert colleges_only("限本系學生") == frozenset()


def test_status_and_special_class():
    assert status_only("僑生（含港澳生）及外國學生修讀（商學院）") == {"僑生", "外籍生"}
    assert status_only("1.僅供外籍交換生與外籍學位生修習") == {"外籍生"}
    assert special_class("國防專班課程。上課地點為藍天樓。") == "國防"
    assert special_class("本課程為原專班學生必修課，不開放非原專班學生修課。") == "原專班"
    assert special_class("本課程為擴大輔系專班，需繳交學分費。") == ""


def test_audience_level():
    assert audience_level("【此科目代碼為碩士班選課用】學碩合開") == "碩士"
    assert audience_level("【此科目代碼為大學部選課用】學碩合開限三年級以上") == "學士"


# ── identity ─────────────────────────────────────────────────────────────────
def test_from_student_id():
    depts = {"703": Dept("703", "資訊", "學士")}
    s = from_student_id("112703001", depts, academic_year=114)
    assert (s.dept_code, s.college, s.level, s.year) == ("703", "資訊", "學士", 3)
    with pytest.raises(ValueError):
        from_student_id("12345", depts)
    with pytest.raises(ValueError):
        from_student_id("112999001", depts)


# ── channel rules ────────────────────────────────────────────────────────────
def test_college_english_section():
    ch = _ch(unit="外文中", targets=frozenset(), dept_code="032", college="", enroll_mode="必修灌檔",
             colleges_only=frozenset({"文", "理", "教育", "資訊"}))
    assert _levels(check_channel(ch, CS3)) == {"assigned_target": "info"}
    assert _levels(check_channel(ch, BA1))["college_only"] == "exclude"


def test_assigned_required_course_of_another_class():
    ch = _ch(unit="企管一甲", targets=unit_targets("企管一甲"), dept_code="304", college="商",
             enroll_mode="必修灌檔")
    assert _levels(check_channel(ch, BA1)) == {"assigned_target": "info"}
    assert _levels(check_channel(ch, CS3))["assigned"] == "warn"


def test_level_and_target():
    assert check_channel(_ch(), CS3) == ()
    assert _levels(check_channel(_ch(), CS_M1)) == {"level_target": "warn"}
    assert _levels(check_channel(_ch(), BA1)) == {"target": "info"}


def test_note_rules():
    assert _levels(check_channel(_ch(note="限大三以上"), replace(CS3, year=2)))["grade_min"] == "exclude"
    assert check_channel(_ch(note="限大三以上"), CS3) == ()
    assert _levels(check_channel(_ch(note="限本系學生"), BA1))["dept_only"] == "warn"
    assert _levels(check_channel(_ch(note="限本系及輔系雙主修生"), BA1))["dept_or_minor"] == "warn"
    assert _levels(check_channel(_ch(note="限碩士生修讀"), CS3))["degree"] == "exclude"
    assert _levels(check_channel(_ch(special_class="國防"), CS3))["special_class"] == "exclude"
    assert check_channel(_ch(special_class="國防"), replace(CS3, statuses=frozenset({"國防"}))) == ()


# ── offering: best channel wins ──────────────────────────────────────────────
def test_judge_picks_matching_channel():
    ug = _ch(audience_level="學士")
    grad = _ch(course_id="1142753852001", sub_num="753852001", unit="資碩工一資碩計一", dept_code="753",
               level="碩士", targets=unit_targets("資碩工一資碩計一"), audience_level="碩士")
    off = Offering(key=("分散式系統", "廖峻鋒", "一D56"), course_ids=(ug.course_id, grad.course_id),
                   channels=(ug, grad))
    a = judge(off, CS3)
    assert a.channel.sub_num == "703059001" and not a.excluded and a.flags == ()
    a = judge(off, CS_M1)
    assert a.channel.sub_num == "753852001" and not a.excluded


def test_judge_excludes_when_all_channels_exclude():
    ch = _ch(colleges_only=frozenset({"商"}))
    off = Offering(key=("x", "y", "z"), course_ids=(ch.course_id,), channels=(ch,))
    assert judge(off, CS3).excluded
    assert not judge(off, BA1).excluded


def test_registry_uses_latest_college():
    from src.v4.catalog import DB
    from src.v4.offering import dept_registry

    if not DB.exists():
        pytest.skip("needs data/1142.db")
    reg = dept_registry()
    assert (reg["703"].college, reg["703"].level) == ("資訊", "學士")   # 理學院 before 資訊學院 existed
    assert (reg["753"].college, reg["753"].level) == ("資訊", "碩士")
