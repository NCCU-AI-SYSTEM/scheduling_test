"""Aliases, prerequisite resolution, LLM cache and course card."""

import sqlite3

from src.v4.structured import fallback, llm
from src.v4.structured.aliases import (Resolver, base_name, build_aliases, load_catalogue, mine_rate,
                                       split_numbering)
from src.v4.structured.card import render_card
from src.v4.structured.clauses import Restriction

NAMES = [("1141000318011", "114", "1", "初級會計學（一）"), ("1142000318011", "114", "2", "初級會計學（二）"),
         ("1142303001001", "114", "2", "個體經濟學"), ("1131303002001", "113", "1", "總體經濟學"),
         ("1142305019001", "114", "2", "行銷管理"), ("1142305020001", "114", "2", "財務管理"),
         ("1142305021001", "114", "2", "人力資源管理"), ("1142305022001", "114", "2", "營運與供應鏈管理"),
         ("1142000001001", "114", "2", "大學英文（二）"), ("1142000002001", "114", "2", "大學英文（三）：商用英文"),
         ("1132302001001", "113", "2", "中級會計學（二）")]
COMMENTS = ["初會老師講得很清楚", "初會作業很多", "初會考試不難", "推薦這堂初會",
            "個經真的很硬", "個經期中很難", "個經要多練習", "總經推推推", "總經考古題有用", "總經老師人好",
            "中會真的好難", "中會二比中會一難", "中會作業多"]


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE COURSE (id TEXT, y TEXT, s TEXT, name TEXT)")
    con.execute("CREATE TABLE RATE (courseId TEXT, content TEXT)")
    con.executemany("INSERT INTO COURSE VALUES (?,?,?,?)", NAMES)
    owner = {"初會": "1141000318011", "個經": "1142303001001", "總經": "1131303002001", "中會": "1132302001001"}
    con.executemany("INSERT INTO RATE VALUES (?,?)",
                    [(owner[next(k for k in owner if k in c)], c) for c in COMMENTS])
    return con


def test_base_and_numbering():
    assert base_name("大學英文（三）：商用英文") == "大學英文"
    assert base_name("初級會計學（一）") == "初級會計學"
    assert split_numbering("初會(一)") == ("初會", "一")
    assert split_numbering("中會二") == ("中會", "二")


def test_mine_rate_finds_abbreviations():
    con = _db()
    cat = load_catalogue(con)
    mined = {a.alias: a.canonical for a in mine_rate(con, cat)}
    assert mined["初會"] == "初級會計學"
    assert mined["個經"] == "個體經濟學"
    assert mined["總經"] == "總體經濟學"
    assert mined["中會"] == "中級會計學"


def test_resolver_numbering_groups_and_substring():
    con = _db()
    cat = load_catalogue(con)
    r = Resolver(build_aliases(con, cat), cat)
    hit = r.resolve("初會(一)")
    assert (hit.base, hit.names, hit.method) == ("初級會計學", ("初級會計學(一)",), "alias:rate")
    assert hit.in_1142            # 初級會計學（二） is offered in 114-2
    group = r.resolve("五管")
    assert group.base == "五管" and "財務管理" in group.names and group.method == "alias:seed"
    assert r.resolve("大英").base == "大學英文"
    assert r.resolve("中級會計").method == "substring"
    assert r.resolve("火星探險導論").method == "unresolved"


def test_prereq_rule_split_and_llm_fallback(tmp_path, monkeypatch):
    con = _db()
    cat = load_catalogue(con)
    r = Resolver(build_aliases(con, cat), cat)
    split = fallback.rule_split("個體經濟學及總體經濟學", r)
    assert [m.res.base for m in split] == ["個體經濟學", "總體經濟學"]
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    cache = llm.Cache("prereq", tmp_path)
    span = "須修畢或同學期修畢五管與某某實習"
    cache.put(span, "test", {"courses": [{"name": "五管", "kind": "course"}, {"name": "某某實習", "kind": "course"}]})
    rs = Restriction("prerequisite", "五管與某某實習", "only", True, "note", span)
    got = fallback.resolve_prereq(rs, r, llm.Cache("prereq", tmp_path))   # reread from disk
    assert [(m.mention, m.res.method, m.method) for m in got] == [("五管", "alias:seed", "llm"),
                                                                    ("某某實習", "unresolved", "llm")]


def test_cache_key_depends_on_prompt_version(tmp_path):
    c = llm.Cache("grading", tmp_path)
    c.put("期中考 50分 期末考 50分", "m", {"items": []})
    assert llm.Cache("grading", tmp_path).get("期中考 50分 期末考 50分") == {"items": []}
    assert llm.cache_key("grading", "x") != llm.cache_key("weekly", "x")


def test_card_line_order():
    course = {"name": "統計學（二）", "cancelled": False, "sub_num": "703001001", "teachers": ["甲"], "teacher_raw": "甲",
              "credits": 3.0, "kind_label": "必修", "ge_category": None, "ge_core": False, "college": "資訊",
              "dept_name": "資訊科學系", "level": "學士", "classroom_raw": "大仁200401", "semester_span": "單學期",
              "enroll_mode": "必修灌檔", "unit_raw": "資訊二", "lang": "中文", "teach_lecture": 60,
              "teach_discussion": 40, "teach_group": 0, "teach_elearning": 0, "teach_other": 0,
              "ai_policy": "conditional", "objective": "介紹推論統計。", "syllabus": None}
    ses = [{"weekday": 3, "start": "13:10", "end": "16:00", "kind": "main", "in_time_field": True},
           {"weekday": 4, "start": "17:10", "end": "19:00", "kind": "lab", "in_time_field": False}]
    tgt = [{"raw": "資訊二"}]
    restr = [{"type": "priority", "value": "資科系", "polarity": "info", "hard": False},
             {"type": "grade_min", "value": "大二", "polarity": "only", "hard": True}]
    ev = [{"category": "期中考", "label": "期中考", "pct": 40.0}, {"category": "期末考", "label": "期末考", "pct": 60.0}]
    wk = [{"week": 1, "topic": "課程介紹"}]
    lines = render_card(course, ses, tgt, restr, ev, wk).splitlines()
    assert lines[0].startswith("【統計學（二）】 703001001｜甲｜3學分｜必修｜資訊學院 資訊科學系 學士")
    assert "（另：實習課 週四 17:10–19:00）" in lines[1] and "選課方式:必修灌檔" in lines[1]
    assert lines[3] == "限制:年級:大二（硬性）；優先:資科系"
    assert lines[4] == "評分:期中考 40%、期末考 60%"
    assert lines[5] == "授課:講述 60%、討論 40%｜AI:有條件開放"
    assert lines[6] == "週次:1 課程介紹" and lines[7] == "簡介:介紹推論統計。"


def test_prereq_multi_numbering_expands():
    con = _db()
    cat = load_catalogue(con)
    r = Resolver(build_aliases(con, cat), cat)
    got = fallback.rule_split("初級會計學(一)(二)", r)
    assert [m.mention for m in got] == ["初級會計學(一)", "初級會計學(二)"]
    assert [m.res.names for m in got] == [("初級會計學(一)",), ("初級會計學(二)",)]
