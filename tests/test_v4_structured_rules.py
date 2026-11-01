"""Rule parsers of src/v4/structured, on strings copied from 114-2 COURSE rows."""

from src.v4.structured import aipolicy, grading, sessions, weekly
from src.v4.structured.clauses import parse_info, parse_note
from src.v4.structured.scalars import classroom, general_ed, teaching
from src.v4.structured.targets import parse_unit
from src.v4.structured.text import clean, merge_fragments, split_teachers, text_lang, textbook_clean


# ── text ─────────────────────────────────────────────────────────────────────
def test_clean_placeholders_and_prefix():
    assert clean("＠備註:無") is None
    assert clean("＠異動資訊:無") is None
    assert clean("＠備註:限大三以上修習") == "限大三以上修習"
    assert clean("A &amp; B") == "A & B"


def test_merge_fragments_glues_phrase_pieces():
    assert merge_fragments("GNSS|、|GIS|與|Map") == "GNSS、GIS與Map"
    assert merge_fragments("課程介紹|金融科技課程介紹") == "課程介紹 | 金融科技課程介紹"
    assert merge_fragments("Midterm 1|20%|Midterm 2|20%") == "Midterm 1 20% | Midterm 2 20%"


def test_textbook_boilerplate_removed():
    raw = ("Management 15th Ed. Robbins (2021)|已申請之圖書館指定參考書目圖書館指定參考書查詢|相關處理要點|"
           "維護智慧財產權，務必使用正版書籍。Respect Copyright.")
    assert textbook_clean(raw) == "Management 15th Ed. Robbins (2021)"


def test_text_lang_and_teachers():
    assert text_lang("This course introduces the basic ideas of statistics and probability.") == "en"
    assert text_lang("本課程介紹統計學的基本概念與機率模型，並以實例說明。") == "zh"
    assert split_teachers("張瑜芸、黃瓊之、賴瑶鍈") == ["張瑜芸", "黃瓊之", "賴瑶鍈"]
    assert split_teachers("詳備註") == []


# ── scalars ──────────────────────────────────────────────────────────────────
def test_teaching_template_with_other_label():
    t = teaching("授課方式|Teaching Approach|35%|講述|Lecture|30%|討論|Discussion|25%|小組活動|Group activity|"
                 "10%|數位學習|E-learning|0%|其他：|Others:")
    assert (t["lecture"], t["discussion"], t["group"], t["elearning"], t["other"], t["sum"]) == (35, 30, 25, 10, 0, 100)
    t2 = teaching("60%|講述|Lecture|0%|討論|Discussion|0%|小組活動|Group activity|0%|數位學習|E-learning|"
                  "40%|其他：|Others:|實作")
    assert t2["other"] == 40 and t2["other_label"] == "實作"


def test_general_ed_and_classroom():
    assert general_ed("跨領域(人文、社會)", 0) == ("跨領域", "人文、社會", False)
    assert general_ed("社會通識", 1) == ("社會", None, True)
    assert general_ed("", 0) == (None, None, False)
    assert classroom("商館260206") == ("商館", "260206")
    assert classroom("綜合中央大教室 (270751)") == ("綜合中央大教室", "270751")
    assert classroom("教室洽系所") == ("教室洽系所", None)


# ── sessions ─────────────────────────────────────────────────────────────────
def test_time_code_minutes_and_blocks():
    s = sessions.parse_time_code("五D56")
    assert [(x.weekday, x.start, x.end, x.periods) for x in s] == [(5, "13:10", "16:00", "D56")]
    two = sessions.parse_time_code("三23456")   # 4 and 5 are not adjacent (C, D between)
    assert [(x.start, x.end) for x in two] == [("09:10", "12:00"), ("14:10", "16:00")]
    assert sessions.parse_time_code("未定或彈性") == []


def test_note_lab_sessions():
    main = sessions.parse_time_code("二D56")
    lab = sessions.note_sessions("英語授課。實習課為二、78。商院ETP學生及外籍生優先。", main)
    assert [(x.weekday, x.start, x.end, x.kind, x.in_time_field) for x in lab] == [(2, "16:10", "18:00", "lab", False)]
    two = sessions.note_sessions("實習課為三、C及五、78", main)
    assert {(x.weekday, x.periods) for x in two} == {(3, "C"), (5, "78")}
    assert sessions.note_sessions("五4為實習課", main)[0].periods == "4"
    disc = sessions.note_sessions("第7、8節為正課，第E節為討論課", sessions.parse_time_code("一78E"))
    assert [(x.weekday, x.periods, x.kind, x.in_time_field) for x in disc] == [(1, "E", "discussion", True)]
    clock = sessions.note_sessions("2. 實習課時間為星期一17：10~19：00於逸仙樓", main)
    assert [(x.weekday, x.start, x.end) for x in clock] == [(1, "17:10", "19:00")]


# ── targets ──────────────────────────────────────────────────────────────────
def test_unit_targets():
    t = parse_unit("法律三甲法律三乙")
    assert [(x.dept_abbr, x.level, x.year, x.section) for x in t] == [("法律", "學士", 3, "甲"), ("法律", "學士", 3, "乙")]
    assert [(x.dept_abbr, x.level, x.year) for x in parse_unit("外交碩一外交博一")] == [("外交", "碩士", 1), ("外交", "博士", 1)]
    assert [(x.year, x.section) for x in parse_unit("科智碩1A科智碩2A")] == [(1, "A"), (2, "A")]
    assert [x.section for x in parse_unit("地三土管地三土資")] == ["土管", "土資"]
    assert parse_unit("外文中心") == []
    assert parse_unit("體育") == []


# ── restrictions ─────────────────────────────────────────────────────────────
def _types(note):
    return {(r.type, r.value, r.polarity, r.hard) for r in parse_note(clean(note))[0]}


def test_note_grade_and_intensive():
    got = _types("＠備註:限大三以上修習，集中授課六週，每次3小時，實際上課日期請參考教學大綱。")
    assert ("grade_min", "大三", "only", True) in got
    assert any(t == "intensive" for t, *_ in got)


def test_note_dept_and_double_major():
    got = _types("＠備註:僅限歐文系及雙主修學生修讀。")
    assert any(t == "dept_only" and v.startswith("歐文系") and h for t, v, _, h in got)


def test_note_prerequisite_and_exam():
    got = _types("＠備註:財管一，第一堂課請務必出席。擋修初會(一)，本課程於115.6.6(六)下午舉行期末會考，"
                 "修課學生皆須參加，無法參加同學請勿修讀。")
    assert ("prerequisite", "初會(一)", "only", True) in got
    assert ("common_exam", "115/06/06下午會考", "info", True) in got
    assert any(t == "first_class_mandatory" for t, *_ in got)


def test_note_college_status_and_not_for_graduation():
    assert ("college_only", "社會科學學院、國際事務學院", "only", True) in _types("＠備註:社會科學、國際事務學院修讀")
    got = _types("＠備註:1.僅供外籍交換生與外籍學位生修習，外籍學位生修此課學分不予以採計。2.第一堂課務必出席，否則將予以退選。")
    assert ("status_only", "外籍生", "only", True) in got
    assert any(t == "not_for_graduation" and v == "外籍學位生" for t, v, *_ in got)
    lang = _types("＠備註:學生所屬國籍或僑居地之官方語言或主要使用語言為德文，修習本科目之成績及學分不予採計。")
    assert any(t == "not_for_graduation" and v == "母語為德文者" for t, v, *_ in lang)


def test_note_add_permit_and_negative_dept():
    got = _types("＠備註:開放遞補，不開放非本院同學選課，不開放加簽。")
    assert ("college_only", "本院", "only", True) in got
    assert any(t == "add_permit" and v == "不開放加簽" and p == "not" for t, v, p, _ in got)
    assert any(t == "dept_only" and p == "not" for t, _, p, _ in _types("＠備註:法律系同學不得選修"))


def test_note_joint_english_priority():
    got = _types("＠備註:英文授課，金融系優先。")
    assert ("english_taught", "英文", "info", False) in got
    assert ("priority", "金融系", "info", False) in got
    assert ("joint_offering", "碩博", "info", False) in _types("＠備註:選修類群/碩博合開")


def test_info_changes():
    ch = parse_info("＠異動資訊:時間異動於2025/12/24;教室異動於2025/12/24;教師異動於2025/12/23);")
    assert [(c.type, c.date) for c in ch] == [("time", "2025/12/24"), ("classroom", "2025/12/24"),
                                              ("teacher", "2025/12/23")]
    assert parse_info("＠異動資訊:停開於2026/03/10;")[0].type == "cancelled"


# ── grading ──────────────────────────────────────────────────────────────────
def test_grading_label_before_percent():
    g = grading.parse("學生出缺席25﹪上課參與與討論25﹪期末書面報告50﹪")
    assert g.reliable and g.pct_sum == 100
    assert [(i.category, i.pct) for i in g.items] == [("出席參與", 25), ("出席參與", 25), ("書面報告", 50)]


def test_grading_label_after_percent():
    g = grading.parse("60%：作業與課堂實作30%：期末評量（本學期休假多，取消期中考試）10%：課程出席與參與度")
    assert [(i.label, i.pct) for i in g.items][0] == ("作業與課堂實作", 60)
    # 期末評量 is an exam (user, 2026/11/01); 取消期中考試 in parentheses must not make it 期中考
    assert g.items[1].category == "期末考"


def test_grading_bilingual_breakdown_kept_once():
    # 體育 五人制足球 (114-2): the breakdown is written in Chinese then English
    g = grading.parse("(一)出席率與學習態度50%(二)測驗40%(三)其他10%"
                      "Attendance and Learning Attitude (50%)Examinations (40%)Others (10%)")
    assert [(i.label, i.pct) for i in g.items] == [("出席率與學習態度", 50), ("測驗", 40), ("其他", 10)]
    assert [i.category for i in g.items] == ["出席參與", "考試", "other"] and g.reliable
    english_first = grading.parse("Midterm Exam 30% Final Exam 40% Homework 30% 期中考 30% 期末考 40% 作業 30%")
    assert [i.label for i in english_first.items] == ["期中考", "期末考", "作業"]


def test_grading_bilingual_check_leaves_nesting_alone():
    nested = grading.parse("平時成績 50% (1)出席 25% (2)作業 25% 期末考 50%")
    assert [(i.label, i.pct) for i in nested.items] == [("平時成績", 50), ("期末考", 50)]
    same_script = grading.parse("小考一 25% 小考二 25% 小考三 25% 小考四 25%")
    assert len(same_script.items) == 4


def test_grading_drops_total_and_children():
    g = grading.parse("Class Performance: 10%Attendance: 10%Quiz: 20%Homework Assignment: 20%"
                      "Midterm Exam: 20%Final Exam: 20%Total 100%")
    assert g.pct_sum == 100 and len(g.items) == 6
    nested = grading.parse("課程出席及發言:10%課堂報告:40%※小組成績為30%※小組成員互評成績為10%"
                           "期末書面報告:50%※小組成績為40%※小組成員互評成績為10%")
    assert nested.pct_sum == 100 and [i.pct for i in nested.items] == [10, 40, 50]


def test_grading_needs_llm_when_sum_off():
    text = "平時成績40%1.聽說練習(音調50%和流暢度50%) ＋小考30%2.課堂表現10%期中考 30%期末考 30%"
    assert grading.needs_llm(grading.parse(text), text)
    assert not grading.needs_llm(grading.parse("期中考 30% 期末考 30% 作業 40%"), "期中考 30% 期末考 30% 作業 40%")


# ── weekly ───────────────────────────────────────────────────────────────────
HEAD = "每周課程進度與作業要求|Course Schedule & Requirements|"


def test_weekly_w_prefixed_with_hours():
    s = (HEAD + "週次|Week|課程主題|Topic|學習投入時間|課堂講授|In-class Hours|課程前後|Outside-of-class Hours|"
         + "|".join(f"W{n:02d}|0{2 + n // 5}/{10 + n}|課程主題第{n}單元|閱讀講義第一章|3|3" for n in range(1, 9)))
    ws = weekly.parse(s)
    assert [w.week for w in ws] == list(range(1, 9))
    assert (ws[0].date, ws[0].topic, ws[0].hours_in, ws[0].hours_out) == ("02/11", "課程主題第1單元", 3.0, 3.0)


def test_weekly_split_di_n_zhou():
    s = HEAD + "|".join(f"第|{n}|周（|3/{n}|）債務不履行{n}" for n in range(1, 8))
    ws = weekly.parse(s)
    assert len(ws) == 7 and ws[2].date == "3/3" and ws[2].topic == "債務不履行3"


def test_weekly_prefers_exact_next_week_over_hours():
    rows = "|".join(f"{n}|{n + 1}|月|2{n}|日|Ch{n} topic|1.|個人作業|3|4-5" for n in range(1, 8))
    ws = weekly.parse(HEAD + "週次|日期|課程主題|學習投入時間|課堂講授|In-class Hours|課程前後|Outside-of-class Hours|" + rows)
    assert [w.week for w in ws] == list(range(1, 8))
    assert ws[0].hours_out == 4.5
    assert all(w.hours_in == 3.0 for w in ws)


def test_weekly_numbered_header_list_does_not_splice():
    s = HEAD + "1、預習教材:詳見講義|2、每週學時數:2小時|3、每週課程進度:|" + "|".join(
        f"第{c}週  3月{n}日|進度:主題{n}" for n, c in zip(range(1, 9), "一二三四五六七八"))
    ws = weekly.parse(s)
    assert ws[0].topic.startswith("進度:主題1") and len(ws) == 8


# ── ai policy ────────────────────────────────────────────────────────────────
def test_ai_policy_category_and_detail():
    p = aipolicy.parse("有條件開放使用:需經教師同意始得使用Conditional Permitted to Use")
    assert (p.category, p.detail, p.detail_source) == ("conditional", "需經教師同意始得使用", "ai_policy")
    assert aipolicy.parse("完全開放使用Completely Permitted to Use").category == "full"
    assert aipolicy.parse("本課程無涉及AI使用This Course Does Not Involve the Use of AI.").category == "none"
    pointer = aipolicy.parse("有條件開放使用:參見簡介Conditional Permitted to Use",
                             syllabus="本課程介紹政策。使用ChatGPT撰寫報告須註明。其餘依校規。")
    assert pointer.detail_source == "ai_policy+syllabus" and "ChatGPT" in pointer.detail
    assert aipolicy.parse("Course Name:AI & Governance").category == "other"


# ── regressions found in the precision samples (2026/10/19) ──────────────────
def test_grading_category_synonyms_and_generic_label():
    assert grading.category("Mid-term exam") == "期中考"
    assert grading.category("中間試験") == "期中考"
    g = grading.parse("課堂參與與個人表現(評分25%)個人或分組報告(評分20%)期中報告:文教機構品質評估(評分25%)"
                      "期末報告:文教機構經營策略規劃(評分30%)")
    assert [i.label for i in g.items][0] == "課堂參與與個人表現"
    assert g.pct_sum == 100


def test_weekly_lesson_is_not_week_and_cjk_numerals():
    lessons = HEAD + "|".join(f"第{c}課(一)|教學活動:聽力練習|課堂教學:3" for c in "二三四五六七八")
    assert weekly.parse(lessons) == []
    cjk = HEAD + "週次|教學內容|" + "|".join(f"{c}|〈作品{n}〉選讀|講述與討論" for n, c in
                                         enumerate("一二三四五六七八九十", 1))
    ws = weekly.parse(cjk)
    assert [w.week for w in ws] == list(range(1, 11)) and ws[9].topic == "〈作品10〉選讀"


def test_weekly_ordinal_prefix_and_dot_date():
    s = HEAD + "|".join(f"{n}|th|week: Topic number {n}" for n in range(1, 8))
    assert weekly.parse(s)[3].topic == "Topic number 4"
    d = HEAD + "Week|Topic|" + "|".join(f"{n} (03.{10 + n})|Ethics topic {n}" for n in range(1, 8))
    assert (weekly.parse(d)[4].date, weekly.parse(d)[4].topic) == ("03.15", "Ethics topic 5")


def test_priority_value_trailing_wei():
    got = _types("＠備註:資管一乙,加簽考量以高年級者必修及雙輔必修為優先。")
    assert ("priority", "高年級者必修及雙輔必修", "info", False) in got


def test_control_characters_removed():
    assert clean("\x9f Tutorial Time II") == "Tutorial Time II"


def test_weekly_skips_weekday_tag_and_mode_column():
    s = HEAD + "|".join(f"{n}|2/{10 + n} (Tue.)|Topic {n} overview" for n in range(1, 8))
    assert weekly.parse(s)[2].topic == "Topic 3 overview"
    m = HEAD + "|".join(f"{n}|3/{n}|實體課程|研究方法第{n}講" for n in range(1, 8))
    assert weekly.parse(m)[0].topic == "研究方法第1講"
    assert merge_fragments("Introduction to|Narrative Essay") == "Introduction to Narrative Essay"


def test_grading_chinese_percent_words():
    g = grading.parse("一.術科測驗百分之四十二.出缺席及學習精神佔百分之四十三.體育常識佔百分之十四、校園環山健走百分之十")
    assert g.reliable and [i.pct for i in g.items] == [40, 40, 10, 10]
