"""Rule table for note clauses. Each rule: a regex, a value extractor, polarity, hard.

hard = the clause binds the student (cannot enrol, is dropped, or the credits do not
count); soft = advice, priority order or a description. polarity: only (limited to
value), not (value is excluded), info (describes the course, no limit).
Types beyond the first list in docs/v4/07_structured_data.md exist because the 845
distinct notes carry them often enough to filter on (合開 272 courses, 遠距 26, 會考 21).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

_COL_FULL = {"文": "文", "理": "理", "商": "商", "法": "法", "社科": "社會科學", "社會科學": "社會科學",
             "國務": "國際事務", "國際事務": "國際事務", "創國": "創新國際", "創新國際": "創新國際",
             "傳播": "傳播", "外語": "外語", "教育": "教育", "資訊": "資訊", "本": "本"}
_COL = "文理商法社科會學國務際事務創新傳播外語教育資訊本"
_CN = "一二三四"


@dataclass(frozen=True)
class Rule:
    type: str
    pattern: re.Pattern
    value: Callable[[re.Match, str], str | None]
    polarity: str | Callable[[re.Match, str], str] = "only"
    hard: bool | None = None      # None: decided by soft / hard words in the span
    once: bool = False            # keep only the first match in a clause


def _g(*idx: int, fmt: str = "{}") -> Callable[[re.Match, str], str | None]:
    def f(m: re.Match, _c: str) -> str | None:
        v = next((m.group(i) for i in idx if m.group(i)), None)
        return fmt.format(v.strip()) if v else None
    return f


def _const(v: str) -> Callable[[re.Match, str], str]:
    return lambda m, c: v


def _part(m: re.Match, clause: str) -> str:
    from src.v4.structured.clauses import _part_of
    return _part_of(clause, m.start())[:60]


def colleges(text: str) -> str | None:
    toks = [t.strip() for t in re.split(r"[、及和與,]", text) if t.strip()]
    names = [_COL_FULL.get(t) or _COL_FULL.get(t.removesuffix("學")) for t in toks]
    if not toks or any(n is None for n in names):
        return None
    return "、".join(dict.fromkeys(n + ("院" if n == "本" else "學院") for n in names))


def _college(m: re.Match, c: str) -> str | None:
    return colleges(m.group(1))


def _status(m: re.Match, c: str) -> str:
    s = m.group(0)
    out = []
    if re.search(r"外籍|外國", s):
        out.append("外籍生")
    if "僑生" in s:
        out.append("僑生")
    if "交換生" in s and not re.search(r"外籍交換生", s):
        out.append("交換生")
    return "、".join(out) or s


def _neg(pat: str) -> Callable[[re.Match, str], str]:
    return lambda m, c: "not" if re.search(pat, c[max(0, m.start() - 6):m.end()]) else "only"


def _degree(m: re.Match, c: str) -> str:
    s = m.group(0)
    if re.search(r"博碩|碩博|研究", s):
        return "研究生"
    if "碩" in s:
        return re.search(r"碩[一二]", s).group(0) if re.search(r"碩[一二]", s) else "碩士"
    if "博" in s:
        return "博士"
    return "學士"


def _no_repeat(m: re.Match, c: str) -> str:
    q = re.search(r"[「\"“]([^」\"”]+)[」\"”]", c)
    if q:
        return q.group(1)
    o = re.search(r"原(?:科目|課程)?名稱為([^，,。；;/]+)|本課程原為([^，,。；;/]+)", c)
    return (o.group(1) or o.group(2)).strip("「」") if o else "同一課程"


def _not_grad_who(m: re.Match, c: str) -> str:
    lang = re.search(r"主要使用語言為([^，,。]{1,6}?)[，,]", c)
    if lang:
        return f"母語為{lang.group(1)}者"
    before = re.split(r"[，,。；]", c[:m.start()])[-1]
    who = re.sub(r"(修(習|讀)?|選修|修此課|修本科目|本科目|此課|學分|之成績及)+$", "", before).strip()
    return who or "所有修課學生"


def _priority(m: re.Match, c: str) -> str | None:
    if "優先順序為" in m.group(0) or "優先順序為" in c[m.start():m.end() + 8]:
        tail = re.search(r"優先順序為([^，,。；;/]+)", c)
        return tail.group(1) if tail else None
    v = re.sub(r"^(加簽考量以|加簽以|以|為|選課)", "", m.group(1).strip()).removesuffix("為").strip()
    return v or None


def _add_permit(m: re.Match, c: str) -> str:
    s = _part(m, c)
    if re.search(r"不(開放|接受|受理|予)?(加簽|加選)|不加簽", s):
        return "不開放加簽"
    if "抽籤" in s:
        return "抽籤"
    if re.search(r"(加簽|加選).{0,16}(第一|首|第1)(次|堂|週|周)|(第一|首|第1)(次|堂|週|周).{0,16}加簽", c):
        return "加簽須第一堂到課"
    if re.search(r"僅同意|只加簽", s):
        return s
    if re.search(r"同意|洽|面談|篩選|人工選課|表單", s):
        return "需教師同意"
    if "不開放遞補" in s:
        return "不開放遞補"
    if re.search(r"開放(加簽|遞補)", s):
        return "開放加簽"
    if re.search(r"不開放(電腦|自行|線上)?選課", s):
        return "不開放自行選課"
    return "加簽規則見備註"


def _prereq(m: re.Match, c: str) -> str | None:
    v = re.sub(r"(課程)?(的同學|者|之學生)?(選課|選修|修習|修讀|方能選課|後再選課|再選課|後再行修習本課程)$", "", m.group(1).strip())
    return v.strip() or None


def _program(m: re.Match, c: str) -> str | None:
    s = _part(m, c)
    return None if re.search(r"洽|辦公室|詳", s) else s


def _first_class(m: re.Match, c: str) -> str:
    if "加簽" in c and not re.search(r"所有|已選上|選上|修課同學|修課者|想修課", c):
        return "加簽者須第一堂到課"
    return "第一堂必到"


def _capacity(m: re.Match, c: str) -> str | None:
    if m.group(1):
        return f"上限{m.group(1)}人"
    if m.group(2):
        return f"{m.group(2)}人以上開課"
    if m.group(3):
        return f"加簽{m.group(3)}名"
    return None


def _english(m: re.Match, c: str) -> str:
    s = m.group(0)
    if re.search(r"英語|英文|全英|EMI|English", s):
        return "英文"
    if "雙語" in s:
        return "雙語"
    return (m.group(1) or s).removeprefix("為").removeprefix("採")


def _online(m: re.Match, c: str) -> str:
    return "非同步遠距" if "非同步" in c else "遠距" if "遠距" in c else "線上"


def _joint(m: re.Match, c: str) -> str | None:
    v = re.sub(r"^(與|和|本科目與|本課程與|課程與)", "", (m.group(1) or "").strip()).removesuffix("課程").strip()
    return v or ("合開" if "合開" in m.group(0) else None)


def _exam(m: re.Match, c: str) -> str:
    d = re.search(r"(\d{3})\.(\d{1,2})\.(\d{1,2})(?:\([一二三四五六日]\))?\s*(上午|下午)?", c)
    return f"{d.group(1)}/{int(d.group(2)):02d}/{int(d.group(3)):02d}{d.group(4) or ''}會考" if d else "會考"


def _assigned(m: re.Match, c: str) -> str:
    s = m.group(0)
    return {"灌檔": "灌檔", "分派": "分派"}.get(s, s)


def _location(m: re.Match, c: str) -> str:
    tail = re.search(r"(?:上課地點|集合地點|上課教室|教室)(?:為|在|:)\s*([^，,。；;/]+)", c)
    return (tail.group(1) if tail else _part(m, c)).strip()[:40]


R = re.compile
RULES: list[Rule] = [
    Rule("grade_min", R(rf"限(?:大學部|學士班)?大?([{_CN}])(?:年級)?(?:以上|學生)|學士班([{_CN}])年級以上|限大([{_CN}])"),
         _g(1, 2, 3, fmt="大{}"), hard=True),
    Rule("degree_only", R(r"限(?:[^，,。；;/]{0,6}?)(?:碩士|碩[一二]?|博士|博|研究)(?:班|生)|限學士班|限大學部|博碩選|碩博選|碩群選"),
         _degree, hard=True, once=True),
    Rule("dept_only", R(r"(?:僅限|只限|限|僅供)((?:本系|本所|本學程|(?![大碩博學修曾有選])[^，,。、；;/限]{0,14}?(?:系|所|學程|ETP|DBA|專班|組))"
                        r"(?:[^，,。；;/]{0,4}?(?:雙主修|輔系|輔雙|雙輔|學程)[^，,。；;/修選]{0,6})?)"),
         _g(1), hard=True, once=True),
    Rule("dept_only", R(r"(本系生|本系學生|[^，,。、；;/]{1,8}系(?:同學|學生)?)不得(?:修讀|選修|修習)"), _g(1), "not", True),
    Rule("dept_only", R(r"不收外系|外系不得"), _const("本系"), hard=True),
    Rule("college_only", R(rf"((?:[{_COL}]{{1,4}}[、及和與]?)+)學院(?:學生)?(?:修讀|選修|修習)|^((?:[{_COL}]{{1,4}}[、及]?)+)學院$"),
         lambda m, c: colleges(m.group(1) or m.group(2)), hard=True),
    Rule("college_only", R(rf"限((?:[{_COL}]{{1,4}}[、及]?)+?)學?院(?:學生|同學)|修讀[（(]([{_COL}]{{1,4}})學院[）)]"),
         lambda m, c: colleges(m.group(1) or m.group(2)), hard=True),
    Rule("college_only", R(r"不開放非本院同學"), _const("本院"), hard=True),
    Rule("status_only", R(r"(?:僅供|只限|限)[^，,。；;/]{0,6}?(?:外籍(?:交換生|學位生|生)|外國學生|僑生|國際交換生|交換生)(?:[與及、]外籍(?:交換生|學位生))?"
                          r"|外國學生修讀|僑生(?:[（(]含港澳生[）)])?(?:及外國學生)?修讀|不開放(?:國際)?交換生|不含[^，,。；;/]{0,4}交換生"),
         _status, _neg(r"不開放|不含"), True, once=True),
    Rule("special_class", R(r"(國防|領導|健康中心|擴大輔系|ETP|原)專班"), _g(1, fmt="{}專班"),
         lambda m, c: "only" if re.search(r"不開放非|限|課程|必修|選修|群修", c) else "info", True, once=True),
    Rule("prerequisite", R(r"(?:擋修(?!單|後)|先修(?:科目|課程)?(?:為|:)?|預修科目:?|須修過|需修過|須修畢(?:或同學期修畢)?|需修畢|"
                           r"需先修(?:習)?|須先修(?:過)?|必須修過|限(?:曾)?修(?:習)?過(?:任一門)?|限曾修習|建議(?:學生須)?先修(?:過)?|建議修畢|"
                           r"(?:須|需|建議)具備|請先完成|有先修課程)\s*([^，,。；;/]{1,60})"),
         _prereq),
    Rule("no_repeat", R(r"(?:不得|不可|請勿|避免|不能)重複(?:修習|選課|選修|修課)?|已修(?:過|習)[^，,。；;/]{0,24}?(?:不得|不須|不需|請勿)|原(?:科目|課程)?名稱為|本課程原為"),
         _no_repeat, hard=True, once=True),
    Rule("not_for_graduation", R(r"不(?:計入|列入|採計|予以?採計|算入)[^，,。；;/]{0,6}?(?:畢業學分|學分)|學分不予(?:以)?採計|成績及學分不予採計|非必選亦不計入"),
         _not_grad_who, hard=True, once=True),
    Rule("priority", R(r"([^，,。；;/]{1,30}?)優先(?!順序)|選課優先順序為[^，,。；;/]+"), _priority, "info", False),
    Rule("add_permit", R(r"加簽|加選|遞補|不開放(?:電腦|自行)選課|人工選課|不開放(?:線上)?選課|選課名單由任課教師篩選|經(?:由)?授課老師同意|徵得[^，,。；;/]{0,6}同意|面談|填寫表單|選課前請先洽"),
         _add_permit, lambda m, c: "not" if re.search(r"不(開放|接受|受理)?(加簽|加選)", _part(m, c)) else "info"),
    Rule("first_class_mandatory", R(r"(?:第一|第1|首)(?:堂|次|週|周)(?:課)?[^，,。；;/集]{0,12}?(?:務必|必須|須|必到|未到|出席|到課|缺席)|出席第一(?:堂|次|週|周)|compulsory to attend the first class|首堂課必到"),
         _first_class, "info", True, once=True),
    Rule("capacity", R(r"(?:人數上限|上限|限)\s*(\d+)\s*(?:人|名)|(\d+)\s*人以上開課|只加簽\s*(\d+)\s*名"), _capacity, "info", False),
    Rule("fee", R(r"收費課程|學分費|需繳[^，,。；;/]{0,4}費"), _const("收費"), _neg(r"不收"), False, once=True),
    Rule("intensive", R(r"密集|集中授課|隔週|隔年開設|兩年開一次|移地教學|異地授課|上八週|(?:\d{1,2}/\d{1,2}[、,]\s*){2,}\d{1,2}/\d{1,2}"),
         _part, "info", False, once=True),
    Rule("location", R(r"上課地點|集合地點|上課教室|教室(?:為|:|請洽|另行)|在[^，,。；;/]{1,16}(?:樓|館|教室|中心|廳)上課|^\d{6}(?:教室)?|研討室|電腦教室|公企中心|中研院"),
         _location, "info", False, once=True),
    Rule("english_taught", R(r"英語授課|英文授課|全英(?:文)?(?:授課)?|EMI|雙語授課|English[- ]taught|([^\s,、/]{1,3}[語文])授課"), _english, "info", False, once=True),
    Rule("pass_fail", R(r"通過/不通過|通過、不通過|pass/fail|P/F", re.I), _const("通過/不通過"), "info", True),
    Rule("online", R(r"遠距|非同步|線上(?:課程|授課|上課)|同步視訊|Teams|NTU COOL"), _online, "info", False, once=True),
    Rule("joint_offering", R(r"([^，,。；;/]{0,24}?)合開|([^，,。；;/]{1,10})主開"), lambda m, c: _joint(m, c) if m.group(1) is not None else m.group(2),
         "info", False),
    Rule("program", R(r"學程|模組|類群|(?<![A-Za-z])群[A-E](?![a-z])|基礎科目|核心課程|Track\s*\d|組(?:必修|選修|群修)|擴大輔系課程|輔系群修|共同必修|必修課程"),
         _program, "info", False, once=True),
    Rule("assigned", R(r"灌檔|分派|不開放自行退選|不開放自由選課"), _assigned, "info", None),
    Rule("id_parity", R(r"(單|雙)號(?:修讀|班|同學)|學號末碼為(單|雙)數"), _g(1, 2, fmt="{}號"), "only", True, once=True),
    Rule("common_exam", R(r"會考"), _exam, "info", True, once=True),
    Rule("equipment", R(r"自備筆電|手提電腦|筆電|必須有課本|自備"), _part, "info", False, once=True),
    Rule("schedule_info", R(r"上課時間(?:需|請|另行|與|為|:)|確切上課時間|時間(?:及|與)?地點另行公告|另行公告|授課日期|上課日期|另行通知"),
         _part, "info", False, once=True),
    Rule("credit_rule", R(r"抵免|可替代|開放替代|認列|採計為"), _part, "info", False, once=True),
    Rule("extra_session", R(r"(?:TA)?(?:實習課|實驗課|討論課|TA課)"), _part, "info", False, once=True),
    Rule("teaching_staff", R(r"授課(?:教師|師資)為|教師為|由[^，,。；;/]{1,24}?(?:老師|教授|醫師)(?:親自)?授課|共同授課|業師|講座教授"),
         _part, "info", False, once=True),
    Rule("cohort", R(r"限(\d{3})(?:學年度)?(以前|以後|級)(?:入學)?(?:學生|新生)?"), lambda m, c: f"{m.group(1)}{m.group(2)}入學",
         "only", True, once=True),
    Rule("open_to", R(r"(?<!不)開放([^，,。；;/]{1,16}?)(?:選修|修課|選課|修讀|選校)"), _g(1), "info", False),
]
