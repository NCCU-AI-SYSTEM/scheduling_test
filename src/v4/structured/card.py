"""Compact course card for LLM prompts, rendered only from structured rows.

    render_card(course, sessions, targets, restrictions, eval_items, weekly) -> str

Every card has the same line order so a prompt can rely on it:
  1 【name】 sub_num｜teachers｜credits｜kind (general-ed category, core)｜college dept level
  2 時間 (+ lab/discussion sessions)｜教室｜span and enrolment mode when not 一般/單學期
  3 對象 target classes｜授課語言
  4 限制 hard restrictions first, then soft ones (at most 6)
  5 評分 items with percent (or 大綱未列比例)
  6 授課 teaching percent｜AI policy
  7 週次 first max_weeks weekly topics
  8 簡介 first 120 characters of objective or syllabus
Lines with nothing to say are omitted. A cancelled course gets 【已停開】 in line 1.
"""

from __future__ import annotations

from src.v4.structured.aipolicy import LABEL_ZH

WEEKDAY = "一二三四五六日"
KIND_ZH = {"lab": "實習課", "discussion": "討論課"}
TEACH = [("teach_lecture", "講述"), ("teach_discussion", "討論"), ("teach_group", "小組"),
         ("teach_elearning", "數位"), ("teach_other", "其他")]
RESTR_ZH = {"dept_only": "限系所", "college_only": "限學院", "grade_min": "年級", "degree_only": "限學制",
            "status_only": "限身分", "special_class": "專班", "cohort": "入學年", "prerequisite": "先修",
            "no_repeat": "勿重複修", "not_for_graduation": "不計畢業學分", "priority": "優先",
            "add_permit": "加簽", "first_class_mandatory": "第一堂", "capacity": "人數", "fee": "收費",
            "intensive": "密集", "location": "地點", "english_taught": "授課語言", "pass_fail": "通過制",
            "online": "遠距", "joint_offering": "合開", "program": "學程", "assigned": "分發",
            "id_parity": "學號", "common_exam": "會考", "equipment": "自備", "schedule_info": "時間",
            "credit_rule": "學分認列", "open_to": "開放", "teaching_staff": "師資", "extra_session": "另有時段",
            "other": "其他"}
SKIP_IN_CARD = {"location", "schedule_info", "extra_session", "teaching_staff", "equipment", "other"}


def _sessions(sessions: list[dict]) -> str:
    main = [s for s in sessions if s["kind"] == "main"]
    extra = [s for s in sessions if s["kind"] != "main" and not s["in_time_field"]]
    fmt = lambda s: f"週{WEEKDAY[s['weekday'] - 1]} {s['start']}–{s['end']}"
    out = "、".join(fmt(s) for s in main) or "未定"
    if extra:
        out += "（另：" + "、".join(f"{KIND_ZH.get(s['kind'], s['kind'])} {fmt(s)}" for s in extra) + "）"
    return out


def _restriction(r: dict) -> str:
    label = RESTR_ZH.get(r["type"], r["type"])
    neg = "非" if r["polarity"] == "not" and r["type"] not in ("add_permit", "fee") else ""
    return f"{label}:{neg}{r['value']}"


def render_card(course: dict, sessions: list[dict], targets: list[dict], restrictions: list[dict],
                eval_items: list[dict], weekly: list[dict], max_weeks: int = 8) -> str:
    c = course
    kind = c["kind_label"]
    if c["ge_category"]:
        kind = f"{c['ge_category']}通識" + ("（核心）" if c["ge_core"] else "")
    org = " ".join(x for x in (c["college"] and f"{c['college']}學院", c["dept_name"], c["level"]) if x)
    head = (f"【{c['name']}】" + ("【已停開】" if c["cancelled"] else "") +
            f" {c['sub_num']}｜{'、'.join(c['teachers']) or c['teacher_raw']}｜{c['credits']:g}學分｜{kind}" +
            (f"｜{org}" if org else ""))
    lines = [head]
    l2 = [f"時間:{_sessions(sessions)}"]
    if c["classroom_raw"]:
        l2.append(f"教室:{c['classroom_raw']}")
    if c["semester_span"] and c["semester_span"] != "單學期":
        l2.append(c["semester_span"])
    if c["enroll_mode"] and c["enroll_mode"] != "一般":
        l2.append(f"選課方式:{c['enroll_mode']}")
    lines.append("｜".join(l2))
    l3 = []
    if targets:
        l3.append("對象:" + "、".join(t["raw"] for t in targets[:6]))
    elif c["unit_raw"]:
        l3.append(f"開課單位:{c['unit_raw']}")
    l3.append(f"授課語言:{c['lang']}")
    lines.append("｜".join(l3))
    shown = [r for r in restrictions if r["type"] not in SKIP_IN_CARD]
    shown = sorted(shown, key=lambda r: (not r["hard"], r["type"]))[:6]
    if shown:
        lines.append("限制:" + "；".join(_restriction(r) + ("（硬性）" if r["hard"] else "") for r in shown))
    scored = [e for e in eval_items if e["pct"] is not None]
    if scored:
        lines.append("評分:" + "、".join(f"{e['label']} {e['pct']:g}%" for e in scored[:8]))
    elif eval_items:
        lines.append("評分:" + "、".join(dict.fromkeys(e["category"] for e in eval_items)) + "（大綱未列比例）")
    teach = "、".join(f"{zh} {c[k]}%" for k, zh in TEACH if c.get(k))
    ai = LABEL_ZH.get(c["ai_policy"] or "", "未填")
    lines.append("｜".join(x for x in (teach and f"授課:{teach}", f"AI:{ai}") if x))
    if weekly:
        topics = [f"{w['week']} {w['topic']}" for w in weekly[:max_weeks] if w["topic"]]
        more = f"…共{len(weekly)}週" if len(weekly) > max_weeks else ""
        lines.append("週次:" + "；".join(topics) + more)
    intro = c["objective"] or c["syllabus"]
    if intro:
        lines.append(f"簡介:{intro[:120]}")
    return "\n".join(lines)
