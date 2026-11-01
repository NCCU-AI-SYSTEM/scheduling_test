"""Query planner: one small-LLM call that turns a student's sentence into a search plan.

Same output schema as the v4 reference labels (scripts/v4/build_eval_v4.py) so the
planner can be scored field by field against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.llm.ollama_http import chat, parse_json_object
from src.v4.assessment import CONDITION_ITEMS, parse_query as parse_assessment
from src.parsers.base import ConditionResult
from src.parsers.ollama_parser import sanitize

PLAN_PROMPT = """你是政大選課助理的查詢規劃器。把學生的一句話拆成 JSON：
- need：想學的主題或內容。去掉時間、語言、學分、系所、課別、評價詞、老師名、課名後剩下的主題；沒有主題就是空字串。
- named_course：明確提到的課名，沒有就 null。
- named_teacher：明確提到的老師姓名，沒有就 null。
- must / must_not：硬條件（must_not 用於不要／避開／除了）。只能用：
  course_lang：授課語言，例如 ["英文"]。想「學」某語言（想學日文）是主題，不是授課語言。
  weekday：1=週一 … 7=週日。 hour_range：早上 [8,12]、中午 [12,13]、下午 [13,18]、晚上 [18,24]、早八 [8,9]。
  point：學分。 kind：["必修"]、["選修"]、["群修"]、["通識"]、["體育"]。 lmt_kind：通識類別如 ["社會通識"]。 unit：系所簡稱如 ["資科"]。
- prefs：軟性偏好，每項 {"aspect": ..., "want": "low" 或 "high"}。aspect 只能是 workload（作業報告考試負擔，涼＝low）、grading（給分，甜＝high）、difficulty（難度，硬＝high）、attendance（點名）、group_work（分組報告）、teaching（老師教學評價）、usefulness（實用）、popularity（熱門、好不好選）。
- intents：可多選 search、lookup、review、eligibility、enroll_chance、compare、out_of_scope。

範例
查詢：週三下午有沒有不用分組的英文授課通識，不要早八
{"need": "", "named_course": null, "named_teacher": null, "must": {"weekday": [3], "hour_range": [13, 18], "course_lang": ["英文"], "kind": ["通識"]}, "must_not": {"hour_range": [8, 9]}, "prefs": [{"aspect": "group_work", "want": "low"}], "intents": ["search"]}
查詢：王小明的統計學甜嗎
{"need": "", "named_course": "統計學", "named_teacher": "王小明", "must": {}, "must_not": {}, "prefs": [{"aspect": "grading", "want": "high"}], "intents": ["lookup", "review"]}

查詢：{query}"""

PLAN_SCHEMA = {"type": "object", "properties": {
    "need": {"type": "string"}, "named_course": {"type": ["string", "null"]},
    "named_teacher": {"type": ["string", "null"]}, "must": {"type": "object"}, "must_not": {"type": "object"},
    "prefs": {"type": "array", "items": {"type": "object", "properties": {
        "aspect": {"type": "string"}, "want": {"type": "string"}}, "required": ["aspect", "want"]}},
    "intents": {"type": "array", "items": {"type": "string"}}},
    "required": ["need", "named_course", "named_teacher", "must", "must_not", "prefs", "intents"]}

ASPECTS = {"workload", "grading", "difficulty", "attendance", "group_work", "teaching", "usefulness", "popularity"}


def _assessment(clauses) -> tuple[str, ...]:
    """Assessment items from a must / must_not object; sanitize() drops this key, so read it first."""
    raw = clauses.get("assessment") if isinstance(clauses, dict) else None
    vals = raw if isinstance(raw, list) else [raw] if raw else []
    return tuple(dict.fromkeys(str(v).strip() for v in vals if str(v).strip() in CONDITION_ITEMS))


@dataclass
class Plan:
    need: str = ""
    named_course: str | None = None
    named_teacher: str | None = None
    cond: ConditionResult = field(default_factory=ConditionResult)
    prefs: list[dict] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    assess_must: tuple[str, ...] = ()
    assess_not: tuple[str, ...] = ()
    seconds: float = 0.0
    ok: bool = True

    @classmethod
    def from_dict(cls, d: dict, seconds: float = 0.0, ok: bool = True) -> "Plan":
        prefs = [p for p in d.get("prefs") or [] if isinstance(p, dict) and p.get("aspect") in ASPECTS
                 and p.get("want") in ("low", "high")]
        return cls(need=(d.get("need") or "").strip(), named_course=d.get("named_course") or None,
                   named_teacher=d.get("named_teacher") or None,
                   cond=ConditionResult(must=sanitize(d.get("must")), must_not=sanitize(d.get("must_not"))),
                   prefs=prefs, intents=[i for i in d.get("intents") or [] if isinstance(i, str)],
                   assess_must=_assessment(d.get("must")), assess_not=_assessment(d.get("must_not")),
                   seconds=seconds, ok=ok)


def plan(query: str, model: str, options: dict | None = None) -> Plan:
    """options pass through to Ollama, e.g. {"num_gpu": 0, "num_thread": 4} on a CPU-only server."""
    res = chat(model, [{"role": "user", "content": PLAN_PROMPT.replace("{query}", query)}], fmt=PLAN_SCHEMA,
               options=options)
    d = parse_json_object(res.content)
    base = Plan(need=query, seconds=res.seconds, ok=False) if d is None else Plan.from_dict(d, res.seconds)
    return with_assessment(base, query)


def with_assessment(p: Plan, query: str) -> Plan:
    """Assessment conditions come from rules on the raw query, merged with any the LLM gave."""
    must, must_not = parse_assessment(query)
    return replace(p, assess_must=tuple(dict.fromkeys(p.assess_must + must)),
                   assess_not=tuple(dict.fromkeys(p.assess_not + must_not)))
