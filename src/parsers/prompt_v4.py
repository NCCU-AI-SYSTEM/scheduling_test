"""Query-understanding prompt shared by every model in the v4 benchmark.

Differences from the v3 prompt (src/parsers/ollama_parser.py):
- explicit rule that wanting to LEARN a language is a topic, not a teaching-language
  condition (v3 gemma4:e4b put course_lang on 45/240 topic queries)
- "不要必修/不要通識" goes to must_not.kind instead of being dropped or inverted
- unit asks for a short department name; matching is substring-based downstream
"""

from __future__ import annotations

PARSE_PROMPT = """你是選課查詢解析器。把學生的查詢拆成 JSON：
- need：學生想學的主題或內容。去掉時間、語言、學分、系所、課別等條件詞後剩下的部分；查詢只有條件沒有主題時為空字串。
- must：必須符合的條件。
- must_not：學生說「不要／避開／除了／不想」的條件。

只能使用以下欄位，沒提到的欄位不要輸出：
  course_lang：授課語言，例如 ["英文"]。注意：學生想「學」某種語言（想學日文、練英文口說）是主題，放進 need，不是授課語言。只有明說用某語言上課（英文授課、全英文、用日文教）才填。
  weekday：星期數字，1=週一 … 7=週日；「週一到週三」= [1,2,3]；「週二或週四」= [2,4]
  hour_range：[開始時, 結束時]。早上 [8,12]、中午 [12,13]、下午 [13,18]、晚上 [18,24]；「早八」= [8,9]；「兩點到五點」= [14,17]
  point：學分數字，例如 3
  kind：["必修"]、["選修"]、["通識"]、["體育"]；「不要必修」放 must_not
  lmt_kind：通識類別，例如 ["社會通識"]、["人文通識"]、["自然通識"]、["外文通識"]、["中文通識"]、["資訊通識"]、["書院通識"]
  unit：開課系所的簡稱，例如 ["企管"]、["資科"]、["外文中心"]

範例
查詢：週三下午英文授課的經濟學，不要早八
{"need": "經濟學", "must": {"weekday": [3], "hour_range": [13, 18], "course_lang": ["英文"]}, "must_not": {"hour_range": [8, 9]}}
查詢：想學日文會話
{"need": "日文會話", "must": {}, "must_not": {}}
查詢：還缺社會通識，不要週五的
{"need": "", "must": {"lmt_kind": ["社會通識"]}, "must_not": {"weekday": [5]}}

查詢：{query}"""

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "need": {"type": "string"},
        "must": {"type": "object"},
        "must_not": {"type": "object"},
    },
    "required": ["need", "must", "must_not"],
}


def render(query: str) -> str:
    # str.format would trip on the JSON braces in the examples
    return PARSE_PROMPT.replace("{query}", query)
