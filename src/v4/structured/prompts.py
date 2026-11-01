"""Prompts for the LLM fallback. Bump VERSION[job] when a prompt changes: the cache key
includes it, so old answers are not reused for a new prompt."""

from __future__ import annotations

VERSION = {"notes": "n1", "prereq": "p1", "grading": "g1", "weekly": "w1"}

_SYS = "你是政治大學課程資料的抽取器。只根據提供的文字回答，不要推測或補充；只輸出 JSON。"

TYPES = """dept_only 限某系所/學程（polarity=not 表示該系不得修）
college_only 限某學院
grade_min 年級下限，value 如 大三
degree_only 限學制，value 學士/碩士/博士/研究生
status_only 限身分，value 外籍生/僑生/交換生
special_class 專班課程
cohort 限入學年度
prerequisite 先修或擋修，value 為課名
no_repeat 與某課不得重複修習，value 為該課名
not_for_graduation 某些學生修了不計畢業學分，value 為哪些學生
priority 某些學生優先，value 為哪些學生
add_permit 加簽或選課方式規則
first_class_mandatory 第一堂必到
capacity 人數上限或開課人數
fee 收費
intensive 密集、隔週、特定日期上課
location 上課地點
english_taught 授課語言，value 英文/中文/雙語/其他語言名
pass_fail 通過/不通過
online 遠距或線上
joint_offering 合開，value 合開對象
program 屬於某學程、模組、群組、組別必修
assigned 灌檔、分派、不可自行退選
id_parity 限學號單號或雙號
common_exam 會考或共同考試
equipment 需自備設備或教材
schedule_info 上課時間說明
credit_rule 抵免、替代、認列
open_to 開放某些人修
teaching_staff 授課教師或業師說明
extra_session 實習課、討論課時段
other 以上皆非但與選課有關
skip 只是前一句的延續或沒有新資訊"""


def notes_prompt(text: str) -> str:
    return f"""下面是一門課的備註全文，以及規則沒有處理到的片段。請只針對「未處理片段」抽出選課資訊。
每個片段可以有 0 到多項，每一項欄位：
- part：片段編號
- type：只能用下列其中一個
{TYPES}
- value：正規化後的短值（系所名、年級、課名、人數、地點、日期），盡量 15 字以內
- polarity：only（限定）、not（排除）、info（說明）
- hard：true 表示不符合就不能修、會被退選或學分不算；false 表示建議、優先順序或說明

回傳 {{"items":[{{"part":1,"type":"...","value":"...","polarity":"...","hard":false}}]}}

{text}"""


def prereq_prompt(text: str) -> str:
    return f"""下面是課程備註裡的先修條件。列出其中每一門先修課的課名。
規則：
- 照原文寫法，保留（一）（二）等編號；「A（一）（二）」拆成「A（一）」與「A（二）」
- 原文用簡稱（如 初會、五管）就照寫簡稱，不要自行展開
- 若是知識或能力而不是課名（如 統計學基礎知識、基礎德文能力），kind 用 knowledge
- logic：all 表示都要修過，any 表示擇一即可
- consent_ok：原文說經老師同意也可以修時為 true

回傳 {{"courses":[{{"name":"...","kind":"course"}}],"logic":"all","consent_ok":false}}

先修條件：{text}"""


def grading_prompt(text: str) -> str:
    return f"""下面是一門課的評分方式。列出最上層的評分項目與所佔百分比，加總應為 100。
規則：
- 某一項底下的細分（例如 平時成績 40% 底下的 小考 20%、作業 20%）不要另外列
- 以分數表示時（例如 出席 20分、總分 100分）換算成百分比
- 「各佔」要拆開：期中考與期末考各佔 30% → 兩項各 30
- 「* 2」「x2」表示乘上次數
- 看不出比例時 pct 用 null
- label 用原文的短名稱，20 字以內

回傳 {{"items":[{{"label":"...","pct":30}}],"basis":"percent"}}，basis 為 percent / points / none

評分方式：{text[:5000]}"""


def weekly_prompt(text: str) -> str:
    return f"""下面是一門課的每週進度（網頁表格被攤平，儲存格之間以 | 分隔）。整理成每週一列：
- week：週次整數；若以單元、場次排列，用其序號
- date：原文日期，沒有就 null
- topic：該週主題，30 字以內，用原文用詞，不要加讀物書目
- 期中考、期末考、放假、停課也要列
若內容不是逐週或逐單元的進度（例如只有一段課程說明），回傳 {{"weeks":[]}}

回傳 {{"weeks":[{{"week":1,"date":"2/24","topic":"..."}}]}}

進度：{text[:6000]}"""


PROMPTS = {"notes": (_SYS, notes_prompt), "prereq": (_SYS, prereq_prompt),
           "grading": (_SYS, grading_prompt), "weekly": (_SYS, weekly_prompt)}
