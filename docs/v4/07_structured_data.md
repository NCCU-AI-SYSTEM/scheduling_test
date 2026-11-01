# v4 結構化課程資料庫（course_struct.db）

建置日期 2026/10/19。對象是 114-2 的 2,795 個課號（含停開 140 個，以 `cancelled` 標記）。原始資料 `data/1142.db` 只以唯讀方式開啟，輸出寫到 `data/v4/course_struct.db`。

重建：

```bash
python -m scripts.v4.build_structured              # 規則 + 已快取的 LLM 回答，約 20 秒
python -m scripts.v4.build_structured --llm cloud  # 快取沒有的輸入才送 gemma4:31b-cloud
python -m scripts.v4.verify_structured [--samples] # 覆蓋率、抽樣精確率、一致性檢查、課名簡稱解析
```

程式在 `src/v4/structured/`（21 個檔案，最大的 `aliases.py` 256 行）。規則分詞產生的簡稱需要 jieba，hopelab 的環境有，筆電沒有；沒有 jieba 時 alias 表少掉 687 列 rule 來源，其餘結果相同。正式版本在 hopelab 建置。

## 1. 為什麼要這一層

03_data_analysis.md 列出的問題裡，和這次相關的是：note、info、evaluation、schedule、ai_policy、teaching_approach 都是一整段文字，schedule 與 evaluation 還是被 `|` 攤平的 HTML 表格；搜尋要篩「限大三以上」「有沒有期中考」「第幾週講什麼」時只能對文字做 regex，給 LLM 的課程描述也長短不一。另一個問題是學生寫的課名簡稱（個經、初會、五管、大英）以及備註裡的擋修初會(一)，原本沒有任何對照。

這一層把這些欄位拆成資料列：條件用來過濾，文字欄去掉樣板後給 embedding，`course.card` 是固定格式的短卡片給 prompt。

## 2. 資料表

| 表 | 列數 | 一列代表 | 用途與設計理由 |
|---|---|---|---|
| course | 2,795 | 一個課號 | 純量欄位與清理後文字，篩選與顯示都從這裡讀 |
| course_path | 3,472 | 課號的一條 dp1/dp2/dp3 路徑 | 同一課號掛在多個系所、學程（最多 7 條），保留全部並附 unit.json 名稱 |
| session | 2,799 | 一段上課時間 | 分鐘精度；44 列來自備註（實習課、討論課） |
| target_class | 4,564 | unit 字串裡的一個班級 | 2,006 個課號有班級；開給誰不等於禁止別人選 |
| restriction | 3,000 | 備註裡的一個條件 | 32 種 type，附原文片段 span 以便回查 |
| prereq | 285 | 先修條件裡的一門課 | 把擋修初會(一) 對到 初級會計學(一) |
| change | 732 | 一筆異動 | 停開 140、教室異動 463、時間異動 73、教師異動 56 |
| eval_item | 9,301 | 一個評分項目 | 8,844 列有百分比 |
| ai_policy | 2,795 | 一個課號的 AI 政策 | 類別加細則文字 |
| weekly | 38,701 | 一週進度 | 2,531 個課號有逐週資料 |
| alias | 815 | 一個課名簡稱 | 簡稱對到課名或課名群組 |
| offering / channel | 2,274 / 2,655 | 一堂課 / 一個選課管道 | 由 `src/v4/offering.py` 的 group_offerings 實體化（排除停開課） |
| meta | 5 | 建置資訊 | 建置時間、來源、LLM 模式 |

### 2.1 course 的欄位

| 欄位 | 內容 | 理由 |
|---|---|---|
| course_id, sub_num, course_group | 13 碼課號、9 碼科目代號、前 6 碼 | course_group 用來把同一份大綱的多個班別併成一組 |
| name, name_en, name_base | 課名、英文課名、去掉編號與副標的課名 | name_base 是簡稱解析與跨學期連結的鍵（大學英文（三）：商用英文 → 大學英文） |
| teachers, teacher_raw | 教師 JSON 陣列、原字串 | 184 門多位教師以 、 分隔；詳備註 轉成空陣列 |
| credits, kind_code, kind_label | 學分、必選群代碼與中文 | kind 依 NCCUCourse/DB.py：1 必修 2 選修 3 群修 4 通識 0 其他 |
| ge_category, ge_detail, ge_core | 通識類別（外文、社會、人文、中文、自然、資訊、跨領域、書院）、跨領域的子類、核心通識 | 通識判斷用 lmtKind，kind=0 的 18 門跨領域通識也收得到 |
| semester_span, enroll_mode, cancelled | 單學期/全學年/三學期、tranTpe、停開 | 全學年 295 門需要先修上學期；灌檔與分派的課不能自由加選 |
| lang, english_taught, syllabus_lang | 授課語言、是否英文授課、大綱文字語言（zh/en/mixed/other） | 英文大綱不等於英文授課，兩者分開存 |
| dept_code, dept_name, college, level | 開課單位代碼、名稱、學院、學制 | dept_name 先查 unit.json，非系所單位（通識中心、外文中心、體育室）用 unit 字串 |
| unit_raw, unit_is_class, is_pe | unit 原字串、是否為班級串接、體育課 | 體育課沒有獨立 kind，以 dp1=03/A6 或 unit=體育 判斷 |
| time_raw, time_tba | 時間代碼、未定或彈性 | 103 門 TBA 不參與空堂比對 |
| classroom_raw, building, room_id | 教室原字串、大樓、6 碼教室代號 | |
| ai_policy | none / full / conditional / prohibited / other | 細則在 ai_policy 表 |
| teach_lecture … teach_other, teach_other_label, teach_sum | 授課方式五類百分比、其他的說明、加總 | 模板固定，規則解析 |
| eval_pct_sum, eval_reliable, eval_method | 評分項目百分比加總、加總在 95–105、rule/llm/none | reliable=0 的課不要拿比例做硬篩選 |
| n_weeks, weekly_method | 解析出的週數、rule/llm/none | |
| enroll_limit, enroll_count | RESULT 1142 的限制人數與選課人數 | 抓取時點未記錄，見 03 第 4.3 節 |
| url | teaSchmUrl | |
| note_clean, info_clean | 去掉 ＠備註:、＠異動資訊: 前綴，無 轉 NULL | |
| syllabus, objective, evaluation | NFKC、控制字元移除、`|` 碎片合併 | |
| schedule_clean | 有逐週資料時是「第n週（日期） 主題」一行一週，否則是去樣板後合併的文字 | 每門課都帶的 `每周課程進度與作業要求|Course Schedule & Requirements|` 已移除 |
| textbook_clean | 去掉圖書館樣板 | 2,786 門都帶同一段樣板 |
| card | LLM 用的課程卡片 | 見第 5 節 |

### 2.2 restriction 的 type

使用者給的清單加上資料裡夠常見、值得篩選的類型。每列有 value（正規化短值）、polarity（only 限定、not 排除、info 說明）、hard、source、span、method。

| type | rule | llm | hard | value 例 |
|---|---|---|---|---|
| joint_offering | 292 | 11 | 0 | 學碩、與教育系 |
| add_permit | 191 | 67 | 126 | 不開放加簽、加簽須第一堂到課、需教師同意 |
| location | 234 | 19 | 0 | 體育館羽球場 |
| program | 230 | 23 | 7 | 主題類群、智財組必修 |
| prerequisite | 163 | 31 | 75 | 初級會計學(一)、五管 |
| schedule_info | 104 | 80 | 1 | 上課時間為週二下午 |
| not_for_graduation | 121 | 34 | 155 | 博士生、英文系學生、母語為德文者 |
| first_class_mandatory | 151 | 4 | 154 | 第一堂必到、加簽者須第一堂到課 |
| english_taught | 127 | 2 | 0 | 英文、雙語、中文、泰語 |
| dept_only | 101 | 28 | 128 | 歐文系及雙主修學生、本系生（not） |
| priority | 117 | 5 | 0 | 金融系、所 |
| other | 0 | 107 | 9 | 不接受旁聽 |
| intensive | 86 | 13 | 2 | 集中授課六週 |
| college_only | 79 | 2 | 81 | 社會科學學院、國際事務學院 |
| extra_session | 61 | 1 | 0 | 實習課為二、78 |
| no_repeat | 40 | 14 | 53 | 原科目名稱 |
| degree_only | 19 | 34 | 47 | 研究生、碩士 |
| teaching_staff | 36 | 14 | 0 | 由臺北市立聯合醫院醫師授課 |
| assigned | 46 | 1 | 16 | 灌檔、分派 |
| status_only | 25 | 18 | 42 | 外籍生、交換生（not） |
| open_to, fee, special_class, credit_rule, online, common_exam, capacity, grade_min, equipment, id_parity, cohort, pass_fail | 各 1–29 | 0–10 | | 大三、單號、114以前入學、115/06/06下午會考 |

新增類型的理由：合開 272 門、地點 155 門、遠距 26 門、會考 21 門、學號單雙號 7 門、入學年度 8 門在備註裡出現，原清單沒有對應類型就只能丟進 other。polarity 欄是因為同一類條件有限定也有排除（限本系 對 本系生不得修讀），不另開 dept_excluded 類型。

hard 的定義：不符合就不能選、會被退選或學分不算為 1；建議、優先、說明為 0。規則依 type 給預設值，prerequisite 與 add_permit 依片段裡的字決定（擋修、須、限 → 1；建議、優先 → 0）。

## 3. 各欄位的抽取方法

| 欄位 | 方法 | 說明 |
|---|---|---|
| session | 規則 | 節次表 A B 1 2 3 4 C D 5 6 7 8 E F G H，每節 hh:10–(hh+1):00；和官方 timeEn 逐門比對 2,692 門全部一致 |
| session（備註） | 規則 | 實習課為四、78／五4為實習課／第E節為討論課／實習課時間為星期一17:10~19:00 等 5 種寫法；備註時段若已在 time 內，in_time_field=1 |
| target_class | 規則 | `(簡稱)(碩/博/專)?(年級)(班別)?` 依序切；地三土管地三土資 這種組別寫法另外處理 |
| restriction | 規則 + LLM | 備註依 。；與 1. 2. 切成子句，31 條規則各自比對；規則沒碰到、長度 4 字以上的片段送 LLM（377 個相異備註），LLM 只能回傳上表的 type；與規則重複或屬於單值類型（會考、第一堂、授課語言等）已有規則結果的 LLM 項目丟掉 |
| prereq | 規則 + LLM | 先試整串、再依 、及和或 切開後逐一解析；任何一段解析失敗才把該條件送 LLM 列課名（59 個相異片段），LLM 被要求保留原文簡稱與編號，A(一)(二) 拆成兩門 |
| change | 規則 | `(停開|教室異動|時間異動|教師異動)於YYYY/MM/DD`；info 只有類型與日期，沒有舊值新值 |
| eval_item | 規則 + LLM | 百分比前後哪邊有標籤就用哪邊；去掉 Total 100%；子項目加總等於上一項時刪掉子項目；百分之四十 轉 40%；加總落在 95–105 算 reliable。不 reliable 而文字有百分比或分數的 273 門（204 份相異文字）送 LLM 取最上層項目，LLM 結果較接近 100 才取代規則結果 |
| eval category | 規則 | src/v4/assessment.py 的 FORMAT_KEYS，加上 考試（未說明期中期末的考試）與 other；Mid-term、中間試験 先正規化 |
| weekly | 規則 + LLM | 在儲存格序列裡找 1, 2, 3 … 的週次標記（純數字、W01、第n週、Week n、一二三、1. 列表、(3/9) 日期），不同寫法分開找取最長；下一週優先於跳週，避免把時數 3|4-5 當成第 3 週。找不到 6 週以上且內容 150 字以上的 329 門（305 份相異文字）送 LLM |
| ai_policy | 規則 | 四個固定中文標籤開頭；細則去掉黏在後面的英文標籤；細則是空的或只寫 參見簡介、Please see the evaluation section 時，到 evaluation、syllabus 找含 AI、ChatGPT、生成式 的句子 |
| alias | 統計 + 規則 + 少量人工 | 見第 6 節 |
| 文字欄 | 規則 | NFKC、html.unescape、C0/C1 控制字元、樣板、`|` 碎片合併（短 CJK 片段或 與、的 等連接字黏回去，欄位邊界保留為 ` | `） |

LLM：`gemma4:31b-cloud`，2 個並發，`src/llm/ollama_http.py` 的 `chat()` 與 `parse_json_object()`。四個 job 共 945 次請求，第一次完整執行 977 秒，0 次錯誤。回答以 sha1（job、prompt 版本、輸入文字）為鍵存在 `data/v4/structured_cache/*.jsonl`，改 prompt 時調 `prompts.VERSION` 就會重問。prompt 摘要：

- notes：給備註全文與未處理片段編號，要求每項回 part、type（限定清單，含 skip 表示只是前句的延續）、value（15 字內）、polarity、hard。
- prereq：列出每門先修課，照原文寫法保留編號與簡稱，知識能力標 knowledge，標出 logic（all/any）與是否可經老師同意。
- grading：只列最上層項目，子項目不列，分數換算百分比，「各佔」拆開，看不出比例填 null。
- weekly：整理成 week、date、topic（30 字內、原文用詞），單元式排列用序號，不是逐週進度就回空陣列。

## 4. 驗證

### 4.1 覆蓋率（全部課號）

分母是原始欄位帶有該資訊的課，判斷原始欄位的條件盡量和抽取程式分開寫（例如週次表用獨立的週次標記與日期計數，限制類別用 03 第 1.7 節的 regex）。

| 欄位 | 覆蓋率 | 分母的定義 |
|---|---|---|
| session | 100.0%（2,692/2,692） | time 有星期加節次 |
| session（備註） | 100.0%（36/36） | 備註寫了實習課或討論課且有星期節次或時刻 |
| target_class | 100.0%（2,006/2,006） | unit 含年級字元 |
| dept_name | 100.0%（2,795/2,795） | |
| college | 100.0%（2,141/2,141） | dp1 有學院列 |
| ge_category | 100.0%（221/221） | lmtKind 非空 |
| teaching | 100.0%（2,785/2,786） | 模板有百分比；got 為五類加總等於 100 |
| ai_policy | 99.1%（2,759/2,783） | 欄位非空 |
| ai_detail | 100.0%（1,123/1,123） | 有條件開放或禁止 |
| restriction | 100.0%（1,499/1,499） | 備註有內容 |
| change | 100.0%（648/648） | 異動資訊有內容 |
| eval_item | 96.0%（2,675/2,786） | evaluation 有內容 |
| eval_pct | 98.5%（2,410/2,447） | evaluation 有阿拉伯數字百分比；got 為加總 95–105 |
| weekly | 99.9%（1,862/1,863） | schedule 是逐週表格（獨立計數 6 個以上連續週次或日期） |
| weekly（寬） | 98.0%（2,526/2,578） | schedule 去樣板後 150 字以上 |
| syllabus、objective、evaluation、schedule_clean | 100.0%（2,786/2,786） | |
| textbook_clean | 100.0%（2,457/2,457） | 去樣板後仍有內容 |

備註限制的逐類核對（分母是 03 第 1.7 節 regex 命中的課，got 為有該 type 的列）：prerequisite 144/144、add_permit 142/142、english_taught 118/118、priority 117/117、not_for_graduation 54/54、first_class_mandatory 78/78、college_only 49/49、intensive 65/65、no_repeat 12/12、grade_min 19/19、common_exam 21/21、fee 18/18、joint_offering 272/272、location 118/118 都是 100%。兩類低於 90%：

- status_only 89.5%（34/38）：沒對到的 4 門寫的是 商院ETP學生及外籍生優先、開放國際交換生選修，存成 priority 與 open_to，不是身分限制。若改成問「這段文字有沒有被某一列收進去」則 38/38。
- online 88.0%（22/25）：沒對到的 3 門寫的是 不開放線上選課，regex 把它當成線上授課，實際不是遠距課。

### 4.2 依通識類別

| 欄位 | 外文 | 社會 | 人文 | 中文 | 自然 | 資訊 | 跨領域 | 書院 | 核心通識 | 非通識 |
|---|---|---|---|---|---|---|---|---|---|---|
| sessions | 100%（56） | 100%（37） | 100%（33） | 100%（31） | 100%（27） | 100%（13） | 100%（18） | 100%（3） | 100%（36） | 100%（2474） |
| teaching | 100%（56） | 100%（37） | 100%（33） | 100%（31） | 100%（27） | 100%（13） | 100%（18） | 100%（6） | 100%（36） | 100%（2565） |
| ai_policy | 100%（56） | 97%（37） | 100%（33） | 100%（31） | 100%（27） | 92%（13） | 100%（18） | 100%（6） | 100%（36） | 99%（2562） |
| ai_detail | 100%（47） | 100%（12） | 100%（17） | 100%（15） | 100%（11） | 100%（8） | 100%（4） | 100%（1） | 100%（18） | 100%（1008） |
| restriction | 100%（56） | 100%（22） | 100%（12） | 100%（31） | 100%（17） | 100%（11） | 100%（8） | 100%（1） | 100%（18） | 100%（1341） |
| eval_item | 100%（56） | 89%（37） | 100%（33） | 100%（31） | 96%（27） | 100%（13） | 100%（18） | 100%（6） | 86%（36） | 96%（2565） |
| eval_pct | 100%（56） | 93%（30） | 97%（32） | 100%（31） | 100%（24） | 100%（13） | 100%（18） | 100%（2） | 97%（30） | 98%（2241） |
| weekly | 100%（52） | 100%（19） | 100%（30） | 100%（27） | 100%（23） | 100%（11） | 100%（13） | 100%（3） | 100%（23） | 100%（1685） |
| weekly（寬） | 100%（56） | 97%（29） | 100%（33） | 100%（31） | 96%（27） | 100%（13） | 100%（17） | 100%（3） | 94%（31） | 98%（2369） |

括號內是分母。target_class 只有非通識課有（通識課的 unit 是 教務處通識教育中心 這類單位名）。

eval_item 在社會通識 89%（33/37）與核心通識 86%（31/36）偏低，沒對到的 5 門核心通識（多數同時是社會通識）evaluation 只寫 請參閱課程附件之授課大綱、見附檔、see attached file，評分方式在附件 PDF，資料庫沒有。資訊通識的 ai_policy 原本是 92%（12/13），原因是第 7 節的課名誤植；2026/11/01 修復後是 100%。

### 4.3 依學院

| 欄位 | 文 | 社會科學 | 商 | 傳播 | 外語 | 法 | 理 | 國際事務 | 教育 | 資訊 | 創新國際 | 非學院單位 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| target_class | 100%（193） | 100%（383） | 100%（367） | 100%（148） | 100%（317） | 100%（175） | 100%（93） | 100%（116） | 100%（81） | 100%（84） | 100%（49） | - |
| ai_policy | 100%（197） | 100%（380） | 100%（405） | 100%（156） | 100%（377） | 100%（175） | 100%（99） | 100%（116） | 100%（85） | 100%（96） | 100%（49） | 100%（648） |
| restriction | 100%（33） | 100%（117） | 100%（311） | 100%（105） | 100%（189） | 100%（76） | 100%（24） | 100%（61） | 100%（35） | 100%（36） | 100%（36） | 100%（476） |
| eval_item | 97%（197） | 98%（383） | 91%（405） | 99%（156） | 95%（377） | 94%（175） | 99%（99） | 96%（116） | 94%（85） | 97%（96） | 98%（49） | 98%（648） |
| eval_pct | 100%（184） | 99%（352） | 98%（336） | 93%（144） | 100%（342） | 99%（109） | 98%（84） | 100%（107） | 97%（69） | 99%（68） | 98%（44） | 99%（608） |
| weekly | 100%（137） | 100%（242） | 100%（243） | 100%（131） | 100%（259） | 99%（78） | 100%（61） | 100%（60） | 100%（74） | 100%（58） | 100%（43） | 100%（477） |
| weekly（寬） | 98%（191） | 98%（368） | 96%（333） | 97%（153） | 100%（370） | 95%（130） | 93%（96） | 99%（106） | 100%（81） | 100%（78） | 96%（49） | 99%（623） |

非學院單位是通識中心、外文中心、體育室、學程等開的課。session、teaching、ai_detail、textbook_clean 各學院都是 100%，表內省略。

低於 90% 的兩格：

- 教育學院 eval_item 原本 88%（75/85）：10 門裡 5 門的 evaluation 欄被爬蟲寫成 Course Name:…（見第 7 節），另外幾門寫 如附件、find attached file。2026/11/01 修復爬蟲錯誤後是 94%（80/85），剩下的都是評分在附件。
- 創新國際學院 ai_policy 原本 86%（42/49），7 門的 ai_policy 欄同樣被寫成 Course Name:…；修復後 100%。

商學院 eval_item 91%（370/405）在 90% 邊上，未對到的多半寫 Please see my website for details 或 Please see the attachment。

### 4.4 精確率（固定種子 20261019 抽樣，逐筆對照原文）

| 欄位 | 精確率 | 錯誤 |
|---|---|---|
| restriction（全部列抽 40） | 100%（40/40） | 2 列 value 太籠統（加簽規則見備註），內容沒錯 |
| restriction（只抽 LLM 列 30） | 77%（23/30） | 國貿一甲 是灌檔對象卻被標成 dept_only；不含選讀生/交換生 講的是灌檔對象，被標成身分限制；115提前入學新生 類型應為 cohort；詳閱教學大綱 被標成 add_permit（2 列）；雙輔生請修習該系課程 被標成 priority；一列 value 憑空加了 用Python學財務計量 |
| eval_item（40） | 98%（39/40） | Midterm and final exams 40% 一項只記成 期中考；另有 4 列 label 帶到前一句文字，但項目、比例、類別都對 |
| weekly（全部列抽 40） | 98%（39/40） | 一列 topic 把活動欄與時數黏在一起（授課報告人/評論人3 8授課） |
| weekly（只抽 LLM 列 30） | 100%（30/30） | 單元式課程的週次是 LLM 依序編號，原文沒有週次時不能當成實際週次 |
| alias（40） | 85%（34/40） | 錯的 6 列都是 rule 來源：初德語、財專研、倫學、健學、國法、資研 的直覺讀法和對到的課不同；rate 來源 7/7 正確 |
| ai_policy 細則（40） | 100%（40/40） | 3 列細則只有 請見課程大綱、See attachment，原始資料裡找不到更多 |
| prereq（40） | 100%（40/40） | 一列 上學期本班（全學年課的連續性要求）抽對但沒有課名可對 |

restriction 整體 3,000 列中 LLM 產生 558 列（18.6%），所以整體精確率大約是規則列的 100% 與 LLM 列的 77% 加權。

抽樣與判定檔在 `data/v4/structured_audit/`：`sample_<field>.jsonl` 是抽出的列與原文，`judged_<field>.jsonl` 是逐筆判定與理由，`report.json` 是覆蓋率與一致性檢查的完整結果。抽查過程中發現並修正的問題（每項都補了測試）：Mid-term 與 中間試験 沒歸到期中考、(評分25%) 被當成標籤、第二課 被當成第二週、純中文數字週次、03.31 日期、(Tue.) 被當成主題、Introduction to|Narrative 被斷開、優先 的值多帶一個 為、先修 A(一)(二) 只對到 (一)、LLM 週次出現 34 與空主題、規則簡稱裡的一般詞（中日）、百分之四十二. 把下一項的編號吃進數字。

### 4.5 課名簡稱解析

| 提及來源 | 解析率 | 對到的課 114-2 有開 | 未解析 |
|---|---|---|---|
| 使用者列的簡稱（個經、總經、初會、中會、五管、大英，加 初會(一)、先修五管） | 100%（8/8） | 8 | 無 |
| forum query bank 裡非完整課名的課程提及（14 個，含 個經 2、總經、中級會計、日文二、計程、經濟學原理） | 100%（14/14） | 11 | 無；日文二、民法總則、政治學概論 114-2 沒開 |
| 114-2 備註的先修課提及（kind=course） | 97.3%（252/259） | 190 | 上學期本班、不提供擋修單、日文語文、智慧財產法、決策科學與企業流程模式、社科統計(上) |
| 其中不是完整課名的提及 | 84.4%（38/45） | 38 | 同上 |

forum query bank 248 題多用 {課} 佔位，真實簡稱只有上表 14 個，樣本小。備註裡沒解析的 7 個中，3 個不是課名（上學期本班、擋修單、日文語文 4 學分），智慧財產法 在課名裡寫作 智慧財產權法總論，社科統計(上) 與 決策科學與企業流程模式 在全庫找不到同名課。先修課多半是上學期的課，114-2 有開的比例（190/252）只代表同名課本學期也有。

## 5. LLM 用的課程卡片

`src/v4/structured/card.py` 的 `render_card(course, sessions, targets, restrictions, eval_items, weekly, max_weeks=8)` 只讀結構化欄位，每門課固定行序，沒有內容的行省略：

```
【策略管理】 305051001｜黃國峯｜3學分｜必修｜商學院 企業管理學系 / 學士
時間:週二 13:10–16:00｜教室:寶來國際廳
對象:企管四甲、企管四乙｜授課語言:中文
限制:先修:五管
評分:Participation 40%、Exam 25%、Group Assignments 35%
授課:講述 20%、討論 70%、小組 10%｜AI:完全開放
週次:1 Course Induction；2 What is Strategy?；…共16週
簡介:Working knowledge on strategic management…
```

限制行先列 hard 的條件，最多 6 項，地點、時間說明、師資、設備、other 不進卡片。卡片存在 `course.card`，重建時一起更新。

## 6. 課名簡稱表

| source | 列數 | 做法 | 信心 |
|---|---|---|---|
| rate | 122 | RATE 評論裡 2–3 字、首字等於課名首字、依序是課名的子序列但不是子字串，且提到這個字串的評論至少 40% 屬於該課（初會 200 則、中會 196、財管 165、個經 142、民訴 119、成管 91、總經 86、高會 54） | 該課評論數 / 提到此字串的評論數 |
| seed | 6 | 人工：五管（企管五大管理，群組）、大英、計程、計概、經原 與 經濟學原理（政大課名是 經濟學；提到 經濟學原理 的 22 則評論有 19 則在 經濟學） | 0.7–0.9，note 欄寫證據 |
| rule | 687 | 114-2 課名的 jieba 詞首字（資料結構 → 資結），本身是常用詞的不收 | 0.5 除以同字串的課名數 |

解析順序是完整課名、去編號的課名、簡稱（可帶 (一)、一 等編號，初會(一) → 初級會計學(一)）、唯一子字串；每一步也會試去掉 三學分之、(大學部)、114-1、一學年 等修飾後的寫法。

五管 的成員是人工定義（營運與供應鏈管理、行銷管理、人力資源管理、研究發展管理、財務管理，其中 研究發展管理 在全庫沒有同名課，表中只存在的 4 門），資料本身沒有定義，信心給 0.7。

## 7. 一致性檢查與意外發現

| 檢查 | 結果 |
|---|---|
| 時間代碼與官方 timeEn | 2,692 門全部一致（同一天兩段在 timeEn 寫成 `wed09-12&14-16`） |
| 備註寫英語授課但 lang 不是英文 | 0 門 |
| 備註寫中文授課但 lang 是英文 | 0 門 |
| kind=4 但沒有 lmtKind | 0；有 lmtKind 但 kind≠4 的 18 門都是跨領域通識（已知） |
| 授課方式加總不是 100 | 1 門 |
| 評分加總超過 105 | 18 門（規則與 LLM 都無法分出上層項目，或原文有加分項） |
| 停開但選課人數大於 0 | 0 |
| lang 英文但大綱是中文 | 7 門 |
| 同一課同一週出現兩列 | 1 門 |
| 同一課同一類同一值的重複限制 | 2 組 |

意外發現：

1. 爬蟲把課名寫進 ai_policy 與 evaluation。24 門課的 ai_policy 是 `Course Name:AI & Governance` 這類字串，7 門的 evaluation 是 `Course Name:Learning Evaluation`、`Course Name:PSYCHOLOGICAL TESTING`。這些課的英文課名都含 AI 或 Evaluation、Assessment、Testing，fetchDescription.py 依英文 h2 文字找段落時把課名標題當成該段。**2026/11/01 已修復**：錯誤來自 4 月那一版爬蟲（用子字串找含 AI、評量 的第一個 h2，先碰到中文課名「科目名稱：AI與治理」）；8 月起的爬蟲已改成英文標題完全比對。重抓這 31 門的大綱頁，修好 35 列（NCCUCourse 的 `repair_course_name_sections.py`，備份 `1142.db.bak-2026-11-01`），NCCUCourse 分支 `fix/section-heading-match`（已 push）加了回歸測試。重建後 ai_policy 覆蓋 100%。
2. 評分方式寫在附件。eval_item 沒對到的 111 門中 50 門只寫 見附檔、Please see the syllabus、Please see my website；另有 7 門是上一條的課名誤植，其餘多是沒有比例也沒有項目名稱的描述（獨立研究、實驗室依服務評分）。
3. 體育課用中文數字寫百分比（術科測驗百分之四十），原本的規則全部漏掉；加上轉換後 eval_item 覆蓋由 2,670 門升到 2,675 門。
4. eval category 的 other 有 1,925 列，比 出席參與（2,016）以外的任何類別都多。期中評量、Literature Review、自評、Design Project 這類項目在 FORMAT_KEYS 沒有對應，篩選期中考、期末考時只看具名類別不受影響，要做分類統計就得擴充詞彙。
5. info 只有異動類型與日期，沒有改成什麼教室或時間；教師異動的字串結尾還多一個右括號。
6. 一門課的評分有時同時有表格和說明文字，百分比重複出現，這是 03 提到加總超過 105 的主因。原本 `catalog.grading` 有 680 門加總不在 90–110（低於 224、高於 456）；子項目刪除規則與 LLM 合起來後，2,447 門有百分比的課只剩 37 門不在 95–105。

## 8. 還不到 90% 或無法處理的部分

| 項目 | 數字 | 原因 |
|---|---|---|
| LLM 產生的限制列精確率 | 77%（23/30） | 主要錯在把灌檔對象、選課提醒當成限制；規則列 40/40 |
| 規則簡稱精確率 | 27/33（82%） | 詞首字縮寫的讀法常和學生習慣不同；學生真的會用的簡稱多半已在 rate 來源 |
| note:status_only、note:online | 89.5%、88.0% | 分母的偵測 regex 比概念寬，沒對到的都是別的類型或不是遠距（4.1） |
| eval_item：社會通識 89%、核心通識 86% | 見 4.2、4.3 | 評分方式在附件 PDF |
| info 舊值新值 | 0 | 原始資料沒有 |
| LLM 週次在單元式大綱 | | 原文沒有週次時是依序編號，不是實際週次 |

## 9. 下游怎麼用

- 硬性篩選只用 `restriction` 中 `method='rule' AND hard=1` 的列（797 列），LLM 列（hard=1 的 214 列）當成提示或在結果加註，不要直接排除課程。和 `src/v4/access.py` 的兩級判定合用時，college_only、status_only、special_class、grade_min、degree_only、cohort、id_parity 對應排除，dept_only 依現行規則是加註。
- 先修：`prereq.resolved_names` 和學生成績單的課名比對（成績單有 subNum 時優先用 subNum）；`kind='knowledge'` 的列只顯示不判斷。
- 時間：衝堂與空堂比對用 `session`，包含 `source='note'` 且 `in_time_field=0` 的實習課；`time_tba=1` 的課不要當成符合空堂。
- 評分條件（不要期中考、要有小考）：`eval_item.category`，並看 `course.eval_reliable`；比例條件只用 reliable=1 的課（2,462 門），沒有項目的課沿用 assessment.py 的 unknown 處理。
- AI 政策：`course.ai_policy` 篩選，`ai_policy.detail` 給 LLM 或使用者看；爬蟲錯誤修復後已沒有課名誤植的 other。
- 週次主題：`weekly.topic` 可加進檢索文件或回答「第幾週講什麼」；`weekly_method='llm'` 的課，週次只當順序。
- 課名簡稱：查詢改寫時先用 `alias` 中 source 為 rate、seed 的列；rule 來源只在其他方法都對不到時使用，並向使用者確認。
- embedding 文字：syllabus、objective、schedule_clean、textbook_clean 已去樣板；多班同一份大綱用 course_group 去重。
- prompt：直接用 `course.card`，長度約 300–600 字。

## 10. 檔案

- 程式：`src/v4/structured/`（text、scalars、sessions、targets、rules、clauses、grading、weekly、aipolicy、aliases、prompts、llm、fallback、assemble、card、store、build、verify、audit、aliaseval）
- 入口：`scripts/v4/build_structured.py`、`scripts/v4/verify_structured.py`
- 測試：`tests/test_v4_structured_rules.py`、`tests/test_v4_structured_aliases.py`（hopelab 上 39 項通過）
- LLM 快取：`data/v4/structured_cache/`（945 列，488 KB，進 git，重建不必再問模型）
- 抽查：`data/v4/structured_audit/`（進 git）
- 輸出：`data/v4/course_struct.db`（26 MB，不進 git，由上面的指令重建）
- `data/v4/unit.json`：qrysub 的系所樹，從 NCCUCourse/data 複製
