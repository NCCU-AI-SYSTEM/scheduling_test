# v4 課程檢索流程

2026/11/02 版。分兩段：資料預備每學期在有網路的機器上跑一次，產出一個檔案 `data/v4/course_struct.db`；線上檢索只讀這個檔案。數字來自 REPORT.md 第 6 節。

```
資料預備（離線，每學期一次）                         線上檢索（每次查詢）
──────────────────────────────                     ──────────────────────────────
NCCUCourse 爬蟲 ──► 1142.db（311 MB，原始）         學生的一句話 ＋ 身分來源
        │                                                  │
        ▼ build_structured                                 ▼ 規劃器（小模型）＋ 評分方式規則
course_struct.db（課程、時段、限制、評分、               │
  每週主題、縮寫、同課合併…）                             ▼ 篩選：條件 → 身分 → 老師 → 成績單 → 評分方式
        │                                                  │
        ▼ build_runtime                                    ▼ 檢索：縮寫展開 → BM25 ＋ 向量 → RRF → 指名優先
＋ dept、aspect、embedding 表                              │
        │                                                  ▼ 重排（只在 GPU）→ 偏好 → 排序 → 同課合併
        ▼ verify_structured                                │
course_struct.db（39 MB）─────────── 部署 ──────────►    ▼ 結果表（每堂課一列，附大綱連結）
```

## 一、資料預備

**一行跑完**（`scripts/v4/prepare_semester.py`）：

```bash
python -m scripts.v4.prepare_semester --semester 115-1 --src data/1151.db --llm cloud --promote
```

依序執行下表的步驟，產出先放在 `data/v4/staging/<學期>/`。過了品質門檻才會換掉上線中的 `course_struct.db`，舊版會先備份到 `data/v4/archive/`。報告寫在 `data/v4/staging/<學期>/prepare_report.md`，結束代碼 0 是通過、1 是門檻沒過、2 是某一步出錯。

學期不再寫死在程式裡，由 `--semester` 或環境變數 `V4_SEMESTER` 決定（`src/v4/semester.py`）。原始資料庫由 `--src` 或 `V4_SOURCE_DB` 指定；線上讀取的資料庫由 `V4_STRUCT_DB` 指定，執行期的學期則讀資料庫的 meta 表。

**品質門檻**

| 檢查 | 條件 | 擋的是什麼 |
|---|---|---|
| 抽取覆蓋率 | 適用課程 50 門以上的欄位，覆蓋率 ≥ 90% | 解析規則失效 |
| 原始資料完整度 | 課程簡介、評分方式、進度表、授課方式、AI 政策，有內容的課 ≥ 80% | 爬蟲漏抓某個欄位；只看抽取覆蓋率時，原始欄位整個是空的也會通過 |
| 不可大幅縮水 | 評分項目、每週主題、限制、上課時段、偏好分數、向量，不得少於上線版本的一半 | 資料不完整的學期蓋掉好的版本 |
| 試搜 | 主題、條件、指名三種查詢都要有結果 | 資料表之間對不起來 |

驗證：114-2 重建後，除了偏好分數多 49 列（依課程家族對上了評鑑），其他每張表的列數都和上線版本相同，門檻通過。114-1 在 1142.db 裡沒有評分方式和進度表的原始文字，門檻列出 7 項問題並擋下，上線版本沒有被動到。

**分步驟執行**

| 步驟 | 指令 | 輸入 → 輸出 | 做什麼 | 時間 |
|---|---|---|---|---|
| 1. 爬資料 | NCCUCourse `python main.py --course` 等 | 選課系統 → `1142.db` | 課程、教師、評鑑、選課結果；修復腳本 `repair_course_name_sections.py` 處理舊爬蟲留下的錯欄 | 數小時（有 rate limit） |
| 2. 結構化 | `python -m scripts.v4.build_structured` | `1142.db` → `course_struct.db` | 規則優先，規則處理不了的才交給 LLM；LLM 答案依輸入文字快取在 `data/v4/structured_cache/` | 有快取約 20 秒；新學期第一次約 16 分鐘（雲端 Gemma 31B，945 次） |
| 3. 執行期資料 | `python -m scripts.v4.build_runtime --encoder e5-large-instruct --aspects data/v4/aspects/gemma4_12b.jsonl` | 加 3 張表 | 系所對照、偏好分數、課程向量 | 有向量快取數秒；重算 e5 向量 GPU 約 1 分鐘、CPU 12 執行緒約 12 分鐘 |
| 4. 檢查 | `python -m scripts.v4.verify_structured` | 覆蓋率表 | 每個欄位的覆蓋率，並依通識類別、學院拆開 | 數秒 |
| 5. 部署 | 複製 `course_struct.db` | — | 線上只需要這個檔和程式 | — |

**結構化做了什麼**（程式在 `src/v4/structured/`，說明在 07_structured_data.md）

| 原始欄位 | 變成 | 方法 |
|---|---|---|
| time | `session`：星期、開始與結束（到分鐘）、節次；note 寫的實習課時段另外列 | 規則 |
| unit | `target_class`：系、學制、年級 | 規則 |
| note、info | `restriction`：一句一列，有類型、正規化的值、是否硬性、原文、規則或 LLM 抽的；`change`：停開、教室、時間、教師異動 | 規則，加 LLM 處理約 845 種相異 note |
| evaluation | `eval_item`：期中考、小考、報告…，含百分比 | 規則，加總不對的交給 LLM；中英對照只取一份 |
| schedule | `weekly`：每週主題 | 規則，抽不到的交給 LLM |
| ai_policy | `ai_policy`：類別加細則 | 規則 |
| teaching_approach | `course` 表的五個比例欄 | 規則 |
| 評鑑留言 | `alias`：課名縮寫（初會 → 初級會計學） | 從留言統計，加人工整理 |
| 同名同師同時段的多個課號 | `offering`、`channel` | 規則 |
| 系所代碼 | `dept`：學院、學制 | 選課分類路徑 dp1/dp2 |
| 評鑑與大綱 | `aspect`：負擔、難度、分組、教學、實用、點名、熱門 | LLM 抽評鑑，只保留人工抽查可信的面向；分組、點名、熱門來自大綱與選課人數 |
| 課程文字 | `embedding`：e5-large-instruct 向量 | 模型編碼 |

系所對照：`unit.json` 是選課系統的系所樹，從 `qrysub.nccu.edu.tw/assets/api/unit.json` 抓來；`nccu_units.tsv` 是校方單位代碼表，來自使用者貼上的網頁下拉選單。`build_runtime` 把兩張合併成資料庫裡的 `unit` 表（446 列），執行時只讀這張表，用來把全人資料或使用者輸入的系所名稱換成代碼。

**評鑑面向**（`extract_aspects`，本機 gemma4:12b）以課程家族（課名＋老師）為單位，已經抽過的家族不會重抽。新學期只需要處理新出現的家族；舊家族在新學期多出來的評鑑，不會自動更新。

## 二、線上檢索

一次查詢依序經過下面這些步驟。程式入口是 `src/v4/pipeline.py` 的 `V4Search.search()`，部署設定在 `src/v4/deploy.py`。

| # | 步驟 | 做什麼 | 程式 | GPU / CPU 時間 |
|---|---|---|---|---|
| 0 | 學生身分 | 全人匯出 > 自行輸入 > 學號，得到系所、學院、學制、年級、雙主修、身分別、大英免修 | `src/v4/identity.py` `resolve()` | 一次 session 一次 |
| 1 | 規劃 | 小模型把句子拆成主題、指名課名與老師、硬條件（要與不要）、偏好、意圖 | `src/v4/planner.py` `plan()` | 12b 0.44 s；CPU 上 e4b 1.3–1.7 s |
| 1b | 評分方式條件 | 不要期中考、要有小考：直接用規則解析原句，不經過模型 | `src/v4/assessment.py` `parse_query()` | <1 ms |
| 2 | 條件篩選 | 資料庫判斷星期、時段、語言、學分、課別、通識類別、系所。剩不到 10 門時依序放寬：系所 → 課別 → 通識類別 → 學分 → 時段 → 星期 → 語言。有指名課名時不篩選，條件改成排序用 | `_cond_mask()`、`src/eval/conditions.py` | 數 ms |
| 3 | 身分篩選 | 每堂課挑最適合這位學生的課號；所有課號都選不到才排除，例如學院限定、身分限定、專班、課號指定學制、年級下限、大英免修。可能選不到的加註，例如限本系、非灌檔對象的灌檔課、開課對象是別的學制 | `src/v4/access.py` `judge()` | 數 ms |
| 4 | 指名老師 | 只留這位老師的課 | `_teacher_mask()` | <1 ms |
| 5 | 成績單 | 已修過、和本學期已選的課衝堂，直接排除 | `src/v4/eligibility.py` | 數 ms |
| 6 | 評分方式 | 大綱明確不符的排除；大綱沒寫的保留，加註後排在後面 | `assessment.verdict()` | <1 ms |
| 7 | 檢索 | 主題文字先展開縮寫（個經 → 個體經濟學），再在篩選後的課裡跑字元 n-gram BM25 和 e5 向量，RRF 融合取前 50；指名課名精確比對的放最前面，保留 (一)、(二) 編號 | `_expand()`、`_is_named()` | GPU 4 ms；CPU 約 80–120 ms |
| 8 | 重排 | Qwen3-Reranker-0.6B 讀前 20 名的完整課程文字。只在 GPU 上開；CPU 上一題要 15–45 秒，所以關掉 | `src/rerankers/models.py` | GPU 約 0.4 s |
| 9 | 偏好 | 依 aspect 分數加減分，例如負擔低、分組少、教學評價高 | `_pref_score()` | <1 ms |
| 10 | 排序 | 先排完全符合原始條件的課，再排評分方式有寫明的課，最後依分數 | `search()` 尾段 | <1 ms |
| 11 | 同課合併 | 同一堂課只留一筆，顯示採用的課號，其他課號列在旁邊 | `_merge()` | <1 ms |
| 12 | 結果表 | 課名、老師、時間、學分、課號與對象、身分加註、評分方式、授課方式、大綱連結 | `src/v4/present.py` | — |

**兩種部署設定**（`src/v4/deploy.py`）

| | gpu preset | cpu preset |
|---|---|---|
| 規劃器 | gemma4:12b | gemma4:e4b（可用 `V4_CPU_PLANNER` 換成 12b，但一題 8–16 s） |
| 向量 | e5，GPU | e5，CPU；文件向量從資料庫讀，不重算 |
| 重排 | Qwen3-Reranker-0.6B | 關閉 |
| 一次查詢 | 約 0.86 s | 12 執行緒約 1.4 s，4 執行緒約 1.8 s |
| 以課程計 nDCG@10 | 0.864 | 0.809（線上系統 0.508） |
| 記憶體 | 16 GB 顯卡放得下 | 約 5 GB RAM，另加規劃器模型 |

服務啟動時要先跑一次查詢，讓 Ollama 把規劃器載入；第一次載入在 CPU 上實測要 86 秒。

## 三、試跑

```bash
# 結構化資料（hopelab，有 1142.db）
python -m scripts.v4.build_structured
python -m scripts.v4.build_runtime --encoder e5-large-instruct --aspects data/v4/aspects/gemma4_12b.jsonl
python -m scripts.v4.verify_structured

# 查詢（三種身分來源擇一）
python -m scripts.v4.demo_search "週三下午的英文授課通識，不要早八" --deploy cpu
python -m scripts.v4.demo_search "想修自然語言處理" --profile data/exportStudentData.json
python -m scripts.v4.demo_search "大英要選哪一班" --dept 資訊科學系 --year 1
STUDENT_ID=112703001 python -m scripts.v4.demo_search "初會(一)有晚上的班嗎"

# 每一步的實際輸出
python -m scripts.v4.analysis.explain_examples
```

每一步的實際範例輸出在 `docs/v4/examples_output.txt`，縮寫對照表在 `docs/v4/aliases.md`。
