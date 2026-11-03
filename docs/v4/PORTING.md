# v4 移植到 CourseLangChain 的計畫

2026/11/03。使用者決定移植 main 上的 v4；v5 的建議設定（`worktree-v5-research`，nDCG 0.899）之後再評估。效果與理由見 `REPORT.md`，資料流程見 `PG_MIGRATION.md`。

## 1. 現在的狀態

| 項目 | 狀態 | 依據 |
|---|---|---|
| 資料包帶 v4 | 完成。`prep.release` 產出的 `data.sql.gz` 含 schema `v4`，`contract.yaml` 第 9 版 | CourseDataPrep#1 |
| 換版部署 | 完成，尚未在真的 app 容器上跑過 | CourseLangChain#8 |
| 執行期不靠 SQLite | 完成。`load_struct_pg` 從 `v4` schema 載入 | 第 5 節 |
| 規劃器走 OpenAI 相容端點 | 完成 | 第 5 節 |
| course_id 對得上後端 | 是。`v4.doc` 2,655 門全部在 `public.course`（114-2），課程詳情和排課不必轉換 | 2026/11/03 查詢 |
| 115-1 資料 | 完成。2026/11/03 爬完並建出成品，放在 hopelab 的 `releases/115-1_20261103-1552`，尚未 promote | 第 6 節 |
| 程式搬進後端 | 未開始 | 第 3 節 |

## 2. 上線時的行為

老師會提供有 GPU 的主機（2026/11/03，口頭說 GTX Spark，推測是 NVIDIA DGX Spark，型號待確認），所以預設用 GPU 設定，CPU 設定只當備援。

學生的話直接交給 `retrieve_tool`。工具內部依序做：規劃器（gemma4:12b）、SQL 硬條件、放寬、pg_search 與 pgvector 排序、RRF、Qwen3-Reranker-0.6B 重排前 20 名（截 512 token，v5 量過品質不變、省 75 ms）、偏好、同課合併。

和現在最大的差別是條件不再由 Brain Agent 先呼叫 `text_to_sql_tool` 產生。原本兩段式的流程要從 `agents/brain_agent.py` 的提示詞拿掉。

## 3. 要搬的程式

放到 CourseLangChain 的新套件 `course_search/`，import 從 `src.v4.*` 改成 `course_search.*`。兩個 repo 沒有共用程式碼，靠 `contract.yaml` 對齊，所以用複製，不做共用套件。

| 新位置 | 來源 | 行數 | 處理 |
|---|---|---|---|
| `pipeline.py` | `src/v4/pipeline.py` | 324 | 拿掉 HyPE 與記憶體 BM25 分支 |
| `planner.py` | `src/v4/planner.py` | 92 | 模型與端點改讀 `paths.py` |
| `pg_backend.py` | `src/v4/pg_backend.py` | 156 | 只留 `where`、`cond_ids`、`PgRanker`；實驗用的建表拿掉 |
| `store.py` | `src/v4/struct_store.py` | 308 | 只留 `load_struct_pg` 與它用到的部分 |
| `access.py`、`identity.py`、`profile.py` | `src/v4/` | 149、171、92 | `profile.py` 改接後端的成績單解析，見第 4 節 |
| `assessment.py`、`offering.py`、`present.py` | `src/v4/` | 110、167、79 | 照搬 |
| `conditions.py` | `src/eval/conditions.py`、`src/parsers/base.py`、`ollama_parser.sanitize` | 約 200 | 合成一檔 |
| `catalog.py` | `src/v4/catalog.py`、`src/doc_builders`、`src/loader/{courses,time_parser}.py` | 約 660 | 只留資料類別與 `_course_metadata`；讀 1142.db 的部分不搬 |
| `llm.py` | `src/llm/openai_http.py` | 60 | 或改用後端的 `utils/llm.py`，要先確認它能傳 `response_format` |
| `encoder.py` | `src/retrievers/encoders.py` | 84 | 只留 e5-large-instruct；後端已經有 sentence-transformers |
| `sparse.py` | `src/retrievers/sparse.py` 的 `_char12` | 約 20 | 查詢端要用同樣的切法，其他不搬 |
| `reranker.py` | `src/rerankers/models.py` | 約 100 | 只留 Qwen3-Reranker-0.6B。v5 重算過，現行設定拿掉重排是顯著的損失（−0.022，CI [−0.035, −0.011]） |

不搬：`src/v4/eligibility.py`（LLM 補判，先不上線）、`src/v4/aspects.py`、`src/v4/structured/`（屬於資料準備）。

新增相依：無。`numpy`、`psycopg2`、`sentence-transformers`（帶 `transformers`）、`torch` 後端都有，但 `pyproject.toml` 把 torch 釘在 CPU index，要改，見第 4 節。e5-large-instruct 第一次查詢要下載約 2 GB，記憶體約 4 GB（REPORT 6.6），`docker-compose.yml` 已經掛載 HF 快取。

## 4. 後端要改的地方

**`tools/retrieve.py`**
- 參數改成 `retrieve_tool(query: str, top_k: int = 10)`，`query` 是學生的原句。
- 每行格式不變：`1. 課名｜course_id: …｜時間: …｜老師: …｜學分: …`。`main.py` 的 `parse_formatted_docs` 靠它產生前端的 `courses` 事件。
- 警告、灌檔、大綱未註明放在下一行，開頭空兩格。`_LINE_RE` 對不上會直接略過，所以前端不受影響，模型讀得到。
- 同一門課有兩個課號時，依學生身分挑一個放進清單，另一個寫在註記。

**切換開關**
- `paths.py` 加 `SEARCH_BACKEND=legacy|v4`，預設 `legacy`。舊路徑留到新版穩定後再刪。
- `agents/brain_agent.py` 依開關決定提示詞和工具清單；`v4` 時不提供 `text_to_sql_tool`。

**設定**（`paths.py`）

| 設定 | 說明 |
|---|---|
| `V4_PLANNER_MODEL` | 規劃器模型，預設 `gemma4:12b`。和 `OPENAI_MODEL` 分開，Brain Agent 用的是大模型 |
| `V4_RERANK` | 預設開啟；沒有 GPU 的環境關閉 |
| 端點 | 沿用 `OPENAI_BASE_URL`、`OPENAI_API_KEY`。LiteLLM 要多加一條規劃器模型的路由 |

**部署機器**（以 DGX Spark 為準，型號確認後再修正）
- CPU 是 ARM64。`paradedb/paradedb` 有 arm64 映像；`ghcr.io/dimitri/pgloader` 只有 amd64，要在這台建資料包就得靠模擬，或把建置留在 x86 機器（hopelab）、只把成品送過去。後者比較單純，`ops.deploy` 本來就支援。
- `pyproject.toml` 的 pytorch-cpu index 和 Dockerfile 要改成 ARM64 的 CUDA 版，容器要 `--gpus all`。
- 128 GB 記憶體由 CPU 與 GPU 共用，規劃器、編碼器、reranker 可以同時常駐。延遲數字都是在 hopelab（x86、RTX 5080）量的，到新機器要重量。

**啟動**：`check_contract()` 多比對 `v4_embed_model`、`v4_embed_dim`、`v4_bm25_tokens`；`contract.yaml` 換成第 9 版（由 `ops.deploy` 處理）。程序啟動時載入一次 `load_struct_pg` 並常駐；換版後 `ops.deploy` 會重啟 app，不需要熱更新。

**學生身分**：`tools/session_profile.py` 留下的是去識別化結果，目前 `parse_transcript` 只有 `department`、`grade_label` 與修課清單，身分篩選還需要學制、身分別（僑生等）、雙主修輔系、大英免修。這幾項不是個資，要在 `transcript_parser.py` 加解析，原始檔照舊不落地。沒上傳成績單時不做身分篩選，只顯示通用警告。自行輸入的表單要前端配合，放到第二階段。學號不在對話裡收。

**前端**：不用改就能動。要顯示警告標籤、讓學生選課號時，才在 `courses` 事件加欄位，舊前端會忽略多出來的欄位。

## 5. 已驗證的項目

用 `scripts/v4/runs/port_readiness.sh` 在 hopelab 對真實成品（114-2）跑：

| 檢查 | 結果 |
|---|---|
| 資料從成品載入 | `load_struct_pg` 和讀 SQLite 的結果完全相同：2,655 門課、2,274 堂、通道順序、別名、評量、面向、系所表、1024 維向量。載入 0.9 秒 |
| 全程不用 SQLite 的搜尋（`--struct-pg --pg whitespace --pg-conditions`） | 以課程計 nDCG@10 0.864、MRR@10 0.918，和先前版本相同。前 10 名重疊 0.992 |
| 延遲（還原後的成品） | BM25 1.4 ms、pgvector 10 ms（全部課程）。建置用的資料庫 BM25 要 65 ms（索引 2 個 segment，還原後是 1 個），不影響線上；確切原因沒有再追 |
| course_id | `v4.doc` 2,655 門全部在 `public.course`；`public.course` 該學期每門課都在 `v4.course` |

規劃器走 OpenAI 相容端點（Ollama 的 `/v1`，`reasoning_effort: none`），R 集 248 句，和原生 API 比：

| 模型 | 條件完全相同 | 整份規劃完全相同 | 條件符合人工答案（原生 → 相容） | p50（原生 → 相容） | 失敗 |
|---|---|---|---|---|---|
| gemma4:12b | 97.2% | 83.1% | 78.2% → 77.8% | 452 → 495 ms | 0 |
| gemma4:e4b | 98.8% | 91.5% | 76.2% → 76.2% | 185 → 184 ms | 0 |

換端點對條件的品質沒有可見的影響。整份規劃的差異還沒逐題看，可能的來源是相容端點用的上下文長度（`num_ctx`）與原生不同。這裡量的是 Ollama 的 `/v1`；換成 LiteLLM 或其他服務要再跑一次 `scripts.v4.analysis.planner_endpoint_parity`。

## 6. 順序

1. **爬 115-1**（2026/11/03 完成）。`main.py --course --fast` 不需登入，22 分鐘，2,932 門課（教務處清單 14 門掃描沒掃到，全部補回），0 個錯誤，填寫率和 114-2 差不到 1%。
   - `1151.db` 是 `1142_texttypes.db` 加上 115-1，保留歷史評鑑，別名與面向才有資料。
   - 新學期要加 `--llm cloud`：`--llm off` 只用快取，新大綱沒有快取，評分比例 89.9%、每週進度 87.6%，門檻擋下。補抽 9 分鐘後是 98.3%、98.2%。`prep.release` 的預設還是 `off`，應該改掉或在文件寫清楚。
   - 成品 212 MB，還原到全新資料庫後，`schema_meta` 第 9 版、115-1，v4 2,813 門可搜尋。8 句真實查詢 p50 600 ms（12b 加重排）。
   - 這 8 句看到兩個問題：「週五沒課的人」被解析成要週五，出來全是週五的課（v5 的否定處理改善這類句子，但這句是否解決未確認）；「週三下午的英文授課通識」前 5 名全是大學英文的不同班。
2. **第一次真的部署**：把第 9 版成品用 `ops.deploy` 換上去，確認 `docker compose --profile app up -d --force-recreate app` 和 `/api/terms` 健康檢查。此時 `SEARCH_BACKEND` 還是 `legacy`，行為不變。
3. **搬程式**，開關預設關閉，測試跟著搬（`tests/test_v4_*` 裡屬於執行期的部分）。
4. **成績單欄位**與身分篩選。
5. **在測試環境打開開關**，用 R 子集和一份真實問句驗收，第 7 節的門檻都過了才打開正式環境。
6. **穩定兩週後**刪舊路徑與 `text_to_sql_tool`。

## 7. 驗收門檻

| 項目 | 門檻 | 怎麼量 |
|---|---|---|
| 檢索品質 | 以課程計 nDCG@10 不低於 0.85（GPU 設定是 0.86） | `run_e2e`，114-2 的 R 子集 |
| 單次搜尋 | p50 不超過 1 秒（規劃加搜尋，hopelab 約 0.72 秒） | `run_e2e` |
| 記憶體 | app 常駐增加不超過 5 GB | 容器統計 |
| 課表與 SSE | `courses`、`plans` 事件照舊，course_id 都能加入課表 | CourseLangChain 既有測試 |
| 回退 | 把開關改回 `legacy` 並重啟就恢復 | 手動 |

## 8. 風險

- **v5 之後才合**：程式搬進後端後，v5 的結論要手動移植，不能直接 merge。見第 9 節。
- **規劃器的模型來源**：正式環境若用 Gemini 等外部端點而沒有 gemma4，規劃品質沒有量過，換模型要先跑 `score_planner`。
- **`alias.in_1142` 欄位名**綁了學期但內容是通用的，搬進 CourseDataPrep 時要改名。
- **R 子集只有 114-2**：115-1 沒有對應的標註，上線前只能驗流程，品質要等 T5（v5 ROADMAP）或真實問句。
- **面向抽取的輸出只在 hopelab**：`data/v4/aspects/` 沒有進 git，換建置機器要整個目錄帶過去，否則要重抽（這次 13 分鐘）。

## 9. 之後轉接 v5

v5 第一階段的建議設定（GPU：e4b 第三版提示詞加規則、只用 e5、重排 512）和 v4 是同一條流程，只換規劃與融合，所以搬程式時把這兩處留成可換的：

| 換什麼 | 改動 | 開關 |
|---|---|---|
| 規劃器 | `planner.py` 換成 `src/v5/planner_open.py`（93 行）加 `src/v5/slots.py`（189 行） | `V4_PLANNER=v4|open` |
| 融合 | `Config.use_bm25=False` | `V4_USE_BM25` |

轉接前要做：人工審 T1b 的句子（目前的 0.96 是未審稿的數字）、用同一份 R 子集和真實問句比較、多人同時查詢的排隊沒有量過。v5 其他項目（T2 到 T5、C 類、蒸餾）是下一步的研究，不擋這次轉接。
