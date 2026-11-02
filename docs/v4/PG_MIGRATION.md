# v4 搬進 PostgreSQL：決策與每學期資料管線

2026/11/02。實驗數字見 `REPORT.md` §6.7。

## 決策：全部改用 PostgreSQL，不並行兩套

線上系統的課程詳情、排課、檢索都讀 ParadeDB。v4 目前讀 `course_struct.db`（SQLite）。實測把 v4 的硬條件、BM25、向量三個環節換成 SQL 之後：

| 項目 | SQLite 版 | PostgreSQL 版 |
|---|---|---|
| 以課程計 nDCG@10，重排 | 0.862 | 0.863（字元 n-gram）、0.865（jieba） |
| 以課程計 nDCG@10，不重排 | 0.840 | 0.841、0.837 |
| MRR@10，重排 | 0.918 | 0.918、0.915 |
| 硬條件選出的課 | — | 346 組條件全部相同 |
| 不重排的整次搜尋 p50 | 13.2 ms | 9.9 ms |

品質和延遲都沒有可量到的差異，所以沒有理由保留 SQLite 執行期。並行兩套的代價在維運：
- 兩份成品各自帶學期，`contract.yaml` 只描述 PostgreSQL 那份，SQLite 那份的學期沒有人檢查。
- 一門課停開或改時段，搜尋讀 SQLite、課程詳情讀 PostgreSQL，兩邊可能在同一個回答裡給出不同資訊。
- 部署要多搬一個檔案並保持同步。

SQLite 仍會出現在兩個地方，都不在執行期：
- 爬蟲輸出（NCCUCourse 不改）。
- 建置過程的中間檔 `course_struct.db`。結構化抽取的程式已經寫好、測過，改成直接寫 PostgreSQL 沒有好處；建好後再整批匯入。這個檔也照樣給 scheduling_test 做離線評估。

## 要改的環節

| 環節 | 現在 | 改成 | 改動 | 品質影響 |
|---|---|---|---|---|
| 爬蟲 | NCCUCourse 輸出 SQLite | 不變 | 無 | 無 |
| 結構化抽取 `build_structured` | 爬蟲 SQLite 寫成 course_struct.db | 不變，當中間檔 | 無 | 無 |
| 執行期表 `build_runtime` | 系所、面向、e5 向量寫進 course_struct.db | 不變 | 無 | 無 |
| 匯入 PostgreSQL | 無 | 新增 `prep.load_v4`：course_struct.db 每張表複製到 schema `v4`，另建 `v4.doc`（文件文字、字元 n-gram、e5 向量 1024 維）和 bm25 索引 | 新寫，可由 `scripts/v4/pg_setup.py` 改 | 無 |
| `contract.yaml` | schema_version 8，bge-base 768 維 | schema_version 9，加 `v4_embed_model`（multilingual-e5-large-instruct）、`v4_embed_dim: 1024`、`v4_bm25_tokens: char12` | 小 | 無；少了它，換模型時檢查不到 |
| 執行期載入 `struct_store.load_struct` | 讀 SQLite | 讀 schema `v4`，SQL 相同，連線改 psycopg2 | 中 | 無 |
| 硬條件 | Python 逐門 | SQL WHERE（`pg_backend.where`） | 已寫 | 無（346 組相同） |
| BM25 | rank_bm25 | pg_search，字元 n-gram 欄位 | 已寫 | +0.001 |
| 向量 | numpy 內積 | pgvector `<#>`（內積，不能用 cosine） | 已寫 | 無 |
| 規劃器、身分、評分條件、重排、偏好、合併、結果表 | Python | 不變 | 無 | 無 |
| CourseLangChain `retrieve_tool` | 自己的 jieba BM25 + bge-base 段落向量 | 換成 v4 搜尋 | 大，見 REPORT §7.2 | 0.508 到 0.86 |
| 課程詳情、排課 | 讀 public schema | 不變 | 無 | 無 |
| bge-base 段落向量 | `retrieve_tool` 在用 | v4 上線後沒人用 | 確認沒有其他工具讀它之後再拿掉 | 無 |

BM25 選字元 n-gram 欄位（Python 先切好，pg_search 用 whitespace 斷詞）而不是 jieba。兩者分數差在雜訊內，選前者是因為它和實驗用的 SQLite 版排序最接近（前 10 名重疊 0.86 對 0.64），之後在 scheduling_test 量到的改進可以直接對應到線上。

## 每學期資料管線

一個入口，一個輸出資料夾，都在 CourseDataPrep：

```bash
uv run python -m prep.release build --source <爬蟲>.db             # 建置、檢查、輸出
uv run python -m prep.release build --source <爬蟲>.db --promote   # 全部通過就切換上線版本
uv run python -m prep.release build --from embed                   # 從最近一次失敗的步驟接著跑
uv run python -m prep.release promote 115-1_20261102-0300
uv run python -m prep.release rollback                             # 切回上一個版本
```

| # | 步驟 | 程式 | 用到 LLM | 誰觸發 |
|---|---|---|---|---|
| 0 | 爬課程、選課結果 | NCCUCourse `main.py --course --fast`、`--result` | 否 | 排程 |
| 0' | 爬教學評鑑、教師 | NCCUCourse `--rate`、`--teacher`，要學校帳密 | 否 | 維護者手動 |
| 1 | 結構化抽取 | `build_structured` | 規則抽不到的才用 gemma4:12b，結果依輸入文字快取 | 自動 |
| 2 | 執行期表 | `build_runtime`（系所、面向、e5 向量） | 面向用 LLM；向量用 GPU，沒有 GPU 也能跑，較慢（REPORT §6.6） | 自動 |
| 3 | 抽取檢查 | `verify_structured`、prepare_semester 的門檻 | 否 | 自動 |
| 4 | 原始資料進 PostgreSQL | 先檢查爬蟲檔欄位型別，再跑 pgloader（掃描輸出的 ERROR）、整數欄位縮窄、`prep.migrate`、`prep.populate_time`、`prep.embed` | 否 | 自動 |
| 5 | v4 進 PostgreSQL | `prep.load_v4` | 否 | 自動 |
| 6 | 成品檢查 | `prep.check`：schema_meta、段落向量、v4 學期、向量維度、bm25 索引 | 否 | 自動 |
| 7 | 輸出 | `pg_dump | gzip`，寫 manifest 與報告 | 否 | 自動 |
| 8 | 上線 | `releases/current` 指到新版本，CourseLangChain 還原並重啟 | 否 | 選課期間門檻通過就自動；其他時間維護者手動 |

學期一律取 `contract.yaml` 的 `course_year`、`course_semester`，換學期要改那份檔並提交。門檻沿用 prepare_semester：抽取覆蓋率 90% 以上、原始欄位填寫率 80% 以上、穩定的表筆數不能少於線上版本的一半、冒煙測試通過。任何一項沒過就停在步驟 7 之前，不輸出成品，報告列出原因。

輸出資料夾：

```
releases/
  115-1_20261102-0300/
    data.sql.gz          schema public 與 v4，在新的 ParadeDB 首次啟動時自動還原
    contract.yaml        學期、schema 版本、兩組向量模型
    manifest.json        各檔 sha256、各表筆數、來源爬蟲檔的 sha256、建置時間、各 repo 的 commit
    prepare_report.md    門檻結果、和上一版的差異（新增、停開、改時段的課）
    course_struct.db     中間檔，給 scheduling_test 離線評估
  115-1_20261103-0300.partial/   沒跑完的建置，用 --from 接著跑
  current  -> 115-1_20261102-0300
  previous -> 115-1_20261101-0300
```

CourseLangChain 端要做兩件事：取得 `current` 底下的 `data.sql.gz` 和 `contract.yaml`，啟動時比對 `schema_meta` 和 `contract.yaml`（含新的 v4 欄位），不符就拒絕服務。

PostgreSQL 容器只在資料目錄是空的時候執行 `db/init/`，所以換版不能只放新檔再重啟。建議的換版方式：在同一個容器裡建 `academic_next`、匯入、比對 schema_meta 並跑固定查詢，通過後把 `academic` 改名為 `academic_prev`、`academic_next` 改名為 `academic`，再換 `contract.yaml` 並重啟 app；失敗就改名換回。後端每次查詢都重新連線，課表存在記憶體，資料庫裡沒有使用者資料，所以可以整個換掉；中斷時間約為改名加重啟。

這支程式是 CourseLangChain 的 `ops.deploy`（分支 `feat/deploy-release`，用法見該 repo 的 `docs/SETUP.md`）。hopelab 用 114-2 的成品實測：匯入 196 MB 的資料包 8–9 秒；換上第 9 版 contract 後，app 自己的 `check_contract()` 通過；同一版再跑會判定不需更新。另開一個每 0.2 秒查一次 `v4.doc` 的程序跑了 4 分鐘，切換期間 1,175 次查詢只失敗 1 次。`rollback` 正確對調線上和上一版，contract.yaml 也跟著換回。

排程建議：選課期間每天凌晨一次，因為課程異動多；其他時間每週一次。步驟 0' 每學期評鑑公布後由維護者手動跑一次。

## 程式要搬到哪裡

| 程式 | 現在 | 搬到 |
|---|---|---|
| `src/v4/structured/`、`build_structured`、`build_runtime`、`verify_structured`、prepare_semester 的門檻 | scheduling_test | CourseDataPrep `prep/v4/` |
| `pg_setup.py` | scheduling_test | 改寫成 CourseDataPrep `prep.load_v4` |
| 搜尋（`pipeline`、`planner`、`access`、`identity`、`assessment`、`present`、`pg_backend`、`struct_store`） | scheduling_test | CourseLangChain，取代 `tools/retrieve.py` |
| 評估（R 子集、評審、`run_e2e`） | scheduling_test | 留在原地，從 `releases/<學期>/course_struct.db` 或 PostgreSQL 讀資料 |

## 試跑（2026/11/02，hopelab，114-2）

`prep.release build` 從完整的爬蟲檔（101,883 列課程）跑到輸出，共 3.5 分鐘：v4 20 秒、pgloader 4 秒、embed 115 秒、dump 48 秒，其餘每步數秒。成品 `data.sql.gz` 196 MB。

試跑抓到並修好的問題：
- 舊爬蟲檔把 `COURSE.smtQty` 宣告成 INTEGER，值卻是單學期科目這類文字，pgloader 會丟掉每一列。現在先檢查型別，不符就停下並說明。
- pgloader 用預設批次大小匯入完整爬蟲檔會記憶體不足，改成每批 2,000 列。
- ParadeDB 映像在每個新資料庫都會自己建 paradedb、tiger、topology schema，完整 dump 還原到新容器會在 `schema "paradedb" already exists` 停下。改成只 dump public 與 v4。

把成品還原到全新的 ParadeDB 容器，v4 搜尋讀 `v4.doc` 的結果和 §6.7 的實驗表相同：以課程計 nDCG@10 0.864、MRR 0.918；不重排 0.842、0.903（`scripts/v4/release_e2e.sh`）。

## 還沒做

- `ops.deploy` 還沒在真的 app 容器上跑過：試跑時重啟指令換成只做 contract 檢查，健康檢查網址也是空的。第一次正式部署時要確認 `docker compose --profile app up -d --force-recreate app` 與 `/api/terms` 健康檢查。
- v4 建置程式仍在 scheduling_test，`prep.release` 透過 `V4_REPO` 呼叫它的 prepare_semester；搬進 CourseDataPrep 是下一步。
- prepare_semester 的 `--crawl` 有錯：NCCUCourse 讀 `config.yaml` 的 target，不讀環境變數 YEAR、SEM，爬蟲路徑也解析錯。搬進 CourseDataPrep 時一起修。
- 爬蟲排程（cron 或 systemd timer）放哪台機器還沒定。
