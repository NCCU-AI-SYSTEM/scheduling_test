# Database Schema

## 1. PostgreSQL — `nccu` database (pgvector)

主要的 dense retrieval 資料庫，含 embedding vector，供 pgvector HNSW ANN 搜尋使用。

### Table: `courses`

| Column | Type | 說明 |
|--------|------|------|
| `course_id` | `text` PK | 課程代碼，格式如 `1142-ABC1234` |
| `year` | `text` | 學年，e.g. `113` |
| `semester` | `text` | 學期，`1` 或 `2` |
| `name` | `text` | 課程名稱（中文） |
| `name_en` | `text` | 課程名稱（英文） |
| `teacher` | `text` | 授課教師 |
| `kind` | `integer` | 課程類型代碼（必修/選修等） |
| `lmt_kind` | `text` | 選課限制類型 |
| `lang` | `text` | 授課語言 |
| `point` | `double precision` | 學分數 |
| `unit` | `text` | 開課系所 |
| `time_raw` | `text` | 原始時間字串，e.g. `"M34 W56"` |
| `sessions` | `jsonb` | 解析後的時段陣列，e.g. `[{"weekday":1,"period":3}, ...]` |
| `weekdays` | `integer[]` | 上課星期陣列，1=Mon … 7=Sun，供 GIN 索引過濾 |
| `has_morning` | `boolean` | 是否有早上課（1-4節） |
| `has_noon` | `boolean` | 是否有中午課（5-6節） |
| `has_afternoon` | `boolean` | 是否有下午課（7-9節） |
| `has_evening` | `boolean` | 是否有晚上課（10+節） |
| `embedding` | `vector(1024)` | BAAI/bge-m3 dense embedding（1024 維） |
| `info` | `text` | 課程說明 |
| `note` | `text` | 備注 |
| `objective` | `text` | 課程目標 |
| `syllabus` | `text` | 課程大綱 |

**Indexes:**

| Index | Type | 說明 |
|-------|------|------|
| `courses_pkey` | btree (course_id) | 主鍵 |
| `idx_courses_embedding` | **hnsw** (vector_cosine_ops) | ANN 向量搜尋 |
| `idx_courses_kind` | btree (kind) | 課程類型過濾 |
| `idx_courses_lang` | btree (lang) | 語言過濾 |
| `idx_courses_point` | btree (point) | 學分過濾 |
| `idx_courses_weekdays` | **gin** (weekdays) | 星期多值過濾 |

**Stats:** ~2,795 rows（1142 學期），table size ~5.2 MB（含 vector index）

**Embedding model:** `BAAI/bge-m3`，dim=1024，cosine similarity

---

## 2. SQLite — `course_meta.db`

GPT-4o-mini 產生的課程 meta 資訊，供 BM25 / hybrid 使用。

### Table: `course_meta_v1`

| Column | Type | 說明 |
|--------|------|------|
| `course_id` | `text` PK | 與 `courses.course_id` 對應 |
| `summary_100` | `text` | 100 字以內課程摘要（繁中） |
| `keywords_json` | `text` | JSON array，關鍵詞列表 |
| `topic_tags_json` | `text` | JSON array，主題標籤 |
| `model` | `text` | 生成所用模型名稱 |
| `generated_at` | `text` | ISO 8601 生成時間 |
| `raw_json` | `text` | 原始 LLM 回傳 JSON |

**Stats:** 2,795 rows，~5.7 MB

---

## 3. SQLite — `1142.db`（來自 NCCUCourse 爬蟲）

原始爬蟲資料，含完整課程、教師、評鑑、選課結果。本 repo 不含此檔（311 MB），
詳見 [NCCUCourse](https://github.com/NCCU-AI-SYSTEM/NCCUCourse)。

**Tables:** `COURSE`, `TEACHER`, `RATE`, `RESULT`

---

## Release Assets

GitHub Release `v1.0-data` 包含：

- `courses_pg_dump.sql.gz` — PostgreSQL `courses` table 完整 dump（含 embedding vector），可用 `psql` 還原
- `course_meta.db` — SQLite meta 資料庫

還原方式：

```bash
# PostgreSQL (需先裝 pgvector extension)
createdb nccu
psql nccu -c "CREATE EXTENSION IF NOT EXISTS vector;"
gunzip -c courses_pg_dump.sql.gz | psql nccu

# SQLite (直接使用)
cp course_meta.db data/processed/course_meta.db
```
