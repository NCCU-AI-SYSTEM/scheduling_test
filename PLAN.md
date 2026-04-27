# NCCU 課程檢索系統 — Retrieval 改進實驗計畫書

版本：v1（1142 學期資料）
工作目錄：`NCCU-AI-SYSTEM/scheduling_test/`
資料來源：`../NCCUCourse/1142.db`

---

## 0. 問題定義

**輸入**：使用者自然語言 query
- 例：「我想學機器學習」、「想找週一沒課的通識」、「李嘉綺老師的課」、「會計系大三必修」、「不要早八的英文授課課」

**輸出**：top-k 課程清單（k=5 / 10 / 20），每筆對應一個 distinct courseId

**評估目標**：給定 query，正確命中與「該 query 在語意上相關」的所有課程。

**研究問題**：
- RQ1 — page_content 只用「課名+老師+時間」是否為當前準確度低的主因？
- RQ2 — 課程內容（objective / syllabus）以何種形式進入索引收益最大？
- RQ3 — 在中文短 query 場景，HyDE 以外哪些 query-side 方法最有效？
- RQ4 — sparse + dense + rerank 三段式 pipeline 對 retrieval 準確度的邊際貢獻為何？
- RQ5 — metadata 結構化過濾（時段、語言、學分、必選修、系所）vs 純語意檢索，使用者實際可用率差距多少？

---

## 1. 資料分析摘要（1142 學期）

| 指標 | 數值 |
|---|---|
| COURSE row | 3,472 |
| distinct courseId | 2,795 |
| 重複率（多系所掛課） | ~24% |
| syllabus / objective / schedule 缺失 | 9 筆（0.3%） |
| classroom 缺失 | 391（11.3%，多為線上課） |
| time / teacher / point / lang | 100% 完整 |
| 中文授課 / 英文授課 | 73% / 21% |
| RATE 對 1142 courseId | 0（需以 teacher 串接 → Future Work） |
| RESULT 對 1142 courseId | 100% 覆蓋（可作熱門度信號） |

**關鍵發現**：
1. dedupe 必做，否則 top-k 重複。
2. syllabus 全部 ≥1000 字，但目前完全未進索引。
3. RATE 跨學期，需另行處理 → 本實驗排除，列 Future Work。

---

## 2. 文本處理 Pipeline

### 2.1 Dedupe
以 courseId 為主鍵聚合多 row：`dp1/dp2/dp3` 收集為 `cross_listed_depts: list`，其餘欄位取第一筆。

### 2.2 欄位清洗
- 全形/半形正規化（NFKC）
- 移除 HTML entity、`@異動資訊`、`Course Schedule|...|...` 中的英文鏡像段
- 中英分離：偵測中英交錯，分別存 `*_zh` 與 `*_en`（跨語檢索用）
- syllabus 切段：以週次行（`\d+\|\d+/\d+\|...`）切，存為 `weekly_topics: list[str]`
- evaluation 結構化：抽 `{type, percent}` pairs（本實驗不用，僅留 metadata）

### 2.3 文件表示（**核心改進**）

baseline（現行）：
```
課程名稱是X, 上課時間是Y, 這堂課的老師是Z
```

新設計（V2）— 「主題密度」優先：
```
[課名] X
[教師] Z　[系所] D　[語言] L　[學分] P　[必選修] K
[課程目標] objective 前 200 字（截斷）
[週次主題] weekly_topics join "、"
[關鍵字] kw1, kw2, ..., kw8（LLM 一次性產生並 cache）
```

新設計（V3）— **multi-field 分欄索引**（推薦）：
不再合併成單一字串，而是同一筆 doc 對多欄位分別 embed，retrieval 時加權合併：
```
field_name        weight  說明
title             3.0     課名
objective_summary 2.0     LLM 摘要 100 字
weekly_topics     1.5     週次標題 join
keywords          1.5     LLM 抽 8 個關鍵字
teacher           1.0     教師名（含同義/別名 resolve）
```

V3 對應的 metadata（不進 embedding，走 SQL/結構化過濾）：
```
courseId, dp1, dp2, dp3, unit, lang, kind (必選修), point (學分),
sessions: [{week, start, end}], smtQty, lmtKind, classroom_id
```

### 2.4 LLM cache 表（一次性離線生成）
新增 `course_meta_v1` 表存 LLM 產出，避免每次 build 重跑：
```sql
CREATE TABLE course_meta_v1 (
  courseId TEXT PRIMARY KEY,
  summary_100 TEXT,        -- 100 字摘要
  keywords_json TEXT,      -- ["機器學習", "深度學習", ...]
  topic_tags_json TEXT,    -- ["AI", "資料科學"] 從受控詞表選
  level TEXT,              -- 入門/進階/研究所
  prereq_inferred TEXT,    -- 推測先備課
  generated_by TEXT,       -- 模型名
  generated_at TEXT
);
```

---

## 3. 模型清單

實驗共用 **6 個模型**：3 個 retrieval 必要、2 個 LLM 用於 eval set 生成（多樣性）、1 個輕量 LLM 跑 query rewriter / meta 生成。

| # | 角色 | 模型 | 部署位置 | 用途 | 必要性 |
|---|---|---|---|---|---|
| M1 | Dense embedding | `BAAI/bge-m3` | 本機（scheduling_test） | 主向量檢索 | 必要 |
| M2 | Reranker | `BAAI/bge-reranker-v2-m3` | 本機 | top-50 → top-10 重排 | 必要 |
| M3 | 輕量生成 LLM | `gemma3:4b`（Ollama，home_mac:11434） | home_mac via SSH tunnel | (a) course meta 生成 (b) HyDE/Q2D/Multi-Query 等 query rewriter (c) Structured Extraction | 必要 |
| M4a | Eval-set 生成 #1 | `claude-opus-4`（Anthropic Message Batches API，50% 折扣） | API | 合成 query（風格 A） | 必要 |
| M4b | Eval-set 生成 #2 | `gpt-4o-mini` 或 `gpt-4.1-mini`（OpenAI Batch API，50% 折扣） | API | 合成 query（風格 B） | 必要 |
| M5 | 替代 dense embedding | `jinaai/jina-embeddings-v3` | 本機 | M1 ablation | Optional |
| M6 | Multi-vector / late interaction | `bge-m3` ColBERT mode | 本機 | RQ4 進階組 | Optional |

### 3.1 M3：Ollama on home_mac via SSH tunnel

連線方式：
```bash
# 在 scheduling_test 開發機建 tunnel（背景執行）
ssh -fNL 11434:127.0.0.1:11434 home_mac
# 之後本機所有程式用 http://127.0.0.1:11434 即可
```

健康檢查：
```bash
curl http://127.0.0.1:11434/api/tags
# 確認 gemma3:4b 已 pull；若無：
ssh home_mac "ollama pull gemma3:4b"
```

呼叫慣例（src 內統一 wrapper，方便切換）：
```python
# src/llm/ollama_client.py
import ollama
client = ollama.Client(host="http://127.0.0.1:11434")
resp = client.chat(model="gemma3:4b", messages=[...], options={"temperature": 0.3})
```

`pyproject.toml` 加入 `ollama` python client。tunnel 中斷時程式應 fail loud，不要 silent fallback。

### 3.2 M4a / M4b：Eval-set 生成用雙模型增加多樣性

策略：每門課跑 **同一 prompt 但兩個模型各生 1 組**，最後合併並去重。

- M4a：Claude Opus，走 Anthropic **Message Batches API**（24h 完成，50% 折扣）
- M4b：GPT-4o-mini 或 GPT-4.1-mini，走 OpenAI **Batch API**（24h 完成，50% 折扣）

兩個模型故意挑「重量級 vs 輕量級」搭配：
- Opus 提供深度語意改寫、學科術語準度
- 4o-mini / 4.1-mini 便宜量大，提供口語、閒聊風格、噪音多樣性
- 風格 bias 互相中和，避免單一模型風格汙染 eval

prompt 一致，溫度 0.7~0.9。每門課各生 3 type，總共 6 query。

**Batch 工作流**：
1. `scripts/build_batch_jobs.py` 把 2,795 課讀出 → 產生兩支 jsonl：
   - `batches/anthropic_eval.jsonl`（custom_id = courseId）
   - `batches/openai_eval.jsonl`（custom_id = courseId）
2. 同時 submit 兩個 batch；輪詢 status；完成後拉回 results
3. `scripts/merge_eval.py` 解析、去重、合併成 `data/eval_synth.jsonl`

成本估算（含 batch 折扣）：
- M4a Claude Opus batch：2,795 × ~600 tokens × $7.5/M（input batch 價）≈ **US$13**
- M4b GPT-4o-mini batch：2,795 × ~600 tokens × $0.075/M ≈ **US$0.5**
- 共 ~US$14，比原估的 50 美降 70%
- M3 gemma3:4b 完全本地，無金錢成本

### 3.3 算力估算

| 階段 | 計算量 | 預估時間 |
|---|---|---|
| M3 meta_gen | 2,795 課 × 1 次 | home_mac M1/M2/M3/M4 級別 ~60 分鐘 |
| M3 query rewriter (E6–E11) | 150 query × 多 pipeline | 每 pipeline ~3 分鐘 |
| M1 build (D-V3) | 2,795 doc × 5 field ≈ 14k 段 | GPU < 30s / CPU ~5min |
| M2 rerank | 150 query × 50 pair | GPU ~15s / CPU ~5min |
| M4a/b eval gen | 2,795 課 × 2 模型 | 並行下 ~30 分鐘 |

---

## 4. 測試資料集取得（**最關鍵**）

不存在公開 NCCU 課程 retrieval benchmark，必須自建。採三來源混合。

### 4.1 來源 A：LLM 合成（量產 train/dev set）

**做法**：對每門課的 (objective + weekly_topics) **同時餵給 M4a (Claude Opus) 與 M4b (GPT-5)**，各自生成 3 種 persona 的 query，合併後去重：
1. **主題型**：「我想學 X」→ 應命中該課
2. **生活型**：「想找週X沒課/早八外的XXX」→ 帶硬約束
3. **同義改寫型**：把課名換成口語（「機率與統計」→「想學機率」）

prompt 範本（要在 `scripts/gen_eval.py`）：
```
給定一門課的目標與週次主題，請產生 3 個學生可能會問的查詢。
要求：
- 不要直接抄課名
- 涵蓋主題、生活約束、口語三類各一
- 輸出 JSON: [{type, query}]
```

**規模**：2,795 課 × 3 type × 2 模型 = 16,770 raw → 去重後保守 ~12,000 (query, gold_courseId) pairs
**檢驗**：抽樣 200 筆人工確認，相關性 ≥ 90% 才採用；同時統計兩模型生成風格差異（平均長度、term overlap）。
**用途**：dev set + ablation。**不可用於最終 test**（LLM 生成 + LLM rerank 有 leakage）。

### 4.2 來源 B：人工金標（test set）

**規模**：150 query（足以區分 5pp 差異）
**生產方式**：
1. 從 PTT NCCU 板、Dcard 政大版、政大選課心得社團爬取「找課程」貼文 → 抽出真實 query 樣本
2. 自己 + 找 3 位政大同學各寫 30 query，內容含：
   - 50 主題型（「我想學深度學習」）
   - 30 教師型（「想修李嘉綺」）
   - 30 約束型（「週二下午的通識」）
   - 20 模糊型（「好過的英文授課課」）
   - 20 多條件複合（「商學院大三選修不要早八」）
3. 每個 query 由標註者瀏覽 1142 課程清單，標出所有相關 courseId（多答案）
4. 兩名標註者交叉，Cohen's κ ≥ 0.7 才合格

**儲存格式**：
```jsonl
{"qid": "q001", "query": "我想學機器學習", "type": "topic",
 "relevant": ["1142001234001", "1142005678002", ...],
 "constraints": {"day_exclude": [], "lang": null, ...}}
```

### 4.3 來源 C：Implicit signal（Future Work，先不做）
RESULT 表的「主修 X 系學生選了什麼課」可作 weak relevance，但需學生個資不可得 → 排除。

### 4.4 切分
- train：來源 A 的 80%（6,700 筆，僅供 query rewriter / reranker fine-tune 用）
- dev：來源 A 的 20%（1,685 筆，調 hyper-params）
- test：來源 B 全部 150 筆（最終報告數字以此為準）

---

## 5. Retrieval 方法清單（不只 HyDE）

### 5.1 Document-side 方法
| 代號 | 方法 | 說明 |
|---|---|---|
| D-Base | name+teacher+time only | 現行 baseline |
| D-Obj | + objective | 加課程目標 |
| D-V2 | + summary + keywords + weekly_topics | LLM 增強單欄文件 |
| D-V3 | multi-field weighted embedding | 分欄索引 + 加權 |
| D-Prop | propositional indexing | 把 syllabus 切成獨立命題句再 embed |
| D-Parent | small-to-big | 小段 embed、命中後回傳整門課 |

### 5.2 Query-side 方法（這是你問的重點）
| 代號 | 方法 | 原理 | 適用場景 |
|---|---|---|---|
| Q-Raw | 原始 query | baseline | — |
| Q-HyDE | Hypothetical Document Embeddings | LLM 生成假設答案文件後 embed | 短 query → 長文件 |
| Q-Q2D | Query2Doc | LLM 生成擴展段落（含原 query） | 比 HyDE 穩定，召回 +5pp |
| Q-Exp | LLM Query Expansion | 加同義詞、領域術語 | 學科術語對映課名口語 |
| Q-Multi | Multi-Query / RAG-Fusion | LLM 改寫成 N 個 paraphrase，各自 retrieve 後 RRF | 模糊 query |
| Q-StepBack | Step-back Prompting | 抽更上層概念（「ML」→「AI/資料科學」）一起檢索 | 過於具體的 query |
| Q-Decomp | Sub-query Decomposition | 把複合 query 拆「想學X」+「不要早八」+「商院」 | 複合條件 query |
| Q-Struct | Structured Extraction | LLM 抽硬約束 → SQL filter，剩餘語意走 retrieval | **約束型 query 必備** |
| Q-Hybrid | Q-Struct + Q-Q2D 組合 | 先抽 metadata 過濾，再對主題做 Q2D | 推薦最終配置 |

### 5.3 Retrieval / Fusion 演算法
| 代號 | 方法 |
|---|---|
| R-BM25 | BM25 + jieba 中文斷詞（baseline 沒斷詞） |
| R-Dense | bge-m3 dense |
| R-Sparse | bge-m3 sparse 模式（learned sparse） |
| R-ColBERT | bge-m3 multi-vector / late interaction |
| F-Weight | 加權合併（baseline 0.5/0.5） |
| F-RRF | Reciprocal Rank Fusion（無需調權） |
| F-Rerank | top-50 → bge-reranker-v2-m3 → top-10 |
| F-LLMRerank | top-20 → LLM listwise rerank（昂貴，僅做 ablation） |

### 5.4 Filter 層
| 代號 | 內容 |
|---|---|
| Filter-Time | 排除使用者已有課表衝突的時段 |
| Filter-Lang | 語言過濾 |
| Filter-Kind | 必/選修、通識類別 |
| Filter-Point | 學分數區間 |
| Filter-Unit | 系所/學院 |

---

## 6. 實驗矩陣（Ablation）

固定：M1 = bge-m3、M2 = bge-reranker-v2-m3、retrieve k=50 → rerank 取 top-10

| 編號 | 文件 | Query | Retrieval | Rerank | Filter | 假設 |
|---|---|---|---|---|---|---|
| E0 | D-Base | Q-Raw | BM25(空格)+Dense 0.5/0.5 | ✗ | ✗ | 重現現行 baseline |
| E1 | D-Base | Q-Raw | BM25(jieba)+Dense RRF | ✗ | ✗ | 中文斷詞收益 |
| E2 | D-Obj | Q-Raw | BM25+Dense RRF | ✗ | ✗ | 加 objective 收益 |
| E3 | D-V2 | Q-Raw | BM25+Dense RRF | ✗ | ✗ | LLM 摘要+關鍵字收益 |
| E4 | D-V3 | Q-Raw | multi-field RRF | ✗ | ✗ | 分欄 embedding 收益 |
| E5 | D-V3 | Q-Raw | RRF | ✓ | ✗ | reranker 收益 |
| E6 | D-V3 | Q-HyDE | RRF | ✓ | ✗ | HyDE 收益 |
| E7 | D-V3 | Q-Q2D | RRF | ✓ | ✗ | Q2D vs HyDE |
| E8 | D-V3 | Q-Multi | RRF | ✓ | ✗ | Multi-query 收益 |
| E9 | D-V3 | Q-StepBack | RRF | ✓ | ✗ | StepBack 收益 |
| E10 | D-V3 | Q-Struct | RRF | ✓ | ✓ all | 結構化抽取收益 |
| E11 | D-V3 | Q-Hybrid (Struct+Q2D) | RRF | ✓ | ✓ all | **預期最佳** |
| E12 | D-V3 | Q-Hybrid | RRF | ✓ + LLM rerank | ✓ all | LLM rerank 是否值得 |
| E13 | D-Prop | Q-Hybrid | RRF | ✓ | ✓ all | propositional 索引 |
| E14 | D-V3 | Q-Hybrid | + R-ColBERT | ✓ | ✓ all | late interaction |

---

## 7. 評估指標

主指標（test set 150 query 平均）：
- **Recall@10、Recall@20**
- **MRR@10**
- **nDCG@10**

副指標：
- 時段衝突率（top-10 中與使用者預設課表衝突比例，越低越好）
- 語言/必選修不符比例
- query 處理 P95 latency
- index build time、index size

統計檢定：
- 對 E0 baseline 做 paired bootstrap (1000 resamples)，95% CI
- 兩兩比較用 paired Wilcoxon signed-rank，p < 0.05

切片分析：
- 按 query type（topic / teacher / constraint / fuzzy / compound）分別出表
- 按系所、語言切片
- error analysis：對 E11 vs E0 的 100 個進步/退步 case 做手動歸因

---

## 8. 預期結果（hypothesis）

| 步驟 | Recall@10 預期 lift |
|---|---|
| E0 baseline | ~25% |
| +jieba (E1) | +3 pp |
| +objective (E2) | +12 pp |
| +V2 摘要關鍵字 (E3) | +6 pp |
| +V3 multi-field (E4) | +3 pp |
| +reranker (E5) | +5 pp |
| +query rewrite (E6/7/8) | +3~6 pp |
| +structured filter (E10/11) | +5 pp |
| **E11 推估** | **~62%** |

業務指標：時段衝突率從 ~30% 降到 < 5%（filter 直接保證）。

---

## 9. 工程結構（在 scheduling_test/）

```
scheduling_test/
├── PLAN.md              ← 本文件
├── pyproject.toml       (uv 管理)
├── data/
│   ├── 1142.db          symlink to ../NCCUCourse/1142.db
│   ├── eval_synth.jsonl 來源 A
│   └── eval_gold.jsonl  來源 B
├── src/
│   ├── loader.py        dedupe + clean → unified Course dataclass
│   ├── meta_gen.py      M3 一次性產生 summary/keywords，寫入 course_meta_v1
│   ├── doc_builders/    D-Base / D-Obj / D-V2 / D-V3 / D-Prop / D-Parent
│   ├── query_rewriters/ HyDE / Q2D / Multi / StepBack / Decomp / Struct
│   ├── retrievers/      BM25 / Dense / Sparse / ColBERT / RRF / Weight
│   ├── rerankers/       bge-reranker / LLM-listwise
│   ├── filters/         time / lang / kind / point / unit
│   └── pipeline.py      組裝 (doc, query, retr, rerank, filter) 配置
├── scripts/
│   ├── gen_eval_synth.py
│   ├── annotate_gold.py 簡易 Flask 標註介面
│   ├── run_experiment.py  跑單一 E 編號
│   └── run_all.py         跑全部 E0..E14 並出表
├── results/
│   ├── runs/            每次跑的 retrieved.jsonl
│   └── tables/          metric 表 csv
└── notebooks/
    ├── 01_data_eda.ipynb
    └── 02_error_analysis.ipynb
```

---

## 10. 時程（單人估算）

| 週 | 工作 |
|---|---|
| W1 | loader + dedupe + EDA + 1142.db 對接；建 D-Base reproduce E0 |
| W2 | M3 跑 meta_gen，建 course_meta_v1；D-Obj / D-V2 |
| W3 | 來源 A 合成 8k eval_synth；抽 200 人工檢查 |
| W4 | 來源 B 標註介面 + 自己標 60 + 找 3 人共標 90 |
| W5 | retrievers + RRF + reranker；跑 E0–E5 |
| W6 | query rewriters；跑 E6–E12 |
| W7 | E13/E14 + error analysis + 統計檢定 |
| W8 | 寫報告 + 整理 reproducibility |

---

## 11. Future Work（明確排除）

1. **RATE 教學評鑑利用** — 跨學期 courseId 不對齊、需 teacher resolve、且涉情感/品質分數，先不做
2. RESULT 熱門度作 ranking signal
3. 學生主修/已修課的 personalization
4. 多輪對話（修正 query）
5. 課表 packing（給定 retrieval 結果做衝突最小化排課）
6. 跨學期歷史課程關聯（先修課推薦）

---

## 12. 風險與緩解

| 風險 | 緩解 |
|---|---|
| 來源 A LLM 生成 query 過度貼近 objective → 高估準確度 | 最終數字一律以 test set (來源 B) 為準 |
| 標註者疲勞、κ 不夠 | 每 query 限制候選 ≤ 30，先用 baseline 預檢索，標註者只審核 |
| bge-m3 對冷門系所術語 OOV | M3 keywords 步驟強制覆蓋學科術語 |
| LLM 成本 | meta_gen 走 home_mac gemma3:4b（免費）；query rewriter 同走 gemma3:4b；只有 eval-set 合成用 Claude/GPT-5（一次性，~US$50） |
| reranker 推論慢 | top-50 → top-10，rerank 規模可控；GPU 不可得時降到 top-30 |
| home_mac SSH tunnel 中斷 | 程式 fail loud，加 reconnect script；長跑任務改用 tmux 在 home_mac 本機跑 |
| gemma3:4b 中文 query rewrite 品質不足 | 抽樣比較 vs Claude，必要時 fallback 到 API 模型，僅換 query 端，不影響檢索層 ablation |

---

## 13. 立即下一步

1. `loader.py` + dedupe + 確認 dataclass schema
2. `gen_eval_synth.py` 跑 1142 全部課程 → 產 8k 合成 query
3. 開 `annotate_gold.py` Flask 介面，先標 30 筆檢驗流程
4. reproduce E0 baseline，確認 metric harness 對得上現行系統
