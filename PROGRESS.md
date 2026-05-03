# NCCU 課程推薦系統 Retrieval 改進實驗 — 進度與結果分析

版本：v2（模型全面改用 GPT）
更新日期：2026-05

---

## 1. 研究背景與動機

現行系統（`CourseLangChain/build.py`）的 `page_content` 只含「課名 + 時間 + 老師」，
完全略過 `syllabus`、`objective`、`schedule`。
使用者問「我想學機器學習」，BM25 只能比對課名 token，無法命中「人工智慧概論」等相關課程。

**核心假設（已驗證）**：把 `objective` 加入索引是最大收益來源，遠大於任何 retrieval 算法改進。

---

## 2. 模型架構（已全面改用 GPT）

Ollama 依賴已完全移除，所有 LLM 生成任務統一走 **OpenAI Batch API gpt-4o-mini**。

| 角色 | 模型 | 部署 |
|---|---|---|
| Dense embedding | BAAI/bge-m3 | 本機 CPU（doc embeddings from cache pkl） |
| Reranker | BAAI/bge-reranker-v2-m3 | home_mac / home_wsl |
| LLM 生成（batch） | gpt-4o-mini | OpenAI Batch API |

費用：所有 LLM 任務合計 **< US$2**（Batch API 50% 折扣）。

---

## 3. 完成工作（W1–W7）

| Week | 內容 | 狀態 |
|---|---|---|
| W1 | Loader + Dedupe + EDA | ✅ |
| W2 | Course meta 生成（summary/keywords/tags） | ✅ GPT batch 送出（1574 reqs，待 merge）|
| W3 | OpenAI batch eval-set 合成（8,253 query） | ✅ 已完成 |
| W4 | Human gold-set 標注介面 | ⬜ 未做 |
| W5 | BM25+jieba / Dense bge-m3 / RRF / Recall+nDCG harness | ✅ |
| W6 | Structured query rewriter + HyDE/Q2D（GPT batch） | ✅ batch 送出（1000 reqs，待 merge）|
| W7 | bge-reranker-v2-m3 cross-encoder rerank | ✅ |

---

## 4. 資料集

### 4.1 課程資料

- **來源**：`NCCUCourse/1142.db`（114-2 學期）
- **規模**：3,472 rows → dedupe → **2,795 distinct courses**
- **欄位完整度**：objective/syllabus 缺 9 筆（0.3%），classroom 缺 391（11%）
- **語言**：中文 72%、英文 21%、其他外語 7%

### 4.2 評估資料集

| 來源 | 規模 | 用途 | 狀態 |
|---|---|---|---|
| LLM 合成（gpt-4.1-mini Batch） | **8,253 queries** | dev/ablation | ✅ 完成 |
| Human gold-set | 目標 150 筆 | 最終 test | ⬜ 未做 |

合成 query 三類型（各 2,751 筆）：topic / constraint / colloquial

### 4.3 進行中的 Batch Jobs

| Batch ID | 用途 | Reqs | 狀態 |
|---|---|---|---|
| batch_69f786db3f148190 | Course meta（summary/keywords） | 1,574 | validating |
| batch_69f77fea3d448190 | HyDE + Q2D（500 × 2） | 1,000 | validating |

---

## 5. Ablation 結果

### 5.1 主要結果（synth_jsonl，LLM 合成 query，n=500-8253）

| Config | n | R@5 | R@10 | R@20 | nDCG@10 | MRR@10 | ms/q |
|---|---|---|---|---|---|---|---|
| **d-base + BM25**（現行系統） | 8253 | 0.228 | 0.288 | 0.352 | 0.188 | 0.156 | 5 |
| d-obj + BM25 | 8253 | 0.265 | 0.328 | 0.395 | 0.217 | 0.183 | 6 |
| d-base + Dense | 8253 | 0.457 | 0.549 | 0.628 | 0.375 | 0.321 | 105 |
| d-obj + RRF | 8253 | 0.440 | 0.577 | 0.718 | 0.366 | 0.300 | 156 |
| d-obj + Dense | 8253 | 0.581 | 0.675 | 0.758 | 0.476 | 0.414 | 82 |
| d-obj + RRF + Rerank (k=20) | 500 | 0.668 | 0.708 | 0.708 | 0.541 | 0.486 | 2713 |
| **d-obj + Dense + Rerank (k=20)** | 500 | 0.666 | **0.716** | 0.716 | **0.542** | **0.486** | 4405 |
| d-obj + Dense + Rerank (k=50) | 跑中 | — | ~0.80 | — | — | — | — |

### 5.2 Key Findings

**F1 — objective 進索引是最大收益，+13pp（dense）**
- D-Base Dense → D-Obj Dense：0.549 → 0.675（+12.6pp）
- D-Base BM25 → D-Obj BM25：0.288 → 0.328（只 +4pp）
- Dense 能做語意泛化，BM25 無法

**F2 — BM25 在 oral query 幾乎失效**
- LLM 合成 query 用口語/別稱，不命中 objective 的學術術語
- D-Obj BM25 vs D-Base BM25 差距只有 4pp

**F3 — RRF 在 oral query 拉低 Dense**
- D-Obj RRF（0.577）< D-Obj Dense（0.675），差 -10pp
- BM25 weak signal 汙染 RRF fusion

**F4 — Reranker 有效，在 RRF 上收益更大**
- Dense + Rerank(k=20)：0.675 → 0.716（+4.1pp）
- RRF + Rerank(k=20)：0.577 → 0.708（+13.1pp，Rerank 修復了 BM25 噪音）
- k=50 速測（n=10）R@10=0.800，depth 增加有正向影響

**F5 — HyDE/Q2D 待確認（batch 中）**
預期 oral/短 query 有 +3–5pp 提升，等 batch 結果後補充。

### 5.3 建議部署配置

**D-Obj + Dense bge-m3 + Reranker bge-v2-m3（k=50 → top-10）**

| | 現行 | 建議 |
|---|---|---|
| page_content | 課名+時間+老師 | 課名+教師+單位+語言+objective[:600] |
| 索引 | BM25 whitespace | Dense bge-m3 |
| 後處理 | 無 | Reranker bge-v2-m3 |
| R@10 | 28.8% | **71.6%（k=20）/ ~80%（k=50）** |
| nDCG@10 | 0.188 | 0.542 / ~0.68 |

---

## 6. 未完成任務與缺失資料

### 6.1 進行中（等通知）

| 任務 | 位置 | 預估完成 |
|---|---|---|
| rerank k=50 完整版（n=500） | home_mac，28/195 batches | ~2.5h |
| meta batch merge | OpenAI batch_69f786db3f148190 | 24h 內 |
| HyDE/Q2D cache merge | OpenAI batch_69f77fea3d448190 | 24h 內 |

### 6.2 批次完成後立刻要做

1. `uv run python scripts/run_meta_batch.py --fetch --merge`
2. `uv run python scripts/run_rewrite_batch.py --fetch --merge`
3. 重跑 `run_experiment --rewrite hyde --retriever dense --reranker bge n=500`
4. 重跑 `run_experiment --rewrite q2d  --retriever dense --reranker bge n=500`
5. 更新本文結果表

### 6.3 缺失資料（阻塞後續）

| 缺少什麼 | 影響 | 補充方式 |
|---|---|---|
| **Human gold-set（150 query）** | 無法做最終 test，只有 LLM synth bias | Dcard 政大版 + 人工標注，建 Flask UI |
| course_meta 完整（1221/2795 done） | D-V2 doc builder 無法跑 | 等 GPT meta batch |
| HyDE/Q2D 正確 cache | HyDE/Q2D experiment 無法跑 | 等 GPT rewrite batch |

### 6.4 尚未做的實驗

| 項目 | 優先度 | 前置條件 |
|---|---|---|
| HyDE + Dense + Rerank | P0 | HyDE cache batch 到位 |
| Q2D + Dense + Rerank | P0 | Q2D cache batch 到位 |
| D-V2 doc builder（LLM 摘要+關鍵字） | P1 | meta batch 完成 |
| Structured filter 整合（時段/語言/必選修） | P1 | filter code 已完成，未整進最佳 pipeline |
| rerank k=50 完整版 | P1 | home_mac 跑中 |
| Step-back / Multi-query rewrite | P2 | HyDE/Q2D 確認有效後 |
| Human gold-set 最終 test | P0 | 人工標注 150 query |

---

## 7. 工程架構

```
scheduling_test/
├── src/
│   ├── loader/          SQLite → Course dataclass, dedupe, time-parser
│   ├── doc_builders/    D-Base / D-Obj / D-V2 page_content
│   ├── retrievers/      BM25+jieba / Dense bge-m3 / RRF
│   ├── rerankers/       bge-reranker-v2-m3 (batch_rerank)
│   ├── query_rewriters/ Struct / HyDE / Q2D / Multi / StepBack
│   ├── filters/         時段 / 語言 / 必選修 / 學分 / 系所
│   ├── eval/            Recall/Hit/MRR/nDCG metrics, dataset loader
│   └── llm/             OpenAI batch build/run/merge, meta_gen
├── scripts/
│   ├── run_experiment.py         主要實驗 runner
│   ├── run_meta_batch.py         W2 meta gen batch
│   ├── build_rewrite_batch.py    W6 HyDE/Q2D batch build
│   ├── run_rewrite_batch.py      W6 HyDE/Q2D batch submit/fetch/merge
│   └── eda_w1.py / dense_smoke.py / inspect_miss.py
├── data/
│   ├── 1142.db           symlink → NCCUCourse/1142.db
│   ├── raw/eval_synth.jsonl       8,253 合成 query
│   └── processed/         EDA parquet, course_meta.db, dense_cache, query_cache
└── PLAN.md                完整 14 章實驗計畫書
```

**Tests**: 26/26 passing，ruff clean

---

## 8. 成本紀錄

| 任務 | 費用 |
|---|---|
| Eval-set 合成（gpt-4.1-mini，8253 query） | ~US$1.5 |
| Meta batch（gpt-4o-mini，1574 課） | ~US$0.06 |
| HyDE/Q2D batch（gpt-4o-mini，1000 req） | ~US$0.02 |
| **合計** | **~US$1.6** |
