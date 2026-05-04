# NCCU 課程推薦系統 Retrieval 改進實驗 — 進度與結果分析

版本：v5（最終結果）
更新日期：2026-05-04

---

## 1. 研究背景

現行系統（`CourseLangChain/build.py`）page_content 只含「課名+時間+老師」，
完全略過 objective/syllabus。

**核心假設（已驗證）**：把 objective 加進索引是最大單一收益來源。

---

## 2. 模型架構

| 角色 | 模型 | 部署 |
|---|---|---|
| Dense embedding | BAAI/bge-m3 | 本機 CPU（doc cache pkl） |
| Reranker | BAAI/bge-reranker-v2-m3 | home_mac（MPS） |
| LLM 生成（全 batch） | gpt-4o-mini | OpenAI Batch API，< US$2 |

---

## 3. 完整 Ablation 結果（synth_jsonl，LLM 合成 query）

| Config | n | R@10 | nDCG@10 | MRR@10 | ms/q |
|---|---|---|---|---|---|
| **d-base+BM25（現行系統）** | 8253 | 0.288 | 0.188 | 0.156 | 5ms |
| d-obj+BM25 | 8253 | 0.328 | 0.217 | 0.183 | 6ms |
| d-v2+BM25（+LLM meta） | 500 | 0.444 | 0.307 | 0.264 | 4ms |
| d-base+Dense | 8253 | 0.549 | 0.375 | 0.321 | 105ms |
| d-obj+RRF | 8253 | 0.577 | 0.366 | 0.300 | 156ms |
| d-obj+Dense+Q2D | 500 | 0.598 | 0.427 | 0.372 | 741ms |
| d-obj+Dense+HyDE | 500 | 0.616 | 0.432 | 0.373 | 794ms |
| d-obj+Dense | 8253 | 0.675 | 0.476 | 0.414 | 82ms |
| d-v2+Dense | 500 | 0.682 | 0.478 | 0.414 | 182ms |
| d-obj+Q2D+Dense+Rerank(k=20) | 500 | 0.660 | 0.508 | 0.447 | — |
| d-obj+HyDE+Dense+Rerank(k=20) | 500 | 0.702 | 0.535 | 0.471 | — |
| d-obj+RRF+Rerank(k=20) | 500 | 0.708 | 0.541 | 0.486 | 2713ms |
| d-obj+Dense+Rerank(k=50) | 500 | 0.708 | 0.521 | 0.461 | 32113ms |
| d-obj+Dense+Rerank(k=20) | 500 | 0.716 | 0.542 | 0.486 | 4405ms |
| **d-v2+Dense+Rerank(k=20)** | 500 | **0.740** | **0.561** | **0.507** | — |

**最佳：D-V2 + Dense bge-m3 + Reranker(k=20)，R@10=0.740**
（vs 現行 0.288，**+45.2pp**）

---

## 4. Key Findings

**F1：Objective 進索引是最大收益**
- D-Base Dense → D-Obj Dense：0.549 → 0.675（+12.6pp）
- D-Base BM25 → D-Obj BM25：0.288 → 0.328（只 +4pp）
- Dense 能做語意泛化，BM25 無法

**F2：LLM meta（摘要+關鍵字）效果依 retriever 而異**
- D-V2 BM25：0.444（vs D-Obj BM25 0.328，**+11.6pp**）→ BM25 靠關鍵字 exact match 大幅受益
- D-V2 Dense：0.682（vs D-Obj Dense 0.675，僅 +0.7pp）→ Dense 已從 objective 學到語意，meta 邊際低
- D-V2 + Rerank：0.740（vs D-Obj + Rerank 0.716，**+2.4pp**）→ Reranker 能利用更豐富文字

**F3：RRF 在 oral query 反不如純 Dense**
- D-Obj RRF 0.577 < D-Obj Dense 0.675（-10pp）
- BM25 weak signal 汙染 fusion；oral query 中 BM25 是噪音

**F4：Reranker 有效，k=20 優於 k=50**
- Dense + Rerank(k=20)：0.675 → 0.716（+4.1pp）
- Dense + Rerank(k=50)：0.675 → 0.708（+3.3pp，更多候選反帶雜訊）
- D-V2 + Rerank(k=20)：0.740（最佳）

**F5：HyDE > Q2D，+Rerank 雙雙有效**
- HyDE+Dense+Rerank：0.702（vs Dense+Rerank 0.716 略低）
- Q2D+Dense+Rerank：0.660（vs Dense+Rerank 0.716 低一些）
- 這個結果顯示 D-Obj + Dense + HyDE 沒有疊加優勢（HyDE text 和 objective 語意重疊高）
- **D-V2+Dense+Rerank 才是最強組合**（LLM meta 豐富了文件，reranker 充分利用）

---

## 5. 建議部署配置

**D-V2 + Dense bge-m3 + Reranker bge-v2-m3（retrieve-k=20 → top-10）**

| | 現行 | 建議 |
|---|---|---|
| page_content | 課名+時間+老師 | 課名+教師+單位+語言+objective+LLM摘要+關鍵字 |
| 索引 | BM25 whitespace | Dense bge-m3（doc embedding cached） |
| 後處理 | 無 | Reranker bge-v2-m3 |
| R@10 | 28.8% | **74.0%** |
| nDCG@10 | 0.188 | 0.561 |
| 延遲 | 5ms | ~82ms dense + ~4s rerank |

---

## 6. 未完成（P0 必做）

| 項目 | 說明 |
|---|---|
| **Human gold-set 150 query** | 現在全靠 LLM synth，有 model bias；需人工標注做最終 test |
| **Structured filter 整合** | code 完成（時段/語言/必選修），未接進最佳 pipeline |
| **Port 回 CourseLangChain** | build.py 要換 D-V2 + dense + reranker |

---

## 7. 成本

| 任務 | 費用 |
|---|---|
| Eval-set 合成（8,253 query） | ~US$1.5 |
| Course meta（2,795 課，摘要+關鍵字） | ~US$0.06 |
| HyDE/Q2D cache（500 query） | ~US$0.02 |
| **合計** | **~US$1.6** |
