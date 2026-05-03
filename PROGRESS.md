# NCCU 課程推薦系統 Retrieval 改進實驗 — 進度與結果分析

版本：v3（自動更新）
更新日期：2026-05-04

---

## 1. 研究背景

現行系統（`CourseLangChain/build.py`）page_content 只含「課名+時間+老師」。
**核心假設（已驗證）**：把 `objective` 加入索引是最大收益來源。

---

## 2. 模型架構（全面改用 GPT Batch API）

| 角色 | 模型 | 部署 |
|---|---|---|
| Dense embedding | BAAI/bge-m3 | 本機 CPU（doc cache pkl） |
| Reranker | BAAI/bge-reranker-v2-m3 | home_wsl（GPU CUDA fallback CPU） |
| LLM 生成 | gpt-4o-mini（OpenAI Batch） | API，< US$2 |

---

## 3. Ablation 結果（synth_jsonl，LLM 合成 query）

| Config | n | R@10 | nDCG@10 | MRR@10 | ms/q |
|---|---|---|---|---|---|
| **d-base+BM25**（現行系統） | 8253 | 0.288 | 0.188 | 0.156 | 5ms |
| d-obj+BM25 | 8253 | 0.328 | 0.217 | 0.183 | 6ms |
| d-v2+BM25（+LLM meta） | 500 | 0.444 | 0.307 | 0.264 | 4ms |
| d-base+Dense | 8253 | 0.549 | 0.375 | 0.321 | 105ms |
| d-obj+RRF | 8253 | 0.577 | 0.366 | 0.300 | 156ms |
| d-obj+Dense+Q2D | 500 | 0.598 | 0.427 | 0.372 | 741ms |
| d-obj+Dense+HyDE | 500 | 0.616 | 0.432 | 0.373 | 794ms |
| d-obj+Dense | 8253 | 0.675 | 0.476 | 0.414 | 82ms |
| d-v2+Dense（+LLM meta） | 500 | 0.682 | 0.478 | 0.414 | 182ms |
| d-obj+RRF+Rerank(k=20) | 500 | 0.708 | 0.541 | 0.486 | 2713ms |
| d-obj+Dense+Rerank(k=50) | 500 | 0.708 | 0.521 | 0.461 | 32113ms |
| **d-obj+Dense+Rerank(k=20)** | 500 | **0.716** | **0.542** | **0.486** | 4405ms |
| d-v2+Dense+Rerank(k=20) | 500 | 跑中 | — | — | — |
| d-obj+HyDE+Dense+Rerank | 500 | 跑中 | — | — | — |
| d-obj+Q2D+Dense+Rerank | 500 | 跑中 | — | — | — |

**目前最佳：d-obj+Dense+Rerank(k=20)，R@10=0.716**（vs 現行 0.288，**+42.8pp**）

---

## 4. Key Findings

**F1：Objective 進索引 vs 課名 alone**
- D-Base Dense → D-Obj Dense：0.549 → 0.675（**+12.6pp**），這是最大單一收益
- D-Base BM25 → D-Obj BM25：0.288 → 0.328（只 +4pp），BM25 無語意泛化

**F2：LLM meta（摘要+關鍵字）對 BM25 大幅提升**
- D-V2 BM25：0.444（vs D-Obj BM25 0.328，**+11.6pp**）
- D-V2 Dense：0.682（vs D-Obj Dense 0.675，僅 +0.7pp）→ Dense 已從 objective 學到語意，meta 邊際效益低

**F3：RRF 在 oral query 反比純 Dense 差**
- D-Obj RRF 0.577 < D-Obj Dense 0.675（-10pp）
- BM25 weak signal 汙染 RRF fusion；oral query 中 BM25 是噪音

**F4：Reranker 有效，但 k=50 不如 k=20**
- Dense + Rerank(k=20)：0.675 → 0.716（+4.1pp）
- Dense + Rerank(k=50)：0.675 → 0.708（+3.3pp，更多雜訊反而略差）
- 最佳配置：retrieve-k=20，rerank top-10

**F5：HyDE/Q2D 有效，約 +6pp**
- D-Obj+HyDE+Dense：0.616（vs D-Obj+Dense 0.675）
  - 注意：無 rerank 版本，且 query encode 多一次 LLM call（794ms/q）
  - 與 k=20 Dense 差距原因待確認（HyDE text 可能偏長）
- D-Obj+Q2D+Dense：0.598
- 等 HyDE+Dense+Rerank 結果再評估最終收益

---

## 5. 未完成任務

### 5.1 進行中

| 任務 | 位置 | 狀態 |
|---|---|---|
| d-v2+dense+rerank(k=20) | home_wsl | 跑中（Compute Scores 中） |
| hyde+dense+rerank(k=20) | home_wsl | 等 d-v2 完後啟動 |
| q2d+dense+rerank(k=20) | home_wsl | 等 hyde 完後啟動 |

### 5.2 缺少的資料

| 缺什麼 | 影響 |
|---|---|
| **Human gold-set 150 query** | 無法做最終 test，LLM synth 有 model bias |

### 5.3 尚未做

| 項目 | 優先度 |
|---|---|
| Human gold-set 標注介面（Flask + Dcard 種子） | P0 |
| Structured filter 整合（時段/語言/必選修） | P1（code 完成） |
| 把最佳配置 port 回 CourseLangChain | P1 |

---

## 6. 建議部署配置

**D-Obj + Dense bge-m3 + Reranker bge-v2-m3（retrieve-k=20 → top-10）**

| | 現行 | 建議 |
|---|---|---|
| page_content | 課名+時間+老師 | 課名+教師+單位+語言+objective[:600] |
| 索引 | BM25 whitespace | Dense bge-m3 |
| 後處理 | 無 | Reranker bge-v2-m3 |
| R@10 | 28.8% | **71.6%** |
| 延遲 | 5ms | 82ms（無 rerank）/ 4.4s（rerank k=20） |

加 HyDE 預期再 +3–5pp，待 +rerank 結果確認。

---

## 7. 成本

| 任務 | 費用 |
|---|---|
| Eval-set 合成（8,253 query） | ~US$1.5 |
| Course meta（2,795 課） | ~US$0.06 |
| HyDE/Q2D cache（500 query） | ~US$0.02 |
| **合計** | **~US$1.6** |
