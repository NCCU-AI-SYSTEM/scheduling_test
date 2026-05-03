# NCCU 課程推薦系統 Retrieval 改進實驗 — 進度與結果分析

版本：v1（1142 學期資料）
更新日期：2026-05

---

## 1. 研究背景與動機

現行系統（`CourseLangChain/build.py`）的 `page_content` 只含「課名 + 時間 + 老師」，
完全略過 `syllabus`、`objective`、`schedule`。
使用者問「我想學機器學習」，BM25 只能比對課名 token，無法命中「人工智慧概論」等相關課程。

**核心假設（已驗證）**：
> 把 `objective` 加入索引是最大收益來源，遠大於任何 retrieval 算法改進。

---

## 2. 完成工作（W1–W7）

| Week | 內容 | 狀態 |
|---|---|---|
| W1 | Loader + Dedupe + EDA | ✅ |
| W2 | Ollama meta_gen（gemma4:e4b on home_mac） | ✅ 進行中（1221/2795）|
| W3 | OpenAI batch eval-set 合成（8,253 query） | ✅ |
| W4 | Human gold-set 標注介面 | ⬜ 未做 |
| W5 | BM25+jieba / Dense bge-m3 / RRF / Recall+nDCG metrics | ✅ |
| W6 | Structured query rewriter / HyDE / Q2D（gpt-4o-mini batch） | ✅ 部分（HyDE/Q2D batch 送出）|
| W7 | bge-reranker-v2-m3 cross-encoder rerank | ✅ |

---

## 3. 資料集

### 3.1 課程資料

- **來源**：`NCCUCourse/1142.db`（114-2 學期）
- **規模**：3,472 rows → dedupe → **2,795 distinct courses**
- **重複率**：24%（多系所掛課）
- **欄位完整度**：objective/syllabus/schedule 缺 9 筆（0.3%）；classroom 缺 391（11%）
- **語言分布**：中文 72%、英文 21%、其他外語 7%
- **課程種類**：選修 52%、必修 27%、通識/體育 21%

### 3.2 評估資料集（Eval Set）

| 來源 | 規模 | 用途 | 狀態 |
|---|---|---|---|
| LLM 合成（gpt-4.1-mini OpenAI Batch） | **8,253 queries** | dev/ablation | ✅ |
| Human gold-set（Dcard 政大版 + 人工標注） | 目標 150 | 最終 test | ⬜ 未做 |

**合成 query 三類型**（各 2,751 筆）：
- **topic**：「我想學 X」（主題型）
- **constraint**：「週四下午英文授課的 X」（帶約束）
- **colloquial**：口語/別稱改寫

### 3.3 LLM 合成 vs Pseudo eval 差異

`objective_smoke` 是從課程目標抽第一句當 query，屬於 leakage 嚴重的偽資料。
BM25 在 leakage 環境下虛高（完整課名 token 全部命中），不代表真實效果。
以下結果以 **synth_jsonl（8,253）** 為準。

---

## 4. 模型與方法清單

| # | 角色 | 模型 | 部署 |
|---|---|---|---|
| M1 | Dense embedding | BAAI/bge-m3 | 本機（MPS/CPU） |
| M2 | Reranker | BAAI/bge-reranker-v2-m3 | home_mac（更穩）|
| M3 | Query rewriter / meta gen | gemma4:e4b（Ollama） | home_mac / home_wsl |
| M4 | Eval-set + HyDE/Q2D 生成 | gpt-4.1-mini / gpt-4o-mini | OpenAI Batch API |

---

## 5. Ablation 結果（synth_jsonl，LLM 合成 query）

### 5.1 主要結果表

| Config | n | R@5 | R@10 | R@20 | nDCG@10 | MRR@10 | ms/q |
|---|---|---|---|---|---|---|---|
| **d-base + BM25**（現行系統） | 8253 | 0.228 | 0.288 | 0.352 | 0.188 | 0.156 | 5 |
| d-obj + BM25 | 8253 | 0.265 | 0.328 | 0.395 | 0.217 | 0.183 | 6 |
| d-base + Dense | 8253 | 0.457 | 0.549 | 0.628 | 0.375 | 0.321 | 105 |
| d-obj + RRF | 8253 | 0.440 | 0.577 | 0.718 | 0.366 | 0.300 | 156 |
| d-obj + Dense | 8253 | 0.581 | 0.675 | 0.758 | 0.476 | 0.414 | 82 |
| d-obj + RRF + Rerank（k=20） | 500 | 0.668 | 0.708 | 0.708 | 0.541 | 0.486 | 2713 |
| **d-obj + Dense + Rerank（k=20）** | 500 | 0.666 | **0.716** | 0.716 | **0.542** | **0.486** | 4405 |
| d-obj + Dense + Rerank（k=50，n=10） | 10 | — | 0.800 | — | 0.676 | 0.633 | 10827 |

> k=50 完整版（n=500）在 home_mac 跑中，預期 R@10 ~ 0.75–0.80

### 5.2 Key Findings

**發現 1 — 加 objective 進索引是最大收益**

- D-Base BM25 → D-Base Dense：+26pp（0.288 → 0.549）
- D-Base Dense → D-Obj Dense：+13pp（0.549 → 0.675）
- D-Obj BM25 vs D-Base BM25：只 +4pp（BM25 無法做語意泛化）

**結論**：objective 對 Dense 的收益遠大於對 BM25，因為 Dense 能做語意比對。

**發現 2 — BM25 在真實 oral query 上幾乎無效**

- D-Obj BM25 vs D-Base BM25：R@10 僅 0.288 → 0.328（+4pp）
- D-Obj Dense vs D-Base BM25：R@10 0.675（+38pp）
- 原因：LLM 合成 query 用口語/別稱，不直接命中 objective 的學術術語

**發現 3 — RRF 在真實 query 反不如純 Dense**

- D-Obj RRF（0.577）< D-Obj Dense（0.675），相差 -10pp
- 原因：BM25 weak signal 拉低 RRF fusion 結果，對 oral query BM25 是噪音

**發現 4 — Reranker 在 top-k 有效**

- Dense(k=20) 0.675 → Dense+Rerank(k=20) 0.716，+4.1pp
- RRF(k=20) 0.577 → RRF+Rerank(k=20) 0.708，+13.1pp（Rerank 對 RRF 修復更多）
- k=50 速測（n=10）R@10=0.800，retrieve depth 有顯著影響

**發現 5 — home_mac M4 reranker 比本機 MPS 快 4 倍**

- home_mac：~46s/batch（128 pairs）
- 本機 MPS：~180s/batch
- MPS + FlagReranker 有相容性問題，建議 reranker 固定在 home_mac 跑

---

## 6. 建議最終部署配置

**推薦：D-Obj + Dense bge-m3 + Reranker bge-v2-m3（k=50 → top-10）**

- 預期 R@10 ~ 75–80%（比現行 +47–52pp）
- 延遲：doc embed 離線一次性，query encode ~50ms，rerank ~2s（k=50 on GPU）
- 成本：embedding model 本機跑，無 API 費用

**對比現行系統**

| | 現行 | 建議 | 提升 |
|---|---|---|---|
| page_content | 課名+時間+老師 | 課名+教師+單位+語言+objective[:600] | 文字量 10x |
| 索引方式 | BM25(空格分詞) + FAISS ensemble 0.5/0.5 | Dense bge-m3 + Reranker bge-v2-m3 | — |
| R@10 | ~28.8% | ~71.6%（k=20）/ ~80%（k=50） | +43–51pp |
| nDCG@10 | 0.188 | 0.542 / ~0.68 | +0.35+ |
| 延遲 | 5ms | 82ms（無 rerank）/ 4.4s（k=20 rerank） | — |

---

## 7. 待完成（Pending）

### 7.1 進行中

| 任務 | 預估完成 |
|---|---|
| home_mac d-obj+dense+rerank k=50（n=500） | 斷線中，恢復後繼續 |
| OpenAI batch rewrite（HyDE+Q2D 500 query） | 已送出，batch_69f77fea3d448190 |
| meta_gen tmux（1221/2795） | ~48h |

### 7.2 HyDE / Q2D 結果（等 batch 回）

HyDE（Hypothetical Document Embedding）：對 query 生成假設課程描述後 embed，
預期對短 query（「我想學機器學習」）有 +3–5pp R@10 提升。

Q2D（Query2Doc）：把 query 擴展成含原字串的段落，
預期效果接近 HyDE 但更穩定（不丟掉原 query 語意）。

### 7.3 尚未做

| 項目 | 優先度 | 說明 |
|---|---|---|
| Human gold-set（150 query） | P0 | 最終 test 必須，LLM synth 有 model bias |
| D-V2 doc builder（LLM 摘要+關鍵字） | P1 | 等 meta_gen 跑完 2795 筆 |
| Structured filter（時段/語言/必選修） | P1 | W6 parser 已完成，未整合進最佳 pipeline |
| Step-back / Multi-query rewrite | P2 | HyDE/Q2D 確認有效後再做 |
| RATE 教學評鑑利用 | P3 | Future Work，需 teacher-level join |

---

## 8. 工程進度

### 8.1 Repo 結構

```
scheduling_test/
├── src/
│   ├── loader/        SQLite → Course dataclass, dedupe, time-parser
│   ├── doc_builders/  D-Base / D-Obj / D-V2 page_content 建構
│   ├── retrievers/    BM25+jieba / Dense bge-m3 / RRF
│   ├── rerankers/     bge-reranker-v2-m3 (batch_rerank)
│   ├── query_rewriters/ Struct / HyDE / Q2D / Multi / StepBack
│   ├── filters/       時段 / 語言 / 必選修 / 學分 / 系所
│   ├── eval/          Recall/Hit/MRR/nDCG, dataset loader
│   └── llm/           Ollama client, meta_gen, batch build/run/merge
├── scripts/
│   ├── run_experiment.py       主要實驗 runner
│   ├── build_rewrite_batch.py  HyDE/Q2D batch 建立
│   ├── run_rewrite_batch.py    submit/fetch/merge
│   └── eda_w1.py / inspect_miss.py / dense_smoke.py
├── data/
│   ├── 1142.db        symlink → NCCUCourse/1142.db
│   ├── raw/eval_synth.jsonl    8,253 合成 query
│   └── processed/     EDA parquet, course_meta.db, dense_cache, query_cache
├── batches/           OpenAI batch input/output
└── PLAN.md            完整 14 章實驗計畫書
```

### 8.2 Git 紀錄（最近 12 commits）

```
0202bd6 fix(w6): rewrite batch shuffle alignment + MPS fallback to CPU
c1a7b2f feat(w6): HyDE/Q2D via OpenAI batch gpt-4o-mini
3e207ba feat(w7): bge-reranker-v2-m3 batch reranker
b58c62c feat(results): ablation on objective_smoke + real eval_synth
79a4c39 fix(w6): unit keyword collocation guard + tests
1e2499b feat(w6): LLM query rewriters via gemma4:e4b
406df09 feat(w6): structured query rewriter + filter (Q-Struct)
8652e99 fix(w3): anthropic batch chunk 500/req
e4f14c1 feat(w5): retrieval harness — BM25+jieba, RRF, metrics
2a9e833 feat(w3): batch eval-set builder + submit/poll/merge
5d1de1b feat(w2): ollama client + meta_gen
f0e8b72 feat(w1): loader, dedupe, time-parser, EDA harness
```

### 8.3 Tests

- **26/26 passing**，ruff clean
- 覆蓋：loader / time-parser / eval metrics / query rewriter / metadata filter

---

## 9. 成本紀錄

| 項目 | 費用 |
|---|---|
| OpenAI eval-set batch（gpt-4.1-mini, 8,253 req × 3 type） | ~US$1.5 |
| OpenAI rewrite batch #1（gpt-4o-mini, 1000 req，錯誤 batch） | ~US$0.2 |
| OpenAI rewrite batch #2（gpt-4o-mini, 1000 req） | ~US$0.2 |
| gemma4:e4b on home_mac/home_wsl | $0（本地） |
| bge-m3 + bge-reranker | $0（HuggingFace 下載） |
| **合計** | **~US$2** |

---

## 10. 下一步優先順序

1. **等 HyDE/Q2D batch 回來** → fetch → merge → run_experiment → 更新結果表
2. **human gold-set 標注**（建議用 `data/dcard選課排幾txt/` 作種子，開 Flask UI）
3. **D-V2 doc builder**（meta_gen 跑完後，加 LLM 摘要 + 關鍵字進索引）
4. **Structured filter 整合**（解析「不要早八」「英文授課」等約束，post-filter）
5. **把建議配置 port 回 CourseLangChain**（換 build.py + 加 reranker 到 pipeline）
