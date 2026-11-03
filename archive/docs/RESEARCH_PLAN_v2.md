# 課程檢索研究計畫 v2（重新定義目標）

更新：2026-10-05　實驗機：hopelab（RTX 5080 16GB、60GB RAM、Ollama）
工作目錄：`hopelab:~/workspace/code/nccu-course/scheduling_test`

---

## 0. 一頁摘要

- **v1 成果**：把課程內容（objective + LLM 摘要/關鍵字）放進索引，再加上 RRF 和 reranker，synthetic R@10 從 0.288 提升到 0.768（E7）。
- **v1 的盲點**：評估集每題只有一個 gold，而且題目是從 gold 課程本身生成的。
  - 114-2 有 58% 的課程有同名的其他班次。只要把「同名課」也算命中，topic 題的 R@10 就是 0.979，已經飽和、測不出差異。
  - constraint 題通常有數百門課都符合條件，所以「單一 gold 的 R@10 = 0.47」主要反映的是評估方法本身的問題，而不是系統有多差。
- **v2 主軸**：
  1. 重建評估：多 gold、分級標註，條件是否滿足由 DB 自動檢查。
  2. 條件優先的混合檢索（parser 抽出條件 → pre-filter 或 soft scoring → 語意排序）。
  3. 用上 v1 完全沒用到的資產：28 學期的歷史資料、26.9 萬則學生評鑑、選課結果。
- **服務重點**：本系統是 agent 的 `retrieve_tool`。對學生來說，最重要的是兩件事：一是列出的課**合格**（時間、語言、課別都對），二是內容**切題**。

---

## 1. 系統全貌（各 repo 怎麼串）

```
NCCUCourse ──爬蟲──► 1142.db（SQLite，101-1 至 114-2，共 28 學期）
      │                  config.yaml 目標已改成 115-1，但本地還沒有 1151.db
      ▼
course-data-prep（另一個 repo，不在本 workspace）── 產生 PostgreSQL dump（.sql.gz）
      ▼
CourseLangChain  origin/feat/mcp-test（2026-09-26，真正的開發線；本地 main 已落後）
  brain_agent（Ollama / Gemini）
   ├ text_to_sql_tool → 只處理時間條件，轉成 sql_filter
   ├ retrieve_tool   → ParadeDB BM25（jieba）+ pgvector 逐欄 chunk（bge-base-zh-v1.5），在 SQL 內做 RRF。沒有 reranker
   ├ course_detail / schedule / my_schedule / preference_order（Dcard 志願序）/ user_profile
   └ MCP：Moodle、校內網站
      ▼
CourseLangChain-frontend  origin/feat/add-to-schedule-from-chat（課表面板、志願序）
      ▼
CourseLangChain-all  docker（nginx + backend）
  submodule 釘在 2026-03/04 的舊版本，部署設定沒跟上新架構；compose 裡也沒有 postgres
  course.andyjjrt.cc 是 NXDOMAIN → 目前沒有線上站
scheduling_test（本 repo）＝ 檢索研究，結果要回饋給 retrieve_tool
```

**我負責的介面**：`retrieve_tool(keyword, top_k, sql_filter)`。研究結論必須能落地到這個簽名，或是擴充它，例如把 `sql_filter` 泛化成結構化的 `conditions`。

---

## 2. v1 已完成的實驗（摘要）

| 階段 | 結論 |
|---|---|
| 文件表示 | 加 objective 帶來 +12.6pp（Dense），是最大的單一收益。LLM meta 主要幫到 BM25（+11.6pp） |
| 檢索 | 用 D-V2 文件時 RRF 才是正收益。reranker 取 k=20 時 +4pp |
| Query 改寫 | HyDE（-1.4pp）、Q2D（-5.6pp）都廢棄 |
| Post-filter | 所有題型都是淨負，已廢棄 |
| BM25 條件擴充 | E7，全集最佳：0.768（topic 0.943／口語 0.893／條件 0.469） |
| Parser | P3 LLM few-shot 的 ExactMatch 是 68%，系所欄位最難（F1 0.89）。Oracle filter 保留 gold 的比例是 99.9% |
| 未完成 | filter + retrieval 串接實驗（序列跑太慢，停在 n=20）；移植回 CourseLangChain |

---

## 3. 資料分析（新）

| 資產 | 規模 | v1 是否使用 | 觀察 |
|---|---|---|---|
| COURSE 114-2 | 2,795 門（3,472 列） | ✓ | 58% 有同名的其他班次。中文課 73%、英文課 21% |
| COURSE 歷史 | 28 學期、約 10 萬列 | ✗ | 80.5% 的 114-2 課程在過去學期出現過同名同師的課 |
| RATE 評鑑 | 26.9 萬則短評（平均 30 字） | ✗ | 65% 的 114-2 課程連得到過去的評鑑（中位數 20 則、p90 211 則）。高頻詞：作業 13.6k、報告 13.1k、考試 12.1k、推 8.3k、分組 2.1k、點名 1.1k、涼 594、甜 462 |
| RESULT 選課結果 | 1142 全覆蓋，另有 1102–1141 的部分資料 | ✗ | 填滿率中位數 0.92，44% 的課額滿 → 可當熱門度／搶手度訊號 |
| Dcard 志願序 | 4,783 行回報（108-1 至 114-2） | 只在 prod 的 preference_order 用到 | 格式為「時段 課名 老師 身分 排N上/沒上」→ 可估「排第幾志願能上」 |
| eval_synth_clean | 7,954 題、單一 gold | ✓ | topic 題已飽和；同名班次造成低估 |
| eval_conditions_v1 | 2,160 題純條件，有 oracle 條件 | ✓（只用在 parser） | 可以直接用 DB 算「條件滿足率」 |

**評估集的三個結構性問題**：
1. **同名班次**：例如「管理學」有 7 個班，評估時只有其中 1 班算對。
2. **自我生成洩漏**：題目由 gold 課程的內容生成，用詞直接重疊，對 dense 和 BM25 都偏簡單。
3. **條件題的 gold 不唯一**：純條件題（如「週一下午中文授課」）有數百門合格課，用 R@10 衡量不合理。

文獻佐證：Rahmani et al., SIGIR 2024 指出，只把來源 passage 當作相關（sparse judgment）時，系統排名與真實排名的相關係數 τ 會從 0.82 掉到 0.16。

---

## 4. 重新定義目標

### 4.1 服務重點

學生在選課期問 agent 時，檢索層要做到下面四件事（依重要性排序）：

1. **合格**：top-k 裡的每門課都滿足學生講出的硬條件（時間、語言、課別、通識類別、系所、學分），否定條件也包含在內。不合格的課會直接造成衝堂，或讓 agent 回答錯誤。
2. **切題**：內容符合學生想學的東西，口語和模糊需求也要能處理。
3. **可決策**：在多個合格又切題的課之間，提供能幫學生做決定的訊號，例如評價、作業量、搶手程度。
4. **夠快**：agent 一輪可能會呼叫 2–3 次檢索，所以 p95 延遲要小於 1.5 秒（GPU）或 3 秒（CPU）。

### 4.2 量化指標（v3 eval）

| 指標 | 定義 | 量測對象 |
|---|---|---|
| **nDCG@10** | 分級相關度 0–3，由 LLM judge 給分並經人工校準 | 切題 |
| **P@10(rel≥2)** | top-10 中相關課程的比例 | 切題 |
| **Precision@10（命中＝滿足硬條件）** | top-10 中滿足全部硬條件的比例，由 DB 自動判定 | 合格 |
| **Success@10**（命中＝至少一門相關且合格） | top-10 中至少有一門課「相關且合格」的題目比例 | 綜合 |
| Sibling-R@10 | 舊 gold 或其同名班次是否進入 top-10 | 和 v1 數字銜接 |
| Latency p50/p95 | 單次 tool call 的耗時 | 夠快 |
| Judge 信度 | 人工標 100 對，計算 Cohen's κ（目標 ≥0.6） | 評估本身 |

### 4.3 v3 評估集

| 子集 | 題數 | 來源 | 相關度怎麼來 |
|---|---|---|---|
| T | 240（topic 120 + 口語 120） | eval_synth_clean | 從 6 個系統各取 top-20 做 pooling，再由 qwen3:14b judge 評 0–3 |
| M | 200 | eval_synth_clean 的 constraint 題 | LLM 拆成「主題 need + 條件」，只保留 gold 有滿足的條件；主題部分由 judge 評，條件部分由 DB 判定 |
| C | 約 300（依 complexity × negation 分層） | eval_conditions_v1 | 只用 DB 判定（oracle 條件） |
| H（待做） | 100 | 學生親寫（Dcard 風格） | 人工標註，同時用來校準 judge |

---

## 5. 相關研究 → 對應到本資料的創新點

| # | 創新點 | 文獻依據 | 為何適合本資料 | 成本 |
|---|---|---|---|---|
| I1 | **評估重建**：多 gold、分級，條件由 DB 判定 | Rahmani SIGIR'24、UMBRELA 2024、Gecko 2024 | 同名班次多、條件題的 gold 不唯一 | 低（本地 LLM） |
| I2 | **條件優先檢索**：LLM parser（系所欄位限定為 enum，用 JSON schema 輸出）→ pre-filter，pool 太小時自動放寬 | HyST 2025、QAM 2025、Fraunhofer 2025 | oracle 保留 gold 99.9%，pool 中位數 538 門 | 低 |
| I3 | **Soft constraint scoring**：每個條件當成模糊因子，否定條件用 1−s | ITLR ACL'25、NS-IR ACL'25 | parser 只有 68% exact，硬過濾錯一個就會把 gold 刪掉 | 低 |
| I4 | **跨學期 + 評鑑擴充文件**：同名同師的歷屆大綱 + RATE 短評摘要成「學生視角」欄位（作業量、考試、給分） | LLM-CR 2026、doc expansion | 65% 的課有評鑑；可以處理「好過、涼、作業少」這類需求，目前的系統完全做不到 | 中（LLM 離線摘要約 1,800 門） |
| I5 | **熱門度／搶手度 prior**：RESULT 的填滿率、Dcard「排幾能上」 | DS@GT TREC ToT 2025 | 資料現成；學生很在意「選不選得上」 | 低 |
| I6 | **Promptriever 式條件 hard negative，微調 reranker**：同主題但條件不合的課當負例；以學期切分訓練（101–114-1 訓練，114-2 測試），避免洩漏 | Promptriever ICLR'25、InPars-v2 | 有 28 學期資料可以零洩漏地切分 | 中（5080 可跑 LoRA） |
| I7 | **換模型**：Qwen3-Embedding-0.6B/4B、Qwen3-Reranker-0.6B/4B，對照 prod 的 bge-base-zh 逐欄 chunk | Qwen3-Embedding 2025 | prod 用的是 768 維小模型 | 低 |
| I8 | **Facet 澄清**：top-20 在系所、時段等面向上的分散度太高時，回傳 facet chips 讓學生選 | ASK ACL'25 | 模糊題型多 | 低（屬於 agent 層，列為後續） |

---

## 6. 實驗進度與預期成效

| Phase | 內容 | 預期 | 成功標準 | 狀態 |
|---|---|---|---|---|
| P0 | hopelab 環境；重現 E7 | R@10 ≈ 0.768 | ±0.5pp 內 | 進行中 |
| P1 | 建 eval v3（T/M/C）+ judge 信度 | — | κ ≥ 0.6；gold 被評 ≥2 的比例 > 85% | 進行中 |
| P2 | 基準重評：BM25、Dense、E7、**prod replica**（bge-base-zh 逐欄 chunk + RRF） | E7 的 Success@10 > prod | 給出 prod 與 E7 的差距 | 待做 |
| P3 | I2 + I3：條件優先（oracle / LLM parser / regex）× hard / soft | M 子集 Precision@10（條件）：0.4 → 0.9 以上；Success@10 +15pp | soft ≥ hard 且 T 子集不退步 | 待做 |
| P4 | I7 換模型 | nDCG@10 +2–4pp | 延遲 p95 < 1.5s | 待做 |
| P5 | I4 + I5 學生視角文件與 prior | 新增的「評價型」題 Success@10 從約 0 提升到 > 0.5 | 需要新的題型（H 子集） | 待做 |
| P6 | I6 reranker 微調（學期切分） | nDCG@10 +3pp，Precision@10（條件）提升 | 在 114-2 上不退步 | 待做 |
| P7 | 移植到 `retrieve_tool`（介面：conditions + reranker） | — | 端到端 agent 測試通過 | 待做 |

---

## 7. 風險

- **LLM judge 偏寬鬆**：絕對分數會偏高。對策是只比較系統之間的相對差，並用人工 κ 校準。
- **judge 與系統用同類模型的循環偏差**：judge 只看原始欄位，不看 LLM 生成的 meta。
- **評鑑資料涉及老師的評價**：只做聚合摘要，不原文外露。實際上線前要和團隊確認是否合規。
- **GPU 共用**：Ollama 和 torch 不能同時佔用 GPU，實驗必須排成序列執行。
