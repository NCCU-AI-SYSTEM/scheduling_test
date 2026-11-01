# v4 文獻調查：政大課程搜尋可用的新方法（2023–2026）

整理日期 2026/10/18。每筆附 APA 簡式與 URL，每筆分發現與套用兩行。標 [未驗證] 者只找到二手來源或沒找到原文；其餘至少核對過 arXiv、會議論文集或官方頁面的標題與年份。部署限制：主 LLM 為 Gemma 4 31B（Ollama Cloud，2026/04/02 釋出，Apache 2.0；Artificial Analysis 頁面 <https://artificialanalysis.ai/models/releases/gemma-4-31b），其餘元件需能在單張> 16GB GPU 或 CPU 上跑。

---

## 1. 大學課程搜尋與推薦

- **AskOski**（UC Berkeley CAHL lab）。<https://cms.askoski.berkeley.edu>
  發現：結合 degree audit、課綱、課表與歷年選課資料的機器學習選課探索系統；官方頁面沒有提到 LLM 對話。
  套用：政大已有 28 學期選課結果與成績單匯出，資料面和 AskOski 同類，可先做歷史熱門度與修課路徑特徵。
- **From Interests to Insights**（Van Deventer et al., 2024），arXiv 預印本。<https://arxiv.org/abs/2412.19312>
  發現：先讓 LLM 依使用者需求寫一份理想課程描述，再拿它的 embedding 找真實課程（HyDE 式）；只做少量範例與公平性檢查，沒有量化評測。
  套用：可做為主題查詢的基線，但查詢端多一次 LLM 呼叫，延遲比 doc 端擴充高。
- **SmartCourse**（Mi et al., 2025）。<https://arxiv.org/abs/2507.22946>
  發現：把成績單與修業規劃放進 prompt，用 Ollama 本地模型給建議；25 題評測中，完整脈絡版比拿掉脈絡版明顯更相關（自訂指標 PlanScore、PersonalScore、Lift、Recall）。
  套用：成績單匯出值得接進來，但題數太少；修課資格建議用規則判斷，LLM 只負責解釋。
- **A Locally Deployed RAG-Based Academic Advising System for Course Selection**（Li & Iwata, 2026），Procedia Computer Science (KES 2026)。<https://arxiv.org/abs/2606.02983>
  發現：本地部署 RAG，支援選課、先修理解、修業規劃。摘要沒有模型與評測細節 [評測細節未驗證]。
  套用：和政大設定最接近的近期論文，可讀全文比較功能清單。
- **Marcel**（Trienes et al., 2025），EMNLP 2025 System Demonstrations。<https://arxiv.org/abs/2507.13937>
  發現：給資源有限學校用的開源學生服務 RAG；加入 FAQ retriever，作者稱優於一般 dense 與 hybrid 檢索，並分享實際上線經驗。
  套用：常見問句（加退選、通識認列）可另建 FAQ 索引，不必全部丟給課程檢索。
- **How Good Are LLMs for Course Recommendation in MOOCs?**（2025）。<https://arxiv.org/abs/2504.08208> [作者未核對]
  發現：zero-shot prompting 只贏隨機推薦；提供歷史選課資料的 few-shot 較好。
  套用：個人化推薦不能只靠 LLM 常識，需要選課紀錄訊號。
- 台灣開源專案：成大 NCKU-Evaluator（把 urschool、nckuhub 評價注入選課系統）<https://github.com/ncchen99/NCKU-Evaluator> ；NCKUpp <https://github.com/WavJaby/NCKUpp> ；元智 yzuCourseBot <https://github.com/Doem/yzuCourseBot> ；交大 NCTUwU 選課模擬器 <https://github.com/splitline/NCTUwU> 。這些都以評價彙整、課表模擬、搶課為主，GitHub 上沒找到台大、清大、成大的 LLM 選課助理。清大電機與微軟合開的課有學生專題做選課與畢業規劃（<https://udn.com/news/story/6928/9511936），未公開程式。>
- Stanford Carta、Michigan Atlas：搜尋沒有找到相關 LLM 功能或論文 [未驗證]。依印象兩者主要提供評價、成績分布與修課路徑。

## 2. 結構化與非結構化混合檢索

- **STaRK**（Wu et al., 2024），NeurIPS 2024 Datasets & Benchmarks。<https://arxiv.org/abs/2404.13207>
  發現：半結構化知識庫（文字描述加實體關係）的檢索基準，涵蓋商品、論文、醫療；現有檢索器與 LLM 系統表現都不理想。查詢由 pipeline 合成再經人工驗證。
  套用：課程表本身就是半結構化 KB（教師、系所、時段是關係，課綱是文字），其查詢合成流程可參考來建測試集。
- **AvaTaR**（Wu et al., 2024），NeurIPS 2024。<https://arxiv.org/abs/2406.11200>
  發現：用正負例對比自動優化 agent 的工具使用 prompt，在四個檢索資料集 Hit@1 平均相對提升 14%。
  套用：parser／planner 的 prompt 可用 v3 評測集的正負例自動調整，不必手改。
- **Autofocus Retrieval (AF-Retriever)**（2025）。<https://arxiv.org/abs/2505.09246> [作者未核對]
  發現：LLM 抽出實體屬性與關係限制，配合向量搜尋與逐步放寬範圍，在 STaRK 三個子集 zero/one-shot 拿到最佳結果。
  套用：查無結果時逐步放寬條件，正好對應課程搜尋中條件過嚴回傳 0 筆的情況。
- **LLM-based Semantic Search for Conversational Queries in E-commerce**（Siddiqui et al., 2026），arXiv。<https://arxiv.org/abs/2601.16492>
  發現：用 LLM 合成資料微調兩個模型，一個把自然語言轉成結構化限制，一個做 embedding，再組合過濾；摘要未給數字。
  套用：可用 Gemma 合成大量中文查詢與 slot JSON 配對，蒸餾到 1–4B 小模型，parser 改在本地跑。
- **Building a Natural Language Interface for Product Search**（Amazon Science）。<https://cdn.amazon.science/a1/97/116ffa7f44e28bcc4bcdfccaceee/building-natural-language-interface-for-product-search.pdf> [年份與會議未驗證]
  發現：微調 LLM 從查詢抽屬性鍵值，exact match 比基線高最多 9.6%，中位延遲約 110 ms。
  套用：slot 抽取用小模型微調可壓到百毫秒級。
- **OmniSQL**（2025）。<https://arxiv.org/abs/2503.02240>
  發現：用 250 萬筆合成資料訓練 7B/14B/32B text-to-SQL，7B 在 Spider test greedy 87.9%。
  套用：自由 SQL 生成風險高；課程 schema 固定，用 slot 填入參數化 SQL 較安全，OmniSQL 只適合當進階查詢備案。
- **Let Me Speak Freely?**（Tam et al., 2024），EMNLP 2024 Industry。<https://aclanthology.org/2024.emnlp-industry.91>
  發現：強制 JSON 等格式會降低推理表現，分類型任務影響小甚至更好；key 的順序也有影響。
  套用：slot 抽取屬分類型，可放心用 constrained decoding；需要推理的欄位（例如判斷是否算通識）放在 JSON 後段或先讓模型寫理由。
- **JSONSchemaBench**（Geng et al., 2025），ICML 2025。<https://arxiv.org/abs/2501.10868>
  發現：比較 Guidance、Outlines、llama.cpp、XGrammar 等六種結構化輸出引擎的效率、覆蓋率與品質，差異很大。
  套用：Ollama 的 format 參數走 llama.cpp grammar；schema 保持扁平、少用複雜限制即可。
- 硬條件與軟條件：上述論文多半把條件當硬過濾。實務上可把明確條件（時段、學分、語言、否定）放 SQL，模糊條件（好過、甜、偏實作）轉成排序特徵或 rerank 指令 [此為推論，未找到專門比較的論文]。

## 3. 適合小算力的新檢索技術

- **BGE M3-Embedding**（Chen et al., 2024）。<https://arxiv.org/abs/2402.03216> [ACL 2024 Findings 未核對]
  發現：同一模型同時輸出 dense、sparse（詞權重）、multi-vector（ColBERT 式），支援 8192 tokens。
  套用：專案已用 bge-m3 dense，sparse 與 multi-vector 可在同一次前向計算取得，額外成本低，值得與 jieba BM25 比較。
- **Jina-ColBERT-v2**（Jha et al., 2024），MRL Workshop 2024。<https://arxiv.org/abs/2408.16672>
  發現：多語 late interaction 模型，向量維度 128 降到 64 幾乎不掉分、儲存減半。
  套用：一學期 2,800 門課、每門約 1–2k tokens，多向量索引在 CPU 上也放得下。
- **Doc2Query--**（Gospodinov et al., 2023），ECIR 2023。<https://arxiv.org/abs/2301.03266>
  發現：先過濾掉低相關的生成查詢再索引，效果最多 +16%、索引縮小 33%。
  套用：現行 LLM 摘要／關鍵字擴充應加一道過濾（用 reranker 分數刪掉與課綱不符的擴充詞）。
- **HyPE: Hypothetical Prompt Embeddings**（Vake et al., 2025），IEEE Access。<https://arxiv.org/abs/2607.29402>
  發現：索引時為每個 chunk 預先生成多個假設問題並以問題做 embedding，查詢時變成問題對問題比對；作者報告 context precision 最多 +42 點。
  套用：離線用 Gemma 為每門課生成學生口吻短問句（例如：想學 Python 的入門課），查詢時零額外延遲。
- **Contextual Retrieval**（Anthropic, 2024/09/19）。<https://www.anthropic.com/news/contextual-retrieval>
  發現：每個 chunk 前加一段由 LLM 生成的文件脈絡再做 embedding 與 BM25，top-20 失敗率由 5.7% 降到 2.9%，加 rerank 再降到 1.9%。
  套用：若課綱分欄位切 chunk（每週進度、評分方式），每個 chunk 前加課名、系所、教師、學期，作法簡單。
- **FollowIR**（Weller et al., 2025），NAACL 2025。<https://aclanthology.org/2025.naacl-long.597>
  發現：多數檢索模型把指令當關鍵字，長指令處理差；微調後的 FollowIR-7B 有明顯進步。
- **Promptriever**（Weller et al., 2025），ICLR 2025。<https://arxiv.org/abs/2409.11136>
  發現：用約 50 萬筆實例級指令資料訓練，檢索器可以像 LM 一樣用自然語言指令調整相關性。
  套用（兩者）：把查詢中的軟條件寫成指令交給 instruction-aware 模型（Qwen3-Embedding／Reranker 都支援 instruction），關鍵字欄位只留主題詞。
- **BRIGHT**（Su et al., 2025），ICLR 2025。<https://arxiv.org/abs/2407.12883>
  發現：MTEB 上 59.0 的模型在需要推理的查詢上只剩約 18 nDCG@10；先讓 LLM 寫推理再檢索可提升最多 12.2 點。
- **ReasonIR**（Shao et al., 2025）。<https://arxiv.org/abs/2504.20595> [作者姓氏未核對]
  發現：8B 檢索器以合成的推理型查詢訓練，BRIGHT 29.9（加 reranker 36.9）。
- **Rank1**（Weller et al., 2025），COLM 2025。<https://arxiv.org/abs/2502.18418>
  發現：蒸餾 R1 推理軌跡訓練小型 reranker，量化後仍保留多數效果；後續重現研究質疑推理步驟的貢獻 [二手來源]。
  套用（三者）：課程查詢多數不需要深推理；資格判斷那類才需要，而那部分交給規則較穩。8B 推理型檢索器在 16GB 卡上可跑但延遲高，優先度低。
- **RankZephyr**（Pradeep et al., 2023）。<https://arxiv.org/abs/2312.02724>
  發現：7B 開源 listwise reranker，部分設定超越 GPT-4，對初始排序有韌性。
- **FIRST**（Reddy et al., 2024），EMNLP 2024。<https://aclanthology.org/2024.emnlp-main.491>
  發現：只看第一個生成 token 的 logits 排序，免去整串生成，listwise rerank 大幅加速。
  套用：Gemma 31B 做 top-20 listwise rerank 只需一次呼叫；FIRST 思路可用在本地小模型。
- **Qwen3 Embedding / Reranker**（Zhang et al., 2025）。<https://arxiv.org/abs/2506.05176>
  發現：0.6B/4B/8B 三種大小，支援 instruction；8B embedding 在 MTEB 多語 70.58；Reranker-0.6B 已勝過先前多數開源 reranker。
  套用：Qwen3-Reranker-0.6B 可直接和現用 bge-reranker-v2-m3（568M）對比，參數量相近。
- **EmbeddingGemma**（Schechter Vera et al., 2025）。<https://arxiv.org/abs/2509.20354>
  發現：308M、2K context、100+ 語言，500M 以下 MTEB 最佳，量化與截斷維度後仍領先。
  套用：CPU 端候選模型；2K context 不夠裝整份課綱，需分欄位切。
- **jina-embeddings-v3**（Sturua et al., 2024）。<https://arxiv.org/abs/2409.10173>
  發現：570M，task LoRA，Matryoshka 可截到 32 維。授權依印象為 CC BY-NC [未驗證]，校內部署前需確認。jina-v4 未查到原文 [未驗證]。
- **Multilingual E5**（Wang et al., 2024）。<https://arxiv.org/abs/2402.05672>
  發現：small/base/large 三種大小與 large-instruct；適合當弱基線。
- 中文斷詞：Nie et al. (2000) 在 IRAL 2000 發現最長匹配詞加單字的組合最好，單用詞或 n-gram 差異不大（<https://www.microsoft.com/en-us/research/?p=151541）。CKIP> 在 SIGHAN 斷詞準確率明顯高於 jieba（AS 97.7 vs 87.1，見 <https://arxiv.org/abs/1901.05816> ，2019），但沒找到 jieba 與 CKIP 在 BM25 檢索上的直接比較；CKIP Transformers 為 GPL-3.0。
  套用：jieba 用簡體語料訓練，課名專有詞（例如：社會語言學）常被切錯。建議三組對照：jieba＋自訂課程詞典、CKIP、字元 bigram 加 unigram，查詢與文件用同一種切法。

## 4. 小模型的 agentic 檢索

- **Search-R1**（Jin et al., 2025），COLM 2025。<https://arxiv.org/abs/2503.09516>
  發現：用 RL 訓練 Qwen2.5-3B/7B 在推理中呼叫搜尋，QA 比 RAG 基線高 20–24%；作者指出只靠 prompting 讓模型呼叫搜尋常常不理想。
- **τ-bench**（Yao et al., 2024）。<https://arxiv.org/abs/2406.12045>
  發現：gpt-4o 等級的 function calling agent 成功率不到 50%，零售情境 pass^8 低於 25%，多次重跑一致性差。
- **Is Agentic RAG worth it?**（Ferrazzi et al., 2026），ACL 2026 Industry。<https://arxiv.org/abs/2601.07711>
  發現：比較 agentic 與強化版 pipeline RAG 的效果與成本，摘要結論是視情境而定 [細節未讀]。
- BFCL 上 Qwen2.5-32B 約 82.2、7B 約 79.6（轉引自 <https://arxiv.org/abs/2602.13665> 表格）[二手數字]。單步工具呼叫已可用，多步規劃仍是弱項。
  套用：Gemma 31B 做開放式 ReAct 迴圈，延遲與穩定性都難控制。較穩的作法是固定流程：一次 planner 呼叫、SQL＋檢索、零結果時最多放寬一次條件，再一次生成；評測時用 pass^k 量一致性。

## 5. 否定與排除

- **NevIR**（Weller et al., 2024），EACL 2024。<https://aclanthology.org/2024.eacl-long.139>
  發現：多數檢索模型對只差一個否定詞的文件對表現接近或低於隨機；cross-encoder 最好，其次 late interaction，bi-encoder 與 sparse 最差。
- **Reproducing NevIR**（van den Elsen et al., 2025），SIGIR 2025。<https://arxiv.org/abs/2502.13506>
  發現：listwise LLM reranker 表現最好，只有 cross-encoder 與 listwise LLM reranker 在 NevIR 與 ExcluIR 兩邊都過得去；在其中一個資料集微調對另一個幫助不穩。
- **ExcluIR**（Zhang et al., 2025），AAAI 2025。<https://arxiv.org/abs/2404.17288>
  發現：3,452 筆排除型查詢評測集，各架構都吃力，生成式檢索相對較好。
- **NS-IR**（Xu et al., 2025），ACL 2025 Findings。<https://arxiv.org/abs/2505.22299>
  發現：用一階邏輯對齊查詢與文件，改善負向限制查詢，附 NegConstraint 資料集。
  套用：不要早八、不要必修這類對應欄位的否定，一律轉成 SQL 排除，不交給 embedding；不要報告太多、不要點名這類落在課綱或評價文字中的否定，用 LLM reranker 或評價的結構化欄位處理。

## 6. 評價運用與修課資格

- **Zero-shot multilingual ABSA**（Wu et al., 2025），International Journal of Machine Learning and Cybernetics。<https://arxiv.org/abs/2412.12564>
  發現：九個 LLM zero-shot 做 aspect-based sentiment，整體不如微調專用模型；簡單 prompt 常比複雜策略好。
  套用：269k 則平均 30 字的評論很適合先人工標 500–1000 則（面向如給分、負擔、教學、點名），用 Gemma 標其餘再訓練小分類器，或直接用簡單 prompt 批次跑。
- review-aware 排序：沒找到 2023 後專門針對課程評價的論文 [未驗證]。實務作法是把每位教師×課名跨 28 學期的面向分數彙整成結構化欄位，配合 RESULT 的選課人數／限修比。
- 修課資格：搜尋沒找到 LLM degree audit 的評測研究。傳統 degree audit 是規則比對（必修、學分、年級門檻）。
  套用：用 LLM 離線把備註欄的限修文字解析成結構（年級、系所、先修課、是否限本系），存成欄位，查詢時用成績單跑規則；LLM 只負責把結果講成人話。

## 7. 評測方法

- **Large Language Models can Accurately Predict Searcher Preferences**（Thomas et al., 2024），SIGIR 2024。<https://arxiv.org/abs/2309.10621>
  發現：用真實使用者偏好挑選 LLM 與 prompt 後，LLM 標注可接近人工。
- **UMBRELA**（Upadhyay et al., 2024）。<https://arxiv.org/abs/2406.06519>
  發現：Thomas et al. 方法的開源重現，在 TREC DL 2019–2023 與系統排名高度相關；後續有換其他 LLM 的研究（<https://arxiv.org/abs/2507.09483）。>
- **LLM-based relevance assessment still can't replace human relevance assessment**（Clarke & Dietz, 2024）。<https://arxiv.org/abs/2412.17156>
  發現：LLM 判官難以區分頂尖系統，也可被針對性操縱；若系統本身用同一 LLM 打分，會有自我偏好。
- **Synthetic Test Collections for Retrieval Evaluation**（Rahmani et al., 2024），SIGIR 2024。<https://arxiv.org/abs/2405.07767>
- **Towards Understanding Bias in Synthetic Data for Evaluation**（Rahmani et al., 2025），CIKM 2025。<https://arxiv.org/abs/2506.10301>
  發現（兩者）：LLM 合成查詢與標注可用來比較系統排名，但絕對分數偏差大。
- **Beyond Benchmark Scores**（Kucia & Gawlik, 2026），CIKM 2026。<https://arxiv.org/abs/2609.14579>
  發現：大學院系資訊系統上，真實學生查詢平均 6.8 字，合成查詢 15.7 字；在合成查詢上調出的 hybrid 設定，遇到真實查詢掉分，延遲還高到 8 倍。
  套用：和政大情境幾乎一樣。v3 評測集需要一批真實短查詢（問卷、Dcard 選課文）做對照組。
- **iEvaLM**（Wang et al., 2023），EMNLP 2023。<https://arxiv.org/abs/2305.13112> ；批評見 **How Reliable is Your Simulator?**（Zhu et al., 2024）<https://arxiv.org/abs/2403.16416>
  發現：LLM 模擬使用者做多輪推薦評測，但有資料洩漏、成功率主要取決於對話歷史。
  套用：多輪對話評測用模擬使用者時，persona 不能直接含答案課號。
- 評測建議：判官模型用和檢索、生成不同家族的模型（例如：生成用 Gemma，判官用 Qwen3），以 pooling 收集各系統 top-k 再判；硬條件用 DB 直接驗證算每個 facet 的 constraint satisfaction，不交給 LLM 判。

## 8. 低階硬體效率

- **Binary and Scalar Embedding Quantization**（Hugging Face, 2024）。<https://huggingface.co/blog/embedding-quantization>
  發現：binary 粗搜加 int8 rescoring，保留約 96% 效果，記憶體與速度最多改善 32 倍。
- **Model2Vec**（MinishLab, 2024）。<https://github.com/MinishLab/model2vec>
  發現：把 sentence transformer 蒸餾成靜態詞向量，體積小 50 倍、快最多 500 倍；有從 bge-m3 蒸餾的 potion-multilingual-128M。
  套用：可當 CPU 上第一階段召回或查詢分類器。
- bge-m3 ONNX：社群匯出版本在 x86 CPU 上 fp32 比 PyTorch 快約 1.27 倍，動態 int8 讓 dense cosine 掉到 0.985（<https://huggingface.co/a-ivanovitch/bge-m3-onnx）[社群數據]。>
- 索引選擇：單學期 2,800 門課、全歷史約 10 萬列，1024 維 float32 約 400MB，FAISS Flat 精確搜尋在 CPU 上即可毫秒級完成，不需要 HNSW 或 IVF [推論]。多向量索引若每門課 500 個 token 向量、64 維 int8，單學期約 90MB。

---

## (a) 八個方法建議（依預期效益排序）

v3 評測集（graded LLM judge × DB 驗證條件、nDCG@10、CSR@10）是所有實驗的前提，另外補一批真實短查詢對照。

1. **結構優先的查詢規劃器，否定直接進 SQL**。Gemma 31B 用 JSON schema 輸出硬條件、軟條件、主題文字三段；硬條件與否定走參數化 SQL，軟條件寫成 reranker 指令。零結果時依 AF-Retriever 方式放寬一次。效益：CSR 與否定查詢最直接。成本：每查詢一次 31B 呼叫。測法：v3 各 facet CSR@10，對比現行 slot parser；否定子集單獨報告。
2. **Doc 端假設問句索引（HyPE＋Doc2Query-- 過濾）**。離線為每門課生成 10–20 個學生口吻短問句，用 reranker 分數刪掉不符者，問句進 BM25 與 dense 索引。效益：拉近短口語查詢與長課綱的落差，查詢時零延遲。成本：一次離線生成（2,800 門課 × 每學期）。測法：真實短查詢子集 nDCG@10，對比現行 LLM 摘要擴充。
3. **換 instruction-aware reranker**。Qwen3-Reranker-0.6B／4B 對比 bge-reranker-v2-m3；top-20 再試 Gemma listwise 一次呼叫。效益：軟條件與文字型否定。成本：0.6B 在 16GB 卡上 top-50 約百毫秒級 [推論]，4B 需量化。測法：v3 nDCG@10，加自建 50 對課綱否定對比題（仿 NevIR）。
4. **評價面向化**。269k 評論用小模型抽面向（給分、負擔、教學、點名、考試型態）與極性，依教師×課名跨學期彙整，加上選課人數／限修比；好過嗎、甜不甜查詢直接查欄位。效益：補上目前完全沒用的訊號。成本：離線批次，標 500 則做驗證。測法：面向抽取 F1；評價型查詢由判官打分。
5. **修課資格規則引擎**。LLM 離線把備註限修文字解析成結構欄位，查詢時以成績單跑規則（年級、系所、先修、已修過）。效益：資格題正確率，避免 LLM 幻覺。成本：離線一次，人工抽驗 100 門。測法：人工標注的資格題集，算 precision／recall。
6. **斷詞與 sparse 對照，加脈絡前綴**。jieba＋課程詞典、CKIP、字元 bigram、bge-m3 sparse 四組；課綱分欄位 chunk 前加課名、系所、教師（Contextual Retrieval）。效益：BM25 端便宜的改進。成本：重建索引而已。測法：v3 nDCG@10 與 R@50。
7. **Embedding 汰換與量化**。bge-m3（dense＋sparse＋multi-vector）、Qwen3-Embedding-0.6B、EmbeddingGemma-300M 比較，各試 int8／binary。效益：可能小幅提升並降低 CPU 延遲。成本：低。測法：v3 指標與 CPU p50／p95 延遲。
8. **有界的 agent 流程**。固定兩步：planner→檢索→（零結果或條件衝突時）放寬或追問，不用開放式 ReAct。效益：複合查詢與多輪對話。成本：最多三次 LLM 呼叫。測法：多輪模擬使用者（persona 不含答案）成功率，加 pass^k 一致性與延遲。

## (b) 值得參考的專案與 repo

- 檢索與排序：FlagOpen/FlagEmbedding、QwenLM/Qwen3-Embedding、castorini/rank_llm、terrierteam/pyterrier_doc2query、MinishLab/model2vec、ckiplab/ckip-transformers
- 半結構化檢索與 agent：snap-stanford/stark、zou-group/avatar、PeterGriffinJin/Search-R1、RUCKBReasoning/OmniSQL、guidance-ai/jsonschemabench
- 否定：zwh-sdu/ExcluIR、xgl-git/NS-IR-main
- 評測：castorini/umbrela、rahmanidashti/BiasSyntheticData、RUCAIBox/iEvaLM-CRS
- 台灣選課工具：ncchen99/NCKU-Evaluator、WavJaby/NCKUpp、Doem/yzuCourseBot、splitline/NCTUwU
