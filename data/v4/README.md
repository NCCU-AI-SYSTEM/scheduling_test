# data/v4

| 路徑 | 內容 | 進 git | 誰用 |
|---|---|---|---|
| `course_struct.db` | 執行期資料：課程、時段、限制、評分、別名、系所、面向、向量。由 `prepare_semester` 或 `build_structured` 加 `build_runtime` 產生 | 否 | 搜尋、評估 |
| `staging/<學期>/` | `prepare_semester` 的輸出與報告，通過門檻才取代上面的檔案 | 否 | `prep.release` |
| `archive/` | 上線過的舊 `course_struct.db` | 否 | 回溯 |
| `unit.json` | 選課系統的系所樹（學院、學制、代碼） | 是 | 身分解析 |
| `nccu_units.tsv` | 校方單位代碼表 | 是 | 身分解析 |
| `aliases_all.tsv` | 課名與縮寫對照的完整匯出，說明在 `docs/v4/aliases.md` | 是 | 人工檢視 |
| `structured_cache/` | LLM 補抽結果，依輸入文字快取；重建時不必再呼叫模型 | 是 | `build_structured` |
| `structured_audit/` | 結構化抽取的抽樣與覆蓋率檢查 | 是 | `verify_structured` |
| `rset/` | R 子集 176 題的題目與人工規劃標準答案 | 是 | `run_e2e` |
| `calibration/` | 評審校準的人工標註 | 是 | `calibration_kappa` |
| `eligibility/` | 修課資格的標準答案與資料 | 是 | `eligibility_bench` |
| `llm_bench/` | 小模型測試題 | 是 | `llm_bench` |
| `query_bank_forum.jsonl` | 論壇來源的問句 | 是 | `build_eval_v4` |

原始爬蟲檔 `data/1142.db` 是指到 NCCUCourse 的 symlink，約 300 MB。全人匯出檔 `data/exportStudentData.json` 有真實個資，已被 `.gitignore` 排除，不能提交。
