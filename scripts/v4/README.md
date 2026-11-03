# scripts/v4

執行方式一律是 `python -m scripts.v4.<名稱>`，在 repo 根目錄跑。2026/11/03 分組；`docs/v4/LAB_NOTEBOOK.md` 寫的是當時的路徑，舊路徑 `scripts/v4/x.py` 對應到下面的新位置。

## 這一層：會一直用到的

| 用途 | 腳本 |
|---|---|
| 每學期資料 | `prepare_semester`（總入口，CourseDataPrep 的 `prep.release` 會呼叫）、`build_structured`、`build_runtime`、`verify_structured`、`extract_aspects`、`export_aliases`、`check_source_db` |
| 評估 | `run_e2e`（plan / run / judge / score）、`compare_runs`、`score_planner`、`build_eval_v4`、`rset_gold` |
| 試用與 PostgreSQL | `demo_search`、`pg_setup` |
| 實驗機 | `bg.sh`（背景執行）、`ops.py`、`status.py` |

## `analysis/`：做過一次的分析與量測

重現報告數字時才用，用 `python -m scripts.v4.analysis.<名稱>` 執行。

- 資料分析：`analyze_data`、`analyze_duplicates`、`format_coverage`、`offering_report`、`peek_structured`、`check_struct_catalog`、`check_synth_leakage`
- 評分與評量解析：`assessment_check`、`bilingual_grading_check`、`debug_grading`、`sample_assess_flips`
- 評鑑面向：`audit_aspects`、`compare_aspects`、`validate_aspects`
- 小模型與資格判斷：`build_llm_bench`、`llm_bench`、`rescore_llm_bench`、`show_bench`、`eligibility_bench`、`eligibility_gold`
- 檢索元件：`retrieval_bench`、`show_retrieval`、`gen_hype`、`calibration_kappa`
- CPU 與 PostgreSQL：`cpu_bench`、`cpu_e2e_bench`、`pg_condition_parity`、`pg_rank_parity`、`pg_dense_debug`
- 範例：`explain_examples`

## `runs/`：實驗當時的整批指令

`phase*.sh` 是 REPORT 第 6 節各階段的完整指令，其他是重建或重跑用的。放在 hopelab 上用 `bg.sh` 跑。
