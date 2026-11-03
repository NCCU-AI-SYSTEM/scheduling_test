# archive：v1–v3 的舊實驗

2026/11/03 從 repo 各處搬來。v4、v5 都用不到這些檔案（依 import、`python -m` 與文字引用逐一檢查過），留著只為了查歷史結果。

**這裡的程式不能直接執行。** 檔案搬離原位後，import 路徑都失效了。要重跑請切回搬移前的版本：

```bash
git worktree add ../st-legacy 2656028
```

| 目錄 | 內容 |
|---|---|
| `docs/root/` | 原本在 repo 根目錄的 PLAN、PROGRESS、EXPERIMENT 文件與 5 張指標圖（Recall@10 階段） |
| `docs/` | RESEARCH_PLAN_v2、查詢分類 v1、舊的 schema、parser 與過濾實驗的圖、評審校準頁 |
| `scripts/` | v1–v3 的評估、改寫、meta、批次、畫圖腳本，以及 Dcard 爬蟲 |
| `src/` | 舊 parser（rule、regex、few-shot、Qwen）、PostgreSQL 過濾、批次工具、soft constraints |
| `batches/` | OpenAI 批次的狀態與 id 對照（`*.jsonl` 不進 git） |
| `data/dcard選課排幾txt/` | 2019–2026 Dcard 選課心得原文，舊評估集的素材 |

v4 的基準線還在用的舊程式沒有搬：`src/prod_replica.py`、`src/pipeline_v3.py`、`src/query_rewriters/`、`src/filters/`、`scripts/build_eval_v3.py`、`scripts/run_experiment.py`、`scripts/rescore_v3.py`。
