# scheduling_test

政大課程檢索的研究與評估。結果移植到 [CourseLangChain](https://github.com/NCCU-AI-SYSTEM/CourseLangChain) 的 `retrieve_tool`；這個 repo 本身不上線。

## 從哪裡開始看

| 想知道 | 看 |
|---|---|
| v4 的結果與建議 | `docs/v4/REPORT.md` §1、§7 |
| 怎麼移植到後端 | `docs/v4/PORTING.md` |
| 每學期資料怎麼產生、為什麼用 PostgreSQL | `docs/v4/PG_MIGRATION.md`、`docs/v4/PIPELINE.md` |
| 每個決定當時的理由 | `docs/v4/LAB_NOTEBOOK.md` |

## 目錄

```
src/v4/          v4 搜尋：規劃器、條件、排序、身分、結果表
src/v4/structured/  從爬蟲資料抽結構化欄位
src/             v4 共用元件（retrievers、rerankers、llm、loader、eval）與基準線（prod_replica）
scripts/v4/      每學期資料與評估的入口；analysis/ 一次性分析，runs/ 實驗整批指令（見 README）
data/v4/         標準答案、快取與 course_struct.db（見 README）
results/v4/      端到端結果
docs/v4/         報告與研究文件
tests/           pytest
archive/         v1–v3 舊實驗（見 README，不能直接執行）
```

## 常用指令

```bash
uv sync
uv run python -m pytest -q
uv run python -m scripts.v4.prepare_semester --semester 114-2 --skip-aspects   # 建資料並檢查門檻
uv run python -m scripts.v4.demo_search --deploy cpu "週三下午的英文通識"        # 試搜尋
uv run python -m scripts.v4.run_e2e score                                     # 重算評分
```

實驗在 hopelab（RTX 5080）上跑，程式用 rsync 同步，背景工作用 `scripts/v4/bg.sh`。

`worktree-v5-research` 分支是下一階段的研究，尚未合併。
