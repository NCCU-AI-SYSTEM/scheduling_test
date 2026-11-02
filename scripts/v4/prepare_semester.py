"""Prepare a semester's course_struct.db end to end, with a quality gate before it goes live.

  python -m scripts.v4.prepare_semester --semester 115-1 --src data/1151.db --llm cloud --promote
  python -m scripts.v4.prepare_semester --semester 114-2 --skip-aspects          # rebuild, check, no promote

Steps (each one is an existing command, run with V4_SEMESTER / V4_SOURCE_DB set):
  0 preflight      the raw DB has COURSE rows for the semester; syllabus fields filled; rows still
                   carrying the old scraper's "Course Name:" bug (NCCUCourse repair_course_name_sections.py)
  1 crawl          optional (--crawl): NCCUCourse main.py --course for the semester
  2 aspects        review aspects with a local LLM (scripts.v4.extract_aspects, resumes by course family)
  3 structure      scripts.v4.build_structured into data/v4/staging/<semester>/course_struct.db
  4 runtime        scripts.v4.build_runtime: dept, unit, aspect, embedding tables
  5 verify         scripts.v4.verify_structured: coverage per field
  6 gate           extraction: every field with >= 50 applicable courses reaches --min-coverage (0.90);
                   source: each syllabus field has raw text for >= --min-source-fill (0.80) of courses;
                   shrink: eval_item, weekly, restriction, session, aspect, embedding keep >= half
                   the live build's rows; a smoke search returns results
  7 diff           against the live course_struct.db: table counts, courses added / removed, coverage
  8 promote        only with --promote and a passing gate: archive the live DB, move staging in

Exit code 0 when the gate passes, 1 when it fails, 2 on a step error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from src.v4.semester import ROOT, parse, source_db, struct_db

PY = sys.executable
NCCUCOURSE = ROOT.parent.parent / "NCCUCourse" if (ROOT.parent.parent / "NCCUCourse").exists() else ROOT.parent / "NCCUCourse"
ASPECTS_DIR = ROOT / "data" / "v4" / "aspects"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], env: dict, cwd: Path = ROOT) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"step failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")


SOURCE_FIELDS = ("syllabus", "evaluation", "schedule", "teaching_approach", "ai_policy")
# tables that must not shrink by half against the live build (a scrape that lost a column)
STABLE_TABLES = ("eval_item", "weekly", "restriction", "session", "aspect", "embedding")


def preflight(src: Path, year: str, term: str) -> dict:
    if not src.exists():
        raise FileNotFoundError(f"raw database {src} not found")
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    q = lambda sql: con.execute(sql, (year, term)).fetchone()[0]  # noqa: E731
    n = q("SELECT count(DISTINCT id) FROM COURSE WHERE y = ? AND s = ?")
    fill = {f: q(f"SELECT count(DISTINCT id) FROM COURSE WHERE y = ? AND s = ? AND coalesce({f}, '') <> ''")
            for f in SOURCE_FIELDS}
    bad = q("SELECT count(*) FROM COURSE WHERE y = ? AND s = ? AND "
            "(ai_policy LIKE 'Course Name:%' OR evaluation LIKE 'Course Name:%')")
    con.close()
    return {"courses": n, "filled": fill, "course_name_bug_rows": bad}


def gate(report: dict, min_cov: float, min_n: int = 50, pre: dict | None = None, min_fill: float = 0.80,
         new_counts: dict | None = None, live_counts: dict | None = None) -> list[str]:
    """Three checks. Extraction: share of applicable courses that got a structured value.
    Source: share of courses whose raw field has content at all; without it the first check
    passes on an empty denominator (114-1 in 1142.db has no evaluation or schedule text).
    Shrink: no stable table loses half its rows against the live build."""
    fails = []
    for name, c in report["coverage"].items():
        hit, n = c["all"]
        if n >= min_n and hit / n < min_cov:
            fails.append(f"抽取 {name} {hit}/{n} = {hit / n:.1%}")
    if pre and pre["courses"]:
        for f, k in pre["filled"].items():
            if k / pre["courses"] < min_fill:
                fails.append(f"原始資料 {f} 只有 {k}/{pre['courses']} 門有內容 = {k / pre['courses']:.1%}")
    if new_counts and live_counts:
        for t in STABLE_TABLES:
            old, new = live_counts.get(t, 0), new_counts.get(t, 0)
            if old and new < old * 0.5:
                fails.append(f"{t} 從上線版本的 {old} 列降到 {new} 列")
    return fails


def smoke(db: Path) -> list[str]:
    """Three plans that need no LLM; each must return courses."""
    from src.parsers.base import ConditionResult
    from src.v4.pipeline import Config, V4Search
    from src.v4.planner import Plan
    from src.v4.struct_store import load_struct

    b = load_struct(db)
    s = V4Search(b.catalog, None, None, b.aspects, offerings=b.offerings, aliases=b.aliases, assess=b.assess)
    cfg = Config(use_dense=False, rerank=False, merge_offerings=True, named_skips_conditions=True)
    plans = {"主題：統計": Plan(need="統計"),
             "條件：週三下午": Plan(cond=ConditionResult(must={"weekday": [3], "hour_range": [13, 18]})),
             "指名：個經": Plan(named_course="個經")}
    return [f"{k} 沒有結果" for k, p in plans.items() if not s.search(p, k, 5, cfg)]


def _counts(db: Path) -> dict[str, int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    out = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in sorted(names)}
    out["_semester"] = dict(con.execute("SELECT key, value FROM meta")).get("semester")
    con.close()
    return out


def diff(new: Path, live: Path) -> str:
    if not live.exists():
        return "（沒有上線中的資料庫可比較）"
    a, b = _counts(live), _counts(new)
    lines = [f"上線中：{a.pop('_semester')}　新版：{b.pop('_semester')}", "",
             "| 表 | 上線中 | 新版 | 差 |", "|---|---|---|---|"]
    for t in sorted(set(a) | set(b)):
        lines.append(f"| {t} | {a.get(t, 0)} | {b.get(t, 0)} | {b.get(t, 0) - a.get(t, 0):+d} |")
    ids = lambda p: {r[0] for r in sqlite3.connect(f"file:{p}?mode=ro", uri=True).execute(  # noqa: E731
        "SELECT course_id FROM course WHERE cancelled = 0")}
    old_ids, new_ids = ids(live), ids(new)
    lines += ["", f"開課（不含停開）：上線中 {len(old_ids)}，新版 {len(new_ids)}，"
                  f"新版才有 {len(new_ids - old_ids)}，新版沒有 {len(old_ids - new_ids)}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--semester", required=True)
    ap.add_argument("--src", type=Path, default=None, help="raw scrape (default V4_SOURCE_DB or data/1142.db)")
    ap.add_argument("--llm", choices=["off", "cloud", "local"], default="off",
                    help="LLM for structure cache misses: cloud gemma4:31b, local gemma4:12b")
    ap.add_argument("--crawl", action="store_true", help="run NCCUCourse main.py --course first")
    ap.add_argument("--skip-aspects", action="store_true")
    ap.add_argument("--aspects-model", default="gemma4:12b")
    ap.add_argument("--encoder", default="e5-large-instruct")
    ap.add_argument("--device", default="cuda" if shutil.which("nvidia-smi") else "cpu")
    ap.add_argument("--min-coverage", type=float, default=0.90, help="extraction coverage per field")
    ap.add_argument("--min-source-fill", type=float, default=0.80, help="share of courses with raw syllabus text")
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()

    sem = parse(a.semester)
    src = (a.src or source_db()).resolve()
    stage = ROOT / "data" / "v4" / "staging" / sem.label
    stage.mkdir(parents=True, exist_ok=True)
    out = stage / "course_struct.db"
    env = {**os.environ, "V4_SEMESTER": sem.label, "V4_SOURCE_DB": str(src), "PYTHONPATH": str(ROOT)}
    aspects_file = ASPECTS_DIR / f"{a.aspects_model.replace(':', '_')}.jsonl"
    summary: dict = {"semester": sem.label, "source": str(src), "staging": str(out)}
    try:
        if a.crawl:
            run(["uv", "run", "python", "main.py", "--course", "--db", src],
                {**env, "YEAR": sem.year, "SEM": sem.term}, cwd=NCCUCOURSE)
        pre = preflight(src, sem.year, sem.term)
        summary["preflight"] = pre
        log(f"preflight {pre}")
        if pre["courses"] == 0:
            raise ValueError(f"{src} has no COURSE rows for {sem.label}")
        if pre["course_name_bug_rows"]:
            log(f"warning: {pre['course_name_bug_rows']} rows carry the Course Name: bug; "
                "run NCCUCourse repair_course_name_sections.py first")
        if not a.skip_aspects:
            run([PY, "-m", "scripts.v4.extract_aspects", "--model", a.aspects_model], env)
        run([PY, "-m", "scripts.v4.build_structured", "--semester", sem.label, "--src", src, "--out", out,
             "--llm", a.llm], env)
        run([PY, "-m", "scripts.v4.build_runtime", "--db", out, "--src", src, "--encoder", a.encoder,
             "--device", a.device] + (["--aspects", aspects_file] if aspects_file.exists() else []), env)
        run([PY, "-m", "scripts.v4.verify_structured", "--db", out, "--src", src, "--report-dir", stage], env)
    except Exception as e:  # noqa: BLE001 - report which step broke, keep the live DB untouched
        log(f"ERROR {e}")
        (stage / "prepare_summary.json").write_text(json.dumps(summary | {"error": str(e)}, ensure_ascii=False, indent=1))
        return 2

    report = json.loads((stage / "report.json").read_text(encoding="utf-8"))
    live = struct_db()
    fails = gate(report, a.min_coverage, pre=pre, min_fill=a.min_source_fill, new_counts=_counts(out),
                 live_counts=_counts(live) if live.exists() else None) + smoke(out)
    summary |= {"gate_failures": fails, "passed": not fails}
    text = [f"# {sem.label} 資料預備結果", "", f"來源：`{src}`　輸出：`{out}`", "",
            f"前置檢查：{pre}", "", "品質門檻：" + ("通過" if not fails else "未通過"), ""]
    text += [f"- {f}" for f in fails] + ["", "## 與上線中版本的差異", "", diff(out, live)]
    (stage / "prepare_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    log(f"report {stage / 'prepare_report.md'}")
    for f in fails:
        log(f"gate: {f}")
    if fails:
        log("gate failed; live database left unchanged")
    elif a.promote:
        archive = ROOT / "data" / "v4" / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        if live.exists():
            old = _counts(live).get("_semester", "unknown")
            shutil.copy2(live, archive / f"course_struct.{old}.{time.strftime('%Y%m%d-%H%M')}.db")
        shutil.copy2(out, live.with_suffix(".db.new"))
        os.replace(live.with_suffix(".db.new"), live)   # atomic on the same filesystem
        summary["promoted"] = str(live)
        log(f"promoted to {live}")
    (stage / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
