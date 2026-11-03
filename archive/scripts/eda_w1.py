"""W1 EDA script: load 1142, dedupe, dump processed parquet + summary stats.

Usage:
    uv run python scripts/eda_w1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.loader import courses_to_records, load_courses

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "1142.db"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"[load] {DB}")
    courses = load_courses(DB, year="114", semester="2")
    print(f"[load] distinct courseId = {len(courses)}")

    records = courses_to_records(courses)
    df = pd.DataFrame(records)

    # persist
    pq_path = OUT_DIR / "courses_1142.parquet"
    df.to_parquet(pq_path, index=False)
    print(f"[write] {pq_path}  ({pq_path.stat().st_size/1024:.1f} KB)")

    # summary stats
    stats: dict[str, object] = {
        "n_courses": int(len(df)),
        "n_cross_listed_avg": float(df["n_cross_listed"].mean()),
        "n_cross_listed_max": int(df["n_cross_listed"].max()),
        "missing": {
            col: int((df[col].astype(str).str.len() == 0).sum())
            for col in [
                "objective",
                "syllabus",
                "schedule",
                "evaluation",
                "textbook",
                "teaching_approach",
                "ai_policy",
                "classroom",
            ]
        },
        "lang_dist": df["lang"].value_counts().head(10).to_dict(),
        "kind_dist": df["kind"].value_counts().to_dict(),
        "session_count_dist": df["sessions"].apply(len).value_counts().to_dict(),
        "objective_len": {
            "p50": int(df["objective_len"].median()),
            "p90": int(df["objective_len"].quantile(0.9)),
            "max": int(df["objective_len"].max()),
            "zero": int((df["objective_len"] == 0).sum()),
        },
        "syllabus_len": {
            "p50": int(df["syllabus_len"].median()),
            "p90": int(df["syllabus_len"].quantile(0.9)),
            "max": int(df["syllabus_len"].max()),
            "zero": int((df["syllabus_len"] == 0).sum()),
        },
        "weekly_topics_dist": {
            "have_any": int((df["n_weekly_topics"] > 0).sum()),
            "p50": int(df["n_weekly_topics"].median()),
            "max": int(df["n_weekly_topics"].max()),
        },
    }

    stats_path = OUT_DIR / "eda_summary.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[write] {stats_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
