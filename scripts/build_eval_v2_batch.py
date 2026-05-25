"""生成新的 eval set（eval_v2），結合：
1. dcard_real_queries.jsonl — 真實使用者 query（爬蟲）
2. query_taxonomy_v1.md — 完整 query 分類體系
3. 課程 objective + meta — 用 GPT-4o-mini batch 生成多樣化合成 query

目標：產出 1000 筆更真實、更多樣的 eval set
  - 涵蓋所有 taxonomy 意圖 × 風格組合
  - 包含時段/語言/學分等硬約束型 query
  - 包含否定、模糊、複合句型
  - 把真實 dcard query 直接加進來

Output: data/raw/eval_v2.jsonl
格式: {"qid": "v2_001", "query": "...", "gold": ["courseId"], "source": "synth_v2|dcard", "intent": "topic|skill|teacher|vague|constraint|compound", "style": "...", "has_negation": bool, "has_constraint": bool}

Usage:
    # Step 1: 生成 batch
    uv run python scripts/build_eval_v2_batch.py --n 800 --out batches/eval_v2.jsonl

    # Step 2: 跑 batch（OpenAI）
    uv run python scripts/run_rewrite_batch.py --input batches/eval_v2.jsonl --purpose eval_v2

    # Step 3: merge + add dcard real queries
    uv run python scripts/build_eval_v2_batch.py --merge --dcard data/raw/dcard_real_queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "1142.db"
META_DB = ROOT / "data" / "processed" / "course_meta.db"
DCARD_QUERIES = ROOT / "data" / "raw" / "dcard_real_queries.jsonl"
OUT_BATCH = ROOT / "batches" / "eval_v2.jsonl"
OUT_FINAL = ROOT / "data" / "raw" / "eval_v2.jsonl"

# ── Taxonomy prompt templates ─────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一位政治大學學生，正在使用課程搜尋系統找課。
根據給定的課程資訊，用「學生自然說話的方式」生成 {n_queries} 個搜尋 query。

規則：
1. 絕對不能直接抄課名或 objective 原文
2. 每個 query 要屬於不同的類型（下面會指定）
3. query 長度：5-35 字
4. 使用繁體中文，用政大學生日常口語

輸出格式（JSON array）：
[
  {{"query": "...", "intent": "topic|skill|teacher|vague|constraint|compound", "style": "正式|口語|疑問句|陳述句|簡短|長句", "has_negation": false, "has_constraint": false}},
  ...
]
"""

# 針對不同 intent 的 sub-prompt
INTENT_GUIDES = {
    "topic": "【主題型】學生知道想學什麼主題，用口語描述，不直接說課名",
    "skill": "【技能型】想學特定技能/工具，描述技能而非課名",
    "vague": "【模糊型】描述感受或目的，不確定具體想學什麼（如「好過的通識」「有趣的課」）",
    "constraint_time": "【時段約束型】含時段限制（週X、早八、下午、晚上），可以有否定（不要早八）",
    "constraint_lang": "【語言約束型】含語言限制（英文授課、中文授課）",
    "constraint_credit": "【學分/必選修型】含學分數或必修/選修/通識類型限制",
    "negation": "【排除型】明確說「不要/不想/不需要」某些條件",
    "compound": "【複合型】同時有 2-3 個不同維度的限制（主題+時段+語言 等）",
}


def load_courses_with_meta() -> list[dict]:
    """Load courses with LLM meta."""
    conn = sqlite3.connect(DB_PATH)
    # 先查 schema 確認欄位名稱
    cols = [r[1] for r in conn.execute("PRAGMA table_info(COURSE)").fetchall()]
    # 欄位名稱映射（id = courseId）
    id_col = "id" if "id" in cols else "courseId"
    name_col = "name" if "name" in cols else "cname"
    obj_col = "objective" if "objective" in cols else "syllabus"
    unit_col = "unit" if "unit" in cols else "dp1"

    rows = conn.execute(f"""
        SELECT DISTINCT c.{id_col}, c.{name_col}, c.{obj_col}, c.teacher,
                        c.{unit_col}, c.lang, c.kind, c.smtQty
        FROM COURSE c
        WHERE c.{obj_col} IS NOT NULL AND c.{obj_col} != ''
    """).fetchall()
    conn.close()

    meta = {}
    if META_DB.exists():
        mconn = sqlite3.connect(META_DB)
        for cid, summary, kw in mconn.execute(
            "SELECT course_id, summary_100, keywords_json FROM course_meta_v1"
        ).fetchall():
            meta[cid] = {"summary": summary or "", "keywords": json.loads(kw) if kw else []}
        mconn.close()

    courses = []
    for cid, cname, obj, teacher, unit, lang, kind, point in rows:
        m = meta.get(cid, {})
        courses.append({
            "courseId": cid,
            "name": cname,
            "objective": (obj or "")[:400],
            "teacher": teacher,
            "unit": unit,
            "lang": lang,
            "kind": kind,
            "point": point,
            "summary": m.get("summary", ""),
            "keywords": m.get("keywords", []),
        })
    return courses


def build_batch(n_synth: int, rng_seed: int = 42) -> None:
    """Build OpenAI batch JSONL for eval v2 generation."""
    courses = load_courses_with_meta()
    rng = random.Random(rng_seed)
    rng.shuffle(courses)

    # Intent distribution（近似 taxonomy 比例）
    intent_dist = [
        ("topic", 0.30),
        ("skill", 0.12),
        ("vague", 0.13),
        ("constraint_time", 0.12),
        ("constraint_lang", 0.08),
        ("constraint_credit", 0.08),
        ("negation", 0.08),
        ("compound", 0.09),
    ]

    requests = []
    counter = 0
    for intent, ratio in intent_dist:
        n_this = round(n_synth * ratio)
        guide = INTENT_GUIDES[intent]
        n_per_course = max(1, min(3, n_this // min(len(courses), n_this)))

        batch_courses = rng.sample(courses, min(len(courses), n_this))
        for c in batch_courses[:n_this]:
            kw_str = "、".join(c["keywords"][:5]) if c["keywords"] else "無"
            user_msg = (
                f"課程：{c['name']}（{c['unit']}，{c['lang']}，{c['point']}學分，{c['kind']}）\n"
                f"教師：{c['teacher']}\n"
                f"摘要：{c['summary'] or c['objective'][:150]}\n"
                f"關鍵字：{kw_str}\n\n"
                f"任務：{guide}\n"
                f"生成 {n_per_course} 個不同風格的 query，JSON array 格式輸出。"
            )
            req = {
                "custom_id": f"evalv2_{intent}_{counter:05d}_{c['courseId']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT.format(n_queries=n_per_course)},
                        {"role": "user", "content": user_msg},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.9,
                    "max_tokens": 400,
                },
            }
            requests.append((req, c["courseId"]))
            counter += 1

    OUT_BATCH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_BATCH.open("w", encoding="utf-8") as f:
        for req, _ in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    # Save courseId mapping
    mapping = {req["custom_id"]: cid for req, cid in requests}
    (OUT_BATCH.parent / "eval_v2_id_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2)
    )

    print(f"[build] {len(requests)} batch requests → {OUT_BATCH}")
    print(f"[build] id map → {OUT_BATCH.parent / 'eval_v2_id_map.json'}")


def merge_results(dcard_path: Path | None = None) -> None:
    """Merge batch results + dcard real queries into eval_v2.jsonl."""
    results_file = OUT_BATCH.with_suffix(".results.jsonl")
    if not results_file.exists():
        print(f"[merge] results file not found: {results_file}")
        return

    id_map_file = OUT_BATCH.parent / "eval_v2_id_map.json"
    id_map = json.loads(id_map_file.read_text()) if id_map_file.exists() else {}

    out = []
    qid_counter = 1
    errors = 0

    with results_file.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            custom_id = row.get("custom_id", "")
            course_id = id_map.get(custom_id, "unknown")
            intent = custom_id.split("_")[1] if "_" in custom_id else "unknown"

            body = row.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            if not choices:
                errors += 1
                continue

            content = choices[0].get("message", {}).get("content", "")
            try:
                parsed = json.loads(content)
                # handle both {"queries": [...]} and [...] and single object formats
                if isinstance(parsed, list):
                    items = parsed
                elif isinstance(parsed, dict) and "query" in parsed:
                    items = [parsed]  # single query object
                else:
                    items = parsed.get("queries", parsed.get("items", [parsed]))
            except Exception:
                errors += 1
                continue

            for item in items:
                q = item.get("query", "").strip()
                if not q or len(q) < 5:
                    continue
                out.append({
                    "qid": f"v2_{qid_counter:05d}",
                    "query": q,
                    "gold": [course_id],
                    "source": "synth_v2",
                    "intent": intent,
                    "style": item.get("style", ""),
                    "has_negation": item.get("has_negation", False),
                    "has_constraint": item.get("has_constraint", False),
                })
                qid_counter += 1

    # Add dcard real queries
    if dcard_path and dcard_path.exists():
        with dcard_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                q = row.get("query", "").strip()
                if not q or len(q) < 5:
                    continue
                out.append({
                    "qid": f"v2_{qid_counter:05d}",
                    "query": q,
                    "gold": [],  # no gold label for real queries
                    "source": row.get("source", "dcard"),
                    "intent": "real",
                    "style": "口語",
                    "has_negation": any(k in q for k in ["不要", "不想", "不需要", "不用"]),
                    "has_constraint": any(k in q for k in ["週", "英文", "中文", "學分", "必修", "通識"]),
                    "url": row.get("url", ""),
                })
                qid_counter += 1

    OUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FINAL.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    intent_counts = Counter(r["intent"] for r in out)
    has_neg = sum(1 for r in out if r.get("has_negation"))
    has_con = sum(1 for r in out if r.get("has_constraint"))
    real = sum(1 for r in out if r["source"] != "synth_v2")

    print(f"\n[merge] total={len(out)}, errors={errors}, real_dcard={real}")
    print(f"  has_negation={has_neg} ({has_neg/len(out)*100:.1f}%)")
    print(f"  has_constraint={has_con} ({has_con/len(out)*100:.1f}%)")
    print(f"  intent dist: {dict(intent_counts.most_common())}")
    print(f"[saved] → {OUT_FINAL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="Number of synth queries to generate")
    ap.add_argument("--merge", action="store_true", help="Merge batch results into eval_v2.jsonl")
    ap.add_argument("--dcard", default=str(DCARD_QUERIES), help="Path to dcard real queries jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.merge:
        dcard_path = Path(args.dcard) if args.dcard else None
        merge_results(dcard_path)
    else:
        build_batch(args.n, rng_seed=args.seed)


if __name__ == "__main__":
    main()
