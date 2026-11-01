"""Benchmark small LLMs on the two jobs the course-search pipeline gives an LLM.

  parse  query -> {need, must, must_not}; scored by EXECUTION against the course DB:
           gold_kept  the gold course still satisfies the parsed conditions
           pool_j     Jaccard(courses passing parsed conds, courses passing oracle conds)
                      (cond rows only; oracle conditions came from the gold course)
           has_cond   share of synth topic/colloquial queries that got any condition
  tool   native tool calling on 30 scenarios: right tool, right arguments, no
         forbidden arguments (e.g. course_lang for "想學日文")

Every model sees the same prompt, temperature 0, think off, one request at a time
(single-user latency). Results go to results/v4/llm_bench/<model>.<task>.jsonl.

  python -m scripts.v4.llm_bench --model qwen3.5:4b --task parse,tool
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.doc_builders import _course_metadata
from src.eval.conditions import satisfies
from src.llm.ollama_http import chat
from src.loader import load_courses
from src.parsers.base import ConditionResult
from src.parsers.ollama_parser import sanitize
from src.parsers.prompt_v4 import PARSE_SCHEMA, render

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "v4" / "llm_bench"
OUT = ROOT / "results" / "v4" / "llm_bench"


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", model)


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


# ── parse ────────────────────────────────────────────────────────────────────

def run_parse(model: str, workers: int, limit: int) -> list[dict]:
    meta = {c.course_id: _course_metadata(c) for c in load_courses()}
    rows = [json.loads(x) for x in (BENCH / "parse_set.jsonl").open(encoding="utf-8")]
    if limit:
        rows = rows[:limit]

    def one(r: dict) -> dict:
        res = chat(model, [{"role": "user", "content": render(r["query"])}], fmt=PARSE_SCHEMA)
        d = _parse_json(res.content)
        out = {"qid": r["qid"], "src": r["src"], "stratum": r["stratum"], "query": r["query"],
               "raw": res.content[:600], "error": res.error, "seconds": round(res.seconds, 3),
               "out_tokens": res.output_tokens, "tok_s": round(res.tok_per_s, 1),
               "json_ok": d is not None}
        cond = ConditionResult(must=sanitize((d or {}).get("must")),
                               must_not=sanitize((d or {}).get("must_not")))
        out["need"] = (d or {}).get("need", "")
        out["pred"] = {"must": cond.must, "must_not": cond.must_not}
        out["has_cond"] = bool(cond.must or cond.must_not)
        out["gold_kept"] = satisfies(meta[r["gold"]], cond)
        if r["src"] == "cond":
            oracle = ConditionResult(must=r["must"], must_not=r["must_not"])
            pred_pool = {c for c, m in meta.items() if satisfies(m, cond)}
            gold_pool = {c for c, m in meta.items() if satisfies(m, oracle)}
            union = pred_pool | gold_pool
            out["pool_j"] = len(pred_pool & gold_pool) / len(union) if union else 1.0
            out["pool_pred"], out["pool_gold"] = len(pred_pool), len(gold_pool)
        return out

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def summarize_parse(rows: list[dict]) -> dict:
    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 4) if xs else None

    cond = [r for r in rows if r["src"] == "cond"]
    topic = [r for r in rows if r["src"] == "synth" and r["stratum"] in ("topic", "colloquial")]
    ok = [r for r in rows if not r["error"]]
    secs = sorted(r["seconds"] for r in ok)
    return {
        "n": len(rows), "errors": len(rows) - len(ok),
        "json_ok": mean(r["json_ok"] for r in rows),
        "gold_kept_all": mean(r["gold_kept"] for r in rows),
        "gold_kept_cond": mean(r["gold_kept"] for r in cond),
        "pool_jaccard_cond": mean(r["pool_j"] for r in cond),
        "exact_pool_cond": mean(r["pool_j"] >= 0.999 for r in cond),
        "neg_pool_jaccard": mean(r["pool_j"] for r in cond if r["stratum"].endswith("-neg")),
        "complex_pool_jaccard": mean(r["pool_j"] for r in cond if r["stratum"].startswith("complex")),
        "topic_has_cond": mean(r["has_cond"] for r in topic),
        "topic_gold_kept": mean(r["gold_kept"] for r in topic),
        "p50_s": secs[len(secs) // 2] if secs else None,
        "p95_s": secs[int(len(secs) * 0.95)] if secs else None,
        "tok_s": mean(r["tok_s"] for r in ok if r["tok_s"]),
    }


# ── tool ─────────────────────────────────────────────────────────────────────

def _match(exp, got) -> bool:
    if isinstance(exp, list):
        try:
            return {int(x) for x in exp} == {int(x) for x in (got if isinstance(got, list) else [got])}
        except (TypeError, ValueError):
            return False
    if isinstance(exp, (int, float)):
        try:
            return float(got) == float(exp)
        except (TypeError, ValueError):
            return False
    g = str(got or "").strip().lower()
    e = str(exp).strip().lower()
    return bool(g) and (e in g or g in e)


STRICT_SUFFIX = ("\n只要學生提到任何主題或條件，就直接呼叫工具搜尋，不要先反問學生偏好。"
                 "學生說「第一門」「那門課」時，從前面的對話找出對應的 course_id。")


def run_tool(model: str, workers: int, limit: int, variant: str = "base") -> list[dict]:
    spec = json.loads((BENCH / "tool_scenarios.json").read_text(encoding="utf-8"))
    if variant == "strict":
        spec["system"] += STRICT_SUFFIX
    scenarios = spec["scenarios"][:limit] if limit else spec["scenarios"]

    def one(s: dict) -> dict:
        msgs = [{"role": "system", "content": spec["system"]}]
        if s.get("with_prior"):
            msgs += [{"role": "user", "content": "幫我找一些課"},
                     {"role": "assistant", "content": spec["prior_results"]}]
        msgs.append({"role": "user", "content": s["user"]})
        res = chat(model, msgs, tools=spec["tools"])
        calls = res.tool_calls
        first = calls[0]["function"] if calls else None
        args = (first or {}).get("arguments") or {}
        if isinstance(args, str):
            args = _parse_json(args) or {}
        exp_tool = s["tool"]
        tool_ok = (first is None) if exp_tool is None else (first is not None and first.get("name") == exp_tool)
        exp_args = s.get("args") or {}
        matched = sum(_match(v, args.get(k)) for k, v in exp_args.items())
        forbid_hit = [k for k in s.get("forbid", []) if args.get(k) not in (None, "", [])]
        arg_score = matched / len(exp_args) if exp_args else 1.0
        return {"id": s["id"], "kind": s["kind"], "error": res.error, "seconds": round(res.seconds, 3),
                "tool": first.get("name") if first else None, "args": args, "n_calls": len(calls),
                "text": res.content[:300], "tool_ok": tool_ok, "arg_score": round(arg_score, 3),
                "forbid_hit": forbid_hit,
                "pass": bool(tool_ok and arg_score == 1.0 and not forbid_hit)}

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, scenarios))


def summarize_tool(rows: list[dict]) -> dict:
    n = len(rows)
    called = [r for r in rows if r["tool"]]
    secs = sorted(r["seconds"] for r in rows if not r["error"])
    return {
        "n": n, "errors": sum(bool(r["error"]) for r in rows),
        "pass": round(sum(r["pass"] for r in rows) / n, 4),
        "tool_acc": round(sum(r["tool_ok"] for r in rows) / n, 4),
        "arg_score": round(sum(r["arg_score"] for r in rows if r["tool_ok"]) /
                           max(1, sum(r["tool_ok"] for r in rows)), 4),
        "forbid_rate": round(sum(bool(r["forbid_hit"]) for r in rows) / n, 4),
        "call_rate": round(len(called) / n, 4),
        "p50_s": secs[len(secs) // 2] if secs else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", default="parse,tool")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tool-variant", default="base", choices=["base", "strict"])
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.jsonl"
    for task in a.task.split(","):
        t0 = time.time()
        rows = (run_parse(a.model, a.workers, a.limit) if task == "parse"
                else run_tool(a.model, a.workers, a.limit, a.tool_variant))
        tag = task if task == "parse" or a.tool_variant == "base" else f"{task}-{a.tool_variant}"
        with (OUT / f"{_slug(a.model)}.{tag}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize_parse(rows) if task == "parse" else summarize_tool(rows)
        s |= {"model": a.model, "task": tag, "wall_s": round(time.time() - t0, 1), "limit": a.limit}
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
