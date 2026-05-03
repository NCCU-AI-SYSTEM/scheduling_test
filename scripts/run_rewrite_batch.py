"""Submit, poll, and fetch a generic OpenAI batch job.

Usage:
    uv run python scripts/run_rewrite_batch.py --submit --input batches/rewrite_eval.jsonl
    uv run python scripts/run_rewrite_batch.py --status
    uv run python scripts/run_rewrite_batch.py --fetch
    uv run python scripts/run_rewrite_batch.py --merge   # write cache files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

STATE_FILE = ROOT / "batches" / "state" / "rewrite.json"
RESULT_FILE = ROOT / "batches" / "rewrite_eval.results.jsonl"
CACHE_DIR = ROOT / "data" / "processed" / "query_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _save_state(obj: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(obj, indent=2))


def _load_state() -> dict:
    if not STATE_FILE.exists():
        raise FileNotFoundError("No rewrite batch state. Run --submit first.")
    return json.loads(STATE_FILE.read_text())


def do_submit(input_path: Path) -> None:
    from openai import OpenAI
    client = OpenAI()
    print(f"[submit] uploading {input_path} ({input_path.stat().st_size/1024:.0f} KB)")
    f = client.files.create(file=input_path.open("rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "nccu-course-rewrite"},
    )
    print(f"[submit] batch_id={batch.id}  status={batch.status}")
    _save_state({"file_id": f.id, "batch_id": batch.id, "status": batch.status})


def do_status() -> dict:
    from openai import OpenAI
    client = OpenAI()
    state = _load_state()
    b = client.batches.retrieve(state["batch_id"])
    rc = b.request_counts
    print(
        f"[status] {b.id}  status={b.status}  "
        f"req={rc.completed}/{rc.total}  failed={rc.failed}"
    )
    state["status"] = b.status
    state["output_file_id"] = b.output_file_id
    _save_state(state)
    return state


def do_fetch() -> None:
    from openai import OpenAI
    client = OpenAI()
    state = _load_state()
    if state["status"] != "completed":
        state = do_status()
    if state["status"] != "completed":
        raise RuntimeError(f"Not completed: {state['status']}")
    content = client.files.content(state["output_file_id"])
    RESULT_FILE.write_bytes(content.read())
    print(f"[fetch] wrote {RESULT_FILE}  ({RESULT_FILE.stat().st_size/1024:.1f} KB)")


def do_merge() -> None:
    """Parse results and write cache files compatible with src/query_rewriters/llm.py."""
    import hashlib
    if not RESULT_FILE.exists():
        raise FileNotFoundError("Run --fetch first.")

    qmap_path = ROOT / "batches" / "rewrite_eval_queries.json"
    if not qmap_path.exists():
        raise FileNotFoundError("rewrite_eval_queries.json missing. Re-run build_rewrite_batch.py.")
    qmap: dict[str, str] = json.loads(qmap_path.read_text())

    # cache key must match src/query_rewriters/llm.py which uses DEFAULT_MODEL
    # We pretend the result came from gemma4:e4b so the cache hits on retrieval
    model = "gemma4:e4b"

    hyde_ok = q2d_ok = 0
    with RESULT_FILE.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            custom_id: str = rec["custom_id"]
            try:
                text = rec["response"]["body"]["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                continue
            method, qid = custom_id.split("|", 1)
            query = qmap.get(qid)
            if not query:
                continue
            # reproduce same cache key as llm.py
            h = hashlib.sha256(f"{method}|{model}|{query}".encode()).hexdigest()[:16]
            cache_path = CACHE_DIR / f"{method}_{h}.json"
            cache_path.write_text(json.dumps({"text": text}, ensure_ascii=False))
            if method == "hyde":
                hyde_ok += 1
            else:
                q2d_ok += 1
    print(f"[merge] hyde={hyde_ok}  q2d={q2d_ok}")
    print(f"[merge] cache dir: {CACHE_DIR}")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--submit", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--fetch", action="store_true")
    g.add_argument("--merge", action="store_true")
    g.add_argument("--wait", action="store_true")
    p.add_argument("--input", default=str(ROOT / "batches" / "rewrite_eval.jsonl"))
    args = p.parse_args()

    if args.submit:
        do_submit(Path(args.input))
    elif args.status:
        do_status()
    elif args.fetch:
        do_fetch()
    elif args.merge:
        do_merge()
    elif args.wait:
        import time
        while True:
            state = do_status()
            if state["status"] == "completed":
                do_fetch()
                break
            time.sleep(60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
