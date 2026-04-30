"""Submit / poll / fetch results for OpenAI and Anthropic batch APIs.

Requires environment variables:
  OPENAI_API_KEY
  ANTHROPIC_API_KEY

Usage:
    # one-shot end-to-end (submit -> wait -> download to results/)
    uv run python -m src.llm.eval_batch_run --provider openai --submit
    uv run python -m src.llm.eval_batch_run --provider openai --status
    uv run python -m src.llm.eval_batch_run --provider openai --fetch

    uv run python -m src.llm.eval_batch_run --provider anthropic --submit
    ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "batches"
STATE_DIR = BATCH_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_INPUT = BATCH_DIR / "openai_eval.jsonl"
ANTHROPIC_INPUT = BATCH_DIR / "anthropic_eval.jsonl"
OPENAI_RESULT = BATCH_DIR / "openai_eval.results.jsonl"
ANTHROPIC_RESULT = BATCH_DIR / "anthropic_eval.results.jsonl"


def _state_path(provider: str) -> Path:
    return STATE_DIR / f"{provider}.json"


def _save_state(provider: str, obj: dict) -> None:
    _state_path(provider).write_text(json.dumps(obj, indent=2))


def _load_state(provider: str) -> dict:
    p = _state_path(provider)
    if not p.exists():
        raise FileNotFoundError(f"No batch state for {provider}; run --submit first")
    return json.loads(p.read_text())


# --- OpenAI -------------------------------------------------------------------

def openai_submit() -> dict:
    from openai import OpenAI

    client = OpenAI()
    print(f"[openai] uploading {OPENAI_INPUT}")
    f = client.files.create(file=open(OPENAI_INPUT, "rb"), purpose="batch")
    print(f"[openai] file_id={f.id}")
    batch = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "nccu-course-eval", "phase": "synth"},
    )
    print(f"[openai] batch_id={batch.id}  status={batch.status}")
    state = {"file_id": f.id, "batch_id": batch.id, "status": batch.status}
    _save_state("openai", state)
    return state


def openai_status() -> dict:
    from openai import OpenAI

    client = OpenAI()
    state = _load_state("openai")
    batch = client.batches.retrieve(state["batch_id"])
    print(
        f"[openai] {batch.id}  status={batch.status}  "
        f"req={batch.request_counts.completed}/{batch.request_counts.total}  "
        f"failed={batch.request_counts.failed}"
    )
    state["status"] = batch.status
    state["output_file_id"] = batch.output_file_id
    state["error_file_id"] = batch.error_file_id
    _save_state("openai", state)
    return state


def openai_fetch() -> Path:
    from openai import OpenAI

    client = OpenAI()
    state = _load_state("openai")
    if not state.get("output_file_id"):
        state = openai_status()
    if state["status"] != "completed":
        raise RuntimeError(f"batch not complete: {state['status']}")
    content = client.files.content(state["output_file_id"])
    OPENAI_RESULT.write_bytes(content.read())
    print(f"[openai] wrote {OPENAI_RESULT}  ({OPENAI_RESULT.stat().st_size/1024:.1f} KB)")
    return OPENAI_RESULT


# --- Anthropic ----------------------------------------------------------------

def anthropic_submit() -> dict:
    import anthropic

    client = anthropic.Anthropic()
    requests = [json.loads(line) for line in ANTHROPIC_INPUT.read_text().splitlines() if line]
    print(f"[anthropic] submitting {len(requests)} requests")
    batch = client.messages.batches.create(requests=requests)
    print(f"[anthropic] batch_id={batch.id}  status={batch.processing_status}")
    state = {"batch_id": batch.id, "status": batch.processing_status}
    _save_state("anthropic", state)
    return state


def anthropic_status() -> dict:
    import anthropic

    client = anthropic.Anthropic()
    state = _load_state("anthropic")
    batch = client.messages.batches.retrieve(state["batch_id"])
    rc = batch.request_counts
    print(
        f"[anthropic] {batch.id}  status={batch.processing_status}  "
        f"succeeded={rc.succeeded}  errored={rc.errored}  "
        f"processing={rc.processing}  cancelled={rc.canceled}"
    )
    state["status"] = batch.processing_status
    state["results_url"] = getattr(batch, "results_url", None)
    _save_state("anthropic", state)
    return state


def anthropic_fetch() -> Path:
    import anthropic

    client = anthropic.Anthropic()
    state = _load_state("anthropic")
    if state["status"] != "ended":
        state = anthropic_status()
        if state["status"] != "ended":
            raise RuntimeError(f"batch not complete: {state['status']}")
    with ANTHROPIC_RESULT.open("w", encoding="utf-8") as out:
        for entry in client.messages.batches.results(state["batch_id"]):
            out.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
    print(f"[anthropic] wrote {ANTHROPIC_RESULT}  ({ANTHROPIC_RESULT.stat().st_size/1024:.1f} KB)")
    return ANTHROPIC_RESULT


# --- CLI ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--submit", action="store_true")
    grp.add_argument("--status", action="store_true")
    grp.add_argument("--fetch", action="store_true")
    grp.add_argument("--wait", action="store_true", help="poll until done then fetch")
    args = p.parse_args(argv)

    actions = {
        ("openai", "submit"): openai_submit,
        ("openai", "status"): openai_status,
        ("openai", "fetch"): openai_fetch,
        ("anthropic", "submit"): anthropic_submit,
        ("anthropic", "status"): anthropic_status,
        ("anthropic", "fetch"): anthropic_fetch,
    }

    if args.wait:
        status_fn = actions[(args.provider, "status")]
        while True:
            state = status_fn()
            done = (
                state["status"] == "completed"
                if args.provider == "openai"
                else state["status"] == "ended"
            )
            if done:
                break
            time.sleep(60)
        actions[(args.provider, "fetch")]()
        return 0

    key = "submit" if args.submit else "status" if args.status else "fetch"
    actions[(args.provider, key)]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
