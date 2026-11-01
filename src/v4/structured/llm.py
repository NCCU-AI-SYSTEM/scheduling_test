"""LLM fallback for the residue the rules miss, cached by input hash.

Four jobs, each over distinct input texts:
  notes    note parts no rule matched -> restriction items
  prereq   prerequisite spans whose course names the resolver cannot split -> names
  grading  evaluations whose percents do not add up to ~100 -> top-level items
  weekly   schedules where no week run was found -> weeks
Cache: data/v4/structured_cache/<job>.jsonl, one JSON object per line
{"key", "model", "output", "error"}. The key hashes the job, prompt version and input,
so a rerun with the same inputs makes no request. Models: gemma4:31b-cloud (at most 2
concurrent requests; 8 returned HTTP 429) or local gemma4:12b.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.v4.structured.prompts import PROMPTS, VERSION

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "v4" / "structured_cache"
MODELS = {"cloud": "gemma4:31b-cloud", "local": "gemma4:12b"}
WORKERS = {"cloud": 2, "local": 2}


def cache_key(job: str, text: str) -> str:
    return hashlib.sha1(f"{job}\x00{VERSION[job]}\x00{text}".encode()).hexdigest()


class Cache:
    """Append-only JSONL per job; later lines win. Thread-safe appends."""

    def __init__(self, job: str, root: Path = CACHE_DIR):
        self.path = root / f"{job}.jsonl"
        self.job = job
        self._lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue   # a torn last line from an interrupted run
                self.data[row["key"]] = row

    def get(self, text: str) -> dict | None:
        row = self.data.get(cache_key(self.job, text))
        return row["output"] if row and row.get("output") is not None else None

    def put(self, text: str, model: str, output: dict | None, error: str = "") -> None:
        row = {"key": cache_key(self.job, text), "model": model, "output": output, "error": error}
        with self._lock:
            self.data[row["key"]] = row
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ask(job: str, text: str, model: str) -> tuple[dict | None, str]:
    from src.llm.ollama_http import chat, parse_json_object  # noqa: PLC0415 - only when calling

    system, build = PROMPTS[job]
    res = chat(model, [{"role": "system", "content": system}, {"role": "user", "content": build(text)}],
               fmt="json" if not model.endswith("-cloud") else None, num_ctx=12288, timeout=300)
    if res.error:
        return None, res.error
    out = parse_json_object(res.content)
    return (out, "") if out is not None else (None, "unparseable: " + res.content[:200])


def run(job: str, texts: list[str], mode: str, log=print) -> Cache:
    """Fill the cache for `texts`; mode 'off' only reads it."""
    cache = Cache(job)
    todo = sorted({t for t in texts if t and cache.get(t) is None})
    log(f"[{job}] {len(texts)} inputs, {len(set(texts)) - len(todo)} cached, {len(todo)} to ask ({mode})")
    if mode == "off" or not todo:
        return cache
    model = MODELS[mode]
    done = [0]

    def one(t: str) -> None:
        out, err = _ask(job, t, model)
        cache.put(t, model, out, err)
        done[0] += 1
        if err or done[0] % 20 == 0:
            log(f"[{job}] {done[0]}/{len(todo)}" + (f" error: {err[:120]}" if err else ""))

    with ThreadPoolExecutor(max_workers=WORKERS[mode]) as pool:
        list(pool.map(one, todo))
    return cache
