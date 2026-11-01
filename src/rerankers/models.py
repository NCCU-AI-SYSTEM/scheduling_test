"""Cross-encoder rerankers behind one interface: score(query, docs) -> list[float].

  bge-reranker-v2-m3   568M XLM-R cross-encoder (current pipeline)
  qwen3-rerank-0.6b/4b instruction-aware; relevance = P("yes") at the answer token,
                       following the Qwen3-Reranker model card
"""

from __future__ import annotations

import time

INSTRUCTION = "根據學生的選課需求，判斷這門大學課程的內容是否符合學生想學的東西"


class BGEReranker:
    name = "bge-reranker-v2-m3"

    def __init__(self, device: str = "cuda"):
        from FlagEmbedding import FlagReranker

        self.model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=device == "cuda", devices=device)

    def score(self, query: str, docs: list[str]) -> list[float]:
        s = self.model.compute_score([[query, d] for d in docs], normalize=True, max_length=1024)
        return [s] if isinstance(s, float) else list(s)


class QwenReranker:
    _PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
               "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
               "<|im_start|>user\n")
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, size: str = "0.6B", max_len: int = 1536, batch_size: int = 8, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf = f"Qwen/Qwen3-Reranker-{size}"
        self.name = f"qwen3-rerank-{size.lower()}"
        self.tok = AutoTokenizer.from_pretrained(hf, padding_side="left")
        dtype = torch.float16 if device == "cuda" else torch.float32  # fp16 matmuls are slow on CPU
        self.model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.max_len = max_len
        self.batch_size = batch_size
        self.torch = torch
        self.pre = self.tok.encode(self._PREFIX, add_special_tokens=False)
        self.suf = self.tok.encode(self._SUFFIX, add_special_tokens=False)

    def _encode(self, query: str, doc: str) -> list[int]:
        body = self.tok.encode(f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}",
                               add_special_tokens=False)
        body = body[: self.max_len - len(self.pre) - len(self.suf)]
        return self.pre + body + self.suf

    def score(self, query: str, docs: list[str]) -> list[float]:
        torch = self.torch
        out: list[float] = []
        for i in range(0, len(docs), self.batch_size):
            ids = [self._encode(query, d) for d in docs[i:i + self.batch_size]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # only the last position is scored; logits_to_keep=1 avoids a (batch, seq, 151k) tensor
                logits = self.model(**batch, logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out.extend(torch.softmax(pair, dim=1)[:, 1].tolist())
        return out


def load(name: str, device: str = "cuda"):
    if name == "bge-reranker-v2-m3":
        return BGEReranker(device)
    if name == "qwen3-rerank-0.6b":
        return QwenReranker("0.6B", device=device)
    if name == "qwen3-rerank-4b":
        return QwenReranker("4B", batch_size=4, device=device)
    raise ValueError(name)


def timed_score(model, query: str, docs: list[str]) -> tuple[list[float], float]:
    t0 = time.time()
    s = model.score(query, docs)
    return s, time.time() - t0
