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
        self.max_len = 1024

    def score(self, query: str, docs: list[str]) -> list[float]:
        s = self.model.compute_score([[query, d] for d in docs], normalize=True, max_length=self.max_len)
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


class CrossEncoderReranker:
    """Small multilingual cross-encoders for CPU serving (sentence-transformers CrossEncoder)."""

    def __init__(self, hf_id: str, name: str, max_len: int = 512, device: str = "cuda", batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.name = name
        self.model = CrossEncoder(hf_id, max_length=max_len, device=device)
        self.batch_size = batch_size

    def score(self, query: str, docs: list[str]) -> list[float]:
        import math

        raw = [float(x) for x in self.model.predict([(query, d) for d in docs], batch_size=self.batch_size,
                                                   show_progress_bar=False)]
        # the pipeline places reranked courses at 1 + score, so scores must lie in [0, 1];
        # mMiniLM returns raw logits (-4.1 ...), bge-reranker-base already returns sigmoid outputs
        if any(x < 0 or x > 1 for x in raw):
            raw = [1 / (1 + math.exp(-x)) for x in raw]
        return raw


SMALL = {"bge-reranker-base": "BAAI/bge-reranker-base",
         "mminilm-l12": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"}


def _int8(model):
    """Dynamic int8 quantisation of every Linear layer (CPU only, x86 fbgemm)."""
    import torch

    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def load(name: str, device: str = "cuda", max_len: int | None = None, int8: bool = False):
    """max_len truncates each (query, course) pair; int8 quantises Linear layers (CPU)."""
    if int8 and device != "cpu":
        raise ValueError("int8 quantisation runs on CPU only")
    if name == "bge-reranker-v2-m3":
        rr = BGEReranker(device)
        if max_len:
            rr.max_len = max_len
    elif name in ("qwen3-rerank-0.6b", "qwen3-rerank-4b"):
        size = "0.6B" if name.endswith("0.6b") else "4B"
        rr = QwenReranker(size, max_len=max_len or 1536, batch_size=8 if size == "0.6B" else 4, device=device)
        if int8:
            rr.model = _int8(rr.model)
    elif name in SMALL:
        rr = CrossEncoderReranker(SMALL[name], name, max_len or 512, device)
        if int8:
            rr.model.model = _int8(rr.model.model)
    else:
        raise ValueError(name)
    return rr


def timed_score(model, query: str, docs: list[str]) -> tuple[list[float], float]:
    t0 = time.time()
    s = model.score(query, docs)
    return s, time.time() - t0
