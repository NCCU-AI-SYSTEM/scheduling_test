"""v4 course search: structure first, then topic, then preferences.

    plan (small LLM)
      -> hard conditions as a DB mask, relaxed one dimension at a time if too few survive
      -> named teacher as a mask, named course as an exact-name boost
      -> eligibility from the transcript (blocked courses removed)
      -> topic retrieval inside the mask: char-ngram BM25 + dense, RRF, rerank top 20
      -> preference re-scoring from review aspects and syllabus breakdowns

Every stage is switchable so experiments can ablate it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from src.eval.conditions import check as check_cond, satisfies
from src.parsers.base import ConditionResult
from src.retrievers.sparse import BM25, tokenize_all
from src.v4.access import Access, StudentIdentity, judge
from src.v4.assessment import AssessmentProfile, profile_from, verdict
from src.v4.catalog import CatalogCourse
from src.v4.eligibility import Flag, check as check_elig
from src.v4.offering import Offering
from src.v4.planner import Plan
from src.v4.profile import StudentProfile

# least reliable first: unit strings are messy, kind/lmt_kind next; time and language
# are the conditions students notice when violated, so they are dropped last
RELAX_ORDER = ["unit", "kind", "lmt_kind", "point", "hour_range", "weekday", "course_lang"]


def doc_text(c: CatalogCourse) -> str:
    x = c.course
    lines = [f"課名: {x.name}" + (f"（{x.name_en}）" if x.name_en else ""), f"教師: {x.teacher}",
             f"開課單位: {x.unit}　類別: {'、'.join(sorted(c.categories))}　語言: {x.lang}"]
    if x.objective:
        lines.append(f"課程目標: {x.objective.strip()[:600]}")
    if c.topics:
        lines.append(f"週次主題: {'、'.join(c.topics)[:500]}")
    if x.syllabus and len(x.objective or "") < 200:
        lines.append(f"課程簡介: {x.syllabus.strip()[:400]}")
    return "\n".join(lines)


def _norm(s: str) -> str:
    return re.sub(r"[（(].*?[)）]|\s", "", s or "")


_NUM = re.compile(r"[（(]\s*([一二三四五六七八九十]+|\d+|[上下])\s*[)）]")


def _numbering(name: str | None) -> str | None:
    """'初會(一)' -> '一'; '統計學（二）' -> '二'; no numbering -> None."""
    m = _NUM.search(name or "")
    return m.group(1) if m else None


@dataclass
class Hit:
    course_id: str
    score: float
    matched: list[str] = field(default_factory=list)
    relaxed: list[str] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    pref: float = 0.0
    offering: Offering | None = None
    access: Access | None = None


@dataclass
class Config:
    use_conditions: bool = True
    relax_min: int = 10
    use_named: bool = True
    use_dense: bool = True
    use_bm25: bool = True
    rerank: bool = True
    rerank_k: int = 20
    use_prefs: bool = True
    pref_weight: float = 0.3
    use_eligibility: bool = True
    use_hype: bool = False
    merge_offerings: bool = False      # one hit per class (same name, teacher, time)
    named_skips_conditions: bool = False  # a named course is looked up, conditions do not filter it out
    use_assessment: bool = True        # 不要期中考 / 要有小考 from the syllabus evaluation text


class V4Search:
    def __init__(self, catalog: list[CatalogCourse], encoder=None, reranker=None,
                 aspects: dict[str, dict] | None = None, doc_fn=doc_text,
                 hype: dict[str, list[str]] | None = None, offerings: list[Offering] | None = None,
                 doc_emb: np.ndarray | None = None, aliases: dict[str, tuple[str, ...]] | None = None,
                 assess: dict[str, AssessmentProfile] | None = None):
        """doc_emb, aliases and assess come from course_struct.db (src.v4.struct_store); without
        them the documents are encoded here and assessment falls back to the keyword scan."""
        self.cat = catalog
        self.offerings = offerings or []
        self.off_of = {cid: o for o in self.offerings for cid in o.course_ids}
        self.ids = [c.course_id for c in catalog]
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self.texts = [doc_fn(c) for c in catalog]
        self.bm25 = BM25(tokenize_all(self.texts, "char12"))
        self.encoder = encoder
        self.demb = doc_emb if doc_emb is not None else (encoder.encode_docs(self.texts)[0] if encoder else None)
        self.reranker = reranker
        self.aspects = aspects or {}
        self.names = [_norm(c.course.name) for c in catalog]
        self.aliases = aliases or {}
        self.assess = [assess[c.course_id] if assess and c.course_id in assess
                       else profile_from([lbl for lbl, _ in c.grading.get("items", [])], c.course.evaluation)
                       for c in catalog]
        self.timing: dict[str, float] = {}
        self.q_owner = np.zeros(0, dtype=int)
        if hype:
            pos = {cid: i for i, cid in enumerate(self.ids)}
            q_texts, owner = [], []
            for cid, qs in hype.items():
                if cid in pos:
                    for q in qs:
                        q_texts.append(q)
                        owner.append(pos[cid])
            self.q_owner = np.asarray(owner)
            self.q_bm25 = BM25(tokenize_all(q_texts, "char12"))
            self.q_emb = encoder.encode_docs(q_texts)[0] if encoder else None

    def _expand(self, text: str) -> tuple[str, list[str]]:
        if not self.aliases:
            return text, []
        from src.v4.struct_store import expand_aliases

        return expand_aliases(text, self.aliases)

    def _is_named(self, i: int, forms: list[str], num: str | None) -> bool:
        """初會(一) matches 初級會計學(一), not 初級會計學(二)."""
        return any(f in self.names[i] for f in forms) and (num is None or _numbering(self.cat[i].course.name) == num)

    def _named_forms(self, named: str) -> list[str]:
        """Normalised names a named course may appear as: itself plus its alias expansions."""
        forms = [_norm(named)]
        forms += [_norm(n) for n in self.aliases.get(named.strip(), ())]
        forms += [_norm(n) for n in self._expand(named)[1]]
        return [f for f in dict.fromkeys(forms) if f]

    def _per_course(self, q_scores: np.ndarray) -> np.ndarray:
        out = np.full(len(self.ids), -np.inf)
        np.maximum.at(out, self.q_owner, q_scores)
        return out

    # ── masks ────────────────────────────────────────────────────────────────
    def _cond_mask(self, cond: ConditionResult, min_pool: int) -> tuple[np.ndarray, list[str]]:
        must, must_not = dict(cond.must), dict(cond.must_not)
        relaxed: list[str] = []
        while True:
            c = ConditionResult(must=must, must_not=must_not)
            mask = np.array([all(check_cond(x.meta, c).values()) for x in self.cat])
            if mask.sum() >= min_pool or not (must or must_not):
                return mask, relaxed
            dim = next((d for d in RELAX_ORDER if d in must or d in must_not), None)
            if dim is None:
                return mask, relaxed
            must.pop(dim, None)
            must_not.pop(dim, None)
            relaxed.append(dim)

    def _teacher_mask(self, name: str) -> np.ndarray | None:
        m = np.array([name in c.course.teacher for c in self.cat])
        return m if m.any() else None

    # ── scoring ──────────────────────────────────────────────────────────────
    def _pref_score(self, cid: str, prefs: list[dict]) -> float:
        a = self.aspects.get(cid, {})
        vals = []
        for p in prefs:
            v = a.get(p["aspect"])
            if v is None:
                continue
            vals.append((3 - v) / 2 if p["want"] == "low" else (v - 3) / 2)
        return float(np.mean(vals)) if vals else 0.0

    def search(self, plan: Plan, raw_query: str, k: int = 10, cfg: Config = Config(),
               profile: StudentProfile | None = None, current: list[CatalogCourse] | None = None,
               identity: StudentIdentity | None = None) -> list[Hit]:
        t0 = time.time()
        n = len(self.cat)
        mask = np.ones(n, dtype=bool)
        relaxed: list[str] = []
        skip_cond = cfg.named_skips_conditions and bool(plan.named_course)
        if cfg.use_conditions and not skip_cond and (plan.cond.must or plan.cond.must_not):
            mask, relaxed = self._cond_mask(plan.cond, cfg.relax_min)
        access: dict[tuple, Access] = {}
        if identity is not None and self.offerings:
            access = {o.key: judge(o, identity) for o in self.offerings}
            mask &= np.array([not access[self.off_of[cid].key].excluded if cid in self.off_of else True
                              for cid in self.ids])
        if cfg.use_named and plan.named_teacher:
            tm = self._teacher_mask(plan.named_teacher)
            if tm is not None:
                mask &= tm
        elig: dict[int, list[Flag]] = {}
        if cfg.use_eligibility and profile:
            for i in np.flatnonzero(mask):
                f = check_elig(self.cat[i], profile, current)
                if any(x.level == "block" for x in f):
                    mask[i] = False
                elif f:
                    elig[i] = f
        unknown_assess: set[int] = set()
        if cfg.use_assessment and (plan.assess_must or plan.assess_not):
            for i in np.flatnonzero(mask):
                v = verdict(self.assess[i], plan.assess_must, plan.assess_not)
                if v == "violate":
                    mask[i] = False
                elif v == "unknown":
                    unknown_assess.add(int(i))
        self.timing["filter"] = time.time() - t0

        # 個經 -> also search 個體經濟學; aliases come only from reviews and hand seeds (user, 2026/11/01)
        text, self.last_expansions = self._expand(plan.need or plan.named_course or raw_query)
        t1 = time.time()
        runs = []
        if cfg.use_bm25:
            s = self.bm25.scores(tokenize_all([text], "char12")[0])
            runs.append(self._rank(s, mask))
        qv = None
        if cfg.use_dense and self.encoder is not None:
            qv = self.encoder.encode_queries([text])[0]
            runs.append(self._rank(self.demb @ qv, mask))
        if cfg.use_hype and len(self.q_owner):
            hm = mask & np.isin(np.arange(n), self.q_owner)
            runs.append(self._rank(self._per_course(self.q_bm25.scores(tokenize_all([text], "char12")[0])), hm))
            if qv is not None and self.q_emb is not None:
                runs.append(self._rank(self._per_course(self.q_emb @ qv), hm))
        fused = self._rrf(runs) if runs else [int(i) for i in np.flatnonzero(mask)][:50]
        named = self._named_forms(plan.named_course) if cfg.use_named and plan.named_course else []
        num = _numbering(plan.named_course) if named else None
        if named:
            exact = [i for i in np.flatnonzero(mask) if self._is_named(i, named, num)]
            fused = list(dict.fromkeys([int(i) for i in exact] + fused))
        self.timing["retrieve"] = time.time() - t1

        t2 = time.time()
        scores = {i: 1.0 / (r + 1) for r, i in enumerate(fused)}
        head = fused[:cfg.rerank_k]
        if cfg.rerank and self.reranker is not None and head and (plan.need or not plan.named_course):
            rs = self.reranker.score(text, [self.texts[i] for i in head])
            for i, s in zip(head, rs):
                scores[i] = 1.0 + s  # reranked head stays above the unreranked tail
            for i in head:
                if self._is_named(i, named, num):
                    scores[i] += 1.0
        self.timing["rerank"] = time.time() - t2

        prefs = plan.prefs if cfg.use_prefs else []
        hits = []
        keep = len(scores) if cfg.merge_offerings else max(k, cfg.rerank_k)  # merging drops duplicates
        for i in sorted(scores, key=lambda j: -scores[j])[:keep]:
            cid = self.ids[i]
            p = self._pref_score(cid, prefs) if prefs else 0.0
            matched = [key for key, ok in check_cond(self.cat[i].meta, plan.cond).items() if ok]
            flags = list(elig.get(i, []))
            if i in unknown_assess:
                flags.append(Flag("info", "assessment_unknown", "大綱未註明評分方式，無法確認是否符合"))
            hits.append(Hit(cid, scores[i] + cfg.pref_weight * p, matched, relaxed, flags, p))
        # Courses meeting every condition the student gave rank first. They differ from the
        # mask only after relaxation or a named lookup that skipped the filter: 禮拜五晚上 2 學分
        # relaxes to 3-credit courses, and the two 2-credit ones must still come first (2026/11/02).
        full = {h.course_id for h in hits if satisfies(self.cat[self.pos[h.course_id]].meta, plan.cond)}
        # syllabi that do not state the assessment rank after every clear match (user, 2026/10/19)
        hits.sort(key=lambda h: (h.course_id not in full,
                                 any(f.kind == "assessment_unknown" for f in h.flags), -h.score))
        if cfg.merge_offerings and self.off_of:
            hits = self._merge(hits, access)
        self.timing["total"] = time.time() - t0
        return hits[:k]

    def _merge(self, hits: list[Hit], access: dict[tuple, Access]) -> list[Hit]:
        """Keep the best-scored code of each class; with an identity, show the channel that fits."""
        out, seen = [], set()
        for h in hits:
            off = self.off_of.get(h.course_id)
            if off is None:
                out.append(h)
                continue
            if off.key in seen:
                continue
            seen.add(off.key)
            a = access.get(off.key)
            out.append(Hit(a.channel.course_id if a else h.course_id, h.score, h.matched, h.relaxed,
                           h.flags, h.pref, off, a))
        return out

    @staticmethod
    def _rank(scores: np.ndarray, mask: np.ndarray, k: int = 50) -> list[int]:
        s = np.where(mask, scores.astype(np.float64), -np.inf)
        kk = min(k, int(mask.sum()))
        if kk <= 0:
            return []
        idx = np.argpartition(-s, kk - 1)[:kk]
        return [int(i) for i in idx[np.argsort(-s[idx])]]

    @staticmethod
    def _rrf(runs: list[list[int]], k: int = 60) -> list[int]:
        acc: dict[int, float] = {}
        for r in runs:
            for rank, i in enumerate(r):
                acc[i] = acc.get(i, 0.0) + 1 / (k + rank + 1)
        return [i for i, _ in sorted(acc.items(), key=lambda x: -x[1])][:50]
