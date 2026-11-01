"""Course-name abbreviations: 個經 -> 個體經濟學, 初會(一) -> 初級會計學(一), 五管 -> a group.

Sources, strongest first:
  rate   mined from RATE comments: a 2-3 character string that starts with the course
         name's first character, is an in-order subsequence of the name but not a
         substring, and appears mostly in that course's comments (share >= MIN_SHARE).
         初會 appears in 200 comments of 初級會計學.
  seed   a short curated list for group names the data cannot define (五管) and
         abbreviations too rare in RATE (大英, 計程); each carries its evidence note.
  rule   initials of the jieba words of a 114-2 course name (資料結構 -> 資結), used only
         when nothing above claims the string; lower confidence.
A mention resolves through: exact name, exact base name, alias (+ numbering), unique
substring of a base name.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_NUMBERING = re.compile(r"\s*[(（]\s*([一二三四五六上下]|[1-6]|I{1,3}|IV|V|VI)\s*[)）]\s*$")
_TAIL_NUM = re.compile(r"(?<=[一-鿿])([一二三四上下])$")
_HAN = re.compile(r"^[一-鿿]+$")
MIN_COUNT = 3
MIN_SHARE = 0.4

# alias -> (canonical base names, evidence). Groups list several members.
SEED: dict[str, tuple[tuple[str, ...], str]] = {
    "五管": (("營運與供應鏈管理", "行銷管理", "人力資源管理", "研究發展管理", "財務管理"),
             "企管五大管理（生產、行銷、人資、研發、財務）；note 先修五管 5 門、RATE 4 則"),
    "大英": (("大學英文",), "RATE 5 則；學生慣用"),
    "計程": (("計算機程式設計",), "forum 程式設計 or 計程；RATE 11 則"),
    "計概": (("計算機概論",), "RATE 73 則"),
    "經原": (("經濟學",), "政大的課名是 經濟學；RATE 提到 經濟學原理 的 22 則有 19 則在 經濟學"),
    "經濟學原理": (("經濟學",), "同上；forum 查詢 沒修過經濟學原理可以直接修個經嗎"),
    "會原": (("會計學原理",), "RATE 3 則"),
}


@dataclass(frozen=True)
class Alias:
    alias: str
    canonical: str               # base name, or the group label for a group
    members: tuple[str, ...]     # base names the alias stands for
    kind: str                    # abbr / group
    source: str                  # rate / seed / rule
    evidence: int                # comments (rate) or 0
    confidence: float
    in_1142: bool
    note: str = ""


@dataclass(frozen=True)
class Resolution:
    mention: str
    base: str | None             # resolved base name or group label
    names: tuple[str, ...]       # catalogue names matched (with numbering when given)
    method: str                  # exact / base / alias / substring / unresolved
    in_1142: bool


def nk(s: str | None) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def split_numbering(name: str) -> tuple[str, str]:
    """初級會計學(一) -> ('初級會計學', '一'); 中會二 -> ('中會', '二')."""
    s = nk(name)
    m = _NUMBERING.search(s) or _TAIL_NUM.search(s)
    return (s[: m.start()], m.group(1)) if m else (s, "")


def base_name(name: str) -> str:
    """大學英文（三）：商用英文 -> 大學英文; 初級會計學（一） -> 初級會計學."""
    s = nk(name).split(":")[0]
    while True:
        b, num = split_numbering(s)
        if not num or b == s:
            return s
        s = b


@dataclass
class Catalogue:
    by_base: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    in_1142: set[str] = field(default_factory=set)       # base names offered in 114-2
    offerings: Counter = field(default_factory=Counter)  # base -> number of course ids
    full: dict[str, str] = field(default_factory=dict)   # normalised full name -> base


def load_catalogue(con: sqlite3.Connection, ys: str = "1142") -> Catalogue:
    cat = Catalogue()
    for name, yss, n in con.execute("SELECT name, y||s, COUNT(DISTINCT id) FROM COURSE GROUP BY name, y||s"):
        b = base_name(name)
        if not b:
            continue
        cat.by_base[b].add(nk(name))
        cat.full[nk(name)] = b
        cat.offerings[b] += n
        if yss == ys:
            cat.in_1142.add(b)
    return cat


def _is_subseq(t: str, s: str) -> bool:
    it = iter(s)
    return all(ch in it for ch in t)


def _abbr_candidates(text: str, base: str) -> set[str]:
    out = set()
    for size in (2, 3):
        for i in range(len(text) - size + 1):
            g = text[i:i + size]
            if g[0] == base[0] and _HAN.match(g) and g not in base and _is_subseq(g, base):
                out.add(g)
    return out


def mine_rate(con: sqlite3.Connection, cat: Catalogue) -> list[Alias]:
    name_of = {cid: base_name(n) for cid, n in con.execute("SELECT DISTINCT substr(id,1,13), name FROM COURSE")}
    comments: dict[str, list[str]] = defaultdict(list)
    for cid, text in con.execute("SELECT courseId, content FROM RATE"):
        b = name_of.get((cid or "")[:13])
        if b and len(text or "") > 4 and len(b) >= 3 and _HAN.match(b[0]):
            comments[b].append(text)
    local: dict[str, Counter] = defaultdict(Counter)
    for b, texts in comments.items():
        for t in texts:
            for g in _abbr_candidates(t, b):
                local[g][b] += 1
    wanted = {g for g, c in local.items() if max(c.values()) >= MIN_COUNT}
    glob: Counter = Counter()
    for texts in comments.values():
        for t in texts:
            grams = {t[i:i + k] for k in (2, 3) for i in range(len(t) - k + 1)}
            glob.update(grams & wanted)
    out = []
    best = {g: local[g].most_common(1)[0] for g in wanted}
    for g in sorted(wanted):
        b, k = best[g]
        share = k / glob[g] if glob[g] else 0
        if k < MIN_COUNT or share < MIN_SHARE or g in cat.by_base:
            continue
        # 中會學 = 中會 + 學, 公關與 = a phrase cut before its noun: not abbreviations
        if len(g) == 3 and (g[-1] in "與的及和之" or best.get(g[:2], ("", 0))[0] == b):
            continue
        out.append(Alias(g, b, (b,), "abbr", "rate", k, round(share, 3), b in cat.in_1142,
                         f"{k}/{glob[g]} comments mentioning {g} belong to {b}"))
    return out


def seed_aliases(cat: Catalogue) -> list[Alias]:
    out = []
    for a, (members, note) in SEED.items():
        present = tuple(m for m in members if m in cat.by_base)
        if not present:
            continue
        kind = "group" if len(members) > 1 else "abbr"
        canon = a if kind == "group" else present[0]
        out.append(Alias(a, canon, present, kind, "seed", 0, 0.9 if kind == "abbr" else 0.7,
                         any(m in cat.in_1142 for m in present), note))
    return out


def rule_aliases(cat: Catalogue, taken: set[str]) -> list[Alias]:
    """Initials of jieba words for 114-2 base names (資料結構 -> 資結). Skipped without jieba."""
    try:
        import jieba  # noqa: PLC0415 - optional dependency, present on hopelab
    except ImportError:
        return []
    jieba.setLogLevel(60)
    claims: dict[str, list[str]] = defaultdict(list)
    for b in sorted(cat.in_1142):
        words = [w for w in jieba.lcut(b) if _HAN.match(w)]
        if len(words) < 2 or len(b) < 4:
            continue
        init = "".join(w[0] for w in words)
        for cand in {init, init[-2:]} if len(init) > 2 else {init}:
            # 中日 is a word (China-Japan); initials that are dictionary words mislead
            if cand in jieba.dt.FREQ and jieba.dt.FREQ[cand] > 0:
                continue
            if 2 <= len(cand) <= 3 and cand[0] == b[0] and cand not in taken and cand not in cat.by_base:
                claims[cand].append(b)
    out = []
    for a, bases in claims.items():
        best = max(bases, key=lambda x: cat.offerings[x])
        out.append(Alias(a, best, (best,), "abbr", "rule", 0, round(0.5 / len(bases), 3), True,
                         f"initials of {best}" + (f"; {len(bases)} names share it" if len(bases) > 1 else "")))
    return out


def build_aliases(con: sqlite3.Connection, cat: Catalogue) -> list[Alias]:
    seeds = seed_aliases(cat)
    rate = [a for a in mine_rate(con, cat) if a.alias not in SEED]
    taken = {a.alias for a in seeds + rate}
    return seeds + rate + rule_aliases(cat, taken)


_QUAL = re.compile(r"^(三學分之|\(大學部\)|大學部|\d{3}-\d)+|(一學年|一年|全學年|相關|等)$")


def _variants(mention: str) -> list[str]:
    m = re.sub(r"^[「『“\"]|[」』”\"]$|課程$|等課$", "", nk(mention))
    q = _QUAL.sub("", _QUAL.sub("", m))
    out = [m, q, re.sub(r"\([^)]*\)$", "", q)]
    return out + [x + "學" for x in out if x and not x.endswith("學")]


class Resolver:
    def __init__(self, aliases: list[Alias], cat: Catalogue):
        self.cat = cat
        self.alias = {a.alias: a for a in sorted(aliases, key=lambda a: a.confidence)}

    def _names(self, bases: tuple[str, ...], num: str) -> tuple[str, ...]:
        names = sorted(n for b in bases for n in self.cat.by_base.get(b, ()))
        if num:
            numbered = [n for n in names if split_numbering(n.split(":")[0])[1] == num]
            return tuple(numbered or names)
        return tuple(names)

    def _hit(self, mention: str, bases: tuple[str, ...], label: str, num: str, method: str) -> Resolution:
        return Resolution(mention, label, self._names(bases, num), method,
                          any(b in self.cat.in_1142 for b in bases))

    def resolve(self, mention: str) -> Resolution:
        """First variant that resolves wins: as written, without qualifiers (三學分之,
        (大學部), 114-1, 一學年, 相關), without a trailing parenthesis, with 學 appended."""
        seen = set()
        for v in _variants(mention):
            if v and v not in seen:
                seen.add(v)
                hit = self._resolve_one(mention, v)
                if hit.method != "unresolved":
                    return hit
        return Resolution(mention, None, (), "unresolved", False)

    def _resolve_one(self, mention: str, m: str) -> Resolution:
        if m in self.cat.full:
            b = self.cat.full[m]
            return Resolution(mention, b, (m,), "exact", b in self.cat.in_1142)
        stem, num = split_numbering(m)
        if stem in self.cat.by_base:
            return self._hit(mention, (stem,), stem, num, "base")
        a = self.alias.get(stem)
        if a:
            return self._hit(mention, a.members, a.canonical, num, f"alias:{a.source}")
        if len(stem) >= 3:
            hits = [b for b in self.cat.by_base if stem in b]
            if len(hits) == 1 or (hits and len(set(hits) & self.cat.in_1142) == 1):
                pick = hits[0] if len(hits) == 1 else next(iter(set(hits) & self.cat.in_1142))
                return self._hit(mention, (pick,), pick, num, "substring")
        return Resolution(mention, None, (), "unresolved", False)
