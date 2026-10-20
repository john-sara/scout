"""A small BM25 index. No dependencies, so the whole retrieval stage is easy to read and test."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from how i if in into is it its of on or that the "
    "this to use using what when where which why with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. `response_model` also yields `response` and `model`."""
    tokens = []
    for raw in re.findall(r"[a-z0-9_]+", text.lower()):
        parts = [p for p in raw.split("_") if p]
        if len(parts) > 1:
            tokens.append(raw)
        tokens.extend(parts)
    return [_stem(t) for t in tokens if t not in _STOP]


def _stem(tok: str) -> str:
    # Deliberately crude: enough to match "parameters" with "parameter".
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        for i, ch in enumerate(chunks):
            toks = tokenize(ch.text + " " + ch.path.replace("/", " ").replace("-", " ").replace(".md", ""))
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            for term in tf:
                self._postings[term].append(i)
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._postings.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in self._postings[term]:
                tf = self._tf[i][term]
                norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg)
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(self.chunks[i], s) for i, s in ranked]

    def search_pages(self, query: str, k: int = 5, depth: int = 50) -> list[str]:
        """Rank whole pages: a page's rank is that of its best chunk."""
        seen: list[str] = []
        for hit in self.search(query, k=depth):
            if hit.chunk.path not in seen:
                seen.append(hit.chunk.path)
            if len(seen) == k:
                break
        return seen
