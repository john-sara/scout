"""Embedding retrieval and hybrid (BM25 + embeddings) retrieval.

Embeddings come from Ollama over plain HTTP, like the chat models, so the project keeps its small
dependency list. Vectors are normalized, so cosine similarity is a dot product, and the search is a
plain loop over about a thousand chunks, which takes tens of milliseconds.

The vectors for a set of chunks are cached on disk. The cache name includes a hash of the model name,
the prefixes and the chunk text, so changing any of them builds a fresh index.
"""
from __future__ import annotations

import hashlib
import json
import math
import operator
import os
import re
from array import array
from pathlib import Path
from typing import Protocol

from .chunking import Chunk
from .llm import ProviderError, _post_with_retry
from .retrieval import BM25Index, Hit, pages_from_hits

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"
_USE_DEFAULT = object()


def cache_dir_from_env() -> Path:
    return Path(os.environ.get("SCOUT_EMBED_CACHE") or DEFAULT_CACHE_DIR)


class Embedder(Protocol):
    model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    def fingerprint(self) -> str: ...


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class OllamaEmbedder:
    """Embeddings from a local Ollama model. nomic-embed-text expects a task prefix on its inputs."""

    def __init__(self, model: str | None = None, host: str | None = None, batch: int = 16, timeout: float = 300,
                 doc_prefix: str = "search_document: ", query_prefix: str = "search_query: ", max_chars: int = 6000):
        self.model = model or os.environ.get("SCOUT_EMBED_MODEL", "nomic-embed-text")
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.batch, self.timeout = batch, timeout
        self.doc_prefix, self.query_prefix, self.max_chars = doc_prefix, query_prefix, max_chars

    def fingerprint(self) -> str:
        return f"ollama|{self.model}|{self.doc_prefix}|{self.query_prefix}|{self.max_chars}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.model, "input": texts}
        data = _post_with_retry(f"{self.host}/api/embed", {}, body, retries=1, timeout=self.timeout, label="Ollama")
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise ProviderError(f"Ollama returned {len(vectors or [])} embeddings for {len(texts)} inputs")
        return [normalize(v) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            part = [self.doc_prefix + t[: self.max_chars] for t in texts[i:i + self.batch]]
            out.extend(self._embed(part))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text[: self.max_chars]])[0]


class FakeEmbedder:
    """Hashed bag of words. Similar texts get similar vectors. Used only in tests."""

    def __init__(self, dim: int = 64):
        self.model, self.dim = f"fake-{dim}", dim
        self.calls = 0

    def fingerprint(self) -> str:
        return f"fake|{self.dim}"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return normalize(v)

    def embed_documents(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def chunk_input(chunk: Chunk) -> str:
    """What gets embedded: the page path and heading give the model the same context a reader has."""
    title = chunk.path.removesuffix(".md").replace("/", " > ")
    return f"{title}" + (f" | {chunk.heading}" if chunk.heading else "") + f"\n{chunk.text}"


def _cache_key(chunks: list[Chunk], embedder: Embedder) -> str:
    h = hashlib.sha1(embedder.fingerprint().encode())
    for ch in chunks:
        h.update(ch.id.encode())
        h.update(chunk_input(ch).encode())
    return h.hexdigest()[:16]


def save_vectors(path: Path, vectors: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = array("f")
    for v in vectors:
        flat.extend(v)
    with open(path, "wb") as f:
        f.write((json.dumps({"n": len(vectors), "dim": len(vectors[0]) if vectors else 0}) + "\n").encode())
        flat.tofile(f)


def load_vectors(path: Path) -> list[list[float]] | None:
    try:
        with open(path, "rb") as f:
            header = json.loads(f.readline())
            flat = array("f")
            flat.fromfile(f, header["n"] * header["dim"])
    except (OSError, ValueError, EOFError, KeyError):
        return None
    dim = header["dim"]
    return [list(flat[i * dim:(i + 1) * dim]) for i in range(header["n"])]


class EmbeddingIndex:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]], embedder: Embedder):
        if len(chunks) != len(vectors):
            raise ValueError("one vector per chunk is required")
        self.chunks, self.vectors, self.embedder = chunks, vectors, embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, cache_dir: Path | str | None = _USE_DEFAULT,
              progress=None) -> "EmbeddingIndex":
        cache = None
        if cache_dir is _USE_DEFAULT:
            cache_dir = cache_dir_from_env()
        if cache_dir is not None:
            safe = re.sub(r"[^a-zA-Z0-9._-]", "_", embedder.model)
            cache = Path(cache_dir) / f"{safe}-{_cache_key(chunks, embedder)}.vec"
            vectors = load_vectors(cache)
            if vectors is not None and len(vectors) == len(chunks):
                return cls(chunks, vectors, embedder)
        vectors = []
        step = 64
        for i in range(0, len(chunks), step):
            vectors.extend(embedder.embed_documents([chunk_input(c) for c in chunks[i:i + step]]))
            if progress:
                progress(min(i + step, len(chunks)), len(chunks))
        if cache is not None:
            save_vectors(cache, vectors)
        return cls(chunks, vectors, embedder)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.embedder.embed_query(query)
        scores = [(sum(map(operator.mul, q, v)), i) for i, v in enumerate(self.vectors)]
        scores.sort(key=lambda t: (-t[0], t[1]))
        return [Hit(self.chunks[i], s) for s, i in scores[:k]]

    def search_pages(self, query: str, k: int = 5, depth: int = 50) -> list[str]:
        return pages_from_hits(self.search(query, k=depth), k)


class HybridIndex:
    """BM25 and embeddings, merged with reciprocal rank fusion: score = sum of 1 / (60 + rank) per list."""

    def __init__(self, bm25: BM25Index, embeddings: EmbeddingIndex, rrf_k: int = 60, depth: int = 50):
        self.bm25, self.embeddings, self.rrf_k, self.depth = bm25, embeddings, rrf_k, depth
        self.chunks = bm25.chunks

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}
        for ranking in (self.bm25.search(query, k=self.depth), self.embeddings.search(query, k=self.depth)):
            for rank, hit in enumerate(ranking, 1):
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
                chunk_by_id[hit.chunk.id] = hit.chunk
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(chunk_by_id[cid], score) for cid, score in ranked]

    def search_pages(self, query: str, k: int = 5, depth: int = 50) -> list[str]:
        return pages_from_hits(self.search(query, k=depth), k)
