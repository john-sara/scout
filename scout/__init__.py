"""Scout: a documentation agent with an evaluation harness."""
from __future__ import annotations

from pathlib import Path

from .chunking import DEFAULT_DOCS_DIR, build_chunks, load_docs
from .retrieval import BM25Index
from .tools import DocTools


RETRIEVAL_MODES = ("bm25", "embed", "hybrid")


def build_index(chunks, retrieval: str = "bm25", embedder=None, progress=None):
    """BM25 by default. 'embed' and 'hybrid' also need an embedder (Ollama by default)."""
    bm25 = BM25Index(chunks)
    if retrieval == "bm25":
        return bm25
    if retrieval not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval must be one of {RETRIEVAL_MODES}, not {retrieval!r}")
    from .embeddings import EmbeddingIndex, HybridIndex, OllamaEmbedder
    emb = EmbeddingIndex.build(chunks, embedder or OllamaEmbedder(), progress=progress)
    return emb if retrieval == "embed" else HybridIndex(bm25, emb)


def build_tools(chunking: str = "headings", docs_dir: Path | str = DEFAULT_DOCS_DIR, enable_code: bool = False,
                retrieval: str = "bm25", embedder=None) -> DocTools:
    docs = load_docs(docs_dir)
    return DocTools(build_index(build_chunks(docs, chunking), retrieval, embedder), docs, enable_code)
