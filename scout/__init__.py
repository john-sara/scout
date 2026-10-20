"""Scout: a documentation agent with an evaluation harness."""
from __future__ import annotations

from pathlib import Path

from .chunking import DEFAULT_DOCS_DIR, build_chunks, load_docs
from .retrieval import BM25Index
from .tools import DocTools


def build_tools(chunking: str = "headings", docs_dir: Path | str = DEFAULT_DOCS_DIR, enable_code: bool = False) -> DocTools:
    docs = load_docs(docs_dir)
    return DocTools(BM25Index(build_chunks(docs, chunking)), docs, enable_code)
