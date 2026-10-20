"""Load documentation pages and split them into chunks.

Two strategies are compared in the evals:

* fixed:    sliding window of N words, ignores document structure.
* headings: split on markdown headings, then window any section that is still too long.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "fastapi_docs"


@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    heading: str
    text: str


_INCLUDE = re.compile(r"^\s*\{\*.*\*\}\s*$")  # code-include directives, the files are not in the corpus
_ADMONITION = re.compile(r"^\s*///\s*\w*.*$")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(\{#.*\})?\s*$")


def clean(text: str) -> str:
    text = _FRONTMATTER.sub("", text)
    text = _COMMENT.sub("", text)
    lines = []
    for line in text.splitlines():
        if _INCLUDE.match(line) or _ADMONITION.match(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def load_docs(root: Path | str = DEFAULT_DOCS_DIR) -> dict[str, str]:
    """Return {relative posix path: cleaned text} for every markdown page under root."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Run: python scripts/fetch_corpus.py")
    docs = {}
    for p in sorted(root.rglob("*.md")):
        docs[p.relative_to(root).as_posix()] = clean(p.read_text(encoding="utf-8"))
    return docs


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words] if words else []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def chunk_fixed(docs: dict[str, str], size: int = 200, overlap: int = 40) -> list[Chunk]:
    chunks = []
    for path, text in docs.items():
        for i, win in enumerate(_windows(text.split(), size, overlap)):
            chunks.append(Chunk(f"{path}#f{i}", path, "", " ".join(win)))
    return chunks


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading path, body) pairs, ignoring '#' inside code fences."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False

    def flush():
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(t for _, t in stack), content))
        body.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)
    flush()
    return sections


def chunk_headings(docs: dict[str, str], max_words: int = 300, overlap: int = 40) -> list[Chunk]:
    chunks = []
    for path, text in docs.items():
        for s_idx, (heading, body) in enumerate(split_sections(text)):
            for w_idx, win in enumerate(_windows(body.split(), max_words, overlap)):
                # The heading is part of the indexed text, so queries can match on it.
                chunks.append(Chunk(f"{path}#h{s_idx}.{w_idx}", path, heading, f"{heading}\n{' '.join(win)}".strip()))
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "headings": chunk_headings}


def build_chunks(docs: dict[str, str], strategy: str) -> list[Chunk]:
    try:
        return STRATEGIES[strategy](docs)
    except KeyError:
        raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
