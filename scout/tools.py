"""Tools the agent can call. Each call returns a ToolResult; errors are returned to the model as
text (is_error=True) so it can recover, instead of crashing the run."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from .retrieval import BM25Index

MAX_PAGE_CHARS = 6000


@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    pages: list[str] = field(default_factory=list)  # doc pages this call surfaced


TOOL_SPECS = [
    {
        "name": "search_docs",
        "description": "Search the FastAPI documentation. Returns the best matching passages with the page path "
                       "they came from. Use short keyword queries, and search again with different words if the "
                       "first results do not help.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for."},
                "k": {"type": "integer", "description": "Number of passages to return (1-8, default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_page",
        "description": "Read a documentation page by its path, for example 'tutorial/body.md'. Use this when a "
                       "search result looks right but you need more detail.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Page path as shown in search results."},
                "offset": {"type": "integer", "description": "Character position to start from. Long pages are returned "
                                                             "in parts; use the offset given in the truncation note to continue."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_pages",
        "description": "List documentation page paths, optionally only those starting with a prefix such as "
                       "'tutorial/security/'.",
        "input_schema": {
            "type": "object",
            "properties": {"prefix": {"type": "string", "description": "Optional path prefix."}},
        },
    },
]


CODE_TIMEOUT_S = 8
MAX_CODE_OUTPUT = 3000

RUN_PYTHON_SPEC = {
    "name": "run_python",
    "description": "Run a short Python snippet to check an answer, for example a minimal FastAPI app exercised "
                   "with fastapi.testclient.TestClient. Returns stdout and stderr. Times out after "
                   f"{CODE_TIMEOUT_S} seconds. Only use it to verify behavior, not to fetch anything.",
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to run."}},
        "required": ["code"],
    },
}

# Environment variables that must never reach model-written code.
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def run_python(code: str, timeout: float = CODE_TIMEOUT_S) -> ToolResult:
    """Run code in a separate Python process: isolated mode, throwaway working directory, secrets removed
    from the environment, hard timeout, capped output.

    This is NOT a security sandbox. The code runs as you and can read your files or use the network.
    That is why the tool is off by default and should only be enabled on a machine you control.
    """
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in _SECRET_HINTS)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp, env=env, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ToolResult(f"ERROR: code timed out after {timeout:g} seconds.", True)
        except OSError as e:
            return ToolResult(f"ERROR: could not start Python: {e}", True)
    out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    if len(out) > MAX_CODE_OUTPUT:
        out = out[:MAX_CODE_OUTPUT] + "\n[output truncated]"
    return ToolResult(out or "(no output)", is_error=proc.returncode != 0)


class DocTools:
    def __init__(self, index: BM25Index, docs: dict[str, str], enable_code: bool = False):
        self.index = index
        self.docs = docs
        self.enable_code = enable_code

    @property
    def specs(self) -> list[dict]:
        return TOOL_SPECS + ([RUN_PYTHON_SPEC] if self.enable_code else [])

    def call(self, name: str, args: dict) -> ToolResult:
        handlers = {"search_docs": self._search, "read_page": self._read, "list_pages": self._list}
        if self.enable_code:
            handlers["run_python"] = self._run
        handler = handlers.get(name)
        if handler is None:
            return ToolResult(f"ERROR: unknown tool {name!r}. Available tools: {', '.join(handlers)}.", True)
        try:
            return handler(args or {})
        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(f"ERROR: bad arguments for {name}: {e}", True)

    def _run(self, args: dict) -> ToolResult:
        code = str(args["code"])
        if not code.strip():
            raise ValueError("code is empty")
        return run_python(code)

    def _search(self, args: dict) -> ToolResult:
        query = str(args["query"]).strip()
        if not query:
            raise ValueError("query is empty")
        k = max(1, min(int(args.get("k", 5)), 8))
        hits = self.index.search(query, k=k)
        if not hits:
            return ToolResult("No results. Try different keywords.")
        lines, pages = [], []
        for h in hits:
            snippet = re.sub(r"\s+", " ", h.chunk.text)[:400]
            where = h.chunk.path + (f" [{h.chunk.heading}]" if h.chunk.heading else "")
            lines.append(f"- {where}\n  {snippet}")
            if h.chunk.path not in pages:
                pages.append(h.chunk.path)
        return ToolResult("\n".join(lines), pages=pages)

    def _read(self, args: dict) -> ToolResult:
        path = str(args["path"]).strip().lstrip("/")
        if path not in self.docs:
            close = [p for p in self.docs if path.split("/")[-1].replace(".md", "") in p][:5]
            hint = f" Did you mean one of: {', '.join(close)}?" if close else " Use list_pages or search_docs to find paths."
            return ToolResult(f"ERROR: no page at {path!r}.{hint}", True)
        text = self.docs[path]
        offset = max(0, int(args.get("offset", 0) or 0))
        end = offset + MAX_PAGE_CHARS
        part, note = text[offset:end], ""
        if offset >= len(text):
            return ToolResult(f"ERROR: offset {offset} is past the end of the page ({len(text)} characters).", True)
        if end < len(text):
            note = f"\n\n[showing characters {offset}-{end} of {len(text)}; call read_page again with offset={end} to continue]"
        return ToolResult(part + note, pages=[path])

    def _list(self, args: dict) -> ToolResult:
        prefix = str(args.get("prefix", "")).lstrip("/")
        paths = [p for p in self.docs if p.startswith(prefix)]
        if not paths:
            return ToolResult(f"No pages start with {prefix!r}.")
        return ToolResult("\n".join(paths))
