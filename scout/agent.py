"""The agent loop: let the model call doc tools until it answers, within a step and token budget."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .llm import Provider, ProviderError
from .pricing import cost_usd
from .tools import DocTools

SYSTEM_PROMPT = """You answer questions about the FastAPI documentation.

Always search the docs before answering, and base your answer on what you find there. Cite the pages you used \
inline with their path in square brackets, for example [tutorial/body.md]. Keep answers short and practical, with a \
small code snippet when it helps.

If the documentation does not cover the question, say so plainly. Do not invent features, options or behavior."""

CODE_NOTE = """

You also have a run_python tool. When an answer depends on exact behavior, you may run a short snippet to \
check it before answering. Do not run code that touches files, the network or the system."""

FINAL_STEP_NOTE = ("You have no tool calls left. Write your final answer now using what you have already found, "
                   "citing the pages you used. If something is missing, say what is missing.")

_CITATION = re.compile(r"\[([\w\-./]+\.md)\]")


@dataclass
class AgentResult:
    answer: str
    stopped: str  # answered | step_limit | token_budget | error
    cited_pages: list[str] = field(default_factory=list)
    retrieved_pages: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def extract_citations(text: str, valid_pages: set[str]) -> list[str]:
    seen: list[str] = []
    for m in _CITATION.finditer(text):
        path = m.group(1)
        if path in valid_pages and path not in seen:
            seen.append(path)
    return seen


def run_agent(
    question: str,
    provider: Provider,
    model: str,
    tools: DocTools,
    max_steps: int = 6,
    token_budget: int = 40_000,
    max_tokens: int = 1024,
    system: str | None = None,
) -> AgentResult:
    """Run one question to completion. Never raises: failures are reported in the result.

    max_steps caps model calls. token_budget caps total (input + output) tokens across the conversation,
    which bounds cost. The run stops as soon as either limit is hit.
    """
    t0 = time.perf_counter()
    if system is None:
        system = SYSTEM_PROMPT + (CODE_NOTE if tools.enable_code else "")
    messages: list[dict] = [{"role": "user", "content": question}]
    res = AgentResult(answer="", stopped="answered")
    last_text = ""

    for step in range(1, max_steps + 1):
        if step == max_steps and step > 1:
            messages.append({"role": "user", "content": FINAL_STEP_NOTE})
        try:
            resp = provider.complete(model, system, messages, tools.specs, max_tokens)
        except ProviderError as e:
            res.stopped, res.error = "error", str(e)
            res.steps.append({"step": step, "kind": "error", "detail": str(e)})
            break

        res.input_tokens += resp.input_tokens
        res.output_tokens += resp.output_tokens
        if resp.text:
            last_text = resp.text

        if not resp.tool_calls:
            res.answer = resp.text
            res.stopped = "answered"
            res.steps.append({"step": step, "kind": "answer", "tokens": [resp.input_tokens, resp.output_tokens]})
            break

        messages.append({"role": "assistant", "text": resp.text, "tool_calls": resp.tool_calls})
        for call in resp.tool_calls:
            result = tools.call(call.name, call.input)
            for p in result.pages:
                if p not in res.retrieved_pages:
                    res.retrieved_pages.append(p)
            res.steps.append({
                "step": step, "kind": "tool", "tool": call.name, "input": call.input,
                "ok": not result.is_error, "result_chars": len(result.text), "pages": result.pages,
            })
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                             "content": result.text, "is_error": result.is_error})

        if res.input_tokens + res.output_tokens >= token_budget:
            res.stopped = "token_budget"
            res.answer = last_text
            break
    else:
        res.stopped = "step_limit"
        res.answer = last_text

    res.latency_s = time.perf_counter() - t0
    res.cost_usd = cost_usd(model, res.input_tokens, res.output_tokens)
    res.cited_pages = extract_citations(res.answer, set(tools.docs))
    return res
