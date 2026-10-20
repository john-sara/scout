"""Model providers behind one small interface.

The agent speaks a provider-neutral message format:

    {"role": "user", "content": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}

Each provider converts that to its own wire format. They call the HTTP APIs directly with `requests`,
so the project has no SDK dependency.
"""
from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import requests


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def complete(self, model: str, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> LLMResponse: ...


# ---------------------------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------------------------

def to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert neutral messages to Anthropic format. Consecutive tool results share one user turn."""
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            blocks = []
            if m.get("text"):
                blocks.append({"type": "text", "text": m["text"]})
            for tc in m.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
            out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if m.get("is_error"):
                block["is_error"] = True
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


class AnthropicProvider:
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str | None = None, retries: int = 4, timeout: float = 120):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.retries, self.timeout = retries, timeout

    def complete(self, model, system, messages, tools, max_tokens):
        body = {"model": model, "max_tokens": max_tokens, "system": system,
                "messages": to_anthropic_messages(messages)}
        if tools:
            body["tools"] = tools
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data = _post_with_retry(self.URL, headers, body, self.retries, self.timeout)
        text, calls = [], []
        for block in data.get("content", []):
            if block["type"] == "text":
                text.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block.get("input", {})))
        usage = data.get("usage", {})
        return LLMResponse("".join(text), calls, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                           data.get("stop_reason", ""))


def _post_with_retry(url, headers, body, retries, timeout, label="API"):
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        except requests.RequestException as e:
            if attempt == retries:
                raise ProviderError(f"{label} request failed: {e}") from e
        else:
            if r.status_code == 200:
                return r.json()
            retryable = r.status_code in (408, 409, 429, 500, 502, 503, 504, 529)
            if not retryable or attempt == retries:
                raise ProviderError(f"{label} returned {r.status_code}: {r.text[:300]}")
        time.sleep(delay)
        delay *= 2
    raise ProviderError("unreachable")


# ---------------------------------------------------------------------------------------------
# Ollama (local models)
# ---------------------------------------------------------------------------------------------

def to_ollama_messages(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("text", "")}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"function": {"name": tc.name, "arguments": tc.input}} for tc in m["tool_calls"]]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_name": m["name"], "content": m["content"]})
    return out


def to_ollama_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in tools]


class OllamaProvider:
    """Local models through Ollama's /api/chat. Needs a model with tool support (for example llama3.1)."""
    name = "ollama"
    _ids = itertools.count(1)

    def __init__(self, host: str | None = None, retries: int = 1, timeout: float = 300):
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.retries, self.timeout = retries, timeout

    def complete(self, model, system, messages, tools, max_tokens):
        body = {"model": model, "stream": False, "messages": to_ollama_messages(system, messages),
                "options": {"num_predict": max_tokens, "num_ctx": 8192}}
        if tools:
            body["tools"] = to_ollama_tools(tools)
        data = _post_with_retry(f"{self.host}/api/chat", {}, body, self.retries, self.timeout, label="Ollama")
        msg = data.get("message", {})
        calls = [ToolCall(f"call_{next(self._ids)}", c["function"]["name"], c["function"].get("arguments") or {})
                 for c in msg.get("tool_calls") or []]
        return LLMResponse(msg.get("content", "") or "", calls, data.get("prompt_eval_count", 0),
                           data.get("eval_count", 0), data.get("done_reason", ""))


# ---------------------------------------------------------------------------------------------
# Fake provider for tests
# ---------------------------------------------------------------------------------------------

class FakeProvider:
    """Plays back scripted responses and records what it was sent. Used only in tests."""
    name = "fake"

    def __init__(self, script: list[LLMResponse | Exception]):
        self.script = list(script)
        self.calls: list[dict] = []

    def complete(self, model, system, messages, tools, max_tokens):
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        if not self.script:
            raise ProviderError("fake provider ran out of scripted responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_provider(spec: str) -> tuple[Provider, str]:
    """Parse 'anthropic:claude-haiku-5-5' or 'ollama:llama3.1' into (provider, model)."""
    if ":" not in spec:
        raise ValueError(f"model spec {spec!r} must look like provider:model, for example anthropic:claude-haiku-5-5")
    kind, model = spec.split(":", 1)
    if kind == "anthropic":
        return AnthropicProvider(), model
    if kind == "ollama":
        return OllamaProvider(), model
    raise ValueError(f"unknown provider {kind!r}; use 'anthropic' or 'ollama'")
