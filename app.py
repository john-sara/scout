"""Scout web app. Run with:  python run.py   (then open http://localhost:8000)

Environment:
  ANTHROPIC_API_KEY    enables the Claude models
  SCOUT_CLAUDE_MODELS  comma list, default claude-haiku-5-5,claude-sonnet-5-5
  SCOUT_ENABLE_CODE    set to 1 to let the agent run Python snippets on this machine (off by default)
  OLLAMA_HOST          where Ollama listens, default http://localhost:11434 (local models appear if it is running)

The app spends money on your API key for every question. It is meant to run on your own machine;
if you ever expose it, keep the rate limit and add real authentication first.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scout import build_tools
from scout.agent import run_agent
from scout.llm import ProviderError, make_provider

FRONTEND = Path(__file__).parent / "frontend"
MAX_QUESTION_CHARS = 500
RATE_LIMIT = 12  # questions per minute per client

app = FastAPI(title="Scout", description="Ask the FastAPI docs, with citations.")


@lru_cache(maxsize=4)
def tools_for(chunking: str):
    return build_tools(chunking, enable_code=os.environ.get("SCOUT_ENABLE_CODE") == "1")


def available_models() -> list[str]:
    models = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        names = os.environ.get("SCOUT_CLAUDE_MODELS", "claude-haiku-5-5,claude-sonnet-5-5").split(",")
        models += [f"anthropic:{n.strip()}" for n in names if n.strip()]
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=0.5)
        models += [f"ollama:{m['name']}" for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError):
        pass
    return models


_hits: dict[str, deque] = defaultdict(deque)


def check_rate_limit(client: str) -> None:
    now = time.monotonic()
    q = _hits[client]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, "Too many questions, wait a minute.")
    q.append(now)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    model: str
    chunking: str = Field(default="headings", pattern="^(fixed|headings)$")


@app.get("/api/config")
def config():
    return {"models": available_models(), "chunking": ["headings", "fixed"]}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request):
    check_rate_limit(request.client.host if request.client else "unknown")
    if body.model not in available_models():
        raise HTTPException(400, f"Model {body.model!r} is not available. Options: {available_models()}")
    try:
        provider, model = make_provider(body.model)
    except (ValueError, ProviderError) as e:
        raise HTTPException(400, str(e))
    result = run_agent(body.question, provider, model, tools_for(body.chunking))
    return result.to_dict()


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
