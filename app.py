"""Scout web app. Run with:  python run.py   (then open http://localhost:8000)

Environment:
  ANTHROPIC_API_KEY    enables the Claude models
  SCOUT_CLAUDE_MODELS  comma list, default claude-haiku-5-5,claude-sonnet-5-5
  SCOUT_RETRIEVAL      bm25 (default), embed or hybrid; the last two need Ollama running an embedding model
  SCOUT_ENABLE_CODE    set to 1 to let the agent run Python snippets on this machine (off by default)
  OLLAMA_HOST          where Ollama listens, default http://localhost:11434 (local models appear if it is running)

Settings for a public demo:
  SCOUT_DISABLE_OLLAMA     set to 1 to skip looking for local models
  SCOUT_DAILY_BUDGET_USD   stop answering for the rest of the UTC day once this much has been spent
  SCOUT_TRUST_PROXY        set to 1 when behind one reverse proxy, so the rate limit uses the real client address

The app spends money on your API key for every question. It is meant to run on your own machine;
if you ever expose it, keep the rate limit and add real authentication first.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scout import build_tools
from scout.agent import run_agent
from scout.llm import ProviderError, make_provider

FRONTEND = Path(__file__).parent / "frontend"
RESULTS = Path(__file__).parent / "results"
MAX_QUESTION_CHARS = 500
RATE_LIMIT = 12  # questions per minute per client
GLOBAL_RATE_LIMIT = 40  # questions per minute across all clients

app = FastAPI(title="Scout", description="Ask the FastAPI docs, with citations.")


@lru_cache(maxsize=4)
def tools_for(chunking: str):
    return build_tools(chunking, enable_code=os.environ.get("SCOUT_ENABLE_CODE") == "1",
                       retrieval=os.environ.get("SCOUT_RETRIEVAL", "bm25"))


def available_models() -> list[str]:
    models = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        names = os.environ.get("SCOUT_CLAUDE_MODELS", "claude-haiku-5-5,claude-sonnet-5-5").split(",")
        models += [f"anthropic:{n.strip()}" for n in names if n.strip()]
    if os.environ.get("SCOUT_DISABLE_OLLAMA") == "1":
        return models
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


def _allow(key: str, limit: int) -> bool:
    now = time.monotonic()
    q = _hits[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        return False
    q.append(now)
    return True


def check_rate_limit(client: str) -> None:
    if not _allow(client, RATE_LIMIT) or not _allow("*all*", GLOBAL_RATE_LIMIT):
        raise HTTPException(429, "Too many questions, wait a minute.")


def client_id(request: Request) -> str:
    """The caller's address. Behind a proxy, use the last X-Forwarded-For entry, which the proxy added itself."""
    if os.environ.get("SCOUT_TRUST_PROXY") == "1":
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded.strip():
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


_spend = {"day": "", "usd": 0.0}


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def daily_budget() -> float | None:
    try:
        return float(os.environ["SCOUT_DAILY_BUDGET_USD"])
    except (KeyError, ValueError):
        return None


def check_budget() -> None:
    cap = daily_budget()
    if cap is None:
        return
    if _spend["day"] != _today():
        _spend.update(day=_today(), usd=0.0)
    if _spend["usd"] >= cap:
        raise HTTPException(429, "This demo's budget for today is used up. Try again tomorrow, or run Scout yourself from GitHub.")


def record_spend(usd: float) -> None:
    if daily_budget() is not None:
        if _spend["day"] != _today():
            _spend.update(day=_today(), usd=0.0)
        _spend["usd"] += usd


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)
    model: str
    chunking: str = Field(default="headings", pattern="^(fixed|headings)$")


@app.get("/api/config")
def config():
    return {"models": available_models(), "chunking": ["headings", "fixed"], "demo": daily_budget() is not None}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/ask")
def ask(body: AskRequest, request: Request):
    check_rate_limit(client_id(request))
    check_budget()
    if body.model not in available_models():
        raise HTTPException(400, f"Model {body.model!r} is not available. Options: {available_models()}")
    try:
        provider, model = make_provider(body.model)
    except (ValueError, ProviderError) as e:
        raise HTTPException(400, str(e))
    result = run_agent(body.question, provider, model, tools_for(body.chunking))
    record_spend(result.cost_usd)
    return result.to_dict()


def _published() -> Path:
    """A committed snapshot of the final benchmark, used before any local runs when present."""
    return RESULTS / "published"


def _latest_run(*dirs: Path):
    for d in dirs:
        files = sorted(d.glob("run-*.json"), key=lambda p: p.stat().st_mtime) if d.exists() else []
        if files:
            return json.loads(files[-1].read_text(encoding="utf-8"))
    return None


@app.get("/api/eval")
def eval_summary():
    """Benchmark numbers from the most recent saved runs, for the strip at the bottom of the page."""
    main = _latest_run(_published(), RESULTS / "v2", RESULTS)
    if not main:
        return {"available": False}
    local = _latest_run(_published() / "ollama", RESULTS / "ollama")
    summaries = dict(main["summaries"])
    if local:
        summaries.update(local["summaries"])
    models = [{"name": name, "score": s["accuracy_answerable"], "abstain": s["abstention_accuracy"],
               "citation": s["citation_hit"], "latency_p50": s["latency_p50_s"], "cost": s["mean_cost_usd"]}
              for name, s in summaries.items()]
    return {"available": True, "n_questions": main["meta"]["n_questions"], "judge": main["meta"]["judge"],
            "models": models, "report": _report_path() is not None}


def _report_path():
    for path in (_published() / "report.html", RESULTS / "report.html"):
        if path.exists():
            return path
    return None


@app.get("/report")
def report():
    path = _report_path()
    if path is None:
        raise HTTPException(404, "No report yet. Run evals/report.py.")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
