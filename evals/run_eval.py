"""Run the agent over the labeled questions for each model and chunking strategy, then report
answer quality, citation accuracy, latency and cost.

    export ANTHROPIC_API_KEY=...          (set it in your shell, never in a file in this repo)
    python evals/run_eval.py --models anthropic:claude-haiku-5-5 anthropic:claude-sonnet-5-5

Answers are graded by a judge model against a short reference answer written from the docs.
Judging with the same model that produced an answer favors that model, so use a judge you are not
also testing when you can, and read the judge's reasons in the JSON output rather than trusting
the totals blindly.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scout import build_tools  # noqa: E402
from scout.agent import run_agent  # noqa: E402
from scout.llm import Provider, ProviderError, make_provider  # noqa: E402
from scout.pricing import cost_usd  # noqa: E402

JUDGE_SYSTEM = """You grade answers to questions about the FastAPI documentation.

You are given a question, a short reference answer, the relevant documentation pages (including pages the candidate cited), and a candidate answer.
Grade the candidate against the reference:
- correct: the candidate gives the same essential solution as the reference. Extra details are fine when the
  documentation pages support them, even if the reference does not mention them.
- partial: the candidate is on the right track but misses an essential part, or states something the
  documentation pages contradict.
- incorrect: the candidate is wrong, off topic, does not actually answer (for example it only says it will keep
  searching), or invents features that appear nowhere in the documentation pages.

If the reference says the documentation does not cover the topic, the candidate is correct only if it
says the documentation does not cover it (it may still add clearly labeled general advice). If it
confidently describes a feature or steps instead, it is incorrect.

Reply with only JSON: {"verdict": "correct" | "partial" | "incorrect", "reason": "<at most 25 words>"}"""

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def load_questions(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_verdict(text: str) -> tuple[str, str]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            verdict = str(data.get("verdict", "")).lower()
            if verdict in SCORES:
                return verdict, str(data.get("reason", ""))
        except json.JSONDecodeError:
            pass
    # A reply cut off mid-sentence can still contain a usable verdict.
    m = re.search(r'"verdict"\s*:\s*"(correct|partial|incorrect)"', text, re.I)
    if m:
        return m.group(1).lower(), "(reason cut off)"
    return "ungraded", f"judge reply could not be parsed: {text[:120]!r}"


def source_text(question: dict, docs: dict[str, str] | None, cited: list[str] | None = None,
                per_page: int = 3000, max_pages: int = 5) -> str:
    """Expected pages first, then pages the candidate cited, so the judge can check claims against them."""
    if not docs:
        return ""
    pages = []
    for p in list(question["expected_pages"][:2]) + list(cited or []):
        if p in docs and p not in pages:
            pages.append(p)
    return "\n\n".join(f"[{p}]\n{docs[p][:per_page]}" for p in pages[:max_pages])


def judge_answer(judge: Provider, judge_model: str, question: dict, answer: str, docs: dict[str, str] | None = None,
                 cited: list[str] | None = None) -> dict:
    if not answer.strip():
        return {"verdict": "incorrect", "reason": "empty answer", "input_tokens": 0, "output_tokens": 0}
    sources = source_text(question, docs, cited)
    prompt = (f"Question: {question['question']}\n\nReference answer: {question['reference']}\n\n"
              + (f"Documentation pages:\n{sources}\n\n" if sources else "")
              + f"Candidate answer:\n{answer}")
    in_tok = out_tok = 0
    for attempt in range(2):  # one retry if the reply is empty or unparseable
        resp = judge.complete(judge_model, JUDGE_SYSTEM, [{"role": "user", "content": prompt}], [], 700)
        in_tok += resp.input_tokens
        out_tok += resp.output_tokens
        verdict, reason = parse_verdict(resp.text)
        if verdict != "ungraded":
            break
    return {"verdict": verdict, "reason": reason, "input_tokens": in_tok, "output_tokens": out_tok}


def grade_row(question: dict, result: dict, verdict: str) -> dict:
    expected = set(question["expected_pages"])
    answerable = bool(expected)
    cited, retrieved = set(result["cited_pages"]), set(result["retrieved_pages"])
    return {
        "answerable": answerable,
        "verdict": verdict,
        "score": SCORES.get(verdict),
        "citation_hit": bool(cited & expected) if answerable else None,
        "retrieval_hit": bool(retrieved & expected) if answerable else None,
    }


def _pct(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def summarize(rows: list[dict]) -> dict:
    """rows: one dict per question with keys from grade_row plus latency_s, cost_usd, tokens, stopped, steps."""
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    graded = [r for r in rows if r["score"] is not None]
    return {
        "n": len(rows),
        "accuracy_answerable": _pct([r["score"] for r in ans if r["score"] is not None]),
        "abstention_accuracy": _pct([r["score"] for r in unans if r["score"] is not None]),
        "overall_accuracy": _pct([r["score"] for r in graded]),
        "citation_hit": _pct([r["citation_hit"] for r in ans]),
        "retrieval_hit": _pct([r["retrieval_hit"] for r in ans]),
        "latency_p50_s": _percentile([r["latency_s"] for r in rows], 0.5),
        "latency_p95_s": _percentile([r["latency_s"] for r in rows], 0.95),
        "mean_tokens": statistics.mean(r["input_tokens"] + r["output_tokens"] for r in rows) if rows else None,
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in rows) if rows else None,
        "mean_model_calls": statistics.mean(r["model_calls"] for r in rows) if rows else None,
        "ungraded": sum(1 for r in rows if r["score"] is None),
        "errors": sum(1 for r in rows if r["stopped"] == "error"),
        "limit_hits": sum(1 for r in rows if r["stopped"] in ("step_limit", "token_budget")),
    }


def _fmt(v, kind):
    if v is None:
        return "n/a"
    return {"pct": f"{v:.0%}", "s": f"{v:.1f}s", "int": f"{v:,.0f}", "usd": f"${v:.4f}", "f": f"{v:.1f}"}[kind]


def render_markdown(summaries: dict[str, dict], failures: dict[str, list[dict]], meta: dict) -> str:
    out = [f"Run {meta['timestamp']}, {meta['n_questions']} questions, judge `{meta['judge']}`.", "",
           "| Config | Answer score | Abstains when it should | Cites right page | Found right page | p50 latency | p95 latency | Mean tokens | Mean cost / question | Model calls |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summaries.items():
        out.append(f"| {name} | {_fmt(s['accuracy_answerable'], 'pct')} | {_fmt(s['abstention_accuracy'], 'pct')} | "
                   f"{_fmt(s['citation_hit'], 'pct')} | {_fmt(s['retrieval_hit'], 'pct')} | "
                   f"{_fmt(s['latency_p50_s'], 's')} | {_fmt(s['latency_p95_s'], 's')} | "
                   f"{_fmt(s['mean_tokens'], 'int')} | {_fmt(s['mean_cost_usd'], 'usd')} | {_fmt(s['mean_model_calls'], 'f')} |")
    out += ["", "Answer score counts correct as 1 and partial as 0.5, on questions the docs can answer.", ""]
    for name, fails in failures.items():
        out.append(f"### Not fully correct: {name}")
        out.append("")
        for f in fails:
            out.append(f"- {f['id']} ({f['verdict']}): {f['question']} Judge: {f['reason']}")
        if not fails:
            out.append("- none")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="provider:model, e.g. anthropic:claude-haiku-5-5 ollama:llama3.1")
    ap.add_argument("--judge", default="anthropic:claude-sonnet-5-5")
    ap.add_argument("--chunking", nargs="+", default=["headings"], choices=["fixed", "headings"])
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--limit", type=int, help="only the first N questions (for a cheap smoke test)")
    ap.add_argument("--enable-code", action="store_true", help="give the agent the run_python tool (runs model-written code on this machine)")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--token-budget", type=int, default=40_000)
    ap.add_argument("--max-cost", type=float, default=5.0, help="abort if total spend passes this many dollars")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    questions = load_questions(args.questions)[: args.limit]
    judge, judge_model = make_provider(args.judge)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    spent = 0.0
    judge_failures = 0
    all_rows: dict[str, list[dict]] = {}
    records: list[dict] = []
    for chunking in args.chunking:
        tools = build_tools(chunking, enable_code=args.enable_code)
        for spec in args.models:
            provider, model = make_provider(spec)
            name = f"{spec} / {chunking}"
            print(f"\n== {name}")
            rows = []
            for q in questions:
                result = run_agent(q["question"], provider, model, tools, args.max_steps, args.token_budget).to_dict()
                try:
                    j = judge_answer(judge, judge_model, q, result["answer"], tools.docs, result["cited_pages"]) if result["stopped"] != "error" else \
                        {"verdict": "ungraded", "reason": result["error"], "input_tokens": 0, "output_tokens": 0}
                    judge_failures = 0
                except ProviderError as e:
                    j = {"verdict": "ungraded", "reason": f"judge failed: {e}", "input_tokens": 0, "output_tokens": 0}
                    judge_failures += 1
                    if judge_failures >= 3:
                        raise SystemExit(f"Stopping: the judge failed 3 times in a row ({e}). Check your API key and credits.")
                spent += result["cost_usd"] + cost_usd(judge_model, j["input_tokens"], j["output_tokens"])
                row = grade_row(q, result, j["verdict"])
                row.update(id=q["id"], question=q["question"], reason=j["reason"], latency_s=result["latency_s"],
                           cost_usd=result["cost_usd"], input_tokens=result["input_tokens"],
                           output_tokens=result["output_tokens"], stopped=result["stopped"],
                           model_calls=sum(1 for _ in {s["step"] for s in result["steps"]}))
                rows.append(row)
                records.append({"config": name, "row": row, "answer": result["answer"], "cited": result["cited_pages"],
                                "retrieved": result["retrieved_pages"], "steps": result["steps"]})
                print(f"  {q['id']} {j['verdict']:<9} {result['latency_s']:5.1f}s  ${result['cost_usd']:.4f}  (total ${spent:.2f})")
                if spent > args.max_cost:
                    print(f"Stopping: spend ${spent:.2f} passed --max-cost ${args.max_cost:.2f}")
                    break
                time.sleep(0.2)
            all_rows[name] = rows
            if spent > args.max_cost:
                break
        if spent > args.max_cost:
            break

    summaries = {n: summarize(r) for n, r in all_rows.items()}
    failures = {n: [x for x in r if x["verdict"] in ("partial", "incorrect", "ungraded")] for n, r in all_rows.items()}
    meta = {"timestamp": stamp, "n_questions": len(questions), "judge": args.judge,
            "max_steps": args.max_steps, "code_tool": args.enable_code, "token_budget": args.token_budget, "total_cost_usd": spent}
    (out_dir / f"run-{stamp}.json").write_text(json.dumps({"meta": meta, "summaries": summaries, "records": records}, indent=2))
    md = render_markdown(summaries, failures, meta)
    (out_dir / "latest.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nTotal spend this run (including judge): ${spent:.2f}")


if __name__ == "__main__":
    main()
