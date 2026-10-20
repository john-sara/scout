"""Re-grade the saved answers from an earlier run with the current judge. Candidate models are not
called again, so this costs only the judge calls.

    python evals/regrade.py results/run-20261020-030107.json --judge anthropic:claude-opus-5-5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.run_eval import grade_row, judge_answer, load_questions, render_markdown, summarize  # noqa: E402
from scout.chunking import load_docs  # noqa: E402
from scout.llm import ProviderError, make_provider  # noqa: E402
from scout.pricing import cost_usd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_json")
    ap.add_argument("--judge", default="anthropic:claude-opus-5-5")
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--max-cost", type=float, default=1.5)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--write-run", action="store_true",
                    help="also save a full run file with the new grades, so the app and report use it")
    args = ap.parse_args()

    run = json.loads(Path(args.run_json).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions(args.questions)}
    docs = load_docs()
    judge, judge_model = make_provider(args.judge)

    spent, rows_by_config, failures = 0.0, {}, {}
    judge_failures = 0
    for rec in run["records"]:
        old = rec["row"]
        q = questions[old["id"]]
        try:
            j = judge_answer(judge, judge_model, q, rec["answer"], docs, rec["cited"])
        except ProviderError as e:
            j = {"verdict": "ungraded", "reason": f"judge failed: {e}", "input_tokens": 0, "output_tokens": 0}
            judge_failures += 1
            if judge_failures >= 3:
                raise SystemExit(f"Stopping: the judge failed 3 times in a row ({e}). Check your API key and credits.")
        else:
            judge_failures = 0
        spent += cost_usd(judge_model, j["input_tokens"], j["output_tokens"])
        result = {"cited_pages": rec["cited"], "retrieved_pages": rec["retrieved"], "steps": rec["steps"]}
        row = grade_row(q, result, j["verdict"])
        row.update({k: old[k] for k in ("id", "question", "latency_s", "cost_usd", "input_tokens", "output_tokens",
                                        "stopped", "model_calls")}, reason=j["reason"], previous_verdict=old["verdict"])
        rows_by_config.setdefault(rec["config"], []).append(row)
        changed = "" if old["verdict"] == j["verdict"] else f"  (was {old['verdict']})"
        print(f"{rec['config']}  {old['id']} {j['verdict']}{changed}  spend ${spent:.2f}")
        if spent > args.max_cost:
            print(f"Stopping: judge spend passed --max-cost ${args.max_cost:.2f}")
            break

    summaries = {n: summarize(r) for n, r in rows_by_config.items()}
    for n, r in rows_by_config.items():
        failures[n] = [x for x in r if x["verdict"] in ("partial", "incorrect", "ungraded")]
    meta = {"timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"), "n_questions": len(questions), "judge": args.judge,
            "regraded_from": args.run_json, "judge_cost_usd": spent}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"regraded-{meta['timestamp']}.json").write_text(json.dumps({"meta": meta, "summaries": summaries,
                                                                         "rows": rows_by_config}, indent=2))
    md = render_markdown(summaries, failures, meta)
    if args.write_run:
        new_rows = {n: {r["id"]: r for r in rows} for n, rows in rows_by_config.items()}
        records = []
        for rec in run["records"]:
            row = new_rows.get(rec["config"], {}).get(rec["row"]["id"])
            if row is not None:
                records.append({**rec, "row": {k: v for k, v in row.items() if k != "previous_verdict"}})
        run_meta = {**run["meta"], "timestamp": meta["timestamp"], "judge": args.judge,
                    "regraded_from": args.run_json, "judge_cost_usd": spent}
        (out / f"run-{meta['timestamp']}.json").write_text(
            json.dumps({"meta": run_meta, "summaries": summaries, "records": records}, indent=2))
        (out / "latest.md").write_text(md, encoding="utf-8")
    (out / "regraded.md").write_text(md, encoding="utf-8")
    print("\n" + md + f"\n\nJudge spend: ${spent:.2f}")


if __name__ == "__main__":
    main()
