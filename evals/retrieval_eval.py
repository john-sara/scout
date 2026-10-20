"""Compare chunking strategies on retrieval alone. Needs no API key and runs in seconds.

For each answerable question, the page ranking is the best-chunk rank per page. A question counts as
a hit at k when any of its expected pages is in the top k pages.

    python evals/retrieval_eval.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scout.chunking import STRATEGIES, build_chunks, load_docs  # noqa: E402
from scout.retrieval import BM25Index  # noqa: E402


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(index, questions, ks=(1, 3, 5)):
    answerable = [q for q in questions if q["expected_pages"]]
    hits = {k: 0 for k in ks}
    rr_total = 0.0
    misses = []
    for q in answerable:
        ranked = index.search_pages(q["question"], k=10)
        expected = set(q["expected_pages"])
        rank = next((i for i, p in enumerate(ranked, 1) if p in expected), None)
        if rank:
            rr_total += 1 / rank
        for k in ks:
            if rank and rank <= k:
                hits[k] += 1
        if not rank or rank > 3:
            misses.append({"id": q["id"], "question": q["question"], "expected": q["expected_pages"], "top3": ranked[:3]})
    n = len(answerable)
    return {"n": n, **{f"hit@{k}": hits[k] / n for k in ks}, "mrr": rr_total / n}, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "retrieval.md"))
    args = ap.parse_args()

    docs = load_docs()
    questions = load_questions(args.questions)
    lines = [
        "| Chunking | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    all_misses = {}
    for name in STRATEGIES:
        chunks = build_chunks(docs, name)
        metrics, misses = evaluate(BM25Index(chunks), questions)
        all_misses[name] = misses
        lines.append(
            f"| {name} | {len(chunks)} | {metrics['hit@1']:.0%} | {metrics['hit@3']:.0%} | "
            f"{metrics['hit@5']:.0%} | {metrics['mrr']:.2f} |"
        )
    n = metrics["n"]
    out = [f"Retrieval on {n} answerable questions over {len(docs)} pages.", "", *lines, ""]
    for name, misses in all_misses.items():
        out.append(f"### Questions not found in the top 3 pages ({name})")
        out.append("")
        for m in misses:
            out.append(f"- {m['id']}: {m['question']}  expected {m['expected']}, got {m['top3']}")
        if not misses:
            out.append("- none")
        out.append("")
    text = "\n".join(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
