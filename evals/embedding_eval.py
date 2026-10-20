"""Compare BM25, embeddings and hybrid retrieval on the labeled questions. Free to run: it needs Ollama
with an embedding model, and no API key.

    ollama pull nomic-embed-text
    python evals/embedding_eval.py

Writes results/embeddings.md (tables and the questions each method gets wrong) and results/embeddings.json
(the page ranking each method gives for every question, so changes can be inspected later).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.retrieval_eval import evaluate, load_questions  # noqa: E402
from scout.chunking import STRATEGIES, build_chunks, load_docs  # noqa: E402
from scout.embeddings import EmbeddingIndex, HybridIndex, OllamaEmbedder  # noqa: E402
from scout.llm import ProviderError  # noqa: E402
from scout.retrieval import BM25Index  # noqa: E402

LABELS = {"bm25": "BM25", "embed": "Embeddings", "hybrid": "Hybrid (BM25 + embeddings)"}


def rankings(index, questions, k=10):
    return {q["id"]: index.search_pages(q["question"], k=k) for q in questions if q["expected_pages"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nomic-embed-text")
    ap.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    embedder = OllamaEmbedder(args.model)
    try:
        embedder.embed_query("test")
    except ProviderError as e:
        raise SystemExit(f"Could not get an embedding from Ollama ({e}).\n"
                         f"Make sure Ollama is running and the model is installed: ollama pull {args.model}")

    docs = load_docs()
    questions = load_questions(args.questions)
    answerable = [q for q in questions if q["expected_pages"]]
    table = ["| Chunking | Retrieval | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|---|"]
    detail, saved = [], {}
    for chunking in STRATEGIES:
        chunks = build_chunks(docs, chunking)
        t0 = time.time()
        emb = EmbeddingIndex.build(chunks, embedder,
                                   progress=lambda done, total, c=chunking: print(f"  embedding {c}: {done}/{total}", end="\r"))
        print(f"\n{chunking}: {len(chunks)} chunks embedded in {time.time() - t0:.0f}s (cached for next time)")
        bm25 = BM25Index(chunks)
        indexes = {"bm25": bm25, "embed": emb, "hybrid": HybridIndex(bm25, emb)}
        ranks = {name: rankings(ix, questions) for name, ix in indexes.items()}
        saved[chunking] = ranks
        for name, ix in indexes.items():
            m, misses = evaluate(ix, questions)
            table.append(f"| {chunking} | {LABELS[name]} | {m['hit@1']:.0%} | {m['hit@3']:.0%} | {m['hit@5']:.0%} | {m['mrr']:.2f} |")
            detail.append(f"### Not in the top 3 pages: {LABELS[name]}, {chunking} chunking\n")
            detail.extend(f"- {x['id']}: {x['question']}  expected {x['expected']}, got {x['top3']}" for x in misses)
            if not misses:
                detail.append("- none")
            detail.append("")
        # questions where the methods disagree about finding the page in the top 5
        diff = ["### Where the methods differ (right page in top 5)", ""]
        for q in answerable:
            exp = set(q["expected_pages"])
            found = {n: any(p in exp for p in ranks[n][q["id"]][:5]) for n in indexes}
            if len(set(found.values())) > 1:
                who = ", ".join(LABELS[n] for n, f in found.items() if f) or "none"
                diff.append(f"- {q['id']}: {q['question']}  found by: {who}")
        detail.extend(diff + [""])
    n = len(answerable)
    text = "\n".join([f"Retrieval on {n} answerable questions over {len(docs)} pages, embeddings from `{args.model}`.", "",
                      *table, "", *detail])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "embeddings.md").write_text(text, encoding="utf-8")
    (out / "embeddings.json").write_text(json.dumps({"model": args.model, "rankings": saved}, indent=2))
    print("\n" + text)


if __name__ == "__main__":
    main()
