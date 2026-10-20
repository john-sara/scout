# Scout

Scout answers questions about the FastAPI documentation with an agent that searches the docs, reads pages, and cites its sources. The interesting part is the evaluation harness around it: a labeled question set, two chunking strategies, and a model comparison that reports answer quality, citation accuracy, latency and cost.

## How it works

The agent gets three tools (`search_docs`, `read_page`, `list_pages`) and calls them in a loop until it can answer. Each run has a step limit and a token budget, tool failures go back to the model as text so it can recover, and every run records what it did. An optional `run_python` tool lets it check an answer by running a snippet. That tool is off by default.

Retrieval is BM25 written from scratch over chunks of the docs, so the whole retrieval stage is under 100 lines with no dependencies. The model providers call the Claude API and Ollama over plain HTTP, so you can compare hosted and local models with the same harness.

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_corpus.py          # downloads the docs at a pinned commit
export ANTHROPIC_API_KEY=...            # or have Ollama running for local models
python run.py                           # http://localhost:8000
```

Run the tests (no key or network needed):

```bash
python -m unittest discover -s tests -t .
```

## Evaluation

The question set is in `evals/questions.jsonl`: 30 questions the docs answer, each with the page or pages that contain the answer and a short reference answer, plus 3 the docs do not cover (rate limiting, a default body size limit, gRPC). The unanswerable ones check whether a model admits the gap or makes something up.

### Retrieval only

`python evals/retrieval_eval.py` needs no API key. A question counts as a hit at k when one of its expected pages is in the top k pages.

| Chunking | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| fixed (200 words) | 695 | 57% | 77% | 87% | 0.69 |
| headings | 1142 | 63% | 80% | 93% | 0.75 |

Splitting on headings does better at every depth, and it also puts the section title into the indexed text, which helps queries that name the topic. Both strategies miss the same kind of question: ones phrased in words the page never uses, like "turn off the interactive docs" for a page about conditional OpenAPI. A keyword index cannot bridge that, and embeddings would be the next thing to try.

### Models

```bash
python evals/run_eval.py --models anthropic:claude-haiku-5-5 anthropic:claude-sonnet-5-5
python evals/run_eval.py --models ollama:llama3.1 --judge anthropic:claude-sonnet-5-5
```

Each question is run through the agent and graded by a judge model against the reference answer. The report has answer score, whether it abstains on unanswerable questions, whether it cites a correct page, p50 and p95 latency, tokens, cost per question, and a list of every answer that was not fully correct with the judge's reason. `--max-cost` stops the run if spending passes a limit.

Results from the latest run go here.

## Limits

- The 33 questions and reference answers were written from the docs and have not been independently reviewed. A few retrieval "misses" are arguably fine, since more than one page can answer a question.
- The judge is an LLM. It favors answers written by the same model, so judge with a different model when you can, and read the reasons in the JSON output.
- Retrieval is lexical only. It cannot match a question that shares no words with the right page.
- `run_python` is not a sandbox. It uses a separate process with a timeout and a stripped environment, but the code runs as you. Leave it off unless you are on a machine you control.
- The web app has a per-client rate limit and no login. It spends your API key on every question, so run it locally.
- Cost numbers use the prices in `scout/pricing.py`, checked on 2026-10-20. Check them before quoting.

## Layout

```
scout/        chunking, BM25 retrieval, tools, model providers, agent loop, pricing
evals/        question set, retrieval eval, model eval harness
tests/        unit tests that run offline with a fake model
app.py        FastAPI app, frontend/ has the page
```

The documentation in `data/` is from [fastapi/fastapi](https://github.com/fastapi/fastapi) (MIT licensed) and is downloaded, not stored in this repo.
