"""Turn the evaluation results into one clean, screenshot-ready page: results/report.html

    python evals/report.py

Reads whatever exists in results/:
  v2/run-*.json        the main Claude run (falls back to results/run-*.json)
  ollama/run-*.json    the local model run
  regraded-*.json      the first run, re-graded with the fixed judge (the "before" column)
and recomputes the retrieval comparison from the question set.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.retrieval_eval import evaluate, load_questions  # noqa: E402
from scout.chunking import STRATEGIES, build_chunks, load_docs  # noqa: E402
from scout.retrieval import BM25Index  # noqa: E402

CORPUS_COMMIT = "03002a0"


def latest(paths):
    paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    return json.loads(paths[-1].read_text(encoding="utf-8")) if paths else None


def short(name: str) -> str:
    """'anthropic:claude-haiku-5-5 / headings' -> 'Claude Haiku 5.5'"""
    model = name.split(" / ")[0].split(":", 1)[-1]
    if model.startswith("claude-"):
        parts = model.split("-")[1:]
        return "Claude " + parts[0].capitalize() + " " + ".".join(parts[1:])
    return model


def pct(v, digits=0):
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def nice_date(stamp: str) -> str:
    from datetime import datetime
    return datetime.strptime(stamp[:8], "%Y%m%d").strftime("%b %d, %Y").replace(" 0", " ")


def esc(s):
    return html.escape(str(s))


def load_results(results: Path):
    main = latest(list((results / "v2").glob("run-*.json"))) or latest(list(results.glob("run-*.json")))
    local = latest(list((results / "ollama").glob("run-*.json")))
    before = latest(list(results.glob("regraded-*.json"))) if (results / "v2").exists() else None
    return main, local, before


LABELS = {"bm25": "BM25", "embed": "Embeddings", "hybrid": "Hybrid"}


def load_embedding_rankings(results: Path):
    for path in (results / "published" / "embeddings.json", results / "embeddings.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def metrics_from_rankings(rankings: dict, questions: list, ks=(1, 3, 5)):
    answerable = [q for q in questions if q["expected_pages"]]
    hits, rr = {k: 0 for k in ks}, 0.0
    for q in answerable:
        ranked = rankings.get(q["id"], [])
        rank = next((i for i, p in enumerate(ranked, 1) if p in set(q["expected_pages"])), None)
        if rank:
            rr += 1 / rank
            for k in ks:
                hits[k] += rank <= k
    n = len(answerable)
    return {f"hit@{k}": hits[k] / n for k in ks} | {"mrr": rr / n}


def retrieval_rows():
    docs = load_docs()
    questions = load_questions(ROOT / "evals" / "questions.jsonl")
    out = []
    for name in STRATEGIES:
        chunks = build_chunks(docs, name)
        metrics, _ = evaluate(BM25Index(chunks), questions)
        out.append((name, len(chunks), metrics))
    return out, len(docs)


CSS = """
:root{--bg:#fff;--ink:#0d0d0d;--body:#2b2b2b;--muted:#6b6b6b;--hair:#e8e8e8;--soft:#f5f5f4;--lime:#d4f45a;--night:#111;--banner:#c9d3ea;
--c1:#ff9d70;--c2:#b8aeff;--c3:#d4f45a;--c4:#9fd8ff;--c5:#ffb5d0;--c6:#ffe37a;
--display:'Inter Tight','Inter',-apple-system,sans-serif;--ui:'Inter',-apple-system,'Segoe UI',sans-serif}
*{box-sizing:border-box;margin:0;padding:0}
html{-webkit-font-smoothing:antialiased}
body{font-family:var(--ui);background:var(--bg);color:var(--ink);line-height:1.5}
.wrap{max-width:1160px;margin:0 auto;padding:0 32px}
.nav{border-bottom:1px solid var(--hair)}
.nav .wrap{display:flex;align-items:center;justify-content:space-between;height:64px}
.logo{display:flex;align-items:center;gap:10px;font-family:var(--display);font-weight:600;font-size:19px;letter-spacing:-.03em;text-decoration:none;color:var(--ink)}
.mark{position:relative;width:22px;height:22px}
.mark i{position:absolute;width:13px;height:13px;border-radius:4px}
.mark i:nth-child(1){left:0;top:0;background:var(--c2)}
.mark i:nth-child(2){left:6px;top:4px;background:var(--c1)}
.mark i:nth-child(3){left:9px;top:9px;background:var(--ink)}
.pill{display:inline-flex;align-items:center;justify-content:center;height:40px;padding:0 20px;border-radius:10px;border:1px solid var(--ink);background:var(--ink);font-size:14px;font-weight:500;text-decoration:none;color:#fff}
.pill:hover{background:#000}
.hero{padding:64px 0 8px}
h1{font-family:var(--display);font-weight:500;font-size:clamp(40px,5.6vw,72px);line-height:1;letter-spacing:-.05em;max-width:860px}
.lede{margin-top:22px;max-width:620px;color:var(--body);font-size:18px}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:24px}
.chip{font-size:13px;padding:5px 12px;border:1px solid var(--hair);border-radius:999px;background:var(--soft);color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:44px}
.stat{border-radius:20px;padding:30px;background:var(--c4);color:var(--ink);min-height:210px;display:flex;flex-direction:column;justify-content:space-between}
.stat.rust{background:var(--c1)}
.stat.dark{background:var(--night);color:#fff}
.stat .big{font-family:var(--display);font-size:72px;line-height:1;font-weight:500;letter-spacing:-.05em}
.stat.dark .big{color:var(--lime)}
.stat .cap{margin-top:18px;font-size:15px;line-height:1.45;color:rgba(13,13,13,.78);max-width:300px}
.stat.dark .cap{color:rgba(255,255,255,.8)}
h2{font-family:var(--display);font-size:32px;font-weight:500;letter-spacing:-.04em;margin:72px 0 6px}
.sub{color:var(--muted);font-size:15px;margin-bottom:6px;max-width:680px}
table{width:100%;border-collapse:collapse;font-size:15px;margin-top:14px}
th,td{text-align:left;padding:14px 20px 14px 0;border-bottom:1px solid var(--hair);white-space:nowrap}
th{font-size:13px;font-weight:500;color:var(--muted);border-bottom:1px solid var(--ink);padding-top:8px;padding-bottom:8px}
td.model{font-family:var(--display);font-size:17px;font-weight:500;letter-spacing:-.02em;white-space:normal}
td.num{font-variant-numeric:tabular-nums}
.bar{display:inline-block;height:8px;border-radius:4px;background:var(--ink);vertical-align:middle;margin-right:10px}
.delta{font-weight:500}
.fail{margin-top:14px;border-top:1px solid var(--ink)}
.fail li{padding:16px 0;border-bottom:1px solid var(--hair);list-style:none;font-size:15px}
.fail b{font-weight:500}
.fail span{color:var(--muted)}
.banner{margin-top:80px;background:var(--banner);padding:36px 0}
.banner .wrap{display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap}
.banner p{font-family:var(--display);font-size:24px;font-weight:500;letter-spacing:-.03em;max-width:640px}
.note{margin:48px 0 64px;padding-top:20px;border-top:1px solid var(--hair);color:var(--muted);font-size:13.5px;max-width:760px}
@media(max-width:860px){.stats{grid-template-columns:1fr}.wrap{padding:0 18px}table{display:block;overflow-x:auto}.stat{min-height:0}.stat .big{font-size:56px}}
"""


def build(results: Path) -> str:
    main, local, before = load_results(results)
    if not main:
        raise SystemExit("No results found. Run evals/run_eval.py first.")
    retr, n_pages = retrieval_rows()
    meta = main["meta"]
    configs = dict(main["summaries"])
    if local:
        configs.update(local["summaries"])
    n_q = meta["n_questions"]

    # headline numbers
    claude = {k: v for k, v in configs.items() if k.startswith("anthropic")}
    best_name = max(claude, key=lambda k: claude[k]["accuracy_answerable"] or 0)
    cheapest = min(claude, key=lambda k: claude[k]["mean_cost_usd"])
    best, cheap = claude[best_name], claude[cheapest]
    ratio = best["mean_cost_usd"] / cheap["mean_cost_usd"] if cheap["mean_cost_usd"] else 0
    abst = [v["abstention_accuracy"] for v in configs.values() if v["abstention_accuracy"] is not None]
    abst_line = (f"Every model admitted when the docs didn't cover a question" if abst and min(abst) == 1.0
                 else "How often models admitted when the docs didn't cover a question")

    rows = []
    max_cost = max(v["mean_cost_usd"] for v in configs.values()) or 1
    for name, s in configs.items():
        bar_w = int((s["accuracy_answerable"] or 0) * 120)
        rows.append(
            f"<tr><td class='model'>{esc(short(name))}</td>"
            f"<td class='num'><span class='bar' style='width:{bar_w}px'></span>{pct(s['accuracy_answerable'])}</td>"
            f"<td class='num'>{pct(s['abstention_accuracy'])}</td>"
            f"<td class='num'>{pct(s['citation_hit'])}</td>"
            f"<td class='num'>{s['latency_p50_s']:.1f}s</td>"
            f"<td class='num'>${s['mean_cost_usd']:.4f}</td></tr>")
    models_table = ("<table><tr><th>Model</th><th>Answer score</th><th>Admits gaps</th><th>Cites right page</th>"
                    "<th>Median latency</th><th>Cost per question</th></tr>" + "".join(rows) + "</table>")

    retr_rows = []
    for name, n_chunks, m in retr:
        retr_rows.append(
            f"<tr><td class='model'>{'Fixed 200-word windows' if name == 'fixed' else 'Split on headings'}</td>"
            f"<td class='num'>{n_chunks:,}</td><td class='num'>{pct(m['hit@1'])}</td><td class='num'>{pct(m['hit@3'])}</td>"
            f"<td class='num'>{pct(m['hit@5'])}</td><td class='num'>{m['mrr']:.2f}</td></tr>")
    retr_table = ("<table><tr><th>Chunking</th><th>Chunks</th><th>Right page in top 1</th><th>Top 3</th><th>Top 5</th>"
                  "<th>MRR</th></tr>" + "".join(retr_rows) + "</table>")
    retr_intro = ("Keyword search (BM25) over two chunking strategies, no model involved. Hit means a correct page "
                  "appears in the top results for a question.")
    emb = load_embedding_rankings(results)
    if emb:
        questions = load_questions(ROOT / "evals" / "questions.jsonl")
        chunk_counts = {name: n for name, n, _ in retr}
        emb_rows = []
        for chunking, by_method in emb["rankings"].items():
            for method, rankings in by_method.items():
                m = metrics_from_rankings(rankings, questions)
                emb_rows.append(
                    f"<tr><td class='model'>{'Fixed 200-word windows' if chunking == 'fixed' else 'Split on headings'}</td>"
                    f"<td>{LABELS[method]}</td><td class='num'>{pct(m['hit@1'])}</td><td class='num'>{pct(m['hit@3'])}</td>"
                    f"<td class='num'>{pct(m['hit@5'])}</td><td class='num'>{m['mrr']:.2f}</td></tr>")
        retr_table = ("<table><tr><th>Chunking</th><th>Retrieval</th><th>Right page in top 1</th><th>Top 3</th><th>Top 5</th>"
                      "<th>MRR</th></tr>" + "".join(emb_rows) + "</table>")
        retr_intro = (f"Keyword search (BM25), embeddings from {esc(emb['model'])} running locally in Ollama, and a hybrid that "
                      "merges both rankings, each over two chunking strategies. No chat model is involved. Hit means a "
                      "correct page appears in the top results. With 30 questions, one question is worth 3 points.")

    before_html = ""
    if before:
        brows = []
        for name, s_after in main["summaries"].items():
            s_before = before["summaries"].get(name)
            if not s_before:
                continue
            d = (s_after["accuracy_answerable"] or 0) - (s_before["accuracy_answerable"] or 0)
            sign = "+" if d >= 0 else ""
            brows.append(
                f"<tr><td class='model'>{esc(short(name))}</td><td class='num'>{pct(s_before['accuracy_answerable'])}</td>"
                f"<td class='num'>{pct(s_after['accuracy_answerable'])}</td>"
                f"<td class='num delta'>{sign}{d * 100:.0f} pts</td>"
                f"<td class='num'>{s_before['limit_hits']} to {s_after['limit_hits']}</td></tr>")
        before_html = ("<h2>What the error analysis fixed</h2><p class='sub'>The first run exposed three problems: long "
                       "pages cut off at 6,000 characters, runs that spent every step searching without answering, and a "
                       "judge that could not see the docs. The first run was re-graded once the judge could see the docs, so "
                       "this compares answers and not judge versions. After the second run, the judge was also shown "
                       "the pages each answer cited, which removed false penalties for correct claims on other real "
                       "pages. Part of the last gain comes from that grading fix, not from the agent.</p>"
                       "<table><tr><th>Model</th><th>Before fixes</th><th>After fixes</th><th>Change</th>"
                       "<th>Runs that hit a limit</th></tr>" + "".join(brows) + "</table>")

    # remaining failures from the main run
    fails = []
    for rec in main["records"]:
        r = rec["row"]
        if r["verdict"] in ("partial", "incorrect", "ungraded"):
            fails.append(f"<li><b>{esc(short(rec['config']))}, {esc(r['id'])}</b> "
                         f"<span>({esc(r['verdict'])})</span><br>{esc(r['question'])}<br><span>{esc(r['reason'])}</span></li>")
    fail_html = ("<h2>Where it still fails</h2><p class='sub'>Every answer the judge did not mark fully correct, with its "
                 "reason.</p><ul class='fail'>" + ("".join(fails[:8]) or "<li>None in this run.</li>") + "</ul>"
                 + (f"<p class='sub' style='margin-top:10px'>{len(fails) - 8} more in the JSON output.</p>" if len(fails) > 8 else ""))

    judge = esc(short(meta["judge"]))
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scout evaluation</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<nav class="nav"><div class="wrap"><a class="logo" href="/"><span class="mark" aria-hidden="true"><i></i><i></i><i></i></span>Scout</a><a class="pill" href="/">Ask Scout</a></div></nav>
<div class="wrap"><div class="hero">
<h1>Which model answers FastAPI questions best?</h1>
<p class="lede">An agent that searches the FastAPI docs and cites its sources, measured on {n_q} hand-labeled questions: answer quality, citation accuracy, latency and cost.</p>
<div class="meta"><span class="chip">{n_q} questions, 3 not covered by the docs</span><span class="chip">{n_pages} doc pages, commit {CORPUS_COMMIT}</span>
<span class="chip">Graded by {judge}</span><span class="chip">{esc(nice_date(meta['timestamp']))}</span></div></div>
<div class="stats">
<div class="stat rust"><div class="big">{pct(best['accuracy_answerable'])}</div><p class="cap">answer score for {esc(short(best_name))}, the best model tested</p></div>
<div class="stat"><div class="big">{ratio:.0f}x</div><p class="cap">price gap per question between {esc(short(best_name))} and {esc(short(cheapest))}, which scored {pct(cheap['accuracy_answerable'])}</p></div>
<div class="stat dark"><div class="big">{pct(max(abst)) if abst else 'n/a'}</div><p class="cap">{abst_line}, instead of inventing an answer</p></div>
</div>
<h2>Models compared</h2>
<p class="sub">Same questions, same tools, same retrieval. The answer score counts correct as 1 and partially correct as 0.5 on the {n_q - 3} questions the docs can answer.</p>
{models_table}
{before_html}
<h2>Retrieval: chunking and embeddings</h2>
<p class="sub">{retr_intro}</p>
{retr_table}
{fail_html}
<p class="note">Questions and reference answers were written from the documentation and have not been independently reviewed. The judge is an LLM, so individual verdicts can be wrong; the reasons are saved in the JSON output. Costs use Anthropic's published prices on 2026-10-20 and local models cost nothing to run.</p>
</div>
<div class="banner"><div class="wrap"><p>Try it yourself. Ask Scout a question about FastAPI and check every source it cites.</p><a class="pill" href="/">Ask Scout</a></div></div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    results = Path(args.results)
    out = Path(args.out) if args.out else results / "report.html"
    out.write_text(build(results), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
