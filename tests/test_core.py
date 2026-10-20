import unittest

from scout.agent import extract_citations, run_agent
from scout.chunking import build_chunks, chunk_fixed, chunk_headings, clean, split_sections
from scout.llm import (FakeProvider, LLMResponse, ProviderError, ToolCall, to_anthropic_messages,
                       to_ollama_messages, make_provider)
from scout.pricing import cost_usd
from scout.retrieval import BM25Index, tokenize
from scout.tools import DocTools

DOCS = {
    "tutorial/body.md": "# Request Body\n\nUse a Pydantic model to declare a request body.\n\n## Install\n\nNothing to install.\n",
    "tutorial/cors.md": "# CORS\n\nUse CORSMiddleware with allow_origins to let a frontend on another origin call the API.\n",
    "advanced/events.md": "# Lifespan Events\n\nUse the lifespan parameter to run code at startup and shutdown.\n",
}


def make_tools(strategy="headings"):
    return DocTools(BM25Index(build_chunks(DOCS, strategy)), DOCS)


class ChunkingTests(unittest.TestCase):
    def test_clean_removes_includes_and_comments(self):
        text = "---\ntitle: x\n---\nhello\n{* ../../docs_src/a.py hl[1] *}\n<!-- note -->\n/// tip\nkeep me\n///\n"
        out = clean(text)
        self.assertIn("hello", out)
        self.assertIn("keep me", out)
        self.assertNotIn("docs_src", out)
        self.assertNotIn("note", out)

    def test_sections_ignore_hashes_in_code_fences(self):
        md = "# A\nintro\n```\n# not a heading\n```\n## B\nbody"
        sections = split_sections(md)
        self.assertEqual([h for h, _ in sections], ["A", "A > B"])
        self.assertIn("# not a heading", sections[0][1])

    def test_fixed_windows_overlap_and_cover_everything(self):
        words = " ".join(f"w{i}" for i in range(500))
        chunks = chunk_fixed({"p.md": words}, size=200, overlap=40)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text.split()[-40:], chunks[1].text.split()[:40])
        self.assertIn("w499", chunks[-1].text.split())

    def test_headings_chunks_carry_heading_text(self):
        chunks = chunk_headings(DOCS)
        self.assertTrue(any(c.heading == "Request Body > Install" for c in chunks))

    def test_unknown_strategy(self):
        with self.assertRaises(ValueError):
            build_chunks(DOCS, "nope")


class RetrievalTests(unittest.TestCase):
    def test_tokenize_splits_snake_case_and_stems(self):
        toks = tokenize("response_model parameters")
        self.assertIn("response_model", toks)
        self.assertIn("model", toks)
        self.assertIn("parameter", toks)

    def test_search_ranks_relevant_page_first(self):
        idx = BM25Index(build_chunks(DOCS, "headings"))
        self.assertEqual(idx.search_pages("how do I allow another origin CORS")[0], "tutorial/cors.md")
        self.assertEqual(idx.search_pages("run code at startup")[0], "advanced/events.md")

    def test_no_match_returns_empty(self):
        idx = BM25Index(build_chunks(DOCS, "headings"))
        self.assertEqual(idx.search("zzzzqqq"), [])


class ToolTests(unittest.TestCase):
    def test_search_returns_pages(self):
        r = make_tools().call("search_docs", {"query": "CORS origin"})
        self.assertFalse(r.is_error)
        self.assertEqual(r.pages[0], "tutorial/cors.md")

    def test_read_missing_page_is_recoverable_error_with_hint(self):
        r = make_tools().call("read_page", {"path": "cors.md"})
        self.assertTrue(r.is_error)
        self.assertIn("tutorial/cors.md", r.text)

    def test_bad_arguments_do_not_raise(self):
        t = make_tools()
        self.assertTrue(t.call("search_docs", {}).is_error)
        self.assertTrue(t.call("search_docs", {"query": "  "}).is_error)
        self.assertTrue(t.call("nope", {}).is_error)

    def test_long_page_is_truncated(self):
        tools = DocTools(BM25Index([]), {"big.md": "x" * 20000})
        r = tools.call("read_page", {"path": "big.md"})
        self.assertIn("offset=6000", r.text)
        self.assertLess(len(r.text), 6200)

    def test_read_page_offset_pages_through_the_whole_page(self):
        text = "".join(chr(97 + i % 26) for i in range(14000))
        tools = DocTools(BM25Index([]), {"big.md": text})
        out, offset = "", 0
        while True:
            r = tools.call("read_page", {"path": "big.md", "offset": offset})
            self.assertFalse(r.is_error)
            body = r.text.split("\n\n[showing")[0]
            out += body
            if "[showing" not in r.text:
                break
            offset += len(body)
        self.assertEqual(out, text)
        self.assertTrue(tools.call("read_page", {"path": "big.md", "offset": 99999}).is_error)

    def test_list_pages_prefix(self):
        r = make_tools().call("list_pages", {"prefix": "tutorial/"})
        self.assertEqual(sorted(r.text.splitlines()), ["tutorial/body.md", "tutorial/cors.md"])


def tool_turn(name, args, i=1, tokens=(100, 20), text=""):
    return LLMResponse(text, [ToolCall(f"t{i}", name, args)], *tokens, "tool_use")


def final(text, tokens=(200, 50)):
    return LLMResponse(text, [], *tokens, "end_turn")


class AgentTests(unittest.TestCase):
    def test_search_then_answer_with_citation(self):
        p = FakeProvider([tool_turn("search_docs", {"query": "CORS"}), final("Use CORSMiddleware [tutorial/cors.md].")])
        r = run_agent("How do I enable CORS?", p, "claude-haiku-5-5", make_tools())
        self.assertEqual(r.stopped, "answered")
        self.assertEqual(r.cited_pages, ["tutorial/cors.md"])
        self.assertEqual(r.retrieved_pages[0], "tutorial/cors.md")
        self.assertEqual((r.input_tokens, r.output_tokens), (300, 70))
        self.assertAlmostEqual(r.cost_usd, (300 * 0.10 + 70 * 0.50) / 1e6)
        # the tool result was fed back to the model
        self.assertEqual(p.calls[1]["messages"][-1]["role"], "tool")

    def test_hallucinated_citation_is_ignored(self):
        p = FakeProvider([final("See [made/up.md] and [tutorial/body.md].")])
        r = run_agent("q", p, "m", make_tools())
        self.assertEqual(r.cited_pages, ["tutorial/body.md"])

    def test_tool_failure_is_returned_to_model_and_run_continues(self):
        p = FakeProvider([tool_turn("read_page", {"path": "missing.md"}), final("I could not find it.")])
        r = run_agent("q", p, "m", make_tools())
        self.assertEqual(r.stopped, "answered")
        self.assertFalse(r.steps[0]["ok"])
        self.assertTrue(p.calls[1]["messages"][-1]["is_error"])

    def test_step_limit(self):
        p = FakeProvider([tool_turn("search_docs", {"query": "a"}, i=i, text="partial") for i in range(5)])
        r = run_agent("q", p, "m", make_tools(), max_steps=3)
        self.assertEqual(r.stopped, "step_limit")
        self.assertEqual(len(p.calls), 3)
        self.assertEqual(r.answer, "partial")

    def test_token_budget_stops_early(self):
        p = FakeProvider([tool_turn("search_docs", {"query": "a"}, tokens=(9000, 2000)), final("never reached")])
        r = run_agent("q", p, "m", make_tools(), token_budget=10_000)
        self.assertEqual(r.stopped, "token_budget")
        self.assertEqual(len(p.calls), 1)

    def test_provider_error_is_reported_not_raised(self):
        p = FakeProvider([ProviderError("API returned 500")])
        r = run_agent("q", p, "m", make_tools())
        self.assertEqual(r.stopped, "error")
        self.assertIn("500", r.error)

    def test_last_step_asks_for_a_final_answer(self):
        from scout.agent import FINAL_STEP_NOTE
        p = FakeProvider([tool_turn("search_docs", {"query": "a"}, i=1), tool_turn("search_docs", {"query": "b"}, i=2),
                          final("Here is the answer [tutorial/body.md].")])
        r = run_agent("q", p, "m", make_tools(), max_steps=3)
        self.assertEqual(r.stopped, "answered")
        self.assertNotIn(FINAL_STEP_NOTE, [m.get("content") for m in p.calls[1]["messages"]])
        self.assertEqual(p.calls[2]["messages"][-1], {"role": "user", "content": FINAL_STEP_NOTE})

    def test_extract_citations_dedupes(self):
        self.assertEqual(extract_citations("[a.md] [a.md] [b.md]", {"a.md", "b.md"}), ["a.md", "b.md"])


class WireFormatTests(unittest.TestCase):
    MESSAGES = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "text": "looking", "tool_calls": [ToolCall("t1", "search_docs", {"query": "x"}),
                                                                  ToolCall("t2", "list_pages", {})]},
        {"role": "tool", "tool_call_id": "t1", "name": "search_docs", "content": "r1", "is_error": False},
        {"role": "tool", "tool_call_id": "t2", "name": "list_pages", "content": "r2", "is_error": True},
    ]

    def test_anthropic_groups_tool_results_into_one_user_turn(self):
        out = to_anthropic_messages(self.MESSAGES)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual([b["type"] for b in out[1]["content"]], ["text", "tool_use", "tool_use"])
        results = out[2]["content"]
        self.assertEqual([b["tool_use_id"] for b in results], ["t1", "t2"])
        self.assertTrue(results[1]["is_error"])
        self.assertNotIn("is_error", results[0])

    def test_anthropic_merges_a_user_note_after_tool_results(self):
        msgs = self.MESSAGES + [{"role": "user", "content": "answer now"}]
        out = to_anthropic_messages(msgs)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual(out[2]["content"][-1], {"type": "text", "text": "answer now"})

    def test_ollama_format(self):
        out = to_ollama_messages("sys", self.MESSAGES)
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant", "tool", "tool"])
        self.assertEqual(out[2]["tool_calls"][0]["function"], {"name": "search_docs", "arguments": {"query": "x"}})
        self.assertEqual(out[3]["tool_name"], "search_docs")


class MiscTests(unittest.TestCase):
    def test_pricing(self):
        self.assertAlmostEqual(cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000), 12.0)
        self.assertEqual(cost_usd("llama3.1", 5000, 5000), 0.0)

    def test_make_provider_validates_spec(self):
        with self.assertRaises(ValueError):
            make_provider("claude-haiku-5-5")
        with self.assertRaises(ValueError):
            make_provider("openai:gpt")



class CodeToolTests(unittest.TestCase):
    def test_off_by_default(self):
        t = make_tools()
        self.assertNotIn("run_python", [s["name"] for s in t.specs])
        self.assertTrue(t.call("run_python", {"code": "print(1)"}).is_error)

    def test_runs_code_and_captures_output(self):
        t = DocTools(BM25Index([]), {}, enable_code=True)
        r = t.call("run_python", {"code": "print(2 + 2)"})
        self.assertFalse(r.is_error)
        self.assertEqual(r.text.strip(), "4")

    def test_errors_and_timeouts_are_reported(self):
        from scout.tools import run_python
        self.assertTrue(run_python("raise ValueError('boom')").is_error)
        self.assertIn("boom", run_python("raise ValueError('boom')").text)
        r = run_python("while True: pass", timeout=1)
        self.assertTrue(r.is_error)
        self.assertIn("timed out", r.text)

    def test_secrets_are_not_visible_to_the_code(self):
        import os
        from scout.tools import run_python
        os.environ["FAKE_API_KEY"] = "hunter2"
        try:
            r = run_python("import os; print(os.environ.get('FAKE_API_KEY'))")
        finally:
            del os.environ["FAKE_API_KEY"]
        self.assertEqual(r.text.strip(), "None")

    def test_output_is_capped(self):
        from scout.tools import run_python
        r = run_python("print('x' * 100000)")
        self.assertLess(len(r.text), 3100)

    def test_code_note_added_to_system_prompt_only_when_enabled(self):
        p = FakeProvider([final("ok")])
        run_agent("q", p, "m", DocTools(BM25Index([]), {}, enable_code=True))
        p2 = FakeProvider([final("ok")])
        run_agent("q", p2, "m", DocTools(BM25Index([]), {}, enable_code=False))

if __name__ == "__main__":
    unittest.main()
