import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals import run_eval
from evals.run_eval import grade_row, judge_answer, parse_verdict, summarize
from scout.llm import FakeProvider, LLMResponse, ToolCall


class VerdictParsingTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_verdict('{"verdict": "correct", "reason": "ok"}'), ("correct", "ok"))

    def test_json_inside_prose_and_case(self):
        v, _ = parse_verdict('Here you go:\n```json\n{"verdict": "Partial", "reason": "missing x"}\n```')
        self.assertEqual(v, "partial")

    def test_truncated_reply_still_yields_the_verdict(self):
        v, _ = parse_verdict('{"verdict": "correct", "reason": "The candidate gives the same essential solution as the ref')
        self.assertEqual(v, "correct")

    def test_garbage_is_ungraded_not_a_crash(self):
        self.assertEqual(parse_verdict("no idea")[0], "ungraded")
        self.assertEqual(parse_verdict('{"verdict": "great"}')[0], "ungraded")


class JudgeTests(unittest.TestCase):
    Q = {"id": "q1", "question": "How?", "expected_pages": ["a.md"], "reference": "Do X."}

    def test_prompt_includes_source_pages_and_unparseable_reply_is_retried(self):
        judge = FakeProvider([LLMResponse("", [], 10, 0, "end_turn"),
                              LLMResponse('{"verdict": "correct", "reason": "ok"}', [], 20, 5, "end_turn")])
        out = judge_answer(judge, "m", self.Q, "Do X.", {"a.md": "SOURCE TEXT HERE"})
        self.assertEqual(out["verdict"], "correct")
        self.assertEqual(len(judge.calls), 2)
        self.assertIn("SOURCE TEXT HERE", judge.calls[0]["messages"][0]["content"])
        self.assertEqual((out["input_tokens"], out["output_tokens"]), (30, 5))

    def test_two_failures_give_ungraded(self):
        judge = FakeProvider([LLMResponse("", [], 1, 0, ""), LLMResponse("nope", [], 1, 0, "")])
        self.assertEqual(judge_answer(judge, "m", self.Q, "Do X.")["verdict"], "ungraded")


class GradingTests(unittest.TestCase):
    Q = {"id": "q1", "question": "?", "expected_pages": ["a.md", "b.md"], "reference": "r"}
    R = {"cited_pages": ["b.md"], "retrieved_pages": ["c.md"], "steps": []}

    def test_grade_row_answerable(self):
        row = grade_row(self.Q, self.R, "partial")
        self.assertEqual((row["score"], row["citation_hit"], row["retrieval_hit"]), (0.5, True, False))

    def test_grade_row_unanswerable_has_no_page_metrics(self):
        row = grade_row({**self.Q, "expected_pages": []}, self.R, "correct")
        self.assertFalse(row["answerable"])
        self.assertIsNone(row["citation_hit"])

    def test_ungraded_score_is_none(self):
        self.assertIsNone(grade_row(self.Q, self.R, "ungraded")["score"])

    def test_empty_answer_is_incorrect_without_calling_judge(self):
        judge = FakeProvider([])
        self.assertEqual(judge_answer(judge, "m", self.Q, "  ")["verdict"], "incorrect")
        self.assertEqual(judge.calls, [])


def row(answerable, score, cite=None, ret=None, lat=1.0, cost=0.01, stopped="answered"):
    return {"answerable": answerable, "score": score, "citation_hit": cite, "retrieval_hit": ret, "latency_s": lat,
            "cost_usd": cost, "input_tokens": 100, "output_tokens": 50, "stopped": stopped, "model_calls": 2}


class SummaryTests(unittest.TestCase):
    def test_summarize(self):
        rows = [row(True, 1.0, True, True, lat=1), row(True, 0.5, False, True, lat=3),
                row(False, 1.0, lat=2), row(True, None, True, True, lat=4, stopped="error")]
        s = summarize(rows)
        self.assertAlmostEqual(s["accuracy_answerable"], 0.75)  # ungraded row is excluded
        self.assertEqual(s["abstention_accuracy"], 1.0)
        self.assertAlmostEqual(s["citation_hit"], 2 / 3)
        self.assertEqual(s["ungraded"], 1)
        self.assertEqual(s["errors"], 1)
        self.assertEqual(s["mean_tokens"], 150)


class EndToEndTests(unittest.TestCase):
    def test_full_run_offline_with_fake_providers(self):
        def cand():
            turns = []
            for i, page in enumerate(["advanced/events.md", "tutorial/background-tasks.md"]):
                turns += [LLMResponse("", [ToolCall(f"t{i}", "search_docs", {"query": "x"})], 100, 10, "tool_use"),
                          LLMResponse(f"Answer [{page}]", [], 200, 40, "end_turn")]
            return FakeProvider(turns)

        judge = FakeProvider([LLMResponse('{"verdict": "correct", "reason": "matches"}', [], 50, 10),
                              LLMResponse('{"verdict": "partial", "reason": "misses add_task"}', [], 50, 10)])
        providers = {"fake:cand": cand(), "fake:judge": judge}

        def fake_make(spec):
            return providers[spec], spec.split(":")[1]

        with tempfile.TemporaryDirectory() as tmp:
            argv = ["run_eval", "--models", "fake:cand", "--judge", "fake:judge", "--limit", "2", "--out", tmp]
            with mock.patch.object(run_eval, "make_provider", fake_make), mock.patch.object(sys, "argv", argv), \
                    mock.patch("time.sleep"):
                run_eval.main()
            md = (Path(tmp) / "latest.md").read_text()
            data = json.loads(next(Path(tmp).glob("run-*.json")).read_text())
        self.assertIn("fake:cand / headings", md)
        self.assertEqual(data["summaries"]["fake:cand / headings"]["accuracy_answerable"], 0.75)
        self.assertEqual(data["summaries"]["fake:cand / headings"]["citation_hit"], 1.0)
        self.assertIn("misses add_task", md)

    def test_regrade_rejudges_saved_answers(self):
        from evals import regrade
        record = {"config": "fake:cand / headings", "answer": "Answer [advanced/events.md]", "cited": ["advanced/events.md"],
                  "retrieved": ["advanced/events.md"], "steps": [{"step": 1, "kind": "answer"}],
                  "row": {"id": "q01", "question": "q", "verdict": "ungraded", "latency_s": 1.0, "cost_usd": 0.01,
                          "input_tokens": 10, "output_tokens": 5, "stopped": "answered", "model_calls": 1}}
        judge = FakeProvider([LLMResponse('{"verdict": "correct", "reason": "fine"}', [], 40, 5, "end_turn")])
        with tempfile.TemporaryDirectory() as tmp:
            run_file = Path(tmp) / "run.json"
            run_file.write_text(json.dumps({"meta": {}, "summaries": {}, "records": [record]}))
            argv = ["regrade", str(run_file), "--judge", "fake:judge", "--out", tmp]
            with mock.patch.object(regrade, "make_provider", lambda spec: (judge, "judge")), \
                    mock.patch.object(sys, "argv", argv):
                regrade.main()
            data = json.loads(next(Path(tmp).glob("regraded-*.json")).read_text())
        row = data["rows"]["fake:cand / headings"][0]
        self.assertEqual((row["verdict"], row["previous_verdict"]), ("correct", "ungraded"))
        self.assertEqual(data["summaries"]["fake:cand / headings"]["accuracy_answerable"], 1.0)


if __name__ == "__main__":
    unittest.main()
