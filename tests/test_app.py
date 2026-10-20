import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient
    import app as scout_app
    HAVE_APP = True
except ImportError:  # fastapi/httpx not installed
    HAVE_APP = False

from scout.llm import FakeProvider, LLMResponse, ToolCall


@unittest.skipUnless(HAVE_APP, "fastapi and httpx are not installed")
class AppTests(unittest.TestCase):
    def setUp(self):
        scout_app._hits.clear()
        self.client = TestClient(scout_app.app)
        self.models = mock.patch.object(scout_app, "available_models", return_value=["fake:m"])
        self.models.start()
        self.addCleanup(self.models.stop)

    def fake_provider(self):
        return FakeProvider([
            LLMResponse("", [ToolCall("t1", "search_docs", {"query": "CORS"})], 100, 10, "tool_use"),
            LLMResponse("Use CORSMiddleware [tutorial/cors.md].", [], 200, 30, "end_turn"),
        ])

    def test_config_lists_models(self):
        r = self.client.get("/api/config")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["models"], ["fake:m"])

    def test_ask_runs_agent_and_returns_citations(self):
        with mock.patch.object(scout_app, "make_provider", return_value=(self.fake_provider(), "m")):
            r = self.client.post("/api/ask", json={"question": "How do I enable CORS?", "model": "fake:m"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["stopped"], "answered")
        self.assertEqual(body["cited_pages"], ["tutorial/cors.md"])

    def test_unknown_model_is_rejected(self):
        r = self.client.post("/api/ask", json={"question": "How do I enable CORS?", "model": "nope:x"})
        self.assertEqual(r.status_code, 400)

    def test_validation(self):
        self.assertEqual(self.client.post("/api/ask", json={"question": "hi", "model": "fake:m"}).status_code, 422)
        self.assertEqual(self.client.post("/api/ask", json={"question": "x" * 600, "model": "fake:m"}).status_code, 422)
        bad = {"question": "How do I enable CORS?", "model": "fake:m", "chunking": "weird"}
        self.assertEqual(self.client.post("/api/ask", json=bad).status_code, 422)

    def test_rate_limit(self):
        with mock.patch.object(scout_app, "RATE_LIMIT", 2), \
                mock.patch.object(scout_app, "make_provider", side_effect=lambda spec: (self.fake_provider(), "m")):
            codes = [self.client.post("/api/ask", json={"question": "How do I enable CORS?", "model": "fake:m"}).status_code
                     for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])

    def test_eval_summary_reads_saved_runs(self):
        import json
        import tempfile
        from pathlib import Path
        run = {"meta": {"n_questions": 33, "judge": "anthropic:claude-opus-5-5"},
               "summaries": {"anthropic:claude-haiku-5-5 / headings": {"accuracy_answerable": 0.9, "abstention_accuracy": 1.0,
                             "citation_hit": 0.9, "latency_p50_s": 6.0, "mean_cost_usd": 0.002}}, "records": []}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run-1.json").write_text(json.dumps(run))
            with mock.patch.object(scout_app, "RESULTS", Path(tmp)):
                body = self.client.get("/api/eval").json()
                self.assertEqual(self.client.get("/report").status_code, 404)
        self.assertTrue(body["available"])
        self.assertEqual(body["models"][0]["score"], 0.9)

    def test_eval_summary_when_no_results(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(scout_app, "RESULTS", Path(tmp)):
            self.assertEqual(self.client.get("/api/eval").json(), {"available": False})

    def test_frontend_is_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Scout", r.text)


if __name__ == "__main__":
    unittest.main()
