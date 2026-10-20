import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scout import build_index, build_tools
from scout.chunking import Chunk
from scout.embeddings import (EmbeddingIndex, FakeEmbedder, HybridIndex, OllamaEmbedder, load_vectors, normalize,
                              save_vectors)
from scout.llm import ProviderError
from scout.retrieval import BM25Index, Hit

CHUNKS = [
    Chunk("cors.md#0", "tutorial/cors.md", "CORS", "Use CORSMiddleware with allow_origins to let a browser frontend call your API"),
    Chunk("sql.md#0", "tutorial/sql-databases.md", "SQL", "Connect to a SQL database with a session and a model table"),
    Chunk("files.md#0", "tutorial/request-files.md", "Files", "Upload files with UploadFile and python-multipart"),
]


class EmbeddingIndexTests(unittest.TestCase):
    def test_normalize(self):
        v = normalize([3.0, 4.0])
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0)
        self.assertEqual(normalize([0.0, 0.0]), [0.0, 0.0])

    def test_search_ranks_the_closest_chunk_first(self):
        index = EmbeddingIndex.build(CHUNKS, FakeEmbedder(), cache_dir=None)
        hits = index.search("browser frontend CORSMiddleware allow_origins", k=2)
        self.assertEqual(hits[0].chunk.path, "tutorial/cors.md")
        self.assertEqual(len(hits), 2)
        self.assertGreaterEqual(hits[0].score, hits[1].score)

    def test_search_pages_returns_unique_pages(self):
        chunks = CHUNKS + [Chunk("cors.md#1", "tutorial/cors.md", "More", "CORS preflight requests and allow_headers")]
        index = EmbeddingIndex.build(chunks, FakeEmbedder(), cache_dir=None)
        pages = index.search_pages("CORS preflight", k=3)
        self.assertEqual(len(pages), len(set(pages)))

    def test_vectors_are_cached_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = FakeEmbedder()
            EmbeddingIndex.build(CHUNKS, first, cache_dir=tmp)
            self.assertEqual(first.calls, 1)
            second = FakeEmbedder()
            index = EmbeddingIndex.build(CHUNKS, second, cache_dir=tmp)
            self.assertEqual(second.calls, 0)  # loaded from the cache
            self.assertEqual(index.search("SQL database", k=1)[0].chunk.path, "tutorial/sql-databases.md")

    def test_changed_text_builds_a_new_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            EmbeddingIndex.build(CHUNKS, FakeEmbedder(), cache_dir=tmp)
            changed = [CHUNKS[0], CHUNKS[1], Chunk("files.md#0", "tutorial/request-files.md", "Files", "Different text")]
            again = FakeEmbedder()
            EmbeddingIndex.build(changed, again, cache_dir=tmp)
            self.assertEqual(again.calls, 1)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.vec"
            vectors = [[0.5, 0.25, -1.0], [0.0, 1.0, 2.0]]
            save_vectors(path, vectors)
            self.assertEqual(load_vectors(path), vectors)
            self.assertIsNone(load_vectors(Path(tmp) / "missing.vec"))
            path.write_bytes(b"garbage")
            self.assertIsNone(load_vectors(path))

    def test_vector_count_must_match_chunks(self):
        with self.assertRaises(ValueError):
            EmbeddingIndex(CHUNKS, [[1.0]], FakeEmbedder())


class HybridTests(unittest.TestCase):
    def ranking(self, ids):
        by_id = {c.id: c for c in CHUNKS}
        return [Hit(by_id[i], 1.0) for i in ids]

    def test_reciprocal_rank_fusion(self):
        bm25 = mock.Mock(chunks=CHUNKS)
        emb = mock.Mock()
        bm25.search.return_value = self.ranking(["cors.md#0", "sql.md#0"])
        emb.search.return_value = self.ranking(["files.md#0", "cors.md#0"])
        hits = HybridIndex(bm25, emb).search("q", k=3)
        # cors is in both lists, so it beats the chunks that appear in only one
        self.assertEqual(hits[0].chunk.id, "cors.md#0")
        self.assertEqual({h.chunk.id for h in hits}, {"cors.md#0", "sql.md#0", "files.md#0"})
        self.assertAlmostEqual(hits[0].score, 1 / 61 + 1 / 62)

    def test_hybrid_end_to_end_with_fake_embedder(self):
        emb = EmbeddingIndex.build(CHUNKS, FakeEmbedder(), cache_dir=None)
        hybrid = HybridIndex(BM25Index(CHUNKS), emb)
        self.assertEqual(hybrid.search_pages("upload files", k=1), ["tutorial/request-files.md"])


class OllamaEmbedderTests(unittest.TestCase):
    def test_request_shape_prefixes_and_normalization(self):
        seen = {}

        def fake_post(url, headers, body, retries, timeout, label="API"):
            seen["url"], seen["body"] = url, body
            return {"embeddings": [[3.0, 4.0] for _ in body["input"]]}

        e = OllamaEmbedder("nomic-embed-text", host="localhost:11434")
        with mock.patch("scout.embeddings._post_with_retry", side_effect=fake_post):
            doc = e.embed_documents(["alpha"])
            query = e.embed_query("beta")
        self.assertEqual(seen["url"], "http://localhost:11434/api/embed")
        self.assertEqual(seen["body"]["input"], ["search_query: beta"])
        self.assertEqual(seen["body"]["model"], "nomic-embed-text")
        self.assertAlmostEqual(doc[0][0], 0.6)
        self.assertAlmostEqual(query[1], 0.8)

    def test_documents_are_sent_in_batches(self):
        sizes = []

        def fake_post(url, headers, body, retries, timeout, label="API"):
            sizes.append(len(body["input"]))
            return {"embeddings": [[1.0, 0.0] for _ in body["input"]]}

        e = OllamaEmbedder(batch=4)
        with mock.patch("scout.embeddings._post_with_retry", side_effect=fake_post):
            vectors = e.embed_documents([f"t{i}" for i in range(10)])
        self.assertEqual(sizes, [4, 4, 2])
        self.assertEqual(len(vectors), 10)

    def test_long_text_is_cut(self):
        seen = {}

        def fake_post(url, headers, body, retries, timeout, label="API"):
            seen["text"] = body["input"][0]
            return {"embeddings": [[1.0]]}

        with mock.patch("scout.embeddings._post_with_retry", side_effect=fake_post):
            OllamaEmbedder(max_chars=50).embed_documents(["x" * 500])
        self.assertLessEqual(len(seen["text"]), 50 + len("search_document: "))

    def test_count_mismatch_is_an_error(self):
        with mock.patch("scout.embeddings._post_with_retry", return_value={"embeddings": []}):
            with self.assertRaises(ProviderError):
                OllamaEmbedder().embed_query("x")

    def test_fingerprint_changes_with_model(self):
        self.assertNotEqual(OllamaEmbedder("a").fingerprint(), OllamaEmbedder("b").fingerprint())


class BuildIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.dict("os.environ", {"SCOUT_EMBED_CACHE": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_is_bm25(self):
        self.assertIsInstance(build_index(CHUNKS), BM25Index)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_index(CHUNKS, "magic")

    def test_modes_use_the_given_embedder(self):
        emb = build_index(CHUNKS, "embed", FakeEmbedder())
        hyb = build_index(CHUNKS, "hybrid", FakeEmbedder())
        self.assertIsInstance(emb, EmbeddingIndex)
        self.assertIsInstance(hyb, HybridIndex)

    def test_build_tools_with_hybrid_retrieval(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "cors.md").write_text("# CORS\n\nUse CORSMiddleware to allow origins for a browser frontend.\n")
            (Path(tmp) / "sql.md").write_text("# SQL\n\nConnect to a SQL database with a session.\n")
            tools = build_tools("headings", docs_dir=tmp, retrieval="hybrid", embedder=FakeEmbedder())
            result = tools.call("search_docs", {"query": "browser CORS origins"})
        self.assertFalse(result.is_error)
        self.assertEqual(result.pages[0], "cors.md")


if __name__ == "__main__":
    unittest.main()
