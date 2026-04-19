"""
Unit tests for the RAG pipeline components.
Tests cover: chunker, vector stores (FAISS, Chroma, Qdrant), embeddings model catalogue,
pipeline config, and tool registry integration.
"""

import asyncio
import os
import tempfile
import pytest

from src.rag.embeddings import EMBEDDING_MODELS, EmbeddingModel
from src.rag.chunker import chunk_documents, load_source, Chunk
from src.rag.vectorstore import create_store, SearchResult, FAISSStore, ChromaStore, QdrantStore
from src.rag.pipeline import RAGConfig, RAGPipeline


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker:
    def _pages(self, text: str):
        return load_source("text", text)

    def test_load_text(self):
        pages = load_source("text", "Hello world.")
        assert len(pages) == 1
        assert pages[0][0] == "Hello world."

    def test_recursive_strategy(self):
        # Use natural sentence separators so recursive splitting works correctly
        text = ("This is a sentence. " * 30) + "\n\n" + ("Another paragraph here. " * 30)
        pages = load_source("text", text)
        chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=20, strategy="recursive")
        assert len(chunks) >= 2
        for c in chunks:
            assert isinstance(c, Chunk)

    def test_code_strategy_splits_on_function(self):
        code = "def foo():\n    return 1\n\ndef bar():\n    return 2\n\ndef baz():\n    return 3\n"
        pages = load_source("text", code)
        chunks = chunk_documents(pages, chunk_size=40, chunk_overlap=0, strategy="code")
        assert len(chunks) >= 2

    def test_semantic_strategy(self):
        text = "This is sentence one. This is sentence two. This is sentence three. Another sentence here."
        pages = load_source("text", text)
        chunks = chunk_documents(pages, chunk_size=50, chunk_overlap=0, strategy="semantic")
        assert all(c.text for c in chunks)

    def test_chunk_metadata(self):
        pages = load_source("text", "word " * 200)
        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=20)
        assert chunks[0].source == "text"
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == len(chunks)

    def test_empty_text_returns_no_chunks(self):
        pages = load_source("text", "   ")
        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=0)
        assert chunks == []

    def test_citation_format(self):
        pages = load_source("text", "sample content " * 30)
        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=0)
        assert "chunk" in chunks[0].citation.lower()


# ── Vector store tests ────────────────────────────────────────────────────────

class TestFAISSStore:
    def _store(self, tmpdir: str) -> FAISSStore:
        return FAISSStore("test_col", persist_dir=tmpdir)

    def test_add_and_query(self, tmp_path):
        store = self._store(str(tmp_path))
        store.add(
            ["hello world", "foo bar baz"],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [{"source": "a", "chunk_index": 0, "total_chunks": 1},
             {"source": "b", "chunk_index": 0, "total_chunks": 1}],
        )
        assert store.count() == 2
        results = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].text == "hello world"
        assert results[0].score > 0.9

    def test_empty_store_returns_empty(self, tmp_path):
        store = self._store(str(tmp_path))
        assert store.query([1.0, 0.0], top_k=5) == []

    def test_persistence(self, tmp_path):
        store1 = self._store(str(tmp_path))
        store1.add(["doc1"], [[0.5, 0.5, 0.0, 0.0]], [{"source": "s", "chunk_index": 0, "total_chunks": 1}])
        # New store instance reads from disk
        store2 = self._store(str(tmp_path))
        assert store2.count() == 1

    def test_delete_collection(self, tmp_path):
        store = self._store(str(tmp_path))
        store.add(["doc"], [[1.0, 0.0]], [{"source": "s", "chunk_index": 0, "total_chunks": 1}])
        store.delete_collection()
        assert store.count() == 0


class TestChromaStore:
    def test_add_and_query(self, tmp_path):
        store = ChromaStore("test_chroma", persist_dir=str(tmp_path))
        store.add(
            ["chromadb test"],
            [[0.1, 0.2, 0.3, 0.4]],
            [{"source": "chroma", "chunk_index": 0, "total_chunks": 1}],
        )
        assert store.count() == 1
        results = store.query([0.1, 0.2, 0.3, 0.4], top_k=1)
        assert len(results) == 1
        assert results[0].text == "chromadb test"


class TestQdrantStore:
    def test_add_and_query(self):
        store = QdrantStore("qdrant_test_unique", dimensions=4)
        store.add(
            ["qdrant result"],
            [[0.9, 0.1, 0.0, 0.0]],
            [{"source": "q", "chunk_index": 0, "total_chunks": 1}],
        )
        results = store.query([0.9, 0.1, 0.0, 0.0], top_k=1)
        assert len(results) >= 1
        assert results[0].score > 0.0


class TestCreateStore:
    def test_faiss_factory(self, tmp_path):
        s = create_store("faiss", "col", persist_dir=str(tmp_path), dimensions=4)
        assert isinstance(s, FAISSStore)

    def test_chroma_factory(self, tmp_path):
        s = create_store("chroma", "chromacol", persist_dir=str(tmp_path), dimensions=4)
        assert isinstance(s, ChromaStore)

    def test_qdrant_factory(self):
        s = create_store("qdrant", "qdrant_factory_col_unique", dimensions=4)
        assert isinstance(s, QdrantStore)

    def test_unknown_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown store_type"):
            create_store("pinecone", "col", persist_dir=str(tmp_path))


# ── Embedding model catalogue ─────────────────────────────────────────────────

class TestEmbeddingModels:
    def test_catalogue_has_expected_models(self):
        assert "openai/text-embedding-3-small" in EMBEDDING_MODELS
        assert "openai/text-embedding-3-large" in EMBEDDING_MODELS
        assert "qwen/qwen3-embedding-8b" in EMBEDDING_MODELS
        assert "baai/bge-m3" in EMBEDDING_MODELS

    def test_model_metadata_complete(self):
        for model_id, meta in EMBEDDING_MODELS.items():
            assert "provider" in meta, f"{model_id} missing provider"
            assert "dimensions" in meta, f"{model_id} missing dimensions"
            assert "description" in meta, f"{model_id} missing description"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            EmbeddingModel("nonexistent/model")

    def test_model_name_extraction(self):
        # OpenRouter requires the full model id (e.g. "openai/text-embedding-3-small")
        m = EmbeddingModel.__new__(EmbeddingModel)
        m.model_id = "openai/text-embedding-3-small"
        m.meta = EMBEDDING_MODELS["openai/text-embedding-3-small"]
        m.provider = "openrouter"
        m.dimensions = 1536
        m._api_key = None
        m._base_url = None
        m._client = None
        assert m._model_name() == "openai/text-embedding-3-small"


# ── Pipeline config ───────────────────────────────────────────────────────────

class TestRAGConfig:
    def test_default_config(self):
        cfg = RAGConfig(config_id="test", name="Test Pipeline")
        assert cfg.embedding_model == "openai/text-embedding-3-small"
        assert cfg.vector_store == "chroma"
        assert cfg.retrieval_strategy == "similarity"
        assert cfg.top_k == 5

    def test_pipeline_initializes(self, tmp_path):
        cfg = RAGConfig(
            config_id="pipe_test",
            name="Test",
            vector_store="faiss",
            persist_dir=str(tmp_path),
        )
        pipeline = RAGPipeline(cfg)
        assert pipeline.chunk_count() == 0


# ── RAG tool registry integration ────────────────────────────────────────────

class TestRAGToolRegistry:
    def test_rag_tools_available(self):
        from src.tools.registry import get_rag_tools
        tools = get_rag_tools()
        tool_names = [t.name for t in tools]
        # get_rag_tools returns both rag_search (general) and perf_search (performance KB)
        assert "rag_search" in tool_names
        assert "perf_search" in tool_names
        assert len(tools) == 2

    def test_rag_tool_has_description(self):
        from src.tools.registry import get_rag_tools
        tools = get_rag_tools()
        rag_tool = next(t for t in tools if t.name == "rag_search")
        assert "knowledge base" in rag_tool.description.lower()

    def test_perf_search_tool_available(self):
        from src.tools.registry import get_rag_tools
        tools = get_rag_tools()
        perf_tool = next((t for t in tools if t.name == "perf_search"), None)
        assert perf_tool is not None
        assert "performance" in perf_tool.description.lower()

    def test_researcher_has_rag_tool_group(self):
        from src.agents.prompts import AGENT_DEFINITIONS
        researcher = next(a for a in AGENT_DEFINITIONS if a["id"] == "researcher")
        assert "rag" in researcher["tools"]
        assert "perf_kb" in researcher["tools"]

    def test_coder_has_rag_tool_group(self):
        from src.agents.prompts import AGENT_DEFINITIONS
        coder = next(a for a in AGENT_DEFINITIONS if a["id"] == "coder")
        assert "rag" in coder["tools"]


# ── MMR retrieval helper ──────────────────────────────────────────────────────
# Tests use _mmr_rerank directly — the LangChain-backed implementation.
# The old _mmr(query_emb, candidates, …) signature is no longer exported.

class TestMMR:
    def test_mmr_returns_top_k(self):
        from src.rag.pipeline import _mmr_rerank
        from src.rag.vectorstore import SearchResult

        candidates = [
            SearchResult("doc1", "src", 0.9, 0, 5, None, {}),
            SearchResult("doc2", "src", 0.8, 1, 5, None, {}),
            SearchResult("doc3", "src", 0.7, 2, 5, None, {}),
            SearchResult("doc4", "src", 0.6, 3, 5, None, {}),
        ]
        result = _mmr_rerank(candidates, top_k=2, lambda_=0.5)
        assert len(result) == 2

    def test_mmr_empty_candidates(self):
        from src.rag.pipeline import _mmr_rerank
        result = _mmr_rerank([], top_k=5, lambda_=0.5)
        assert result == []


# ── BM25 scoring ─────────────────────────────────────────────────────────────
# Tests use rank_bm25.BM25Okapi directly — the library used internally by
# LangChain's BM25Retriever inside the hybrid retrieval strategy.

class TestBM25:
    def test_matching_doc_scores_higher(self):
        from rank_bm25 import BM25Okapi
        texts = ["python is great", "java is verbose", "python rocks"]
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        scores = bm25.get_scores("python".split())
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_no_match_gives_zero(self):
        from rank_bm25 import BM25Okapi
        texts = ["hello world", "foo bar"]
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        scores = bm25.get_scores(["zzz_unknown"])
        assert all(s == 0.0 for s in scores)
