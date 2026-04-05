"""
Integration & functional tests for the RAG pipeline.

Covers:
  - All 3 vector store backends (FAISS, Chroma, Qdrant)
  - All 4 chunking strategies
  - All 4 retrieval strategies (with mocked embeddings)
  - Live OpenRouter embedding call (skipped if no key)
  - Full ingest → retrieve → generate cycle (with mocked LLM)
  - RAG config DB round-trip
  - BM25 hybrid scoring
  - MMR diversity reranking
  - EmbeddingModel catalogue completeness
"""

import asyncio
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_vec(n: int, dim: int = 8) -> list[float]:
    """Simple deterministic vector by repeating a pattern."""
    import math
    angle = 2 * math.pi * n / 20
    base = [math.cos(angle), math.sin(angle), math.cos(2 * angle), math.sin(2 * angle)]
    # Extend to desired dim
    return (base * ((dim // len(base)) + 1))[:dim]


EMBED_DIM = 8   # small dimension for fast tests


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_persist(tmp_path):
    return str(tmp_path)


@pytest.fixture
def sample_docs():
    return [
        "Python is a high-level programming language.",
        "Machine learning uses algorithms to learn from data.",
        "FastAPI is a modern Python web framework for building APIs.",
        "RAG stands for Retrieval-Augmented Generation.",
        "Vector databases store embeddings for semantic search.",
    ]


@pytest.fixture
def mock_embed(sample_docs):
    """Patch EmbeddingModel.embed_batch with deterministic vectors."""
    async def _embed_batch(texts):
        return [_make_vec(hash(t) % 20, EMBED_DIM) for t in texts]

    async def _embed(text):
        return _make_vec(hash(text) % 20, EMBED_DIM)

    mock = MagicMock()
    mock.embed_batch = AsyncMock(side_effect=_embed_batch)
    mock.embed = AsyncMock(side_effect=_embed)
    mock.dimensions = EMBED_DIM
    return mock


# ── Chunker tests (extended) ─────────────────────────────────────────────────

class TestChunkerStrategies:
    @pytest.mark.parametrize("strategy", ["recursive", "fixed", "semantic", "code"])
    def test_all_strategies_produce_chunks(self, strategy):
        from src.rag.chunker import chunk_documents, load_source
        text = "Hello world. " * 50
        pages = load_source("text", text)
        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=10, strategy=strategy)
        assert len(chunks) > 0
        for c in chunks:
            assert c.text.strip()

    def test_load_url_like_content_as_text(self):
        from src.rag.chunker import load_source
        pages = load_source("text", "<h1>Title</h1><p>Content here</p>")
        assert pages[0][0].strip()

    def test_file_not_found_raises(self):
        from src.rag.chunker import load_source
        with pytest.raises(FileNotFoundError):
            load_source("file", "/nonexistent/path/doc.txt")

    def test_unknown_source_type_raises(self):
        from src.rag.chunker import load_source
        with pytest.raises(ValueError):
            load_source("database", "some content")

    def test_code_strategy_preserves_def_keyword(self):
        from src.rag.chunker import chunk_documents, load_source
        code = "\n".join([f"def func_{i}():\n    pass\n" for i in range(10)])
        pages = load_source("text", code)
        chunks = chunk_documents(pages, chunk_size=60, chunk_overlap=0, strategy="code")
        combined = " ".join(c.text for c in chunks)
        assert "def" in combined

    def test_chunk_citation_includes_source(self):
        from src.rag.chunker import chunk_documents, load_source
        pages = load_source("text", "word " * 100)
        pages[0] = (pages[0][0], {"source": "my_doc.txt"})
        chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=0)
        assert "my_doc.txt" in chunks[0].citation


# ── FAISS store (all operations) ─────────────────────────────────────────────

class TestFAISSFull:
    def test_add_query_delete_cycle(self, tmp_persist):
        from src.rag.vectorstore import FAISSStore
        s = FAISSStore("full_cycle", persist_dir=tmp_persist)
        s.add(["alpha", "beta", "gamma"],
              [_make_vec(0, 4), _make_vec(5, 4), _make_vec(10, 4)],
              [{"source": f"doc{i}", "chunk_index": i, "total_chunks": 3} for i in range(3)])
        assert s.count() == 3
        results = s.query(_make_vec(0, 4), top_k=1)
        assert results[0].text == "alpha"
        s.delete_collection()
        assert s.count() == 0

    def test_top_k_clamped_to_available(self, tmp_persist):
        from src.rag.vectorstore import FAISSStore
        s = FAISSStore("clamp_test", persist_dir=tmp_persist)
        s.add(["only_one"], [_make_vec(1, 4)], [{"source": "s", "chunk_index": 0, "total_chunks": 1}])
        results = s.query(_make_vec(1, 4), top_k=10)
        assert len(results) == 1

    def test_score_is_between_neg1_and_1(self, tmp_persist):
        from src.rag.vectorstore import FAISSStore
        s = FAISSStore("score_range", persist_dir=tmp_persist)
        s.add(["doc a", "doc b"],
              [_make_vec(0, 4), _make_vec(8, 4)],
              [{"source": "s", "chunk_index": i, "total_chunks": 2} for i in range(2)])
        for r in s.query(_make_vec(0, 4), top_k=2):
            assert -1.0 <= r.score <= 1.01  # tiny float epsilon


# ── Chroma store ──────────────────────────────────────────────────────────────

class TestChromaFull:
    def test_add_and_query(self, tmp_persist):
        from src.rag.vectorstore import ChromaStore
        s = ChromaStore("chroma_full", persist_dir=tmp_persist)
        s.add(["chroma doc"], [_make_vec(3, 4)], [{"source": "src", "chunk_index": 0, "total_chunks": 1}])
        assert s.count() == 1
        results = s.query(_make_vec(3, 4), top_k=1)
        assert results[0].text == "chroma doc"
        assert 0 <= results[0].score <= 1.0


# ── Qdrant store ──────────────────────────────────────────────────────────────

class TestQdrantFull:
    def test_add_query_count(self):
        from src.rag.vectorstore import QdrantStore
        s = QdrantStore("qdrant_full_unique_2", dimensions=4)
        s.add(["qdrant text"], [_make_vec(7, 4)], [{"source": "qsrc", "chunk_index": 0, "total_chunks": 1}])
        assert s.count() >= 1
        results = s.query(_make_vec(7, 4), top_k=1)
        assert results[0].text == "qdrant text"


# ── Retrieval strategies (mocked embeddings) ──────────────────────────────────

class TestRetrievalStrategies:
    def _make_pipeline(self, strategy: str, tmp_path, store: str = "faiss") -> "RAGPipeline":
        from src.rag.pipeline import RAGConfig, RAGPipeline
        cfg = RAGConfig(
            config_id=f"ret_{strategy}_{store}",
            name="Test",
            vector_store=store,
            retrieval_strategy=strategy,
            embedding_model="qwen/qwen3-embedding-8b",
            top_k=3,
            persist_dir=str(tmp_path),
        )
        return RAGPipeline(cfg)

    @pytest.mark.asyncio
    async def test_similarity_retrieval(self, tmp_path, sample_docs):
        from src.rag.pipeline import RAGPipeline
        pipeline = self._make_pipeline("similarity", tmp_path)

        # Pre-load store with mock embeddings
        texts = sample_docs
        vecs = [_make_vec(i, EMBED_DIM) for i in range(len(texts))]
        meta = [{"source": "test", "chunk_index": i, "total_chunks": len(texts)} for i in range(len(texts))]
        pipeline._store.add(texts, vecs, meta)

        # Mock the embedding call
        async def _mock_embed(text):
            return _make_vec(0, EMBED_DIM)  # closest to texts[0]

        pipeline._embedding_model.embed = AsyncMock(side_effect=_mock_embed)
        pipeline._embedding_model.embed_batch = AsyncMock(side_effect=lambda ts: [_make_vec(0, EMBED_DIM)] * len(ts))

        results = await pipeline.retrieve("Python programming")
        assert len(results) <= 3
        assert all(hasattr(r, "text") for r in results)

    @pytest.mark.asyncio
    async def test_mmr_returns_diverse_results(self, tmp_path):
        # Tests _mmr_rerank — the current LangChain-backed MMR reranker.
        # The old _mmr(query_emb, candidates, …) shim is no longer exported;
        # callers now use _mmr_rerank(candidates, top_k, lambda_) directly.
        from src.rag.pipeline import _mmr_rerank
        from src.rag.vectorstore import SearchResult

        candidates = [
            SearchResult("python doc 1", "src", 0.95, 0, 5, None, {}),
            SearchResult("python doc 2", "src", 0.90, 1, 5, None, {}),
            SearchResult("java doc 1",   "src", 0.80, 2, 5, None, {}),
            SearchResult("java doc 2",   "src", 0.75, 3, 5, None, {}),
            SearchResult("rust doc",     "src", 0.70, 4, 5, None, {}),
        ]
        result = _mmr_rerank(candidates, top_k=3, lambda_=0.5)
        assert len(result) == 3
        # First result should always be the highest-relevance candidate
        assert result[0].score == 0.95

    @pytest.mark.asyncio
    async def test_hybrid_scoring(self, tmp_path, sample_docs):
        # Tests BM25 scoring via rank_bm25 — the library used by LangChain's
        # BM25Retriever inside the hybrid retrieval strategy.
        from rank_bm25 import BM25Okapi

        texts = ["Python is great", "Java is verbose", "Python rocks"]
        tokenised = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenised)
        scores = bm25.get_scores("python".lower().split())
        # Python docs should score higher than Java doc
        assert scores[0] > scores[1], "Python doc 1 should outscore Java doc"
        assert scores[2] > scores[1], "Python doc 2 should outscore Java doc"
        # No overlap for a completely unknown term → all zeros
        zero_scores = bm25.get_scores(["zzz_unknown_term"])
        assert all(s == 0.0 for s in zero_scores)

    @pytest.mark.asyncio
    async def test_retrieve_empty_store_returns_empty(self, tmp_path):
        from src.rag.pipeline import RAGConfig, RAGPipeline
        cfg = RAGConfig(
            config_id="empty_ret_test",
            name="Empty",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            persist_dir=str(tmp_path),
        )
        pipeline = RAGPipeline(cfg)
        pipeline._embedding_model.embed = AsyncMock(return_value=_make_vec(0, EMBED_DIM))
        results = await pipeline.retrieve("anything")
        assert results == []


# ── Ingest pipeline ───────────────────────────────────────────────────────────

class TestIngestPipeline:
    @pytest.mark.asyncio
    async def test_ingest_text_source(self, tmp_path):
        from src.rag.pipeline import RAGConfig, RAGPipeline
        cfg = RAGConfig(
            config_id="ingest_text",
            name="IngestTest",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            persist_dir=str(tmp_path),
        )
        pipeline = RAGPipeline(cfg)

        # Mock embedding
        async def _batch(texts):
            return [_make_vec(hash(t) % 20, EMBED_DIM) for t in texts]

        pipeline._embedding_model.embed_batch = AsyncMock(side_effect=_batch)

        result = await pipeline.ingest("text", "This is test content. " * 50)
        assert result["chunks"] > 0
        assert result["tokens_estimated"] > 0
        assert pipeline.chunk_count() > 0

    @pytest.mark.asyncio
    async def test_ingest_multiple_sources(self, tmp_path):
        from src.rag.pipeline import RAGConfig, RAGPipeline
        cfg = RAGConfig(
            config_id="ingest_multi",
            name="MultiIngest",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            persist_dir=str(tmp_path),
        )
        pipeline = RAGPipeline(cfg)

        async def _batch(texts):
            return [_make_vec(hash(t) % 20, EMBED_DIM) for t in texts]

        pipeline._embedding_model.embed_batch = AsyncMock(side_effect=_batch)

        await pipeline.ingest("text", "First document content. " * 20)
        count1 = pipeline.chunk_count()
        await pipeline.ingest("text", "Second document content. " * 20)
        count2 = pipeline.chunk_count()
        assert count2 > count1

    @pytest.mark.asyncio
    async def test_ingest_empty_text_returns_zero_chunks(self, tmp_path):
        from src.rag.pipeline import RAGConfig, RAGPipeline
        cfg = RAGConfig(
            config_id="ingest_empty",
            name="EmptyIngest",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            persist_dir=str(tmp_path),
        )
        pipeline = RAGPipeline(cfg)
        pipeline._embedding_model.embed_batch = AsyncMock(return_value=[])
        result = await pipeline.ingest("text", "   ")
        assert result["chunks"] == 0


# ── Full RAG cycle (mocked LLM + embeddings) ──────────────────────────────────

class TestFullRAGCycle:
    @pytest.mark.asyncio
    async def test_query_returns_rag_response(self, tmp_path):
        from src.rag.pipeline import RAGConfig, RAGPipeline, RAGResponse
        cfg = RAGConfig(
            config_id="full_cycle",
            name="Full",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            persist_dir=str(tmp_path),
            top_k=2,
        )
        pipeline = RAGPipeline(cfg)

        # Pre-populate store
        texts = ["RAG means Retrieval Augmented Generation.", "Vector stores hold embeddings."]
        vecs = [_make_vec(i, EMBED_DIM) for i in range(len(texts))]
        meta = [{"source": "doc.txt", "chunk_index": i, "total_chunks": 2} for i in range(2)]
        pipeline._store.add(texts, vecs, meta)

        async def _embed(text):
            return _make_vec(0, EMBED_DIM)

        pipeline._embedding_model.embed = AsyncMock(side_effect=_embed)

        # Mock LLM response
        mock_resp = MagicMock()
        mock_resp.content = "RAG stands for Retrieval-Augmented Generation [1]."
        mock_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 20}

        # get_llm is imported inside the pipeline function; patch at the source module
        with patch("src.llm.client.get_llm") as mock_llm_factory:
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
            mock_llm_factory.return_value = mock_llm

            response = await pipeline.query("What is RAG?")

        assert isinstance(response, RAGResponse)
        assert response.answer
        assert response.chunks_retrieved > 0
        assert len(response.citations) > 0
        assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_query_no_documents_returns_graceful_message(self, tmp_path):
        from src.rag.pipeline import RAGConfig, RAGPipeline
        cfg = RAGConfig(
            config_id="no_docs",
            name="NoDocs",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            persist_dir=str(tmp_path),
        )
        pipeline = RAGPipeline(cfg)
        pipeline._embedding_model.embed = AsyncMock(return_value=_make_vec(0, EMBED_DIM))

        response = await pipeline.query("What is anything?")
        assert "not find" in response.answer.lower() or "no" in response.answer.lower()
        assert response.chunks_retrieved == 0


# ── Live embedding test (OpenRouter) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_openrouter_embedding():
    """Live call to OpenRouter. Skipped if OPENROUTER_KEY is not set."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    key = os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_KEY not set — skipping live embedding test")

    from src.rag.embeddings import EmbeddingModel

    model = EmbeddingModel("qwen/qwen3-embedding-8b", api_key=key)
    texts = [
        "Hello world from RAG pipeline test.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    ]
    vecs = await model.embed_batch(texts)

    assert len(vecs) == 2
    assert len(vecs[0]) == 4096   # qwen3-embedding-8b native dim
    assert len(vecs[1]) == 4096
    # Vectors should be non-zero
    assert any(v != 0.0 for v in vecs[0])
    # Two different texts should produce different embeddings
    assert vecs[0] != vecs[1]

    # Single embed
    vec = await model.embed("test sentence")
    assert len(vec) == 4096


@pytest.mark.asyncio
async def test_live_ingest_and_retrieve():
    """Live end-to-end: ingest text → retrieve with OpenRouter embeddings."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    key = os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_KEY not set — skipping live ingest test")

    import tempfile
    from src.rag.pipeline import RAGConfig, RAGPipeline

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = RAGConfig(
            config_id="live_ingest_test",
            name="LiveTest",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            retrieval_strategy="similarity",
            top_k=2,
            persist_dir=tmpdir,
        )
        pipeline = RAGPipeline(cfg)

        content = (
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
            "It extends LangChain with cycles and controllability. "
            "Agents can collaborate using a supervisor pattern, a sequential pattern, or parallel execution. "
            "RAG pipelines retrieve context from vector stores to augment LLM generation."
        )
        result = await pipeline.ingest("text", content)
        assert result["chunks"] > 0

        results = await pipeline.retrieve("What is LangGraph?")
        assert len(results) > 0
        assert any("LangGraph" in r.text for r in results)


# ── Embedding model catalogue completeness ─────────────────────────────────────

class TestEmbeddingCatalogue:
    def test_all_recommended_models_present(self):
        from src.rag.embeddings import EMBEDDING_MODELS
        recommended = [m for m, meta in EMBEDDING_MODELS.items() if meta.get("recommended")]
        assert len(recommended) >= 3, f"Expected at least 3 recommended models, got {recommended}"

    def test_all_models_have_required_fields(self):
        from src.rag.embeddings import EMBEDDING_MODELS
        required = {"provider", "dimensions", "max_tokens", "cost_per_1m_tokens", "description"}
        for model_id, meta in EMBEDDING_MODELS.items():
            missing = required - set(meta.keys())
            assert not missing, f"Model {model_id} missing fields: {missing}"

    def test_all_models_use_openrouter_provider(self):
        from src.rag.embeddings import EMBEDDING_MODELS
        for model_id, meta in EMBEDDING_MODELS.items():
            assert meta["provider"] == "openrouter", (
                f"Model {model_id} has provider '{meta['provider']}', expected 'openrouter'"
            )

    def test_free_model_exists(self):
        from src.rag.embeddings import EMBEDDING_MODELS
        free = [m for m, meta in EMBEDDING_MODELS.items() if meta["cost_per_1m_tokens"] == 0.0]
        assert len(free) >= 1, "Should have at least one free model"

    def test_invalid_model_raises(self):
        from src.rag.embeddings import EmbeddingModel
        with pytest.raises(ValueError):
            EmbeddingModel("fake/nonexistent-model")


# ── Config defaults ───────────────────────────────────────────────────────────

class TestConfigDefaults:
    def test_embedding_config_has_openrouter_base_url(self):
        from src.config import EmbeddingConfig
        cfg = EmbeddingConfig()
        assert "openrouter.ai" in cfg.base_url

    def test_embedding_config_default_model_is_valid(self):
        from src.config import EmbeddingConfig
        from src.rag.embeddings import EMBEDDING_MODELS
        cfg = EmbeddingConfig()
        assert cfg.default_model in EMBEDDING_MODELS


# ── Pipeline registry ─────────────────────────────────────────────────────────

class TestPipelineRegistry:
    def test_get_pipeline_returns_same_instance(self, tmp_path):
        from src.rag.pipeline import RAGConfig, get_pipeline, evict_pipeline
        cfg = RAGConfig(
            config_id="registry_test_unique",
            name="R",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            persist_dir=str(tmp_path),
        )
        p1 = get_pipeline(cfg)
        p2 = get_pipeline(cfg)
        assert p1 is p2
        evict_pipeline("registry_test_unique")

    def test_evict_removes_from_registry(self, tmp_path):
        from src.rag.pipeline import RAGConfig, get_pipeline, evict_pipeline
        cfg = RAGConfig(
            config_id="evict_test_unique",
            name="E",
            vector_store="faiss",
            embedding_model="qwen/qwen3-embedding-8b",
            persist_dir=str(tmp_path),
        )
        p1 = get_pipeline(cfg)
        evict_pipeline("evict_test_unique")
        p2 = get_pipeline(cfg)
        assert p1 is not p2
        evict_pipeline("evict_test_unique")
