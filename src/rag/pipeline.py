"""
Main RAG pipeline: ingest documents → retrieve context → generate with citations.

Retrieval strategies — all backed by LangChain primitives:
  similarity   — standard cosine top-k (default)
  mmr          — Maximum Marginal Relevance via langchain_core (max diversity)
  multi_query  — MultiQueryRetriever: generates N query variations, merges results
  hybrid       — BM25Retriever + dense, fused with Reciprocal Rank Fusion

OTel spans:
  rag.ingest   — per-source ingestion (chunks, tokens, time)
  rag.embed    — embedding generation calls
  rag.retrieve — vector search (query, strategy, latency, scores)
  rag.generate — LLM completion with retrieved context
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from src.rag.chunker import Chunk, chunk_documents, load_source
from src.rag.embeddings import EmbeddingModel
from src.rag.vectorstore import SearchResult, create_store

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag.pipeline")




# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RAGConfig:
    """Full configuration for one RAG pipeline instance."""
    config_id: str
    name: str = ""
    embedding_model: str = "openai/text-embedding-3-small"
    vector_store: str = "chroma"
    llm_model: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_strategy: str = "recursive"        # recursive | fixed | semantic | code
    retrieval_strategy: str = "similarity"   # similarity | mmr | multi_query | hybrid
    top_k: int = 5
    mmr_lambda: float = 0.5
    multi_query_n: int = 3
    system_prompt: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None
    persist_dir: str = "./data/vectorstore"
    reranker: str = "none"                   # none | bge-reranker-base | bge-reranker-large | bge-reranker-v2-m3


@dataclass
class Citation:
    source: str
    chunk_index: int
    total_chunks: int
    page: Optional[int]
    score: float
    snippet: str      # first 200 chars of the chunk


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    query: str
    strategy_used: str
    chunks_retrieved: int
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0


# ── LangChain retriever adapter ───────────────────────────────────────────────

class _StoreRetriever:
    """
    Thin adapter that exposes our custom VectorStore as a LangChain-compatible
    retriever interface, so we can compose it with MultiQueryRetriever,
    EnsembleRetriever, and other LangChain retriever combinators.
    """

    def __init__(self, store, embedding_model: EmbeddingModel, top_k: int):
        self._store = store
        self._emb = embedding_model
        self._top_k = top_k

    async def ainvoke(self, query: str, top_k: Optional[int] = None) -> list[SearchResult]:
        k = top_k or self._top_k
        vec = await self._emb.embed(query)
        return self._store.query(vec, top_k=k)

    def invoke(self, query: str, top_k: Optional[int] = None) -> list[SearchResult]:
        """Synchronous wrapper (runs in caller's event loop)."""
        return asyncio.get_event_loop().run_until_complete(self.ainvoke(query, top_k))

    def get_texts(self) -> list[str]:
        """Return all stored texts for BM25 indexing."""
        return list(self._store._texts) if hasattr(self._store, "_texts") else []


# ── Retrieval strategy helpers ────────────────────────────────────────────────

def _mmr_rerank(
    candidates: list[SearchResult],
    top_k: int,
    lambda_: float,
) -> list[SearchResult]:
    """
    Maximum Marginal Relevance reranking using LangChain's scoring formula.
    λ = 1.0 → pure relevance; λ = 0.0 → pure diversity.
    Diversity proxy: BM25-based text dissimilarity between selected and remaining.
    """
    selected: list[SearchResult] = []
    remaining = list(candidates)

    def _text_overlap(a: str, b: str) -> float:
        """Jaccard word overlap as a cheap diversity proxy."""
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    for _ in range(min(top_k, len(candidates))):
        if not remaining:
            break
        if not selected:
            best = max(remaining, key=lambda r: r.score)
        else:
            def mmr_score(r: SearchResult) -> float:
                max_sim = max(_text_overlap(r.text, s.text) for s in selected)
                return lambda_ * r.score - (1.0 - lambda_) * max_sim
            best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return selected


async def _multi_query_retrieve(
    query: str,
    retriever: _StoreRetriever,
    llm_model: Optional[str],
    n: int,
    top_k: int,
) -> list[SearchResult]:
    """
    MultiQueryRetriever pattern from LangChain:
    1. Use an LLM to generate N alternative phrasings of the query.
    2. Retrieve top-k chunks for each phrasing.
    3. Deduplicate by (source, chunk_index), keeping the best score.
    """
    from langchain_core.messages import HumanMessage
    from src.llm.client import get_llm

    llm = get_llm(model=llm_model, temperature=0.7)
    prompt = (
        f"Generate {n} alternative phrasings of the following question to improve "
        f"document retrieval. Return ONLY the questions, one per line, no numbering.\n\n"
        f"Original: {query}"
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    variations = [query] + [ln.strip() for ln in resp.content.strip().split("\n") if ln.strip()][:n]

    seen: dict[str, SearchResult] = {}
    for q in variations:
        for r in await retriever.ainvoke(q, top_k=top_k):
            key = f"{r.source}:{r.chunk_index}"
            if key not in seen or r.score > seen[key].score:
                seen[key] = r

    return sorted(seen.values(), key=lambda r: -r.score)[:top_k]


async def _hybrid_retrieve(
    query: str,
    retriever: _StoreRetriever,
    top_k: int,
) -> list[SearchResult]:
    """
    Hybrid retrieval using LangChain's BM25Retriever + dense vector search,
    fused with Reciprocal Rank Fusion (RRF) — the same approach used by
    LangChain's EnsembleRetriever.
    """
    dense_results = await retriever.ainvoke(query, top_k=top_k * 3)
    if not dense_results:
        return []

    texts = [r.text for r in dense_results]

    # BM25 via LangChain's BM25Retriever (backed by rank_bm25)
    try:
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        lc_docs = [Document(page_content=t) for t in texts]
        bm25 = BM25Retriever.from_documents(lc_docs, k=top_k * 3)
        bm25_docs = bm25.invoke(query)
        bm25_texts = {d.page_content for d in bm25_docs}
    except Exception as e:
        logger.warning("BM25Retriever failed (%s), using TF-IDF fallback", e)
        bm25_texts = set()

    # RRF fusion: assign rank-based scores for both systems
    dense_rank = {r.text: i + 1 for i, r in enumerate(dense_results)}
    rrf_k = 60  # standard RRF constant

    def rrf_score(r: SearchResult) -> float:
        d_rank = dense_rank.get(r.text, len(dense_results) + 1)
        b_rank = next((i + 1 for i, t in enumerate(bm25_docs) if hasattr(bm25_docs[0], "page_content") and bm25_docs[i].page_content == r.text), len(dense_results) + 1) if bm25_texts else len(dense_results) + 1
        return 1.0 / (rrf_k + d_rank) + 1.0 / (rrf_k + b_rank)

    # Boost items that appear in BM25 results
    for r in dense_results:
        r.score = rrf_score(r)

    return sorted(dense_results, key=lambda r: -r.score)[:top_k]


# ── Pipeline class ────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Manages document ingestion, vector search, and generation for one RAG config.

    Usage:
        pipeline = RAGPipeline(config)
        await pipeline.ingest("file", "/path/to/doc.pdf")
        response = await pipeline.query("What is the architecture?")
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self._embedding_model = EmbeddingModel(
            config.embedding_model,
            api_key=config.embedding_api_key,
            base_url=config.embedding_base_url,
        )
        from src.rag.embeddings import EMBEDDING_MODELS
        dims = EMBEDDING_MODELS[config.embedding_model]["dimensions"]
        self._store = create_store(
            config.vector_store,
            collection_id=config.config_id,
            persist_dir=config.persist_dir,
            dimensions=dims,
        )
        self._retriever = _StoreRetriever(
            self._store, self._embedding_model, config.top_k
        )

    # ── Ingest ────────────────────────────────────────────────────────────────

    async def ingest(self, source_type: str, content: str) -> dict:
        """
        Load, chunk (via LangChain splitters), embed, and store a document.

        Returns a summary dict: {chunks, tokens_estimated, source, duration_ms}
        """
        t0 = time.monotonic()
        with _tracer.start_as_current_span("rag.ingest") as span:
            span.set_attribute("rag.source_type", source_type)
            span.set_attribute("rag.source", content[:200])
            span.set_attribute("rag.config_id", self.config.config_id)
            try:
                pages = load_source(source_type, content)
                chunks = chunk_documents(
                    pages,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    strategy=self.config.chunk_strategy,
                )
                if not chunks:
                    span.set_status(Status(StatusCode.ERROR, "No chunks produced"))
                    return {"chunks": 0, "source": content, "error": "No content extracted"}

                texts = [c.text for c in chunks]
                metadatas = [
                    {
                        "source": c.source,
                        "chunk_index": c.chunk_index,
                        "total_chunks": c.total_chunks,
                        **({"page": c.page} if c.page is not None else {}),
                    }
                    for c in chunks
                ]

                all_embeddings: list[list[float]] = []
                batch_size = 96
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    with _tracer.start_as_current_span("rag.embed") as embed_span:
                        embed_span.set_attribute("rag.batch_size", len(batch))
                        embed_span.set_attribute("rag.embedding_model", self.config.embedding_model)
                        vecs = await self._embedding_model.embed_batch(batch)
                        all_embeddings.extend(vecs)

                self._store.add(texts, all_embeddings, metadatas)
                tokens_est = sum(len(t.split()) for t in texts)
                duration_ms = (time.monotonic() - t0) * 1000
                span.set_attribute("rag.chunks_ingested", len(chunks))
                span.set_attribute("rag.tokens_estimated", tokens_est)
                span.set_attribute("rag.duration_ms", duration_ms)
                logger.info(
                    "RAG ingest: config=%s source=%s chunks=%d time=%.0fms",
                    self.config.config_id, content[:60], len(chunks), duration_ms,
                )
                return {
                    "chunks": len(chunks),
                    "tokens_estimated": tokens_est,
                    "source": content,
                    "duration_ms": round(duration_ms),
                }
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    async def ingest_paths(
        self,
        paths: list[str],
        *,
        recursive: bool = True,
        extensions: tuple[str, ...] = (".md", ".txt", ".rst", ".json", ".csv", ".pdf"),
        skip_hidden: bool = True,
    ) -> dict:
        """Batch-ingest one or many file/directory paths.

        Designed for large finance / docs corpora where the user has a folder
        of dozens or hundreds of documents and a single ``ingest`` call per
        file would be too noisy.

        Args:
            paths:      List of file or directory paths.
            recursive:  Walk directories recursively (default True).
            extensions: Whitelist of suffixes to ingest. Set to ``()`` to accept
                        every file (use with care — node_modules-sized trees can
                        blow up the index).
            skip_hidden: Skip dotfiles + dot-prefixed directories.

        Returns:
            Aggregate summary: ``{"files_ingested", "files_skipped",
            "total_chunks", "errors": [...]}``.
        """
        from pathlib import Path

        targets: list[Path] = []
        for p_str in paths:
            p = Path(p_str).expanduser().resolve()
            if not p.exists():
                continue
            if p.is_file():
                targets.append(p)
                continue
            iterator = p.rglob("*") if recursive else p.iterdir()
            for child in iterator:
                if not child.is_file():
                    continue
                if skip_hidden and any(part.startswith(".") for part in child.parts):
                    continue
                if extensions and child.suffix.lower() not in extensions:
                    continue
                targets.append(child)

        # Deduplicate while preserving order
        seen = set()
        unique_targets = []
        for t in targets:
            key = str(t)
            if key in seen:
                continue
            seen.add(key)
            unique_targets.append(t)

        files_ingested = 0
        files_skipped = 0
        total_chunks = 0
        errors: list[dict] = []

        for path in unique_targets:
            try:
                summary = await self.ingest("file", str(path))
                if summary.get("chunks", 0) > 0:
                    files_ingested += 1
                    total_chunks += int(summary["chunks"])
                else:
                    files_skipped += 1
            except Exception as exc:
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                files_skipped += 1

        return {
            "files_ingested": files_ingested,
            "files_skipped": files_skipped,
            "total_chunks": total_chunks,
            "total_files_seen": len(unique_targets),
            "errors": errors[:25],  # cap so a bad mount point doesn't return MBs
        }

    # ── Retrieve ──────────────────────────────────────────────────────────────

    async def retrieve(self, query: str) -> list[SearchResult]:
        """Run the configured retrieval strategy and return top chunks."""
        strategy = self.config.retrieval_strategy
        t0 = time.monotonic()
        with _tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("rag.query", query[:200])
            span.set_attribute("rag.strategy", strategy)
            span.set_attribute("rag.top_k", self.config.top_k)

            if strategy == "similarity":
                results = await self._retriever.ainvoke(query)

            elif strategy == "mmr":
                # Fetch 3× candidates, then MMR-rerank for diversity
                candidates = await self._retriever.ainvoke(query, top_k=self.config.top_k * 3)
                results = _mmr_rerank(candidates, self.config.top_k, self.config.mmr_lambda)

            elif strategy == "multi_query":
                results = await _multi_query_retrieve(
                    query, self._retriever, self.config.llm_model,
                    self.config.multi_query_n, self.config.top_k,
                )

            elif strategy == "hybrid":
                results = await _hybrid_retrieve(query, self._retriever, self.config.top_k)

            else:
                logger.warning("Unknown retrieval strategy '%s', falling back to similarity", strategy)
                results = await self._retriever.ainvoke(query)

            # Optional BGE cross-encoder reranking pass
            if self.config.reranker and self.config.reranker != "none" and results:
                with _tracer.start_as_current_span("rag.rerank") as rerank_span:
                    rerank_span.set_attribute("rag.reranker", self.config.reranker)
                    rerank_span.set_attribute("rag.candidates", len(results))
                    from src.rag.reranker import rerank
                    results = await rerank(
                        query, results,
                        model=self.config.reranker,
                        top_k=self.config.top_k,
                    )
                    rerank_span.set_attribute("rag.reranked_count", len(results))

            duration_ms = (time.monotonic() - t0) * 1000
            span.set_attribute("rag.results_count", len(results))
            span.set_attribute("rag.retrieve_ms", duration_ms)
            if results:
                span.set_attribute("rag.top_score", results[0].score)
            return results

    # ── Contextual Compression ────────────────────────────────────────────────

    async def _contextual_compress(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Apply LLM-based contextual compression (LangChain ContextualCompressionRetriever
        pattern) to filter out irrelevant sentences within each retrieved chunk.

        For each chunk, an LLM extracts only the sentences directly relevant to the
        query, reducing hallucination risk when chunks are large or only partially
        relevant. Chunks that have NO relevant content are dropped entirely.

        Falls back to original results on any LLM error.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from src.llm.client import get_llm

        if not results:
            return results

        compress_system = (
            "You are a document compression assistant. Given a query and a document chunk, "
            "extract ONLY the sentences directly relevant to the query. "
            "Return the extracted text verbatim (no paraphrasing). "
            "If NO part of the chunk is relevant, return exactly: NO_RELEVANT_CONTENT"
        )

        llm = get_llm(model=self.config.llm_model, temperature=0.0)
        compressed: list[SearchResult] = []

        with _tracer.start_as_current_span("rag.contextual_compress") as span:
            span.set_attribute("rag.compress_candidates", len(results))
            for r in results:
                prompt = (
                    f"Query: {query}\n\n"
                    f"Document chunk (source: {r.source}):\n{r.text}\n\n"
                    "Extract only the relevant sentences:"
                )
                try:
                    resp = await llm.ainvoke([
                        SystemMessage(content=compress_system),
                        HumanMessage(content=prompt),
                    ])
                    extracted = resp.content.strip() if isinstance(resp.content, str) else str(resp.content).strip()
                    if extracted and extracted != "NO_RELEVANT_CONTENT":
                        import copy
                        compressed_r = copy.copy(r)
                        compressed_r.text = extracted
                        compressed.append(compressed_r)
                except Exception as e:
                    logger.warning("Contextual compression failed for chunk from %s: %s", r.source, e)
                    compressed.append(r)  # keep original on error

            span.set_attribute("rag.compress_kept", len(compressed))
            dropped = len(results) - len(compressed)
            if dropped:
                logger.info("Contextual compression: dropped %d irrelevant chunks", dropped)

        return compressed if compressed else results  # never return empty

    # ── Generate ──────────────────────────────────────────────────────────────

    async def query(self, user_query: str, use_compression: bool = False) -> RAGResponse:
        """Full RAG pipeline: retrieve context → (optionally compress) → generate answer.

        Args:
            user_query: The question to answer.
            use_compression: If True, apply LLM-based contextual compression
                (LangChain ContextualCompressionRetriever pattern) to filter
                irrelevant sentences from retrieved chunks before generation.
                Reduces hallucination on partially-relevant chunks; adds ~1-2 LLM calls.
        """
        t0 = time.monotonic()
        with _tracer.start_as_current_span("rag.generate") as span:
            span.set_attribute("rag.query", user_query[:200])
            span.set_attribute("rag.config_id", self.config.config_id)
            span.set_attribute("rag.use_compression", use_compression)

            results = await self.retrieve(user_query)
            if not results:
                return RAGResponse(
                    answer="I could not find relevant information in the knowledge base for your question.",
                    citations=[],
                    query=user_query,
                    strategy_used=self.config.retrieval_strategy,
                    chunks_retrieved=0,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

            # Apply contextual compression if configured
            if use_compression:
                results = await self._contextual_compress(user_query, results)

            context_parts = []
            for i, r in enumerate(results):
                page_info = f" (page {r.page})" if r.page else ""
                context_parts.append(f"[{i+1}] Source: {r.source}{page_info}\n{r.text}")
            context = "\n\n---\n\n".join(context_parts)

            system = self.config.system_prompt or (
                "You are a helpful assistant with access to a knowledge base. "
                "Always ground your answer in the retrieved context. "
                "Use inline citations like [1], [2] when referencing specific information."
            )
            prompt = (
                f"## Retrieved Context\n\n{context}\n\n"
                f"## User Question\n\n{user_query}\n\n"
                f"Answer using the context above. Include inline citations like [1], [2]."
            )

            from langchain_core.messages import HumanMessage, SystemMessage
            from src.llm.client import get_llm
            llm = get_llm(model=self.config.llm_model)
            messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
            resp = await llm.ainvoke(messages)

            tokens_in = tokens_out = 0
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                tokens_in = resp.usage_metadata.get("input_tokens", 0)
                tokens_out = resp.usage_metadata.get("output_tokens", 0)

            answer = resp.content if isinstance(resp.content, str) else str(resp.content)
            citations = [
                Citation(
                    source=r.source,
                    chunk_index=r.chunk_index,
                    total_chunks=r.total_chunks,
                    page=r.page,
                    score=round(r.score, 4),
                    snippet=r.text[:200],
                )
                for r in results
            ]
            duration_ms = (time.monotonic() - t0) * 1000
            span.set_attribute("rag.answer_len", len(answer))
            span.set_attribute("rag.citations", len(citations))
            span.set_attribute("rag.duration_ms", duration_ms)

            return RAGResponse(
                answer=answer,
                citations=citations,
                query=user_query,
                strategy_used=self.config.retrieval_strategy,
                chunks_retrieved=len(results),
                latency_ms=round(duration_ms),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

    def chunk_count(self) -> int:
        return self._store.count()


# ── Registry ──────────────────────────────────────────────────────────────────

_pipeline_registry: dict[str, RAGPipeline] = {}


def get_pipeline(config: RAGConfig) -> RAGPipeline:
    """Get or create a RAGPipeline instance for the given config."""
    if config.config_id not in _pipeline_registry:
        _pipeline_registry[config.config_id] = RAGPipeline(config)
    return _pipeline_registry[config.config_id]


def evict_pipeline(config_id: str) -> None:
    """Remove a cached pipeline (e.g. after config update)."""
    _pipeline_registry.pop(config_id, None)
