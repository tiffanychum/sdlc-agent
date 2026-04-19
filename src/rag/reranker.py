"""
BGE Cross-Encoder Reranker — CPU-friendly reranking for RAG pipelines.

Supported models (all run on CPU via sentence-transformers):
  bge-reranker-base  — BAAI/bge-reranker-base   (~278 MB)  Fast, good quality
  bge-reranker-large — BAAI/bge-reranker-large  (~670 MB)  Slower, better quality
  bge-reranker-v2-m3 — BAAI/bge-reranker-v2-m3 (~570 MB)  Multilingual

Usage:
    from src.rag.reranker import rerank
    results = await rerank(query, results, model="bge-reranker-base", top_k=5)

The module lazy-loads and caches model instances — first call downloads
the model (~30-60 s on first run) then is instant on subsequent calls.
"""

import asyncio
import logging
import time
from functools import lru_cache

from src.rag.vectorstore import SearchResult

logger = logging.getLogger(__name__)

# ── Model catalog ──────────────────────────────────────────────────────────────

RERANKER_MODELS: dict[str, dict] = {
    "none": {
        "label": "No Reranker",
        "description": "Skip reranking — use raw retrieval scores.",
        "hf_id": None,
    },
    "bge-reranker-base": {
        "label": "BGE Reranker Base",
        "description": "BAAI/bge-reranker-base — fast, ~278 MB, CPU-friendly.",
        "hf_id": "BAAI/bge-reranker-base",
    },
    "bge-reranker-large": {
        "label": "BGE Reranker Large",
        "description": "BAAI/bge-reranker-large — higher quality, ~670 MB, slower on CPU.",
        "hf_id": "BAAI/bge-reranker-large",
    },
    "bge-reranker-v2-m3": {
        "label": "BGE Reranker v2-m3",
        "description": "BAAI/bge-reranker-v2-m3 — multilingual, ~570 MB, CPU-friendly.",
        "hf_id": "BAAI/bge-reranker-v2-m3",
    },
}


# ── Lazy model loader ──────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _load_cross_encoder(hf_id: str):
    """Load a sentence-transformers CrossEncoder and cache in memory."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    logger.info("Loading BGE reranker '%s' (first run downloads model)…", hf_id)
    t0 = time.monotonic()
    # device="cpu" is explicit — no GPU assumed
    model = CrossEncoder(hf_id, device="cpu", max_length=512)
    logger.info("Reranker '%s' loaded in %.1fs", hf_id, time.monotonic() - t0)
    return model


# ── Public API ─────────────────────────────────────────────────────────────────

async def rerank(
    query: str,
    results: list[SearchResult],
    model: str = "bge-reranker-base",
    top_k: int | None = None,
) -> list[SearchResult]:
    """
    Rerank *results* using a BGE cross-encoder model.

    Runs inference in a thread pool to avoid blocking the event loop.
    Returns up to *top_k* results sorted by reranker score (descending).
    """
    if model == "none" or not results:
        return results

    meta = RERANKER_MODELS.get(model)
    if meta is None or meta["hf_id"] is None:
        logger.warning("Unknown reranker '%s', skipping.", model)
        return results

    hf_id: str = meta["hf_id"]
    pairs = [(query, r.text) for r in results]

    def _score() -> list[float]:
        cross_encoder = _load_cross_encoder(hf_id)
        return cross_encoder.predict(pairs).tolist()

    loop = asyncio.get_event_loop()
    scores: list[float] = await loop.run_in_executor(None, _score)

    # Attach reranker scores and sort
    reranked = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    import math
    k = top_k or len(results)
    final: list[SearchResult] = []
    for r, score in reranked[:k]:
        # Replace the original retrieval score with the cross-encoder score
        # (normalised to [0, 1] via sigmoid so it's comparable with retrieval scores)
        normalised = 1.0 / (1.0 + math.exp(-score))
        final.append(SearchResult(
            text=r.text,
            score=round(normalised, 4),
            source=r.source,
            chunk_index=r.chunk_index,
            total_chunks=r.total_chunks,
            page=r.page,
            metadata=getattr(r, "metadata", {}),
        ))
    return final
