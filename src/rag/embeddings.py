"""
Embedding model abstraction for the RAG pipeline.

All models are served via OpenRouter's embeddings API
(https://openrouter.ai/api/v1 — OpenAI-compatible).
This works from any region including HK.

The provider is always "openrouter" unless the user explicitly sets
EMBED_BASE_URL to something else.

Dimensions below are the native output sizes reported by each model.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Embedding model catalogue (all available on OpenRouter) ─────────────────

EMBEDDING_MODELS: dict[str, dict] = {
    # ── Best recommended defaults ────────────────────────────────────────────
    "openai/text-embedding-3-small": {
        "provider": "openrouter",
        "dimensions": 1536,
        "max_tokens": 8192,
        "cost_per_1m_tokens": 0.02,
        "description": "Fast, cheap, great all-around. Best default for most use cases.",
        "recommended": True,
    },
    "openai/text-embedding-3-large": {
        "provider": "openrouter",
        "dimensions": 3072,
        "max_tokens": 8192,
        "cost_per_1m_tokens": 0.13,
        "description": "Highest OpenAI accuracy. Best for large semantic corpora.",
        "recommended": False,
    },
    # ── Free tier (no API cost) ──────────────────────────────────────────────
    "nvidia/llama-nemotron-embed-vl-1b-v2:free": {
        "provider": "openrouter",
        "dimensions": 2048,
        "max_tokens": 131072,
        "cost_per_1m_tokens": 0.0,
        "description": "Free tier. Long context (131K tokens). Good for large documents.",
        "recommended": False,
    },
    # ── Perplexity ───────────────────────────────────────────────────────────
    "perplexity/pplx-embed-v1-4b": {
        "provider": "openrouter",
        "dimensions": 4096,
        "max_tokens": 32000,
        "cost_per_1m_tokens": 0.03,
        "description": "Perplexity large embed. Long context, high-dimensional. Best quality.",
        "recommended": True,
    },
    "perplexity/pplx-embed-v1-0.6b": {
        "provider": "openrouter",
        "dimensions": 1024,
        "max_tokens": 32000,
        "cost_per_1m_tokens": 0.004,
        "description": "Perplexity lightweight. Very cheap, long context, good speed.",
        "recommended": False,
    },
    # ── Qwen3 (multilingual, excellent for Asian languages + code) ───────────
    "qwen/qwen3-embedding-8b": {
        "provider": "openrouter",
        "dimensions": 4096,
        "max_tokens": 32000,
        "cost_per_1m_tokens": 0.01,
        "description": "Qwen3 large. Best multilingual + code embedding. Great for HK/CN.",
        "recommended": True,
    },
    "qwen/qwen3-embedding-4b": {
        "provider": "openrouter",
        "dimensions": 2560,
        "max_tokens": 32768,
        "cost_per_1m_tokens": 0.02,
        "description": "Qwen3 medium. Balanced multilingual performance.",
        "recommended": False,
    },
    # ── BGE (BAAI — excellent open-source models) ────────────────────────────
    "baai/bge-m3": {
        "provider": "openrouter",
        "dimensions": 1024,
        "max_tokens": 8192,
        "cost_per_1m_tokens": 0.01,
        "description": "BGE-M3: multilingual (100+ langs), hybrid retrieval support.",
        "recommended": True,
    },
    "baai/bge-large-en-v1.5": {
        "provider": "openrouter",
        "dimensions": 1024,
        "max_tokens": 512,
        "cost_per_1m_tokens": 0.01,
        "description": "BGE Large English. Strong semantic similarity for English docs.",
        "recommended": False,
    },
    # ── Mistral (great for code + technical docs) ────────────────────────────
    "mistralai/codestral-embed-2505": {
        "provider": "openrouter",
        "dimensions": 1024,
        "max_tokens": 8192,
        "cost_per_1m_tokens": 0.15,
        "description": "Mistral code embedding. Best for source code search RAG.",
        "recommended": False,
    },
    "mistralai/mistral-embed-2312": {
        "provider": "openrouter",
        "dimensions": 1024,
        "max_tokens": 8192,
        "cost_per_1m_tokens": 0.10,
        "description": "Mistral general embedding. Strong multilingual performance.",
        "recommended": False,
    },
    # ── Google ───────────────────────────────────────────────────────────────
    "google/gemini-embedding-001": {
        "provider": "openrouter",
        "dimensions": 3072,
        "max_tokens": 20000,
        "cost_per_1m_tokens": 0.15,
        "description": "Google Gemini embedding. High quality, long context.",
        "recommended": False,
    },
    # ── GTE family (lightweight, reliable) ──────────────────────────────────
    "thenlper/gte-large": {
        "provider": "openrouter",
        "dimensions": 1024,
        "max_tokens": 512,
        "cost_per_1m_tokens": 0.01,
        "description": "GTE Large. Efficient, good English semantic search.",
        "recommended": False,
    },
}


class EmbeddingModel:
    """
    Unified embedding interface — all models via OpenRouter's OpenAI-compatible API.

    Usage:
        model = EmbeddingModel("qwen/qwen3-embedding-8b")
        vec  = await model.embed("def fibonacci(n): ...")
        vecs = await model.embed_batch(["chunk1", "chunk2"])
    """

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        if model_id not in EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown embedding model '{model_id}'. "
                f"Available: {list(EMBEDDING_MODELS)}"
            )
        self.model_id = model_id
        self.meta = EMBEDDING_MODELS[model_id]
        self.provider = self.meta["provider"]
        self.dimensions = dimensions or self.meta["dimensions"]
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            # Resolve API key: explicit > OPENROUTER_KEY > POE_API_KEY
            key = (
                self._api_key
                or os.getenv("OPENROUTER_KEY")
                or os.getenv("OPENROUTER_API_KEY")
                or os.getenv("POE_API_KEY", "")
            )
            # Resolve base URL: explicit > EMBED_BASE_URL > OpenRouter
            base = (
                self._base_url
                or os.getenv("EMBED_BASE_URL")
                or "https://openrouter.ai/api/v1"
            )
            self._client = AsyncOpenAI(
                api_key=key,
                base_url=base,
                default_headers={
                    "HTTP-Referer": "https://sdlc-agent.local",
                    "X-Title": "SDLC Agent RAG",
                },
            )
        return self._client

    def _model_name(self) -> str:
        """Return raw model name for the API call (strip 'openrouter/' prefix if any)."""
        return self.model_id

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if not texts:
            return []
        client = self._get_client()
        model_name = self._model_name()
        # Strip the ':free' suffix some OpenRouter models use in the catalogue
        # but keep it in the actual API call as OpenRouter needs it
        resp = await client.embeddings.create(
            model=model_name,
            input=texts,
        )
        # Sort by index to preserve order
        items = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in items]
