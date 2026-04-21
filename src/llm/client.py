"""LLM client configured for Poe API (OpenAI-compatible).

Provides role-specific factory functions so agents, judges, and RCA
can each use the most appropriate model without touching call sites.
"""

import httpx
from langchain_openai import ChatOpenAI
from src.config import config

# Models that require extended thinking to be enabled via extra_body.
# These models reject requests without thinking.budget_tokens >= 1024.
# Only Poe-compatible model IDs that have been verified to accept the thinking header.
_THINKING_MODELS: set[str] = {
    "claude-sonnet-4.6",
}

_THINKING_BUDGET_TOKENS = 5000

# Hard timeout for all LLM calls — prevents indefinite hangs on slow models.
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=5.0)


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    resolved_model = model or config.llm.model
    resolved_temp = temperature if temperature is not None else config.llm.temperature
    resolved_max_tokens = max_tokens or config.llm.max_tokens

    kwargs: dict = dict(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=resolved_model,
        temperature=resolved_temp,
        max_tokens=resolved_max_tokens,
        http_client=httpx.Client(timeout=_HTTP_TIMEOUT),
        http_async_client=httpx.AsyncClient(timeout=_HTTP_TIMEOUT),
    )

    if resolved_model in _THINKING_MODELS:
        # Thinking models require budget_tokens >= 1024 and max_tokens >= budget_tokens.
        budget = _THINKING_BUDGET_TOKENS
        effective_max = max(resolved_max_tokens, budget + 1024)
        kwargs["max_tokens"] = effective_max
        kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": budget}}

    return ChatOpenAI(**kwargs)


def get_judge_llm(temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
    """LLM used for G-Eval scoring, semantic similarity, and DeepEval trace metrics."""
    return get_llm(model=config.llm.judge_model, temperature=temperature, max_tokens=max_tokens)


def get_rubric_judge_llm(
    temperature: float = 0.0,
    max_tokens: int | None = 8000,
) -> ChatOpenAI:
    """Opus-class judge for pairwise A/B output-quality analysis.

    Separate from `get_judge_llm` so DeepEval / G-Eval scoring keeps its pinned
    cheaper model while the A/B rubric judge can use Opus for better pairwise
    reasoning — at user-triggered (not automatic) cost only.
    """
    return get_llm(
        model=config.llm.rubric_judge_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_rca_llm(temperature: float = 0.0, max_tokens: int | None = None) -> ChatOpenAI:
    """LLM used for root-cause analysis on regression failures."""
    return get_llm(model=config.llm.rca_model, temperature=temperature, max_tokens=max_tokens)


def get_router_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Lightweight LLM for routing and supervisor decisions.

    Deliberately bypasses thinking-model detection — routing is a simple
    single-token decision ("router_decides" / "sequential" etc.) that does
    not benefit from extended thinking, and sending thinking tokens causes
    unnecessary provider 500s with some model variants.
    Max tokens is capped at 256 to keep routing calls fast and cheap.
    """
    return ChatOpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=config.llm.router_model,
        temperature=temperature,
        max_tokens=256,
        http_client=httpx.Client(timeout=_HTTP_TIMEOUT),
        http_async_client=httpx.AsyncClient(timeout=_HTTP_TIMEOUT),
    )
