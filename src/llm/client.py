"""LLM client configured for Poe API (OpenAI-compatible)."""

from langchain_openai import ChatOpenAI
from src.config import config


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=model or config.llm.model,
        temperature=temperature if temperature is not None else config.llm.temperature,
        max_tokens=max_tokens or config.llm.max_tokens,
    )
