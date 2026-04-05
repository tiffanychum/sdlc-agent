import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("POE_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.poe.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-5.3-codex"))
    judge_model: str = field(default_factory=lambda: os.getenv("LLM_JUDGE_MODEL", "gpt-4o"))
    rca_model: str = field(default_factory=lambda: os.getenv("LLM_RCA_MODEL", "gpt-4o"))
    router_model: str = field(default_factory=lambda: os.getenv("LLM_ROUTER_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingConfig:
    """Configuration for RAG embedding calls — routed through OpenRouter by default."""
    # API key: prefers OPENROUTER_KEY (works from any region including HK)
    api_key: str = field(
        default_factory=lambda: (
            os.getenv("OPENROUTER_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("POE_API_KEY", "")
        )
    )
    # Base URL: OpenRouter's embedding endpoint (OpenAI-compatible)
    base_url: str = field(
        default_factory=lambda: os.getenv("EMBED_BASE_URL", "https://openrouter.ai/api/v1")
    )
    # Default model if not set per-pipeline; Qwen3 is excellent for multilingual + code
    default_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "qwen/qwen3-embedding-8b")
    )


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    eval_output_dir: str = "eval/results"


config = AppConfig()
