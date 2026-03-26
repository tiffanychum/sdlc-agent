import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("POE_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.poe.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-5.3-codex"))
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    eval_output_dir: str = "eval/results"


config = AppConfig()
