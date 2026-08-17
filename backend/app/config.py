from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data.db"

    # "transformers"  - loads base+adapter via transformers/peft (needs GPU for realistic latency)
    # "ollama"        - calls a local/remote Ollama server serving the GGUF export
    # "mock"          - deterministic fake scorer, no model weights needed (default for local dev)
    model_backend: str = "mock"

    base_model_path: str = "unsloth/gemma-3-4b-it-bnb-4bit"
    adapter_path: str = "../outputs/adapter"

    ollama_host: str = "http://localhost:11434"
    ollama_finetuned_model: str = "resume-jd-scorer"
    ollama_base_model: str = "gemma3:4b"

    anthropic_api_key: str | None = None
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
