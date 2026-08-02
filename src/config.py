"""
Application configuration — loaded from .env via Pydantic Settings.

All secrets live in .env (never hardcoded). This is the single source of truth
for configuration across the entire app. Import `settings` anywhere you need config.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: .../src/config.py -> .../
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env.local", PROJECT_ROOT / ".env"),
        extra="ignore",
    )

    # Database
    database_url: str

    # Embedding
    embedding_provider: str = "cohere"
    cohere_key: str = ""
    cohere_model: str = "embed-english-v3.0"

    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # Observability
    observability_provider: str = "none"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # App
    app_env: str = "development"
    log_level: str = "INFO"


settings = Settings()
