"""
Application configuration — loaded from .env via Pydantic Settings.

All secrets live in .env (never hardcoded). This is the single source of truth
for configuration across the entire app. Import `settings` anywhere you need config.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str

    # Embedding
    embedding_provider: str = "openai"
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

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
