from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Enterprise Knowledge Copilot"
    app_env: Literal["development", "test", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    api_prefix: str = "/api"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = True

    database_url: str = (
        "postgresql+asyncpg://knowledge_copilot:change-me@localhost:5432/knowledge_copilot"
    )
    database_echo: bool = False

    upload_directory: str = "data/uploads"
    max_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    chunk_size: int = Field(default=400, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "knowledge_chunks_multilingual_e5_small_v1"
    qdrant_timeout_seconds: float = Field(default=30.0, gt=0)

    embedding_provider: Literal["sentence_transformers"] = "sentence_transformers"
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_model_revision: str = "0e60b8d9d2166d80387f86e3b48ec9ced55f4d15"
    embedding_dimension: int = Field(default=384, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_device: str = "cpu"
    embedding_normalize: bool = True
    embedding_cache_directory: str = "data/model_cache"

    llm_provider: Literal["openai_compatible", "ollama"] | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_max_output_tokens: int = Field(default=700, ge=1)
    rag_max_context_characters: int = Field(default=12000, ge=500)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings snapshot."""

    return Settings()
