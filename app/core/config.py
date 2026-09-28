from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
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
    cors_allowed_origins: list[str] = Field(default_factory=list)

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
    citation_excerpt_max_characters: int = Field(default=500, ge=1)
    dense_candidate_limit: int = Field(default=20, ge=1, le=100)
    lexical_candidate_limit: int = Field(default=20, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    evidence_min_dense_score: float | None = None
    evidence_min_rrf_score: float | None = None
    evidence_min_reranker_score: float | None = None
    reranker_provider: Literal["cross_encoder"] = "cross_encoder"
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_model_revision: str = "1427fd652930e4ba29e8149678df786c240d8825"
    reranker_device: str = "cpu"
    reranker_batch_size: int = Field(default=8, ge=1, le=128)
    reranker_max_length: int = Field(default=512, ge=32, le=8192)
    reranker_cache_directory: str = "data/model_cache"
    rerank_candidate_limit: int = Field(default=20, ge=1, le=100)
    rerank_final_top_k: int = Field(default=5, ge=1, le=100)
    comparison_content_similarity_threshold: float = Field(default=0.6, ge=0, le=1)
    comparison_semantic_similarity_threshold: float = Field(default=0.95, ge=-1, le=1)
    comparison_max_sections: int = Field(default=100, ge=1, le=1000)
    agent_max_tool_steps: int = Field(default=4, ge=1, le=10)
    agent_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, value: list[str]) -> list[str]:
        return [origin.rstrip("/") for origin in value if origin.strip()]

    @model_validator(mode="after")
    def validate_reranking_limits(self) -> Settings:
        if self.rerank_final_top_k > self.rerank_candidate_limit:
            raise ValueError("RERANK_FINAL_TOP_K cannot exceed RERANK_CANDIDATE_LIMIT")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP cannot equal or exceed CHUNK_SIZE")
        if self.app_env == "production":
            if "change-me" in self.database_url:
                raise ValueError("production DATABASE_URL must not use placeholder credentials")
            if "*" in self.cors_allowed_origins:
                raise ValueError("production CORS_ALLOWED_ORIGINS must not contain '*'")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide immutable settings snapshot."""

    return Settings()
