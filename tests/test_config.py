import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings


def test_settings_load_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_PORT", "8100")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")

    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert settings.app_port == 8100
    assert isinstance(settings.openai_api_key, SecretStr)
    assert "not-a-real-key" not in repr(settings.openai_api_key)


def test_settings_reject_invalid_port() -> None:
    with pytest.raises(ValidationError):
        Settings(app_port=70000)


def test_settings_reject_final_top_k_above_rerank_pool() -> None:
    with pytest.raises(ValidationError):
        Settings(rerank_candidate_limit=3, rerank_final_top_k=4)


def test_production_rejects_placeholder_credentials_and_wildcard_cors() -> None:
    with pytest.raises(ValidationError, match="placeholder credentials"):
        Settings(app_env="production")
    with pytest.raises(ValidationError, match="must not contain"):
        Settings(
            app_env="production",
            database_url="postgresql+asyncpg://service:secure@db/service",
            cors_allowed_origins=["*"],
        )


def test_chunk_defaults_and_agent_deadline_are_validated() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValidationError):
        Settings(agent_timeout_seconds=0)
