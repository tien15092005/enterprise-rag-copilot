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
