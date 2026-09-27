import pytest

from app.core.config import Settings
from app.generation.errors import LLMConfigurationError
from app.generation.providers.factory import create_llm_provider
from app.generation.providers.ollama import OllamaProvider
from app.generation.providers.openai_compatible import OpenAICompatibleProvider


def test_no_provider_is_valid_for_non_generation_paths() -> None:
    assert create_llm_provider(Settings(llm_provider=None)) is None


def test_openai_compatible_provider_construction() -> None:
    provider = create_llm_provider(
        Settings(
            llm_provider="openai_compatible",
            openai_api_key="placeholder-test-key",
            openai_model="test-model",
        )
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model_name == "test-model"


def test_ollama_provider_construction() -> None:
    provider = create_llm_provider(Settings(llm_provider="ollama", ollama_model="small-test"))
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "small-test"


def test_selected_provider_requires_configuration() -> None:
    with pytest.raises(LLMConfigurationError):
        create_llm_provider(Settings(llm_provider="openai_compatible"))
