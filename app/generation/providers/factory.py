from app.core.config import Settings
from app.generation.errors import LLMConfigurationError
from app.generation.providers.base import LLMProvider
from app.generation.providers.ollama import OllamaProvider
from app.generation.providers.openai_compatible import OpenAICompatibleProvider


def create_llm_provider(settings: Settings) -> LLMProvider | None:
    if settings.llm_provider is None:
        return None
    if settings.llm_provider == "openai_compatible":
        if settings.openai_api_key is None or not settings.openai_model:
            raise LLMConfigurationError(
                "OPENAI_API_KEY and OPENAI_MODEL are required for openai_compatible"
            )
        return OpenAICompatibleProvider(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if not settings.ollama_model:
        raise LLMConfigurationError("OLLAMA_MODEL is required for ollama")
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
