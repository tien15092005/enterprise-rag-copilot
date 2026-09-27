from __future__ import annotations

from app.core.exceptions import AppError


class LLMConfigurationError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="llm_not_configured", status_code=503)


class LLMProviderError(AppError):
    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(
            message,
            code="llm_provider_failed",
            status_code=502,
            details={"provider": provider},
        )


class CitationValidationError(AppError):
    def __init__(self, message: str, *, unknown_keys: list[str] | None = None) -> None:
        super().__init__(
            message,
            code="invalid_llm_citations",
            status_code=502,
            details={"unknown_citation_keys": unknown_keys or []},
        )
