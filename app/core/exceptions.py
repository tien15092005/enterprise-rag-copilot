from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base exception for expected application failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "application_error",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
