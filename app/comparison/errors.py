from uuid import UUID

from app.core.exceptions import AppError


class ComparisonError(AppError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "comparison_invalid",
        status_code: int = 409,
    ) -> None:
        super().__init__(message, code=code, status_code=status_code)


class ComparisonVersionNotFoundError(ComparisonError):
    def __init__(self, version_id: UUID) -> None:
        super().__init__(
            f"document version '{version_id}' was not found",
            code="comparison_version_not_found",
            status_code=404,
        )
