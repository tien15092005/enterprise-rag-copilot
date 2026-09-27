from app.core.exceptions import AppError


class RerankerError(AppError):
    def __init__(self, message: str = "reranker execution failed") -> None:
        super().__init__(message, code="reranker_failed", status_code=503)


class RerankerProvenanceError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="reranker_provenance_violation", status_code=500)
