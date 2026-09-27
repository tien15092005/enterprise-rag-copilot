from app.core.exceptions import AppError


class VersionNotReadyError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="version_not_ready", status_code=409)


class VersionNotIndexedError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="version_not_indexed", status_code=409)


class VectorIndexingError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="vector_indexing_failed", status_code=502)


class RetrievalProvenanceError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="retrieval_provenance_mismatch", status_code=500)
