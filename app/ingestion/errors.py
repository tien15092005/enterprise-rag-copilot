from app.core.exceptions import AppError


class IngestionError(AppError):
    def __init__(self, message: str, *, code: str = "ingestion_failed") -> None:
        super().__init__(message, code=code, status_code=422)


class UnsupportedDocumentError(IngestionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsupported_document")


class DuplicateDocumentVersionError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="duplicate_document_version", status_code=409)


class DuplicateLogicalDocumentError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="duplicate_logical_document", status_code=409)


class DocumentNotFoundError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="document_not_found", status_code=404)


class StoredFileTooLargeError(AppError):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(
            f"uploaded file exceeds the {max_bytes}-byte limit",
            code="file_too_large",
            status_code=413,
        )
