from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import ValidationError

from app.api.dependencies import (
    get_catalog_service,
    get_ingestion_service,
    get_request_settings,
)
from app.core.config import Settings
from app.core.exceptions import AppError
from app.db.models import DocumentVersion, LogicalDocument
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.models.documents import (
    DocumentVersionCreate,
    DocumentVersionDetail,
    DocumentVersionSummary,
    IngestionAcceptedResponse,
    LogicalDocumentCreate,
    LogicalDocumentDetail,
    LogicalDocumentResponse,
)
from app.models.versioning import DocumentLanguage

router = APIRouter(tags=["documents"])
CatalogService = Annotated[DocumentCatalogService, Depends(get_catalog_service)]
VersionService = Annotated[IngestionService, Depends(get_ingestion_service)]
AppSettings = Annotated[Settings, Depends(get_request_settings)]


def version_summary(version: DocumentVersion) -> DocumentVersionSummary:
    return DocumentVersionSummary.model_validate(version)


async def version_detail(
    version: DocumentVersion, service: IngestionService
) -> DocumentVersionDetail:
    summary = version_summary(version)
    return DocumentVersionDetail(
        **summary.model_dump(),
        effective_from=version.effective_from,
        effective_to=version.effective_to,
        published_at=version.published_at,
        supersedes_version_id=version.supersedes_version_id,
        storage_key=version.storage_key,
        ingestion_config=version.ingestion_config,
        embedding_model=version.embedding_model,
        embedding_revision=version.embedding_revision,
        embedding_dimension=version.embedding_dimension,
        vector_indexed_at=version.vector_indexed_at,
        vector_index_error=version.vector_index_error,
        chunk_count=await service.chunk_count(version.document_version_id),
        ingestion_job=version.ingestion_job,
    )


def document_detail(document: LogicalDocument) -> LogicalDocumentDetail:
    return LogicalDocumentDetail(
        **LogicalDocumentResponse.model_validate(document).model_dump(),
        versions=[version_summary(version) for version in document.versions],
    )


@router.post(
    "/documents",
    response_model=LogicalDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    request: LogicalDocumentCreate,
    service: CatalogService,
) -> LogicalDocument:
    return await service.create(request)


@router.get("/documents", response_model=list[LogicalDocumentResponse])
async def list_documents(service: CatalogService) -> list[LogicalDocument]:
    return await service.list()


@router.get("/documents/{logical_document_id}", response_model=LogicalDocumentDetail)
async def get_document(
    logical_document_id: UUID,
    service: CatalogService,
) -> LogicalDocumentDetail:
    return document_detail(await service.get(logical_document_id))


@router.post(
    "/documents/{logical_document_id}/versions",
    response_model=IngestionAcceptedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document_version(
    logical_document_id: UUID,
    service: VersionService,
    settings: AppSettings,
    file: Annotated[UploadFile, File()],
    version_label: Annotated[str, Form(min_length=1, max_length=100)],
    version_ordinal: Annotated[int, Form(ge=1)],
    language: Annotated[DocumentLanguage, Form()],
    revision_number: Annotated[int, Form(ge=1)] = 1,
    effective_from: Annotated[datetime | None, Form()] = None,
    effective_to: Annotated[datetime | None, Form()] = None,
    published_at: Annotated[datetime | None, Form()] = None,
    supersedes_version_id: Annotated[UUID | None, Form()] = None,
    activate: Annotated[bool, Form()] = False,
    chunk_size: Annotated[int | None, Form(ge=1)] = None,
    chunk_overlap: Annotated[int | None, Form(ge=0)] = None,
) -> IngestionAcceptedResponse:
    try:
        request = DocumentVersionCreate(
            version_label=version_label,
            revision_number=revision_number,
            version_ordinal=version_ordinal,
            language=language,
            effective_from=effective_from,
            effective_to=effective_to,
            published_at=published_at,
            supersedes_version_id=supersedes_version_id,
            activate=activate,
            chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
            chunk_overlap=(chunk_overlap if chunk_overlap is not None else settings.chunk_overlap),
        )
    except ValidationError as exc:
        raise AppError(
            "invalid document version metadata",
            code="invalid_ingestion_request",
            status_code=422,
            details={"errors": exc.errors(include_url=False, include_context=False)},
        ) from exc
    filename = file.filename or "document"
    version = await service.ingest(
        logical_document_id,
        request,
        source=file.file,
        filename=filename,
    )
    return IngestionAcceptedResponse(
        document_version=await version_detail(version, service),
        message="document version ingested successfully",
    )


@router.get(
    "/documents/{logical_document_id}/versions",
    response_model=list[DocumentVersionSummary],
)
async def list_document_versions(
    logical_document_id: UUID,
    service: VersionService,
) -> list[DocumentVersionSummary]:
    versions = await service.list_versions(logical_document_id)
    return [version_summary(version) for version in versions]


@router.get(
    "/document-versions/{document_version_id}",
    response_model=DocumentVersionDetail,
)
async def get_document_version(
    document_version_id: UUID,
    service: VersionService,
) -> DocumentVersionDetail:
    return await version_detail(await service.get_version(document_version_id), service)
