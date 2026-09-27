from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.versioning import (
    DocumentLanguage,
    IngestionJobStatus,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
)


class LogicalDocumentCreate(BaseModel):
    stable_key: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    title: str = Field(min_length=1, max_length=500)
    description: str | None = None
    owner: str | None = Field(default=None, max_length=200)
    domain: str | None = Field(default=None, max_length=200)


class DocumentVersionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_version_id: UUID
    logical_document_id: UUID
    version_label: str
    revision_number: int
    version_ordinal: int
    language: DocumentLanguage
    source_filename: str
    checksum: str
    business_status: VersionBusinessStatus
    indexing_status: VersionIndexingStatus
    vector_index_status: VectorIndexStatus
    created_at: datetime
    is_current: bool


class LogicalDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    logical_document_id: UUID
    stable_key: str
    title: str
    description: str | None
    owner: str | None
    domain: str | None
    created_at: datetime
    updated_at: datetime


class LogicalDocumentDetail(LogicalDocumentResponse):
    versions: list[DocumentVersionSummary]


class DocumentVersionCreate(BaseModel):
    version_label: str = Field(min_length=1, max_length=100)
    revision_number: int = Field(default=1, ge=1)
    version_ordinal: int = Field(ge=1)
    language: DocumentLanguage
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    published_at: datetime | None = None
    supersedes_version_id: UUID | None = None
    activate: bool = False
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_configuration(self) -> DocumentVersionCreate:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to <= self.effective_from
        ):
            raise ValueError("effective_to must be later than effective_from")
        return self


class IngestionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ingestion_job_id: UUID
    status: IngestionJobStatus
    error_code: str | None
    error_message: str | None
    started_at: datetime
    completed_at: datetime | None


class DocumentVersionDetail(DocumentVersionSummary):
    effective_from: datetime | None
    effective_to: datetime | None
    published_at: datetime | None
    supersedes_version_id: UUID | None
    storage_key: str
    ingestion_config: dict[str, Any]
    embedding_model: str | None
    embedding_revision: str | None
    embedding_dimension: int | None
    vector_indexed_at: datetime | None
    vector_index_error: str | None
    chunk_count: int
    ingestion_job: IngestionJobResponse | None


class IngestionAcceptedResponse(BaseModel):
    document_version: DocumentVersionDetail
    message: str
