from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentLanguage(StrEnum):
    ENGLISH = "en"
    VIETNAMESE = "vi"


class VersionBusinessStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"


class VersionIndexingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETING = "deleting"


class IngestionJobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class VectorIndexStatus(StrEnum):
    NOT_INDEXED = "not_indexed"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class VersionSelectorKind(StrEnum):
    CURRENT = "current"
    EXACT = "exact"
    ALL = "all"
    AS_OF = "as_of"


class LogicalDocumentIdentity(BaseModel):
    """Stable identity shared by all immutable versions of one logical document."""

    model_config = ConfigDict(frozen=True)

    logical_document_id: UUID
    key: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)


class DocumentVersionSnapshot(BaseModel):
    """Immutable source-semantic view of one document version.

    Operational persistence fields are intentionally excluded from this frozen boundary.
    """

    model_config = ConfigDict(frozen=True)

    document_version_id: UUID
    logical_document_id: UUID
    version_label: str = Field(min_length=1, max_length=100)
    revision_number: int = Field(default=1, ge=1)
    version_ordinal: int = Field(ge=1)
    language: DocumentLanguage
    source_filename: str = Field(min_length=1, max_length=500)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    supersedes_version_id: UUID | None = None
    business_status: VersionBusinessStatus
    indexing_status: VersionIndexingStatus

    @model_validator(mode="after")
    def validate_version_invariants(self) -> DocumentVersionSnapshot:
        if (
            self.business_status is VersionBusinessStatus.ACTIVE
            and self.indexing_status is not VersionIndexingStatus.READY
        ):
            raise ValueError("an active document version must be ready for retrieval")
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to <= self.effective_from
        ):
            raise ValueError("effective_to must be later than effective_from")
        return self


class VersionSelector(BaseModel):
    """Explicit retrieval scope; current has no heuristic fallback."""

    model_config = ConfigDict(frozen=True)

    kind: VersionSelectorKind = VersionSelectorKind.CURRENT
    document_version_ids: tuple[UUID, ...] = ()
    as_of: datetime | None = None

    @model_validator(mode="after")
    def validate_selector_arguments(self) -> VersionSelector:
        if self.kind is VersionSelectorKind.EXACT and not self.document_version_ids:
            raise ValueError("exact version selection requires at least one document_version_id")
        if self.kind is not VersionSelectorKind.EXACT and self.document_version_ids:
            raise ValueError("document_version_ids are valid only for exact selection")
        if self.kind is VersionSelectorKind.AS_OF and self.as_of is None:
            raise ValueError("as_of selection requires a timestamp")
        if self.kind is not VersionSelectorKind.AS_OF and self.as_of is not None:
            raise ValueError("as_of timestamp is valid only for as_of selection")
        return self
