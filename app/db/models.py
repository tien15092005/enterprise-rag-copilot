from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.db.base import Base
from app.models.versioning import (
    DocumentLanguage,
    IngestionJobStatus,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
)


def enum_type(enum_class: type[Any], name: str) -> Enum:
    return Enum(
        enum_class,
        name=name,
        native_enum=False,
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )


class LogicalDocument(Base):
    __tablename__ = "logical_documents"

    logical_document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4
    )
    stable_key: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    owner: Mapped[str | None] = mapped_column(String(200))
    domain: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="logical_document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "logical_document_id",
            "version_label",
            "revision_number",
            name="uq_document_versions_label_revision",
        ),
        UniqueConstraint(
            "logical_document_id",
            "version_ordinal",
            name="uq_document_versions_ordinal",
        ),
        UniqueConstraint(
            "logical_document_id",
            "checksum",
            name="uq_document_versions_checksum",
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL OR effective_to > effective_from",
            name="effective_period",
        ),
        CheckConstraint(
            "business_status <> 'active' OR indexing_status = 'ready'",
            name="active_requires_ready",
        ),
        Index(
            "uq_document_versions_one_active",
            "logical_document_id",
            unique=True,
            postgresql_where=text("business_status = 'active'"),
            sqlite_where=text("business_status = 'active'"),
        ),
    )

    document_version_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4
    )
    logical_document_id: Mapped[UUID] = mapped_column(
        ForeignKey("logical_documents.logical_document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_label: Mapped[str] = mapped_column(String(100), nullable=False)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    version_ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    supersedes_version_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("document_versions.document_version_id", ondelete="SET NULL")
    )
    language: Mapped[DocumentLanguage] = mapped_column(
        enum_type(DocumentLanguage, "document_language"), nullable=False
    )
    source_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1000), nullable=False, unique=True)
    business_status: Mapped[VersionBusinessStatus] = mapped_column(
        enum_type(VersionBusinessStatus, "version_business_status"),
        nullable=False,
        default=VersionBusinessStatus.DRAFT,
    )
    indexing_status: Mapped[VersionIndexingStatus] = mapped_column(
        enum_type(VersionIndexingStatus, "version_indexing_status"),
        nullable=False,
        default=VersionIndexingStatus.PENDING,
    )
    ingestion_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    vector_index_status: Mapped[VectorIndexStatus] = mapped_column(
        enum_type(VectorIndexStatus, "vector_index_status"),
        nullable=False,
        default=VectorIndexStatus.NOT_INDEXED,
        server_default=VectorIndexStatus.NOT_INDEXED.value,
    )
    embedding_model: Mapped[str | None] = mapped_column(String(300))
    embedding_revision: Mapped[str | None] = mapped_column(String(100))
    embedding_dimension: Mapped[int | None] = mapped_column(Integer)
    vector_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    vector_index_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    logical_document: Mapped[LogicalDocument] = relationship(back_populates="versions")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    ingestion_job: Mapped[IngestionJob | None] = relationship(
        back_populates="document_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    @property
    def is_current(self) -> bool:
        return (
            self.business_status is VersionBusinessStatus.ACTIVE
            and self.indexing_status is VersionIndexingStatus.READY
        )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_version_id", "ordinal", name="uq_chunks_version_ordinal"),
    )

    chunk_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_versions.document_version_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[list[str] | None] = mapped_column(JSON)
    language: Mapped[DocumentLanguage] = mapped_column(
        enum_type(DocumentLanguage, "chunk_language"), nullable=False
    )
    section_fingerprint: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    ingestion_job_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid4
    )
    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_versions.document_version_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    status: Mapped[IngestionJobStatus] = mapped_column(
        enum_type(IngestionJobStatus, "ingestion_job_status"), nullable=False
    )
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document_version: Mapped[DocumentVersion] = relationship(back_populates="ingestion_job")


_IMMUTABLE_VERSION_FIELDS = (
    "logical_document_id",
    "version_label",
    "revision_number",
    "version_ordinal",
    "effective_from",
    "effective_to",
    "published_at",
    "supersedes_version_id",
    "language",
    "source_filename",
    "checksum",
    "storage_key",
    "ingestion_config",
)


@event.listens_for(DocumentVersion, "before_update")
def prevent_ready_version_source_mutation(
    _mapper: object, _connection: object, target: DocumentVersion
) -> None:
    state = inspect(target)
    status_history = state.attrs.indexing_status.history
    original_status = (
        status_history.deleted[0] if status_history.deleted else target.indexing_status
    )
    if original_status is not VersionIndexingStatus.READY:
        return

    changed = [
        field for field in _IMMUTABLE_VERSION_FIELDS if state.attrs[field].history.has_changes()
    ]
    if changed:
        fields = ", ".join(changed)
        raise ValueError(f"ready document version source snapshot is immutable: {fields}")


__all__ = ["Base", "Chunk", "DocumentVersion", "IngestionJob", "LogicalDocument"]
