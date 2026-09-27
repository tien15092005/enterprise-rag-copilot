from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import BinaryIO
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Chunk, DocumentVersion, IngestionJob, LogicalDocument
from app.ingestion.chunker import ChunkingConfig, TokenWindowChunker
from app.ingestion.errors import (
    DocumentNotFoundError,
    DuplicateDocumentVersionError,
    DuplicateLogicalDocumentError,
    IngestionError,
)
from app.ingestion.parsers import ParserRegistry
from app.ingestion.storage import DocumentStorage
from app.models.documents import DocumentVersionCreate, LogicalDocumentCreate
from app.models.versioning import (
    IngestionJobStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
)

logger = logging.getLogger(__name__)


class DocumentCatalogService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, request: LogicalDocumentCreate) -> LogicalDocument:
        document = LogicalDocument(**request.model_dump())
        self._session.add(document)
        try:
            await self._session.commit()
        except IntegrityError as exc:
            await self._session.rollback()
            raise DuplicateLogicalDocumentError(
                f"logical document key '{request.stable_key}' already exists"
            ) from exc
        await self._session.refresh(document)
        return document

    async def list(self) -> list[LogicalDocument]:
        result = await self._session.scalars(
            select(LogicalDocument).order_by(LogicalDocument.stable_key)
        )
        return list(result)

    async def get(self, logical_document_id: UUID) -> LogicalDocument:
        document = await self._session.scalar(
            select(LogicalDocument)
            .where(LogicalDocument.logical_document_id == logical_document_id)
            .options(selectinload(LogicalDocument.versions))
        )
        if document is None:
            raise DocumentNotFoundError(f"logical document '{logical_document_id}' was not found")
        document.versions.sort(key=lambda version: version.version_ordinal)
        return document


class IngestionService:
    def __init__(
        self,
        session: AsyncSession,
        storage: DocumentStorage,
        parsers: ParserRegistry,
        chunker: TokenWindowChunker,
    ) -> None:
        self._session = session
        self._storage = storage
        self._parsers = parsers
        self._chunker = chunker

    async def ingest(
        self,
        logical_document_id: UUID,
        request: DocumentVersionCreate,
        *,
        source: BinaryIO,
        filename: str,
    ) -> DocumentVersion:
        logical_document = await self._session.get(LogicalDocument, logical_document_id)
        if logical_document is None:
            raise DocumentNotFoundError(f"logical document '{logical_document_id}' was not found")

        parser = self._parsers.for_filename(filename)
        chunk_config = ChunkingConfig(request.chunk_size, request.chunk_overlap)
        version_id = uuid4()
        stored = await self._storage.save(
            source,
            document_version_id=version_id,
            original_filename=filename,
        )

        duplicate = await self._session.scalar(
            select(DocumentVersion.document_version_id).where(
                DocumentVersion.logical_document_id == logical_document_id,
                DocumentVersion.checksum == stored.checksum,
            )
        )
        if duplicate is not None:
            await self._storage.delete(stored.storage_key)
            raise DuplicateDocumentVersionError(
                "the same source content already exists in this logical document"
            )

        if request.supersedes_version_id is not None:
            predecessor = await self._session.get(DocumentVersion, request.supersedes_version_id)
            if predecessor is None or predecessor.logical_document_id != logical_document_id:
                await self._storage.delete(stored.storage_key)
                raise DocumentNotFoundError(
                    "supersedes_version_id does not belong to this logical document"
                )

        version = DocumentVersion(
            document_version_id=version_id,
            logical_document_id=logical_document_id,
            version_label=request.version_label,
            revision_number=request.revision_number,
            version_ordinal=request.version_ordinal,
            effective_from=request.effective_from,
            effective_to=request.effective_to,
            published_at=request.published_at,
            supersedes_version_id=request.supersedes_version_id,
            language=request.language,
            source_filename=stored.original_filename,
            checksum=stored.checksum,
            storage_key=stored.storage_key,
            business_status=VersionBusinessStatus.DRAFT,
            indexing_status=VersionIndexingStatus.PROCESSING,
            ingestion_config={
                "chunk_size": chunk_config.chunk_size,
                "chunk_overlap": chunk_config.chunk_overlap,
                "strategy": "regex_token_window_v1",
                "source_size_bytes": stored.size_bytes,
            },
        )
        job = IngestionJob(
            document_version_id=version_id,
            status=IngestionJobStatus.PROCESSING,
        )
        version.ingestion_job = job
        self._session.add(version)
        try:
            await self._session.commit()
        except IntegrityError as exc:
            await self._session.rollback()
            await self._storage.delete(stored.storage_key)
            raise DuplicateDocumentVersionError(
                "version label/revision, ordinal, checksum, or storage key conflicts "
                "with an existing version"
            ) from exc

        try:
            content = await self._storage.read(stored.storage_key)
            parsed = parser.parse(content)
            chunks = self._chunker.chunk(
                parsed,
                document_version_id=version_id,
                language=request.language,
                config=chunk_config,
            )
        except Exception as exc:
            await self._mark_failed(version_id, exc)
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(str(exc)) from exc

        self._session.add_all(
            [
                Chunk(
                    chunk_id=chunk.chunk_id,
                    document_version_id=chunk.document_version_id,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    page=chunk.page,
                    section_path=list(chunk.section_path) or None,
                    language=chunk.language,
                    section_fingerprint=chunk.section_fingerprint,
                )
                for chunk in chunks
            ]
        )
        # Persist source chunks while the version is still processing. PostgreSQL's
        # immutability trigger correctly rejects chunk inserts once the snapshot is ready.
        await self._session.flush()
        refreshed_version = await self._session.get(DocumentVersion, version_id)
        if refreshed_version is None:
            raise RuntimeError("document version disappeared during ingestion")
        refreshed_version.indexing_status = VersionIndexingStatus.READY
        refreshed_job = await self._session.scalar(
            select(IngestionJob).where(IngestionJob.document_version_id == version_id)
        )
        if refreshed_job is None:
            raise RuntimeError("ingestion job disappeared during ingestion")
        refreshed_job.status = IngestionJobStatus.SUCCEEDED
        refreshed_job.completed_at = datetime.now(UTC)

        if request.activate:
            await self._session.execute(
                update(DocumentVersion)
                .where(
                    DocumentVersion.logical_document_id == logical_document_id,
                    DocumentVersion.business_status == VersionBusinessStatus.ACTIVE,
                    DocumentVersion.document_version_id != version_id,
                )
                .values(business_status=VersionBusinessStatus.SUPERSEDED)
            )
            await self._session.flush()
            refreshed_version.business_status = VersionBusinessStatus.ACTIVE

        await self._session.commit()
        logger.info(
            "document ingestion completed",
            extra={
                "event": "document_ingestion_completed",
                "logical_document_id": str(logical_document_id),
                "document_version_id": str(version_id),
                "chunk_count": len(chunks),
            },
        )
        return await self.get_version(version_id)

    async def _mark_failed(self, version_id: UUID, exc: Exception) -> None:
        await self._session.rollback()
        version = await self._session.get(DocumentVersion, version_id)
        job = await self._session.scalar(
            select(IngestionJob).where(IngestionJob.document_version_id == version_id)
        )
        if version is not None:
            version.indexing_status = VersionIndexingStatus.FAILED
        if job is not None:
            job.status = IngestionJobStatus.FAILED
            job.error_code = exc.code if isinstance(exc, IngestionError) else "ingestion_failed"
            job.error_message = str(exc)[:2000]
            job.completed_at = datetime.now(UTC)
        await self._session.commit()
        logger.warning(
            "document ingestion failed",
            extra={
                "event": "document_ingestion_failed",
                "document_version_id": str(version_id),
                "error_type": type(exc).__name__,
            },
        )

    async def list_versions(self, logical_document_id: UUID) -> list[DocumentVersion]:
        exists = await self._session.get(LogicalDocument, logical_document_id)
        if exists is None:
            raise DocumentNotFoundError(f"logical document '{logical_document_id}' was not found")
        result = await self._session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.logical_document_id == logical_document_id)
            .options(selectinload(DocumentVersion.ingestion_job))
            .order_by(DocumentVersion.version_ordinal)
        )
        return list(result)

    async def get_version(self, document_version_id: UUID) -> DocumentVersion:
        version = await self._session.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_version_id == document_version_id)
            .options(selectinload(DocumentVersion.ingestion_job))
        )
        if version is None:
            raise DocumentNotFoundError(f"document version '{document_version_id}' was not found")
        return version

    async def chunk_count(self, document_version_id: UUID) -> int:
        count = await self._session.scalar(
            select(func.count(Chunk.chunk_id)).where(
                Chunk.document_version_id == document_version_id
            )
        )
        return count or 0
