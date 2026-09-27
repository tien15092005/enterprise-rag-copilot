from __future__ import annotations

import logging
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, DocumentVersion
from app.embeddings.base import EmbeddingProvider
from app.ingestion.errors import DocumentNotFoundError
from app.models.retrieval import (
    DenseRetrievalBatch,
    DenseRetrievalCandidate,
    DenseRetrievalResponse,
    VectorIndexResponse,
)
from app.models.versioning import VectorIndexStatus, VersionIndexingStatus
from app.retrieval.errors import (
    RetrievalProvenanceError,
    VectorIndexingError,
    VersionNotIndexedError,
    VersionNotReadyError,
)
from app.vector_store.base import VectorPoint, VectorStore

logger = logging.getLogger(__name__)


class VectorIndexingService:
    def __init__(
        self,
        session: AsyncSession,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._session = session
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def index_version(self, document_version_id: UUID) -> VectorIndexResponse:
        total_started = perf_counter()
        version = await self._session.get(DocumentVersion, document_version_id)
        if version is None:
            raise DocumentNotFoundError(f"document version '{document_version_id}' was not found")
        if version.indexing_status is not VersionIndexingStatus.READY:
            raise VersionNotReadyError(
                f"document version '{document_version_id}' has not completed ingestion"
            )

        chunks = list(
            await self._session.scalars(
                select(Chunk)
                .where(Chunk.document_version_id == document_version_id)
                .order_by(Chunk.ordinal)
            )
        )
        if not chunks:
            raise VersionNotReadyError(f"document version '{document_version_id}' has no chunks")

        version.vector_index_status = VectorIndexStatus.INDEXING
        version.vector_index_error = None
        await self._session.commit()

        try:
            embedding_started = perf_counter()
            vectors = await self._embeddings.embed_passages([chunk.text for chunk in chunks])
            embedding_latency_ms = (perf_counter() - embedding_started) * 1000

            qdrant_started = perf_counter()
            await self._vector_store.ensure_collection(self._embeddings.dimension)
            await self._vector_store.upsert(
                [
                    VectorPoint(
                        point_id=chunk.chunk_id,
                        vector=vector,
                        payload={
                            "chunk_id": str(chunk.chunk_id),
                            "document_version_id": str(document_version_id),
                            "logical_document_id": str(version.logical_document_id),
                            "language": chunk.language.value,
                            "page": chunk.page,
                            "section_path": chunk.section_path,
                            "ordinal": chunk.ordinal,
                        },
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            )
            qdrant_latency_ms = (perf_counter() - qdrant_started) * 1000
        except Exception as exc:
            await self._session.rollback()
            failed_version = await self._session.get(DocumentVersion, document_version_id)
            if failed_version is not None:
                failed_version.vector_index_status = VectorIndexStatus.FAILED
                failed_version.vector_index_error = str(exc)[:2000]
                await self._session.commit()
            logger.exception(
                "vector indexing failed",
                extra={
                    "event": "vector_indexing_failed",
                    "document_version_id": str(document_version_id),
                },
            )
            raise VectorIndexingError(str(exc)) from exc

        refreshed = await self._session.get(DocumentVersion, document_version_id)
        if refreshed is None:
            raise RuntimeError("document version disappeared during vector indexing")
        refreshed.vector_index_status = VectorIndexStatus.INDEXED
        refreshed.embedding_model = self._embeddings.model_name
        refreshed.embedding_revision = self._embeddings.revision
        refreshed.embedding_dimension = self._embeddings.dimension
        refreshed.vector_indexed_at = datetime.now(UTC)
        refreshed.vector_index_error = None
        await self._session.commit()

        total_latency_ms = (perf_counter() - total_started) * 1000
        logger.info(
            "vector indexing completed",
            extra={
                "event": "vector_indexing_completed",
                "document_version_id": str(document_version_id),
                "chunk_count": len(chunks),
                "embedding_latency_ms": round(embedding_latency_ms, 3),
                "qdrant_indexing_latency_ms": round(qdrant_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
            },
        )
        return VectorIndexResponse(
            document_version_id=document_version_id,
            status=VectorIndexStatus.INDEXED,
            chunks_indexed=len(chunks),
            embedding_model=self._embeddings.model_name,
            embedding_revision=self._embeddings.revision,
            embedding_dimension=self._embeddings.dimension,
            embedding_latency_ms=embedding_latency_ms,
            qdrant_latency_ms=qdrant_latency_ms,
            total_latency_ms=total_latency_ms,
        )


class DenseRetrievalService:
    def __init__(
        self,
        session: AsyncSession,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._session = session
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def search(
        self, query: str, document_version_id: UUID, top_k: int
    ) -> DenseRetrievalResponse:
        batch = await self.search_versions(query, [document_version_id], top_k)
        return DenseRetrievalResponse(
            document_version_id=document_version_id,
            embedding_model=batch.embedding_model,
            candidates=batch.candidates,
            embedding_latency_ms=batch.embedding_latency_ms,
            retrieval_latency_ms=batch.retrieval_latency_ms,
            total_latency_ms=batch.total_latency_ms,
        )

    async def search_versions(
        self, query: str, document_version_ids: list[UUID], top_k: int
    ) -> DenseRetrievalBatch:
        total_started = perf_counter()
        if not document_version_ids:
            return DenseRetrievalBatch(
                document_version_ids=[],
                embedding_model=self._embeddings.model_name,
                candidates=[],
                embedding_latency_ms=0,
                retrieval_latency_ms=0,
                total_latency_ms=(perf_counter() - total_started) * 1000,
            )
        versions = list(
            await self._session.scalars(
                select(DocumentVersion).where(
                    DocumentVersion.document_version_id.in_(document_version_ids)
                )
            )
        )
        versions_by_id = {version.document_version_id: version for version in versions}
        missing = [
            version_id for version_id in document_version_ids if version_id not in versions_by_id
        ]
        if missing:
            raise DocumentNotFoundError(f"document version '{missing[0]}' was not found")
        unindexed = [
            version_id
            for version_id in document_version_ids
            if versions_by_id[version_id].vector_index_status is not VectorIndexStatus.INDEXED
        ]
        if unindexed:
            raise VersionNotIndexedError(f"document version '{unindexed[0]}' is not vector-indexed")

        embedding_started = perf_counter()
        query_vector = await self._embeddings.embed_query(query)
        embedding_latency_ms = (perf_counter() - embedding_started) * 1000
        retrieval_started = perf_counter()
        hits = await self._vector_store.search_versions(query_vector, document_version_ids, top_k)
        retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000

        chunk_ids = [hit.point_id for hit in hits]
        chunks = {
            chunk.chunk_id: chunk
            for chunk in await self._session.scalars(
                select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))
            )
        }
        candidates: list[DenseRetrievalCandidate] = []
        for hit in hits:
            chunk = chunks.get(hit.point_id)
            if chunk is None:
                raise RetrievalProvenanceError(
                    f"Qdrant chunk '{hit.point_id}' does not exist in PostgreSQL"
                )
            version = versions_by_id.get(chunk.document_version_id)
            if version is None:
                raise RetrievalProvenanceError(
                    f"Qdrant returned chunk '{hit.point_id}' outside the resolved version scope"
                )
            self._validate_provenance(hit.payload, chunk, version.logical_document_id)
            candidates.append(
                DenseRetrievalCandidate(
                    chunk_id=chunk.chunk_id,
                    document_version_id=chunk.document_version_id,
                    logical_document_id=version.logical_document_id,
                    text=chunk.text,
                    score=hit.score,
                    ordinal=chunk.ordinal,
                    page=chunk.page,
                    section_path=chunk.section_path,
                    language=chunk.language,
                )
            )

        total_latency_ms = (perf_counter() - total_started) * 1000
        logger.info(
            "dense retrieval completed",
            extra={
                "event": "dense_retrieval_completed",
                "document_version_ids": [str(version_id) for version_id in document_version_ids],
                "embedding_latency_ms": round(embedding_latency_ms, 3),
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
                "candidate_count": len(candidates),
            },
        )
        return DenseRetrievalBatch(
            document_version_ids=document_version_ids,
            embedding_model=self._embeddings.model_name,
            candidates=candidates,
            embedding_latency_ms=embedding_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
        )

    @staticmethod
    def _validate_provenance(
        payload: dict[str, object], chunk: Chunk, logical_document_id: UUID
    ) -> None:
        expected = {
            "chunk_id": str(chunk.chunk_id),
            "document_version_id": str(chunk.document_version_id),
            "logical_document_id": str(logical_document_id),
            "language": chunk.language.value,
            "page": chunk.page,
            "section_path": chunk.section_path,
            "ordinal": chunk.ordinal,
        }
        mismatches = [key for key, value in expected.items() if payload.get(key) != value]
        if mismatches:
            raise RetrievalProvenanceError(
                "Qdrant payload does not match PostgreSQL provenance: " + ", ".join(mismatches)
            )
