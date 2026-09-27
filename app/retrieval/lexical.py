from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, cast
from uuid import UUID

from sqlalchemy import func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.db.models import Chunk, DocumentVersion
from app.ingestion.errors import DocumentNotFoundError
from app.models.retrieval import (
    LexicalRetrievalBatch,
    LexicalRetrievalCandidate,
    LexicalRetrievalResponse,
)
from app.models.versioning import DocumentLanguage, VectorIndexStatus, VersionIndexingStatus
from app.retrieval.errors import InvalidRetrievalScopeError, VersionNotIndexedError

logger = logging.getLogger(__name__)


class LexicalRetrievalService:
    """PostgreSQL FTS retrieval with language-specific baseline configurations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self, query: str, document_version_id: UUID, top_k: int
    ) -> LexicalRetrievalResponse:
        batch = await self.search_versions(query, [document_version_id], top_k)
        return LexicalRetrievalResponse(
            document_version_id=document_version_id,
            candidates=batch.candidates,
            retrieval_latency_ms=batch.retrieval_latency_ms,
            total_latency_ms=batch.total_latency_ms,
        )

    async def search_versions(
        self, query: str, document_version_ids: list[UUID], top_k: int
    ) -> LexicalRetrievalBatch:
        total_started = perf_counter()
        if not document_version_ids:
            raise InvalidRetrievalScopeError()
        versions = list(
            await self._session.scalars(
                select(DocumentVersion).where(
                    DocumentVersion.document_version_id.in_(document_version_ids)
                )
            )
        )
        versions_by_id = {version.document_version_id: version for version in versions}
        missing = [item for item in document_version_ids if item not in versions_by_id]
        if missing:
            raise DocumentNotFoundError(f"document version '{missing[0]}' was not found")
        unavailable = [
            item
            for item in document_version_ids
            if versions_by_id[item].indexing_status is not VersionIndexingStatus.READY
            or versions_by_id[item].vector_index_status is not VectorIndexStatus.INDEXED
        ]
        if unavailable:
            raise VersionNotIndexedError(f"document version '{unavailable[0]}' is not queryable")

        retrieval_started = perf_counter()
        candidates: list[LexicalRetrievalCandidate] = []
        for language, config in (
            (DocumentLanguage.ENGLISH, "english"),
            (DocumentLanguage.VIETNAMESE, "simple"),
        ):
            candidates.extend(
                await self._search_language(
                    query,
                    document_version_ids,
                    language=language,
                    config=config,
                    limit=top_k,
                )
            )
        candidates = sorted(
            candidates,
            key=lambda item: (-item.lexical_score, str(item.chunk_id)),
        )[:top_k]
        retrieval_latency_ms = (perf_counter() - retrieval_started) * 1000
        total_latency_ms = (perf_counter() - total_started) * 1000
        logger.info(
            "lexical retrieval completed",
            extra={
                "event": "lexical_retrieval_completed",
                "document_version_ids": [str(item) for item in document_version_ids],
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
                "candidate_count": len(candidates),
            },
        )
        return LexicalRetrievalBatch(
            document_version_ids=document_version_ids,
            candidates=candidates,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
        )

    async def _search_language(
        self,
        query: str,
        document_version_ids: list[UUID],
        *,
        language: DocumentLanguage,
        config: str,
        limit: int,
    ) -> list[LexicalRetrievalCandidate]:
        configuration: ColumnElement[Any] = literal_column(f"'{config}'::regconfig")
        search_vector = func.to_tsvector(configuration, Chunk.text)
        search_query = func.websearch_to_tsquery(configuration, query)
        lexical_score = func.ts_rank_cd(search_vector, search_query).label("lexical_score")
        rows = (
            await self._session.execute(
                select(Chunk, DocumentVersion.logical_document_id, lexical_score)
                .join(
                    DocumentVersion,
                    DocumentVersion.document_version_id == Chunk.document_version_id,
                )
                .where(
                    Chunk.document_version_id.in_(document_version_ids),
                    Chunk.language == language,
                    search_vector.op("@@")(search_query),
                )
                .order_by(lexical_score.desc(), Chunk.chunk_id)
                .limit(limit)
            )
        ).all()
        candidates: list[LexicalRetrievalCandidate] = []
        for row in rows:
            chunk = cast(Chunk, row[0])
            logical_document_id = cast(UUID, row[1])
            score = cast(float, row[2])
            candidates.append(
                LexicalRetrievalCandidate(
                    chunk_id=chunk.chunk_id,
                    document_version_id=chunk.document_version_id,
                    logical_document_id=logical_document_id,
                    text=chunk.text,
                    lexical_score=float(score),
                    ordinal=chunk.ordinal,
                    page=chunk.page,
                    section_path=chunk.section_path,
                    language=chunk.language,
                )
            )
        return candidates
