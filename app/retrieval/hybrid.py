from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import UUID

from app.models.retrieval import (
    DenseRetrievalBatch,
    HybridRetrievalBatch,
    HybridRetrievalResponse,
    RetrievalBatch,
    RetrievalMode,
)
from app.retrieval.errors import RetrievalProvenanceError
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import LexicalRetrievalService
from app.retrieval.service import DenseRetrievalService

if TYPE_CHECKING:
    from app.retrieval.reranked import RerankedRetrievalService

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    def __init__(
        self,
        dense: DenseRetrievalService,
        lexical: LexicalRetrievalService,
        *,
        dense_candidate_limit: int,
        lexical_candidate_limit: int,
        rrf_k: int,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._dense_candidate_limit = dense_candidate_limit
        self._lexical_candidate_limit = lexical_candidate_limit
        self._rrf_k = rrf_k

    async def search(
        self, query: str, document_version_id: UUID, top_k: int
    ) -> HybridRetrievalResponse:
        batch = await self.search_versions(query, [document_version_id], top_k)
        return HybridRetrievalResponse(
            document_version_id=document_version_id,
            candidates=batch.candidates,
            dense_latency_ms=batch.dense_latency_ms,
            lexical_latency_ms=batch.lexical_latency_ms,
            fusion_latency_ms=batch.fusion_latency_ms,
            total_latency_ms=batch.total_latency_ms,
            dense_candidate_count=batch.dense_candidate_count,
            lexical_candidate_count=batch.lexical_candidate_count,
            merged_duplicate_count=batch.merged_duplicate_count,
        )

    async def search_versions(
        self, query: str, document_version_ids: list[UUID], top_k: int
    ) -> HybridRetrievalBatch:
        total_started = perf_counter()
        dense = await self._dense.search_versions(
            query,
            document_version_ids,
            max(top_k, self._dense_candidate_limit),
        )
        lexical = await self._lexical.search_versions(
            query,
            document_version_ids,
            max(top_k, self._lexical_candidate_limit),
        )
        allowed_versions = set(document_version_ids)
        leaked_dense = [
            candidate.document_version_id
            for candidate in dense.candidates
            if candidate.document_version_id not in allowed_versions
        ]
        leaked_lexical = [
            candidate.document_version_id
            for candidate in lexical.candidates
            if candidate.document_version_id not in allowed_versions
        ]
        leaked = leaked_dense + leaked_lexical
        if leaked:
            raise RetrievalProvenanceError(
                f"hybrid branch returned evidence outside resolved scope: '{leaked[0]}'"
            )
        fusion_started = perf_counter()
        all_fused = reciprocal_rank_fusion(
            dense.candidates,
            lexical.candidates,
            k=self._rrf_k,
        )
        fusion_latency_ms = (perf_counter() - fusion_started) * 1000
        merged_duplicate_count = len(dense.candidates) + len(lexical.candidates) - len(all_fused)
        candidates = all_fused[:top_k]
        total_latency_ms = (perf_counter() - total_started) * 1000
        logger.info(
            "hybrid retrieval completed",
            extra={
                "event": "hybrid_retrieval_completed",
                "document_version_ids": [str(item) for item in document_version_ids],
                "dense_retrieval_latency_ms": round(dense.total_latency_ms, 3),
                "lexical_retrieval_latency_ms": round(lexical.total_latency_ms, 3),
                "fusion_latency_ms": round(fusion_latency_ms, 3),
                "total_latency_ms": round(total_latency_ms, 3),
                "dense_candidate_count": len(dense.candidates),
                "lexical_candidate_count": len(lexical.candidates),
                "merged_duplicate_count": merged_duplicate_count,
                "result_count": len(candidates),
            },
        )
        return HybridRetrievalBatch(
            document_version_ids=document_version_ids,
            candidates=candidates,
            embedding_latency_ms=dense.embedding_latency_ms,
            dense_latency_ms=dense.total_latency_ms,
            lexical_latency_ms=lexical.total_latency_ms,
            fusion_latency_ms=fusion_latency_ms,
            total_latency_ms=total_latency_ms,
            dense_candidate_count=len(dense.candidates),
            lexical_candidate_count=len(lexical.candidates),
            merged_duplicate_count=merged_duplicate_count,
        )


class ModeAwareRetrievalService:
    def __init__(
        self,
        dense: DenseRetrievalService,
        hybrid: HybridRetrievalService,
        reranked: RerankedRetrievalService | None = None,
    ) -> None:
        self._dense = dense
        self._hybrid = hybrid
        self._reranked = reranked

    async def search_versions(
        self,
        query: str,
        document_version_ids: list[UUID],
        top_k: int,
        mode: RetrievalMode,
    ) -> RetrievalBatch:
        if mode is RetrievalMode.HYBRID:
            return await self._hybrid.search_versions(query, document_version_ids, top_k)
        if mode is RetrievalMode.HYBRID_RERANK:
            if self._reranked is None:
                raise ValueError("hybrid reranking service is not configured")
            return await self._reranked.search_versions(query, document_version_ids, top_k)
        dense: DenseRetrievalBatch = await self._dense.search_versions(
            query, document_version_ids, top_k
        )
        return dense
