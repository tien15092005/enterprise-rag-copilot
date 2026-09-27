from __future__ import annotations

import logging
from time import perf_counter
from uuid import UUID

from app.models.retrieval import RerankedRetrievalBatch
from app.reranking.base import Reranker
from app.reranking.errors import RerankerProvenanceError
from app.retrieval.hybrid import HybridRetrievalService

logger = logging.getLogger(__name__)


class RerankedRetrievalService:
    """Bounded post-RRF reranking that cannot add candidates or alter provenance."""

    def __init__(
        self,
        hybrid: HybridRetrievalService,
        reranker: Reranker,
        *,
        candidate_limit: int,
        final_top_k: int,
    ) -> None:
        if candidate_limit < 1 or final_top_k < 1:
            raise ValueError("reranking limits must be positive")
        if final_top_k > candidate_limit:
            raise ValueError("rerank final_top_k cannot exceed candidate_limit")
        self._hybrid = hybrid
        self._reranker = reranker
        self._candidate_limit = candidate_limit
        self._final_top_k = final_top_k

    async def search_versions(
        self, query: str, document_version_ids: list[UUID], top_k: int
    ) -> RerankedRetrievalBatch:
        total_started = perf_counter()
        hybrid = await self._hybrid.search_versions(
            query,
            document_version_ids,
            self._candidate_limit,
        )
        pool = hybrid.candidates[: self._candidate_limit]
        allowed = {candidate.chunk_id: candidate for candidate in pool}
        rerank_started = perf_counter()
        try:
            reranked = await self._reranker.rerank(query, pool)
        except Exception:
            logger.exception(
                "reranker execution failed",
                extra={
                    "event": "reranker_failed",
                    "reranker_model": self._reranker.model_name,
                    "reranker_revision": self._reranker.revision,
                    "candidate_count_before_reranking": len(pool),
                },
            )
            raise
        reranking_latency_ms = (perf_counter() - rerank_started) * 1000
        seen: set[UUID] = set()
        for expected_rank, candidate in enumerate(reranked, start=1):
            original = allowed.get(candidate.chunk_id)
            if original is None:
                raise RerankerProvenanceError(
                    f"reranker introduced unknown chunk '{candidate.chunk_id}'"
                )
            if candidate.chunk_id in seen:
                raise RerankerProvenanceError(f"reranker duplicated chunk '{candidate.chunk_id}'")
            seen.add(candidate.chunk_id)
            original_metadata = original.model_dump()
            returned_metadata = candidate.model_dump(exclude={"reranker_score", "reranker_rank"})
            if returned_metadata != original_metadata:
                raise RerankerProvenanceError(
                    f"reranker changed retrieval metadata for chunk '{candidate.chunk_id}'"
                )
            if candidate.reranker_rank != expected_rank:
                raise RerankerProvenanceError("reranker returned inconsistent rank metadata")
        final_limit = min(top_k, self._final_top_k)
        candidates = reranked[:final_limit]
        total_latency_ms = (perf_counter() - total_started) * 1000
        logger.info(
            "reranked retrieval completed",
            extra={
                "event": "reranked_retrieval_completed",
                "reranker_model": self._reranker.model_name,
                "reranker_revision": self._reranker.revision,
                "candidate_count_before_reranking": len(pool),
                "candidate_count_after_reranking": len(candidates),
                "reranking_latency_ms": round(reranking_latency_ms, 3),
                "total_retrieval_latency_ms": round(total_latency_ms, 3),
            },
        )
        return RerankedRetrievalBatch(
            document_version_ids=document_version_ids,
            candidates=candidates,
            embedding_latency_ms=hybrid.embedding_latency_ms,
            dense_latency_ms=hybrid.dense_latency_ms,
            lexical_latency_ms=hybrid.lexical_latency_ms,
            fusion_latency_ms=hybrid.fusion_latency_ms,
            reranking_latency_ms=reranking_latency_ms,
            total_latency_ms=total_latency_ms,
            dense_candidate_count=hybrid.dense_candidate_count,
            lexical_candidate_count=hybrid.lexical_candidate_count,
            merged_duplicate_count=hybrid.merged_duplicate_count,
            pre_rerank_candidate_count=len(pool),
            post_rerank_candidate_count=len(candidates),
            reranker_model=self._reranker.model_name,
            reranker_revision=self._reranker.revision,
        )
