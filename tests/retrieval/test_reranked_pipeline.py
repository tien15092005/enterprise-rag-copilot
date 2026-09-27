from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.models.retrieval import (
    DenseRetrievalBatch,
    HybridRetrievalBatch,
    HybridRetrievalCandidate,
    RerankedRetrievalBatch,
    RerankedRetrievalCandidate,
    RetrievalMode,
)
from app.models.versioning import DocumentLanguage
from app.reranking.errors import RerankerProvenanceError
from app.retrieval.hybrid import ModeAwareRetrievalService
from app.retrieval.reranked import RerankedRetrievalService


def candidate(version_id: UUID, rank: int) -> HybridRetrievalCandidate:
    return HybridRetrievalCandidate(
        chunk_id=uuid4(),
        document_version_id=version_id,
        logical_document_id=uuid4(),
        text=f"evidence {rank}",
        ordinal=rank,
        page=rank,
        section_path=["Policy"],
        language=DocumentLanguage.ENGLISH,
        dense_rank=rank,
        dense_score=1 - rank / 10,
        lexical_rank=rank,
        lexical_score=1 / rank,
        rrf_score=1 / (60 + rank),
        rrf_rank=rank,
    )


class FakeHybrid:
    def __init__(self, candidates: list[HybridRetrievalCandidate]) -> None:
        self.candidates = candidates
        self.requested_top_k = 0
        self.version_ids: list[UUID] = []

    async def search_versions(
        self, _query: str, document_version_ids: list[UUID], top_k: int
    ) -> HybridRetrievalBatch:
        self.requested_top_k = top_k
        self.version_ids = document_version_ids
        return HybridRetrievalBatch(
            document_version_ids=document_version_ids,
            candidates=self.candidates[:top_k],
            embedding_latency_ms=1,
            dense_latency_ms=2,
            lexical_latency_ms=3,
            fusion_latency_ms=0.5,
            total_latency_ms=6.5,
            dense_candidate_count=len(self.candidates),
            lexical_candidate_count=len(self.candidates),
            merged_duplicate_count=len(self.candidates),
        )


class FakeReranker:
    model_name = "fake-multilingual"
    revision = "test-revision"

    async def rerank(
        self, _query: str, candidates: list[HybridRetrievalCandidate]
    ) -> list[RerankedRetrievalCandidate]:
        return [
            RerankedRetrievalCandidate(
                **item.model_dump(),
                reranker_score=float(rank),
                reranker_rank=index,
            )
            for index, (rank, item) in enumerate(
                zip(range(len(candidates), 0, -1), reversed(candidates), strict=True),
                start=1,
            )
        ]


class InjectingReranker(FakeReranker):
    async def rerank(
        self, query: str, candidates: list[HybridRetrievalCandidate]
    ) -> list[RerankedRetrievalCandidate]:
        result = await super().rerank(query, candidates)
        return [result[0].model_copy(update={"chunk_id": uuid4()})]


@pytest.mark.asyncio
async def test_candidate_limit_final_top_k_reordering_and_metadata_preservation() -> None:
    version = uuid4()
    original = [candidate(version, rank) for rank in range(1, 5)]
    hybrid = FakeHybrid(original)
    service = RerankedRetrievalService(
        hybrid,  # type: ignore[arg-type]
        FakeReranker(),
        candidate_limit=3,
        final_top_k=2,
    )

    result = await service.search_versions("policy", [version], 10)

    assert hybrid.requested_top_k == 3
    assert [item.chunk_id for item in result.candidates] == [
        original[2].chunk_id,
        original[1].chunk_id,
    ]
    assert result.candidates[0].dense_score == original[2].dense_score
    assert result.candidates[0].lexical_score == original[2].lexical_score
    assert result.candidates[0].rrf_score == original[2].rrf_score
    assert result.candidates[0].rrf_rank == 3
    assert result.candidates[0].reranker_rank == 1
    assert result.pre_rerank_candidate_count == 3
    assert result.post_rerank_candidate_count == 2
    assert result.reranking_latency_ms >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("version_count", [1, 2])
async def test_reranking_preserves_exact_current_all_and_as_of_version_sets(
    version_count: int,
) -> None:
    versions = [uuid4() for _ in range(version_count)]
    items = [candidate(version, rank) for rank, version in enumerate(versions, start=1)]
    hybrid = FakeHybrid(items)
    result = await RerankedRetrievalService(
        hybrid,  # type: ignore[arg-type]
        FakeReranker(),
        candidate_limit=5,
        final_top_k=5,
    ).search_versions("query", versions, 5)

    assert hybrid.version_ids == versions
    assert {item.document_version_id for item in result.candidates} <= set(versions)


@pytest.mark.asyncio
async def test_reranker_cannot_introduce_candidates_or_versions() -> None:
    version = uuid4()
    service = RerankedRetrievalService(
        FakeHybrid([candidate(version, 1)]),  # type: ignore[arg-type]
        InjectingReranker(),
        candidate_limit=5,
        final_top_k=5,
    )
    with pytest.raises(RerankerProvenanceError):
        await service.search_versions("query", [version], 5)


class FakeDense:
    async def search_versions(
        self, _query: str, document_version_ids: list[UUID], _top_k: int
    ) -> DenseRetrievalBatch:
        return DenseRetrievalBatch(
            document_version_ids=document_version_ids,
            embedding_model="fake",
            candidates=[],
            embedding_latency_ms=0,
            retrieval_latency_ms=0,
            total_latency_ms=0,
        )


@pytest.mark.asyncio
async def test_mode_compatibility_and_hybrid_rerank_dispatch() -> None:
    version = uuid4()
    item = candidate(version, 1)
    hybrid = FakeHybrid([item])
    reranked = RerankedRetrievalService(
        hybrid,  # type: ignore[arg-type]
        FakeReranker(),
        candidate_limit=5,
        final_top_k=5,
    )
    service = ModeAwareRetrievalService(
        FakeDense(),  # type: ignore[arg-type]
        hybrid,  # type: ignore[arg-type]
        reranked,
    )

    assert (await service.search_versions("q", [version], 5, RetrievalMode.DENSE)).candidates == []
    hybrid_result = await service.search_versions("q", [version], 5, RetrievalMode.HYBRID)
    reranked_result = await service.search_versions("q", [version], 5, RetrievalMode.HYBRID_RERANK)
    assert hybrid_result.candidates[0].chunk_id == item.chunk_id
    assert isinstance(reranked_result, RerankedRetrievalBatch)
    assert reranked_result.candidates[0].reranker_score == 1
