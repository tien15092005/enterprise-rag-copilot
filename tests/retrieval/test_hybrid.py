from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.models.retrieval import (
    DenseRetrievalBatch,
    DenseRetrievalCandidate,
    LexicalRetrievalBatch,
    LexicalRetrievalCandidate,
)
from app.models.versioning import DocumentLanguage
from app.retrieval.errors import RetrievalProvenanceError
from app.retrieval.hybrid import HybridRetrievalService


class FakeDense:
    def __init__(self, candidates: list[DenseRetrievalCandidate]) -> None:
        self.candidates = candidates
        self.version_ids: list[UUID] = []

    async def search_versions(
        self, _query: str, document_version_ids: list[UUID], _top_k: int
    ) -> DenseRetrievalBatch:
        self.version_ids = document_version_ids
        return DenseRetrievalBatch(
            document_version_ids=document_version_ids,
            embedding_model="fake",
            candidates=self.candidates,
            embedding_latency_ms=1,
            retrieval_latency_ms=2,
            total_latency_ms=3,
        )


class FakeLexical:
    def __init__(self, candidates: list[LexicalRetrievalCandidate]) -> None:
        self.candidates = candidates
        self.version_ids: list[UUID] = []

    async def search_versions(
        self, _query: str, document_version_ids: list[UUID], _top_k: int
    ) -> LexicalRetrievalBatch:
        self.version_ids = document_version_ids
        return LexicalRetrievalBatch(
            document_version_ids=document_version_ids,
            candidates=self.candidates,
            retrieval_latency_ms=2,
            total_latency_ms=2,
        )


def dense_candidate(chunk_id: UUID, version_id: UUID) -> DenseRetrievalCandidate:
    return DenseRetrievalCandidate(
        chunk_id=chunk_id,
        document_version_id=version_id,
        logical_document_id=DOCUMENT_ID,
        text="matching evidence",
        score=0.9,
        ordinal=0,
        page=1,
        section_path=["Policy"],
        language=DocumentLanguage.ENGLISH,
    )


def lexical_candidate(chunk_id: UUID, version_id: UUID) -> LexicalRetrievalCandidate:
    return LexicalRetrievalCandidate(
        chunk_id=chunk_id,
        document_version_id=version_id,
        logical_document_id=DOCUMENT_ID,
        text="matching evidence",
        lexical_score=0.5,
        ordinal=0,
        page=1,
        section_path=["Policy"],
        language=DocumentLanguage.ENGLISH,
    )


DOCUMENT_ID = uuid4()


@pytest.mark.asyncio
async def test_hybrid_passes_same_scope_and_preserves_provenance() -> None:
    version_ids = [uuid4(), uuid4()]
    chunk_id = uuid4()
    dense = FakeDense([dense_candidate(chunk_id, version_ids[0])])
    lexical = FakeLexical([lexical_candidate(chunk_id, version_ids[0])])
    service = HybridRetrievalService(
        dense,  # type: ignore[arg-type]
        lexical,  # type: ignore[arg-type]
        dense_candidate_limit=10,
        lexical_candidate_limit=10,
        rrf_k=60,
    )

    result = await service.search_versions("policy", version_ids, 5)

    assert dense.version_ids == lexical.version_ids == version_ids
    assert result.candidates[0].chunk_id == chunk_id
    assert result.candidates[0].document_version_id == version_ids[0]
    assert result.candidates[0].dense_rank == 1
    assert result.candidates[0].lexical_rank == 1
    assert result.merged_duplicate_count == 1


@pytest.mark.asyncio
async def test_hybrid_rejects_branch_version_leakage() -> None:
    allowed = uuid4()
    leaked = uuid4()
    service = HybridRetrievalService(
        FakeDense([dense_candidate(uuid4(), leaked)]),  # type: ignore[arg-type]
        FakeLexical([]),  # type: ignore[arg-type]
        dense_candidate_limit=10,
        lexical_candidate_limit=10,
        rrf_k=60,
    )

    with pytest.raises(RetrievalProvenanceError):
        await service.search_versions("policy", [allowed], 5)
