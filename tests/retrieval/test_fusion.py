from __future__ import annotations

from uuid import UUID, uuid4

from app.models.retrieval import DenseRetrievalCandidate, LexicalRetrievalCandidate
from app.models.versioning import DocumentLanguage
from app.retrieval.fusion import reciprocal_rank_fusion


def dense(chunk_id: UUID, score: float) -> DenseRetrievalCandidate:
    return DenseRetrievalCandidate(
        chunk_id=chunk_id,
        document_version_id=VERSION_ID,
        logical_document_id=DOCUMENT_ID,
        text=f"chunk {chunk_id}",
        score=score,
        ordinal=0,
        page=None,
        section_path=["Policy"],
        language=DocumentLanguage.ENGLISH,
    )


def lexical(chunk_id: UUID, score: float) -> LexicalRetrievalCandidate:
    return LexicalRetrievalCandidate(
        chunk_id=chunk_id,
        document_version_id=VERSION_ID,
        logical_document_id=DOCUMENT_ID,
        text=f"chunk {chunk_id}",
        lexical_score=score,
        ordinal=0,
        page=None,
        section_path=["Policy"],
        language=DocumentLanguage.ENGLISH,
    )


DOCUMENT_ID = uuid4()
VERSION_ID = uuid4()


def test_rrf_merges_duplicates_and_preserves_branch_metadata() -> None:
    shared = uuid4()
    dense_only = uuid4()
    lexical_only = uuid4()

    fused = reciprocal_rank_fusion(
        [dense(shared, 0.9), dense(dense_only, 0.8)],
        [lexical(shared, 0.7), lexical(lexical_only, 0.6)],
        k=60,
    )

    assert fused[0].chunk_id == shared
    assert {candidate.chunk_id for candidate in fused[1:]} == {dense_only, lexical_only}
    assert fused[0].dense_rank == fused[0].lexical_rank == 1
    assert fused[0].dense_score == 0.9
    assert fused[0].lexical_score == 0.7
    by_id = {candidate.chunk_id: candidate for candidate in fused}
    assert by_id[dense_only].lexical_rank is None
    assert by_id[lexical_only].dense_rank is None
    assert fused[0].rrf_score == 2 / 61


def test_rrf_is_deterministic_and_k_is_configurable() -> None:
    first = uuid4()
    second = uuid4()
    dense_list = [dense(first, 0.9), dense(second, 0.8)]

    small_k = reciprocal_rank_fusion(dense_list, [], k=1)
    large_k = reciprocal_rank_fusion(dense_list, [], k=100)
    repeated = reciprocal_rank_fusion(dense_list, [], k=1)

    assert [item.chunk_id for item in small_k] == [first, second]
    assert [item.chunk_id for item in repeated] == [first, second]
    assert small_k[0].rrf_score == 1 / 2
    assert large_k[0].rrf_score == 1 / 101
