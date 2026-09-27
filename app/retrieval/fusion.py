from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.models.retrieval import (
    DenseRetrievalCandidate,
    HybridRetrievalCandidate,
    LexicalRetrievalCandidate,
)
from app.retrieval.errors import RetrievalProvenanceError


@dataclass(slots=True)
class _FusedState:
    candidate: DenseRetrievalCandidate | LexicalRetrievalCandidate
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float = 0


def reciprocal_rank_fusion(
    dense: list[DenseRetrievalCandidate],
    lexical: list[LexicalRetrievalCandidate],
    *,
    k: int = 60,
) -> list[HybridRetrievalCandidate]:
    """Fuse ranked lists using sum(1 / (k + rank)), with one-based ranks."""

    if k < 1:
        raise ValueError("RRF k must be positive")
    states: dict[UUID, _FusedState] = {}

    for rank, dense_candidate in enumerate(dense, start=1):
        state = states.setdefault(dense_candidate.chunk_id, _FusedState(candidate=dense_candidate))
        _validate_same_chunk(state.candidate, dense_candidate)
        if state.dense_rank is None:
            state.dense_rank = rank
            state.dense_score = dense_candidate.score
            state.rrf_score += 1 / (k + rank)

    for rank, lexical_candidate in enumerate(lexical, start=1):
        state = states.setdefault(
            lexical_candidate.chunk_id, _FusedState(candidate=lexical_candidate)
        )
        _validate_same_chunk(state.candidate, lexical_candidate)
        if state.lexical_rank is None:
            state.lexical_rank = rank
            state.lexical_score = lexical_candidate.lexical_score
            state.rrf_score += 1 / (k + rank)

    fused = [
        HybridRetrievalCandidate(
            chunk_id=state.candidate.chunk_id,
            document_version_id=state.candidate.document_version_id,
            logical_document_id=state.candidate.logical_document_id,
            text=state.candidate.text,
            ordinal=state.candidate.ordinal,
            page=state.candidate.page,
            section_path=state.candidate.section_path,
            language=state.candidate.language,
            dense_rank=state.dense_rank,
            dense_score=state.dense_score,
            lexical_rank=state.lexical_rank,
            lexical_score=state.lexical_score,
            rrf_score=state.rrf_score,
        )
        for state in states.values()
    ]
    return sorted(
        fused,
        key=lambda candidate: (
            -candidate.rrf_score,
            min(candidate.dense_rank or 10**9, candidate.lexical_rank or 10**9),
            str(candidate.chunk_id),
        ),
    )


def _validate_same_chunk(
    first: DenseRetrievalCandidate | LexicalRetrievalCandidate,
    second: DenseRetrievalCandidate | LexicalRetrievalCandidate,
) -> None:
    comparable = (
        "document_version_id",
        "logical_document_id",
        "text",
        "ordinal",
        "page",
        "section_path",
        "language",
    )
    mismatches = [name for name in comparable if getattr(first, name) != getattr(second, name)]
    if mismatches:
        raise RetrievalProvenanceError(
            f"retrieval branches disagree on chunk '{first.chunk_id}': " + ", ".join(mismatches)
        )
