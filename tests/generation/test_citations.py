from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.generation.citations import CitationFactory, CitationSource, CitationValidator
from app.generation.context import EvidenceContextBuilder
from app.generation.errors import CitationValidationError
from app.models.retrieval import DenseRetrievalCandidate
from app.models.versioning import DocumentLanguage


def source(
    text: str = "Evidence text",
    *,
    chunk_id: UUID | None = None,
    language: DocumentLanguage = DocumentLanguage.ENGLISH,
) -> CitationSource:
    candidate = DenseRetrievalCandidate(
        chunk_id=chunk_id or uuid4(),
        document_version_id=uuid4(),
        logical_document_id=uuid4(),
        text=text,
        score=0.9,
        ordinal=0,
        page=1,
        section_path=["Policy"],
        language=language,
    )
    return CitationSource(candidate, "Policy", "2026", "policy.md")


def test_key_assignment_is_deterministic_and_deduplicates_chunks() -> None:
    first = source("First")
    duplicate = CitationSource(
        first.candidate.model_copy(update={"score": 0.4}), "Policy", "2026", "policy.md"
    )
    second = source("Second")

    built = EvidenceContextBuilder(2000).build([first, duplicate, second])

    assert [item.citation.citation_key for item in built.evidence] == ["S1", "S2"]
    assert [item.citation.rank for item in built.evidence] == [1, 2]
    assert [item.citation.chunk_id for item in built.evidence] == [
        first.candidate.chunk_id,
        second.candidate.chunk_id,
    ]


def test_excerpt_is_bounded_unicode_safe_and_from_supplied_source() -> None:
    vietnamese = "Nhân viên được hưởng mười hai ngày nghỉ phép có lương mỗi năm."
    record = CitationFactory(17).create(
        source(vietnamese, language=DocumentLanguage.VIETNAMESE), rank=1, supplied_text=vietnamese
    )

    assert record.excerpt == vietnamese[:17]
    assert len(record.excerpt) == 17
    assert record.language is DocumentLanguage.VIETNAMESE


def test_validator_normalizes_repeated_references_and_tracks_omitted_evidence() -> None:
    built = EvidenceContextBuilder(2000).build([source("First"), source("Second")])
    result = CitationValidator().validate(
        "First claim [S1]. Repeated support [S1].", tuple(item.citation for item in built.evidence)
    )

    assert result.referenced_keys == ("S1", "S1")
    assert result.normalized_keys == ("S1",)
    assert result.duplicate_keys == ("S1",)
    assert result.omitted_evidence_keys == ("S2",)
    assert [citation.citation_key for citation in result.citations] == ["S1"]


@pytest.mark.parametrize("answer", ["Unsupported claim [S9].", "Unsupported claim without a key."])
def test_validator_rejects_unknown_or_missing_citations(answer: str) -> None:
    built = EvidenceContextBuilder(2000).build([source()])

    with pytest.raises(CitationValidationError):
        CitationValidator().validate(answer, tuple(item.citation for item in built.evidence))
