from uuid import uuid4

from app.generation.context import EvidenceContextBuilder, EvidenceMetadata
from app.models.retrieval import DenseRetrievalCandidate
from app.models.versioning import DocumentLanguage


def candidate(text: str, score: float) -> DenseRetrievalCandidate:
    return DenseRetrievalCandidate(
        chunk_id=uuid4(),
        document_version_id=uuid4(),
        logical_document_id=uuid4(),
        text=text,
        score=score,
        ordinal=0,
        page=3,
        section_path=["Leave", "Annual"],
        language=DocumentLanguage.ENGLISH,
    )


def test_context_preserves_ranking_metadata_and_stable_keys() -> None:
    first = candidate("Highest ranked evidence", 0.9)
    second = candidate("Second ranked evidence", 0.8)
    built = EvidenceContextBuilder(2000).build(
        [
            EvidenceMetadata(first, "Handbook", "2026", "handbook.pdf"),
            EvidenceMetadata(second, "Handbook", "2025", "handbook.pdf"),
        ]
    )

    assert [item.citation.citation_key for item in built.evidence] == ["S1", "S2"]
    assert built.text.index("Highest ranked") < built.text.index("Second ranked")
    assert built.evidence[0].citation.page == 3
    assert built.evidence[0].citation.section_path == ["Leave", "Annual"]
    assert built.evidence[0].citation.filename == "handbook.pdf"


def test_context_limit_keeps_highest_ranked_evidence_first() -> None:
    first = candidate("A" * 1000, 0.9)
    second = candidate("B" * 1000, 0.8)

    built = EvidenceContextBuilder(300).build(
        [
            EvidenceMetadata(first, "Handbook", "2026", "handbook.pdf"),
            EvidenceMetadata(second, "Handbook", "2025", "handbook.pdf"),
        ]
    )

    assert len(built.text) <= 300
    assert built.truncated is True
    assert len(built.evidence) == 1
    assert "B" * 20 not in built.text
