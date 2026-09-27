from __future__ import annotations

from uuid import uuid4

import pytest

from app.comparison.alignment import SectionAligner, SectionSnapshot
from app.comparison.models import AlignmentMethod
from app.models.versioning import DocumentLanguage


class FakeEmbeddings:
    model_name = "fake"
    revision = "test"
    dimension = 3

    async def embed_query(self, _text: str) -> list[float]:
        return [1, 0, 0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "device loss" in text or "incident reporting" in text:
                vectors.append([1, 0, 0])
            elif "obsolete" in text or text == "alpha":
                vectors.append([0, 1, 0])
            else:
                vectors.append([0, 0, 1])
        return vectors


def section(
    path: tuple[str, ...] | None,
    text: str,
    ordinal: int,
    fingerprint: str | None = None,
) -> SectionSnapshot:
    chunk_id = uuid4()
    return SectionSnapshot(
        document_version_id=uuid4(),
        logical_document_id=uuid4(),
        chunk_ids=(chunk_id,),
        citation_chunk_id=chunk_id,
        citation_text=text,
        text=text,
        section_path=path,
        section_fingerprint=fingerprint,
        ordinal=ordinal,
        page=None,
        language=DocumentLanguage.ENGLISH,
    )


@pytest.mark.asyncio
async def test_alignment_stages_and_stable_unmatched_order() -> None:
    left = [
        section(("Policy", "Remote Work"), "one day", 0),
        section(("Policy", "Annual-Leave"), "ten days", 1),
        section(("Policy", "Old title"), "nearly identical policy sentence", 2),
        section(("Policy", "Device loss"), "alpha device loss protocol", 3),
        section(("Policy", "Removed"), "obsolete unique words", 4),
    ]
    right = [
        section(("Policy", "Remote Work"), "two days", 0),
        section(("policy", "annual leave"), "twelve days", 1),
        section(("Policy", "New title"), "nearly identical policy sentences", 2),
        section(("Policy", "Incident report"), "omega incident reporting procedure", 3),
        section(("Policy", "Added"), "brand new unique material", 4),
    ]
    result = await SectionAligner(
        FakeEmbeddings(),  # type: ignore[arg-type]
        content_threshold=0.8,
        semantic_threshold=0.9,
    ).align(left, right)

    assert [pair.method for pair in result.pairs] == [
        AlignmentMethod.EXACT_PATH,
        AlignmentMethod.NORMALIZED_PATH,
        AlignmentMethod.CONTENT_SIMILARITY,
        AlignmentMethod.SEMANTIC,
        AlignmentMethod.UNALIGNED,
        AlignmentMethod.UNALIGNED,
    ]
    assert result.pairs[4].left is left[4] and result.pairs[4].right is None
    assert result.pairs[5].left is None and result.pairs[5].right is right[4]
    assert result.semantic_latency_ms >= 0


@pytest.mark.asyncio
async def test_low_similarity_is_not_silently_aligned() -> None:
    result = await SectionAligner(
        FakeEmbeddings(),  # type: ignore[arg-type]
        content_threshold=0.95,
        semantic_threshold=1.01,
    ).align(
        [section(("Old",), "alpha", 0)],
        [section(("New",), "omega", 0)],
    )

    assert len(result.pairs) == 2
    assert all(pair.method is AlignmentMethod.UNALIGNED for pair in result.pairs)


@pytest.mark.asyncio
async def test_section_fingerprint_alignment_is_explicit() -> None:
    result = await SectionAligner(
        FakeEmbeddings(),  # type: ignore[arg-type]
        content_threshold=1,
        semantic_threshold=1.01,
    ).align(
        [section(None, "old wording", 0, "stable-fingerprint")],
        [section(None, "new wording", 0, "stable-fingerprint")],
    )

    assert result.pairs[0].method is AlignmentMethod.SECTION_FINGERPRINT


@pytest.mark.asyncio
async def test_leaf_heading_alignment_ignores_versioned_document_title() -> None:
    aligner = SectionAligner(
        FakeEmbeddings(),  # type: ignore[arg-type]
        content_threshold=1,
        semantic_threshold=1.01,
    )
    exact = await aligner.align(
        [section(("Policy 2025", "Remote Work"), "one day", 0)],
        [section(("Policy 2026", "Remote Work"), "two days", 0)],
    )
    normalized = await aligner.align(
        [section(("Policy 2025", "Annual-Leave"), "ten days", 0)],
        [section(("Policy 2026", "annual leave"), "twelve days", 0)],
    )

    assert exact.pairs[0].method is AlignmentMethod.EXACT_HEADING
    assert normalized.pairs[0].method is AlignmentMethod.NORMALIZED_HEADING
