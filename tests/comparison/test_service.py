from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.comparison.errors import ComparisonError, ComparisonVersionNotFoundError
from app.comparison.models import ChangeType, ComparisonRequest, ComparisonSide
from app.comparison.service import DocumentComparisonService
from app.core.config import Settings
from app.db.models import Chunk, DocumentVersion, LogicalDocument
from app.db.session import Database
from app.generation.errors import CitationValidationError
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage
from app.models.versioning import (
    DocumentLanguage,
    VersionBusinessStatus,
    VersionIndexingStatus,
)


class FakeEmbeddings:
    model_name = "fake"
    revision = "test"
    dimension = 3

    async def embed_query(self, _text: str) -> list[float]:
        return [1, 0, 0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [
            [1, 0, 0] if "obsolete" in text else [0, 1, 0] if "new benefit" in text else [0, 0, 1]
            for text in texts
        ]


class FakeLLM:
    provider_name = "fake"
    model_name = "fake-comparison"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(self.text, LLMTokenUsage(20, 10, 30))

    async def close(self) -> None:
        return None


async def add_version_pair(
    session: AsyncSession,
    *,
    language: DocumentLanguage = DocumentLanguage.ENGLISH,
) -> tuple[LogicalDocument, DocumentVersion, DocumentVersion]:
    document = LogicalDocument(stable_key=f"compare-{uuid4().hex}", title="Policy")
    left = _version(document, "2025", 1, language)
    right = _version(document, "2026", 2, language)
    if language is DocumentLanguage.ENGLISH:
        left_sections = [
            ("Remote Work", "Employees may work remotely up to one day per week."),
            ("Legacy Benefit", "This obsolete travel allowance is available."),
            ("Security", "Company laptops must use encryption."),
        ]
        right_sections = [
            ("Remote Work", "Employees may work remotely up to two days per week."),
            ("New Benefit", "A new benefit provides a wellness allowance."),
            ("Security", "Company laptops must use encryption."),
        ]
    else:
        left_sections = [("Nghỉ phép", "Nhân viên được nghỉ mười ngày mỗi năm.")]
        right_sections = [("Nghỉ phép", "Nhân viên được nghỉ mười hai ngày mỗi năm.")]
    session.add_all(
        [
            document,
            left,
            right,
            *[
                _chunk(left, index, heading, text, language)
                for index, (heading, text) in enumerate(left_sections)
            ],
            *[
                _chunk(right, index, heading, text, language)
                for index, (heading, text) in enumerate(right_sections)
            ],
        ]
    )
    await session.commit()
    return document, left, right


def _version(
    document: LogicalDocument, label: str, ordinal: int, language: DocumentLanguage
) -> DocumentVersion:
    return DocumentVersion(
        logical_document=document,
        version_label=label,
        revision_number=1,
        version_ordinal=ordinal,
        language=language,
        source_filename=f"policy-{label}.md",
        checksum=(str(ordinal) * 64)[:64],
        storage_key=f"comparison/{uuid4().hex}.md",
        business_status=VersionBusinessStatus.DRAFT,
        indexing_status=VersionIndexingStatus.READY,
        ingestion_config={},
    )


def _chunk(
    version: DocumentVersion,
    ordinal: int,
    heading: str,
    text: str,
    language: DocumentLanguage,
) -> Chunk:
    return Chunk(
        chunk_id=uuid4(),
        document_version=version,
        ordinal=ordinal,
        text=text,
        page=None,
        section_path=[heading],
        section_fingerprint=None,
        language=language,
    )


def comparison_settings(base: Settings) -> Settings:
    return base.model_copy(
        update={
            "comparison_content_similarity_threshold": 0.8,
            "comparison_semantic_similarity_threshold": 0.99,
        }
    )


@pytest.mark.asyncio
async def test_deterministic_comparison_classifies_all_changes_and_citations(
    app: Any, settings: Settings
) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document, left, right = await add_version_pair(session)
        service = DocumentComparisonService(
            session, FakeEmbeddings(), None, comparison_settings(settings)
        )
        result = await service.compare(
            ComparisonRequest(
                logical_document_id=document.logical_document_id,
                left_document_version_id=left.document_version_id,
                right_document_version_id=right.document_version_id,
                include_unchanged=True,
                debug=True,
            )
        )

    assert result.summary is None
    assert result.counts.model_dump() == {"added": 1, "removed": 1, "modified": 1, "unchanged": 1}
    assert [item.change_type for item in result.items] == [
        ChangeType.MODIFIED,
        ChangeType.REMOVED,
        ChangeType.UNCHANGED,
        ChangeType.ADDED,
    ]
    modified = result.items[0]
    assert modified.diff is not None
    assert "one day" in modified.diff.removed[0]
    assert "two days" in modified.diff.added[0]
    modified_citations = [item for item in result.citations if item.item_order == modified.order]
    assert {item.side for item in modified_citations} == {
        ComparisonSide.LEFT,
        ComparisonSide.RIGHT,
    }
    assert {item.citation.document_version_id for item in modified_citations} == {
        left.document_version_id,
        right.document_version_id,
    }
    assert all(item.citation.dense_score is None for item in modified_citations)
    assert result.debug is not None
    assert result.timing.generation_ms == 0


@pytest.mark.asyncio
async def test_vietnamese_comparison_preserves_source_text(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document, left, right = await add_version_pair(
            session, language=DocumentLanguage.VIETNAMESE
        )
        result = await DocumentComparisonService(
            session, FakeEmbeddings(), None, comparison_settings(settings)
        ).compare(
            ComparisonRequest(
                logical_document_id=document.logical_document_id,
                left_document_version_id=left.document_version_id,
                right_document_version_id=right.document_version_id,
            )
        )

    assert result.counts.modified == 1
    assert "mười ngày" in (result.items[0].left_text or "")
    assert "mười hai ngày" in (result.items[0].right_text or "")
    assert result.summary_language is DocumentLanguage.VIETNAMESE


@pytest.mark.asyncio
async def test_fake_generated_summary_is_validated_and_cites_both_versions(
    app: Any, settings: Settings
) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document, left, right = await add_version_pair(session)
        llm = FakeLLM("Remote work increased from one to two days [S1] [S2].")
        result = await DocumentComparisonService(
            session, FakeEmbeddings(), llm, comparison_settings(settings)
        ).compare(
            ComparisonRequest(
                logical_document_id=document.logical_document_id,
                left_document_version_id=left.document_version_id,
                right_document_version_id=right.document_version_id,
                generate_summary=True,
                debug=True,
            )
        )

    assert result.summary is not None
    assert {item.side for item in result.citations} == {
        ComparisonSide.LEFT,
        ComparisonSide.RIGHT,
    }
    assert result.token_usage is not None and result.token_usage.total_tokens == 30
    assert result.debug is not None
    assert result.debug.citation_validation.status == "validated"
    assert "Requested summary language: en" in llm.calls[0].user_prompt


@pytest.mark.asyncio
async def test_unknown_summary_citation_is_rejected(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document, left, right = await add_version_pair(session)
        service = DocumentComparisonService(
            session,
            FakeEmbeddings(),
            FakeLLM("Invented [S99]."),
            comparison_settings(settings),
        )
        with pytest.raises(CitationValidationError):
            await service.compare(
                ComparisonRequest(
                    logical_document_id=document.logical_document_id,
                    left_document_version_id=left.document_version_id,
                    right_document_version_id=right.document_version_id,
                    generate_summary=True,
                )
            )


@pytest.mark.asyncio
async def test_rejects_versions_from_different_logical_documents(
    app: Any, settings: Settings
) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document, left, _right = await add_version_pair(session)
        other_document, other_left, _other_right = await add_version_pair(session)
        service = DocumentComparisonService(
            session, FakeEmbeddings(), None, comparison_settings(settings)
        )
        with pytest.raises(ComparisonError, match="same logical document"):
            await service.compare(
                ComparisonRequest(
                    logical_document_id=document.logical_document_id,
                    left_document_version_id=left.document_version_id,
                    right_document_version_id=other_left.document_version_id,
                )
            )
        assert other_document.logical_document_id != document.logical_document_id


@pytest.mark.asyncio
async def test_explicit_pair_cannot_leak_a_third_version(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document, left, right = await add_version_pair(session)
        third = _version(document, "2027", 3, DocumentLanguage.ENGLISH)
        session.add_all(
            [
                third,
                _chunk(
                    third,
                    0,
                    "Remote Work",
                    "Employees may work remotely every day.",
                    DocumentLanguage.ENGLISH,
                ),
            ]
        )
        await session.commit()
        result = await DocumentComparisonService(
            session, FakeEmbeddings(), None, comparison_settings(settings)
        ).compare(
            ComparisonRequest(
                logical_document_id=document.logical_document_id,
                left_document_version_id=left.document_version_id,
                right_document_version_id=right.document_version_id,
                include_unchanged=True,
            )
        )

    assert {citation.citation.document_version_id for citation in result.citations} == {
        left.document_version_id,
        right.document_version_id,
    }
    assert all(
        third.document_version_id not in {item.citation.document_version_id}
        for item in result.citations
    )


@pytest.mark.asyncio
async def test_missing_explicit_version_is_rejected(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        document, left, _right = await add_version_pair(session)
        service = DocumentComparisonService(
            session, FakeEmbeddings(), None, comparison_settings(settings)
        )
        with pytest.raises(ComparisonVersionNotFoundError):
            await service.compare(
                ComparisonRequest(
                    logical_document_id=document.logical_document_id,
                    left_document_version_id=left.document_version_id,
                    right_document_version_id=uuid4(),
                )
            )


def test_same_version_pair_is_rejected_by_request_schema() -> None:
    version_id = uuid4()
    with pytest.raises(ValidationError):
        ComparisonRequest(
            logical_document_id=uuid4(),
            left_document_version_id=version_id,
            right_document_version_id=version_id,
        )
