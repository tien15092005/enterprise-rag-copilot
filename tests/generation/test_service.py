from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.db.models import DocumentVersion, LogicalDocument
from app.db.session import Database
from app.generation.context import EvidenceContextBuilder
from app.generation.errors import CitationValidationError, LLMConfigurationError, LLMProviderError
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage
from app.generation.service import RagAnswerService
from app.models.query import QueryRequest
from app.models.retrieval import DenseRetrievalBatch, DenseRetrievalCandidate
from app.models.versioning import (
    DocumentLanguage,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
)
from app.retrieval.version_scope import VersionScopeResolver


class FakeRetrieval:
    def __init__(self, candidates: list[DenseRetrievalCandidate]) -> None:
        self.candidates = candidates
        self.requested_version_ids: list[object] = []

    async def search_versions(
        self, _query: str, document_version_ids: list[object], _top_k: int
    ) -> DenseRetrievalBatch:
        self.requested_version_ids = document_version_ids
        return DenseRetrievalBatch(
            document_version_ids=document_version_ids,
            embedding_model="fake-e5",
            candidates=self.candidates,
            embedding_latency_ms=1.5,
            retrieval_latency_ms=2.5,
            total_latency_ms=4,
        )


class FakeLLM:
    provider_name = "fake"
    model_name = "fake-grounded-v1"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(self.response, LLMTokenUsage(100, 20, 120))

    async def close(self) -> None:
        return None


class FailingLLM(FakeLLM):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError("controlled failure", provider="fake")


async def ready_scope(database: Database) -> tuple[Any, DocumentVersion]:
    async with database.session_factory() as session:
        document = LogicalDocument(stable_key=f"rag-{uuid4().hex}", title="Employee Handbook")
        version = DocumentVersion(
            logical_document=document,
            version_label="2026",
            revision_number=1,
            version_ordinal=1,
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
            language=DocumentLanguage.ENGLISH,
            source_filename="employee_handbook_en.md",
            checksum=uuid4().hex * 2,
            storage_key=f"rag/{uuid4().hex}.md",
            business_status=VersionBusinessStatus.ACTIVE,
            indexing_status=VersionIndexingStatus.READY,
            vector_index_status=VectorIndexStatus.INDEXED,
            ingestion_config={},
        )
        session.add_all([document, version])
        await session.commit()
        return document.logical_document_id, version


def make_candidate(version: DocumentVersion, text: str) -> DenseRetrievalCandidate:
    return DenseRetrievalCandidate(
        chunk_id=uuid4(),
        document_version_id=version.document_version_id,
        logical_document_id=version.logical_document_id,
        text=text,
        score=0.91,
        ordinal=0,
        page=None,
        section_path=["Remote Work"],
        language=version.language,
    )


async def build_service(
    database: Database,
    retrieval: FakeRetrieval,
    llm: FakeLLM | None,
    settings: Settings,
) -> tuple[RagAnswerService, Any]:
    session = database.session_factory()
    service = RagAnswerService(
        session=session,
        scope_resolver=VersionScopeResolver(session),
        retrieval=retrieval,  # type: ignore[arg-type]
        context_builder=EvidenceContextBuilder(2000),
        llm=llm,
        settings=settings,
    )
    return service, session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "llm_answer", "expected_language"),
    [
        (
            "How many remote work days are allowed?",
            "Employees may work remotely two days per week [S1].",
            DocumentLanguage.ENGLISH,
        ),
        (
            "Nhân viên được nghỉ phép bao nhiêu ngày?",
            "Nhân viên được nghỉ mười hai ngày [S1].",
            DocumentLanguage.VIETNAMESE,
        ),
    ],
)
async def test_grounded_bilingual_answer_and_structured_citations(
    app: Any,
    settings: Settings,
    question: str,
    llm_answer: str,
    expected_language: DocumentLanguage,
) -> None:
    database: Database = app.state.database
    logical_document_id, version = await ready_scope(database)
    retrieval = FakeRetrieval([make_candidate(version, "Employees may work remotely two days.")])
    llm = FakeLLM(llm_answer)
    service, session = await build_service(database, retrieval, llm, settings)
    try:
        result = await service.answer(
            QueryRequest(question=question, logical_document_id=logical_document_id, debug=True)
        )
    finally:
        await session.close()

    assert result.answer_language is expected_language
    assert result.citations[0].citation_key == "S1"
    assert result.citations[0].version_label == "2026"
    assert result.citations[0].filename == "employee_handbook_en.md"
    assert result.token_usage is not None and result.token_usage.total_tokens == 120
    assert result.debug_candidates is not None
    assert retrieval.requested_version_ids == [version.document_version_id]
    assert expected_language.value in {"en", "vi"}
    expected_prompt_language = "Vietnamese" if expected_language.value == "vi" else "English"
    assert f"Requested answer language: {expected_prompt_language}" in llm.calls[0].user_prompt


@pytest.mark.asyncio
async def test_zero_evidence_refuses_without_calling_llm(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    logical_document_id, _version = await ready_scope(database)
    llm = FakeLLM("must not be used")
    service, session = await build_service(database, FakeRetrieval([]), llm, settings)
    try:
        result = await service.answer(
            QueryRequest(
                question="What is the cafeteria menu?", logical_document_id=logical_document_id
            )
        )
    finally:
        await session.close()

    assert result.insufficient_evidence is True
    assert result.citations == []
    assert result.timing.generation_ms == 0
    assert llm.calls == []


@pytest.mark.asyncio
async def test_retrieval_only_succeeds_without_llm(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    logical_document_id, version = await ready_scope(database)
    retrieval = FakeRetrieval([make_candidate(version, "Known evidence")])
    service, session = await build_service(database, retrieval, None, settings)
    try:
        result = await service.answer(
            QueryRequest(
                question="What does the policy say?",
                logical_document_id=logical_document_id,
                retrieval_only=True,
                debug=True,
            )
        )
    finally:
        await session.close()

    assert result.answer is None
    assert [citation.citation_key for citation in result.citations] == ["S1"]
    assert result.citations[0].document_version_id == version.document_version_id
    assert result.timing.generation_ms == 0
    assert result.token_usage is None
    assert result.debug_candidates is not None


@pytest.mark.asyncio
async def test_generation_without_llm_reports_configuration_error(
    app: Any, settings: Settings
) -> None:
    database: Database = app.state.database
    logical_document_id, version = await ready_scope(database)
    retrieval = FakeRetrieval([make_candidate(version, "Known evidence")])
    service, session = await build_service(database, retrieval, None, settings)
    try:
        with pytest.raises(LLMConfigurationError):
            await service.answer(
                QueryRequest(
                    question="What does the policy say?", logical_document_id=logical_document_id
                )
            )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unknown_citation_key_is_rejected(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    logical_document_id, version = await ready_scope(database)
    retrieval = FakeRetrieval([make_candidate(version, "Known evidence")])
    service, session = await build_service(database, retrieval, FakeLLM("Claim [S99]."), settings)
    try:
        with pytest.raises(CitationValidationError) as exc_info:
            await service.answer(
                QueryRequest(question="Question", logical_document_id=logical_document_id)
            )
    finally:
        await session.close()
    assert exc_info.value.details["unknown_citation_keys"] == ["S99"]


@pytest.mark.asyncio
async def test_provider_failure_is_structured(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    logical_document_id, version = await ready_scope(database)
    retrieval = FakeRetrieval([make_candidate(version, "Known evidence")])
    service, session = await build_service(database, retrieval, FailingLLM(""), settings)
    try:
        with pytest.raises(LLMProviderError) as exc_info:
            await service.answer(
                QueryRequest(question="Question", logical_document_id=logical_document_id)
            )
    finally:
        await session.close()
    assert exc_info.value.code == "llm_provider_failed"
