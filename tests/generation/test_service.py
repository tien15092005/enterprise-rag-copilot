from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.db.models import DocumentVersion, LogicalDocument
from app.db.session import Database
from app.generation.citations import CitationScopeError
from app.generation.context import EvidenceContextBuilder
from app.generation.errors import CitationValidationError, LLMConfigurationError, LLMProviderError
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage
from app.generation.service import RagAnswerService
from app.models.query import QueryRequest, QueryVersionSelector
from app.models.retrieval import (
    DenseRetrievalBatch,
    DenseRetrievalCandidate,
    HybridRetrievalBatch,
    HybridRetrievalCandidate,
    RerankedRetrievalBatch,
    RerankedRetrievalCandidate,
    RetrievalMode,
)
from app.models.versioning import (
    DocumentLanguage,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
    VersionSelectorKind,
)
from app.retrieval.version_scope import VersionScopeResolver


class FakeRetrieval:
    def __init__(self, candidates: list[DenseRetrievalCandidate]) -> None:
        self.candidates = candidates
        self.requested_version_ids: list[object] = []
        self.requested_mode: RetrievalMode | None = None

    async def search_versions(
        self,
        _query: str,
        document_version_ids: list[object],
        _top_k: int,
        _mode: RetrievalMode,
    ) -> DenseRetrievalBatch:
        self.requested_version_ids = document_version_ids
        self.requested_mode = _mode
        candidates = [
            candidate
            for candidate in self.candidates
            if candidate.document_version_id in document_version_ids
        ]
        return DenseRetrievalBatch(
            document_version_ids=document_version_ids,
            embedding_model="fake-e5",
            candidates=candidates,
            embedding_latency_ms=1.5,
            retrieval_latency_ms=2.5,
            total_latency_ms=4,
        )


class LeakyRetrieval(FakeRetrieval):
    async def search_versions(
        self,
        _query: str,
        document_version_ids: list[object],
        _top_k: int,
        _mode: RetrievalMode,
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


class FakeHybridRetrieval:
    def __init__(self, candidate: HybridRetrievalCandidate) -> None:
        self.candidate = candidate
        self.requested_mode: RetrievalMode | None = None

    async def search_versions(
        self,
        _query: str,
        document_version_ids: list[object],
        _top_k: int,
        mode: RetrievalMode,
    ) -> HybridRetrievalBatch:
        self.requested_mode = mode
        return HybridRetrievalBatch(
            document_version_ids=document_version_ids,
            candidates=[self.candidate],
            embedding_latency_ms=1,
            dense_latency_ms=2,
            lexical_latency_ms=3,
            fusion_latency_ms=0.5,
            total_latency_ms=6.5,
            dense_candidate_count=1,
            lexical_candidate_count=1,
            merged_duplicate_count=1,
        )


class FakeRerankedRetrieval:
    def __init__(self, candidate: RerankedRetrievalCandidate) -> None:
        self.candidate = candidate
        self.requested_mode: RetrievalMode | None = None

    async def search_versions(
        self,
        _query: str,
        document_version_ids: list[object],
        _top_k: int,
        mode: RetrievalMode,
    ) -> RerankedRetrievalBatch:
        self.requested_mode = mode
        return RerankedRetrievalBatch(
            document_version_ids=document_version_ids,
            candidates=[self.candidate],
            embedding_latency_ms=1,
            dense_latency_ms=2,
            lexical_latency_ms=3,
            fusion_latency_ms=0.5,
            reranking_latency_ms=4,
            total_latency_ms=10.5,
            dense_candidate_count=1,
            lexical_candidate_count=1,
            merged_duplicate_count=1,
            pre_rerank_candidate_count=1,
            post_rerank_candidate_count=1,
            reranker_model="fake-multilingual",
            reranker_revision="pinned",
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


async def versioned_scope(database: Database) -> tuple[Any, DocumentVersion, DocumentVersion]:
    async with database.session_factory() as session:
        document = LogicalDocument(stable_key=f"citation-{uuid4().hex}", title="Policy")
        v1 = DocumentVersion(
            logical_document=document,
            version_label="2025",
            revision_number=1,
            version_ordinal=1,
            effective_from=datetime(2025, 1, 1, tzinfo=UTC),
            effective_to=datetime(2026, 1, 1, tzinfo=UTC),
            language=DocumentLanguage.ENGLISH,
            source_filename="policy-2025.md",
            checksum=uuid4().hex * 2,
            storage_key=f"citation/{uuid4().hex}.md",
            business_status=VersionBusinessStatus.SUPERSEDED,
            indexing_status=VersionIndexingStatus.READY,
            vector_index_status=VectorIndexStatus.INDEXED,
            ingestion_config={},
        )
        v2 = DocumentVersion(
            logical_document=document,
            version_label="2026",
            revision_number=1,
            version_ordinal=2,
            effective_from=datetime(2026, 1, 1, tzinfo=UTC),
            language=DocumentLanguage.ENGLISH,
            source_filename="policy-2026.md",
            checksum=uuid4().hex * 2,
            storage_key=f"citation/{uuid4().hex}.md",
            business_status=VersionBusinessStatus.ACTIVE,
            indexing_status=VersionIndexingStatus.READY,
            vector_index_status=VectorIndexStatus.INDEXED,
            ingestion_config={},
        )
        session.add_all([document, v1, v2])
        await session.commit()
        return document.logical_document_id, v1, v2


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
    assert result.citations[0].source_filename == "employee_handbook_en.md"
    assert result.token_usage is not None and result.token_usage.total_tokens == 120
    assert result.debug is not None
    assert result.debug.evidence[0].retrieval_score == 0.91
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
    assert result.debug is not None


@pytest.mark.asyncio
async def test_retrieval_only_hybrid_preserves_fusion_debug(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    logical_document_id, version = await ready_scope(database)
    base = make_candidate(version, "Known hybrid evidence")
    retrieval = FakeHybridRetrieval(
        HybridRetrievalCandidate(
            chunk_id=base.chunk_id,
            document_version_id=base.document_version_id,
            logical_document_id=base.logical_document_id,
            text=base.text,
            ordinal=base.ordinal,
            page=base.page,
            section_path=base.section_path,
            language=base.language,
            dense_rank=2,
            dense_score=0.8,
            lexical_rank=1,
            lexical_score=0.5,
            rrf_score=0.0325,
        )
    )
    service, session = await build_service(database, retrieval, None, settings)  # type: ignore[arg-type]
    try:
        result = await service.answer(
            QueryRequest(
                question="What does the policy say?",
                logical_document_id=logical_document_id,
                retrieval_mode=RetrievalMode.HYBRID,
                retrieval_only=True,
                debug=True,
            )
        )
    finally:
        await session.close()

    assert result.answer is None
    assert retrieval.requested_mode is RetrievalMode.HYBRID
    assert result.citations[0].dense_rank == 2
    assert result.citations[0].lexical_rank == 1
    assert result.citations[0].rrf_score == 0.0325
    assert result.timing.lexical_retrieval_ms == 3
    assert result.timing.fusion_ms == 0.5


@pytest.mark.asyncio
async def test_retrieval_only_rerank_preserves_all_scores_and_skips_llm(
    app: Any, settings: Settings
) -> None:
    database: Database = app.state.database
    logical_document_id, version = await ready_scope(database)
    base = make_candidate(version, "Known reranked evidence")
    retrieval = FakeRerankedRetrieval(
        RerankedRetrievalCandidate(
            chunk_id=base.chunk_id,
            document_version_id=base.document_version_id,
            logical_document_id=base.logical_document_id,
            text=base.text,
            ordinal=base.ordinal,
            page=base.page,
            section_path=base.section_path,
            language=base.language,
            dense_rank=3,
            dense_score=0.7,
            lexical_rank=2,
            lexical_score=0.4,
            rrf_score=0.03,
            rrf_rank=2,
            reranker_score=1.75,
            reranker_rank=1,
        )
    )
    service, session = await build_service(database, retrieval, None, settings)  # type: ignore[arg-type]
    try:
        result = await service.answer(
            QueryRequest(
                question="What does the policy say?",
                logical_document_id=logical_document_id,
                retrieval_mode=RetrievalMode.HYBRID_RERANK,
                retrieval_only=True,
                debug=True,
            )
        )
    finally:
        await session.close()

    citation = result.citations[0]
    assert result.answer is None
    assert retrieval.requested_mode is RetrievalMode.HYBRID_RERANK
    assert citation.retrieval_score == 1.75
    assert citation.dense_rank == 3
    assert citation.lexical_rank == 2
    assert citation.rrf_rank == 2
    assert citation.reranker_rank == 1
    assert citation.reranker_score == 1.75
    assert result.timing.reranking_ms == 4
    assert result.timing.generation_ms == 0


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


@pytest.mark.asyncio
async def test_citations_preserve_exact_current_all_and_as_of_provenance(
    app: Any, settings: Settings
) -> None:
    database: Database = app.state.database
    logical_document_id, v1, v2 = await versioned_scope(database)
    retrieval = FakeRetrieval(
        [make_candidate(v1, "2025 evidence"), make_candidate(v2, "2026 evidence")]
    )
    service, session = await build_service(
        database, retrieval, FakeLLM("Claim [S1] [S2]."), settings
    )
    try:
        all_result = await service.answer(
            QueryRequest(
                question="Question",
                logical_document_id=logical_document_id,
                version_selector=QueryVersionSelector(kind=VersionSelectorKind.ALL),
                retrieval_mode=RetrievalMode.HYBRID_RERANK,
            )
        )
        assert [citation.document_version_id for citation in all_result.citations] == [
            v1.document_version_id,
            v2.document_version_id,
        ]

        assert isinstance(service._llm, FakeLLM)
        service._llm.response = "Claim [S1]."
        exact_result = await service.answer(
            QueryRequest(
                question="Question",
                version_selector=QueryVersionSelector(
                    kind=VersionSelectorKind.EXACT, document_version_id=v1.document_version_id
                ),
                retrieval_mode=RetrievalMode.HYBRID_RERANK,
            )
        )
        assert exact_result.citations[0].document_version_id == v1.document_version_id

        current_result = await service.answer(
            QueryRequest(
                question="Question",
                logical_document_id=logical_document_id,
                retrieval_mode=RetrievalMode.HYBRID_RERANK,
            )
        )
        assert current_result.citations[0].document_version_id == v2.document_version_id

        as_of_result = await service.answer(
            QueryRequest(
                question="Question",
                logical_document_id=logical_document_id,
                version_selector=QueryVersionSelector(
                    kind=VersionSelectorKind.AS_OF,
                    as_of=datetime(2025, 6, 1, tzinfo=UTC),
                ),
                retrieval_mode=RetrievalMode.HYBRID_RERANK,
            )
        )
        assert as_of_result.citations[0].document_version_id == v1.document_version_id
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_citation_construction_rejects_version_leakage(app: Any, settings: Settings) -> None:
    database: Database = app.state.database
    logical_document_id, v1, v2 = await versioned_scope(database)
    service, session = await build_service(
        database,
        LeakyRetrieval([make_candidate(v2, "Leaked version evidence")]),
        FakeLLM("Claim [S1]."),
        settings,
    )
    try:
        with pytest.raises(CitationScopeError):
            await service.answer(
                QueryRequest(
                    question="Question",
                    logical_document_id=logical_document_id,
                    version_selector=QueryVersionSelector(
                        kind=VersionSelectorKind.EXACT, document_version_id=v1.document_version_id
                    ),
                )
            )
    finally:
        await session.close()
