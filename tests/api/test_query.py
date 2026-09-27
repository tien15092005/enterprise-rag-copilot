from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.api.dependencies import get_rag_service
from app.models.query import (
    Citation,
    CitationValidationDebug,
    QueryDebug,
    QueryRequest,
    QueryResponse,
    QueryTiming,
    ResolvedScope,
    ResolvedVersion,
)
from app.models.versioning import DocumentLanguage
from app.retrieval.version_scope import VersionScopeError


class FakeRagService:
    async def answer(self, request: QueryRequest) -> QueryResponse:
        version_id = request.version_selector.document_version_id or uuid4()
        return QueryResponse(
            answer="Grounded answer [S1].",
            answer_language=DocumentLanguage.ENGLISH,
            citations=[],
            resolved_scope=ResolvedScope(
                kind=request.version_selector.kind,
                logical_document_id=request.logical_document_id or uuid4(),
                versions=[
                    ResolvedVersion(
                        document_version_id=version_id,
                        version_label="2026",
                        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
                        effective_to=None,
                    )
                ],
            ),
            timing=QueryTiming(
                scope_resolution_ms=1,
                query_embedding_ms=2,
                retrieval_ms=3,
                context_construction_ms=1,
                generation_ms=4,
                total_ms=11,
            ),
            token_usage=None,
            prompt_version="citation_baseline_v2",
            insufficient_evidence=False,
            debug=(
                QueryDebug(
                    evidence=[
                        Citation(
                            citation_key="S1",
                            logical_document_id=request.logical_document_id or uuid4(),
                            document_version_id=version_id,
                            version_label="2026",
                            chunk_id=uuid4(),
                            source_filename="handbook.md",
                            page=None,
                            section_path=None,
                            excerpt="Grounded evidence.",
                            rank=1,
                            retrieval_score=0.9,
                            language=DocumentLanguage.ENGLISH,
                            effective_from=None,
                            effective_to=None,
                        )
                    ],
                    citation_validation=CitationValidationDebug(
                        status="validated",
                        referenced_keys=["S1"],
                        normalized_keys=["S1"],
                        duplicate_keys=[],
                        omitted_evidence_keys=[],
                    ),
                )
                if request.debug
                else None
            ),
        )


class FailingScopeService:
    async def answer(self, _request: QueryRequest) -> QueryResponse:
        raise VersionScopeError("no explicitly active, ready, indexed current version exists")


@pytest.mark.asyncio
async def test_query_response_schema_and_debug_mode(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    document_id = uuid4()
    response = await client.post(
        "/api/query",
        json={
            "question": "What changed?",
            "logical_document_id": str(document_id),
            "version_selector": {"kind": "current"},
            "debug": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "answer",
        "answer_language",
        "citations",
        "resolved_scope",
        "timing",
        "token_usage",
        "prompt_version",
        "insufficient_evidence",
        "debug",
    }
    assert payload["resolved_scope"]["kind"] == "current"
    assert payload["debug"]["evidence"][0]["retrieval_score"] == 0.9
    assert payload["debug"]["citation_validation"]["status"] == "validated"
    assert payload["prompt_version"] == "citation_baseline_v2"


@pytest.mark.asyncio
async def test_query_accepts_retrieval_only_mode(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    response = await client.post(
        "/api/query",
        json={"question": "What changed?", "retrieval_only": True},
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_validation(client: AsyncClient) -> None:
    response = await client.post(
        "/api/query",
        json={"question": "", "version_selector": {"kind": "exact"}, "top_k": 0},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_domain_error_mapping(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[get_rag_service] = lambda: FailingScopeService()
    response = await client.post(
        "/api/query",
        json={
            "question": "Question",
            "logical_document_id": str(uuid4()),
            "version_selector": {"kind": "current"},
        },
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "version_scope_unresolvable"
