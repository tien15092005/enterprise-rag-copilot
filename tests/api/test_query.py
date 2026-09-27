from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.api.dependencies import get_rag_service
from app.models.query import (
    DebugCandidate,
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
            prompt_version="rag_baseline_v1",
            insufficient_evidence=False,
            debug_candidates=(
                [
                    DebugCandidate(
                        citation_key="S1",
                        chunk_id=uuid4(),
                        document_version_id=version_id,
                        score=0.9,
                        ordinal=0,
                        page=None,
                        section_path=None,
                        language=DocumentLanguage.ENGLISH,
                    )
                ]
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
    assert payload["resolved_scope"]["kind"] == "current"
    assert payload["debug_candidates"][0]["score"] == 0.9
    assert payload["prompt_version"] == "rag_baseline_v1"


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
