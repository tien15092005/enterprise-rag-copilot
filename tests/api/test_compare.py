from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.api.dependencies import get_comparison_service
from app.comparison.models import (
    ComparisonCounts,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonTiming,
    ComparisonVersion,
)
from app.models.versioning import DocumentLanguage


class FakeComparisonService:
    def __init__(self) -> None:
        self.request: ComparisonRequest | None = None

    async def compare(self, request: ComparisonRequest) -> ComparisonResponse:
        self.request = request
        version = dict(
            logical_document_id=request.logical_document_id,
            language=DocumentLanguage.ENGLISH,
            source_filename="policy.md",
            effective_from=None,
            effective_to=None,
        )
        return ComparisonResponse(
            left_version=ComparisonVersion(
                document_version_id=request.left_document_version_id,
                version_label="2025",
                **version,
            ),
            right_version=ComparisonVersion(
                document_version_id=request.right_document_version_id,
                version_label="2026",
                **version,
            ),
            items=[],
            counts=ComparisonCounts(),
            summary=None,
            summary_language=DocumentLanguage.ENGLISH,
            citations=[],
            timing=ComparisonTiming(
                version_load_ms=1,
                alignment_ms=1,
                semantic_fallback_ms=0,
                classification_diff_ms=1,
                generation_ms=0,
                total_ms=3,
            ),
            token_usage=None,
            prompt_version="version_comparison_v1",
        )


@pytest.mark.asyncio
async def test_compare_api_is_strongly_typed(app: FastAPI, client: AsyncClient) -> None:
    service = FakeComparisonService()
    app.dependency_overrides[get_comparison_service] = lambda: service
    document_id, left_id, right_id = uuid4(), uuid4(), uuid4()

    response = await client.post(
        "/api/compare",
        json={
            "logical_document_id": str(document_id),
            "left_document_version_id": str(left_id),
            "right_document_version_id": str(right_id),
            "include_unchanged": True,
            "generate_summary": False,
            "debug": True,
        },
    )

    assert response.status_code == 200
    assert service.request is not None
    assert service.request.left_document_version_id == left_id
    assert response.json()["left_version"]["version_label"] == "2025"


@pytest.mark.asyncio
async def test_compare_api_rejects_same_or_missing_versions(client: AsyncClient) -> None:
    version_id = uuid4()
    same = await client.post(
        "/api/compare",
        json={
            "logical_document_id": str(uuid4()),
            "left_document_version_id": str(version_id),
            "right_document_version_id": str(version_id),
        },
    )
    missing = await client.post("/api/compare", json={})

    assert same.status_code == 422
    assert missing.status_code == 422
