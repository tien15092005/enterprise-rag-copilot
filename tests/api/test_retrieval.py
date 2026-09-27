from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_dense_retrieval_request_validation(client: AsyncClient) -> None:
    response = await client.post(
        "/api/retrieval/dense",
        json={"query": "", "document_version_id": str(uuid4()), "top_k": 0},
    )

    assert response.status_code == 422


def test_phase_three_routes_are_in_openapi(app: FastAPI) -> None:
    schema = app.openapi()

    assert "/api/document-versions/{document_version_id}/index" in schema["paths"]
    assert "/api/retrieval/dense" in schema["paths"]
    dense_schema = schema["components"]["schemas"]["DenseRetrievalRequest"]
    assert dense_schema["required"] == ["query", "document_version_id"]
    assert dense_schema["properties"]["top_k"]["minimum"] == 1
