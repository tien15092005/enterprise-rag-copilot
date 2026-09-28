from __future__ import annotations

from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.agent.provider import AgentDecision, AgentMessage, AgentToolCall, AgentToolDefinition
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage
from app.models.query import TokenUsage
from app.vector_store.base import VectorPoint, VectorSearchHit


class E2EEmbeddings:
    model_name = "fake-e2e-multilingual"
    revision = "test"
    dimension = 2

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "remote" in text.casefold() else [0.0, 1.0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "remote" in text.casefold() else [0.0, 1.0] for text in texts]


class E2EVectorStore:
    def __init__(self) -> None:
        self.points: dict[UUID, VectorPoint] = {}

    async def ensure_collection(self, dimension: int) -> None:
        assert dimension == 2

    async def upsert(self, points: list[VectorPoint]) -> None:
        self.points.update({point.point_id: point for point in points})

    async def search_exact_version(
        self, vector: list[float], document_version_id: UUID, top_k: int
    ) -> list[VectorSearchHit]:
        return await self.search_versions(vector, [document_version_id], top_k)

    async def search_versions(
        self, vector: list[float], document_version_ids: list[UUID], top_k: int
    ) -> list[VectorSearchHit]:
        allowed = {str(item) for item in document_version_ids}
        matches = [
            point
            for point in self.points.values()
            if point.payload["document_version_id"] in allowed
        ]
        ranked = sorted(
            matches,
            key=lambda point: -sum(a * b for a, b in zip(vector, point.vector, strict=True)),
        )
        return [
            VectorSearchHit(
                point.point_id,
                sum(a * b for a, b in zip(vector, point.vector, strict=True)),
                point.payload,
            )
            for point in ranked[:top_k]
        ]

    async def delete_version(self, document_version_id: UUID) -> None:
        self.points = {
            key: point
            for key, point in self.points.items()
            if point.payload["document_version_id"] != str(document_version_id)
        }

    async def close(self) -> None:
        return None


class E2EProvider:
    provider_name = "fake-e2e"
    model_name = "fake-e2e-v1"

    def __init__(self) -> None:
        self.decisions: list[AgentDecision] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        answer = (
            "Employees may work remotely up to one day per week [S1]."
            if "one day" in request.user_prompt
            else "Employees may work remotely up to two days per week [S1]."
        )
        return LLMResponse(answer, LLMTokenUsage(50, 10, 60))

    async def decide(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentToolDefinition],
        temperature: float,
        max_output_tokens: int,
    ) -> AgentDecision:
        return self.decisions.pop(0)

    async def close(self) -> None:
        return None


async def upload(
    client: AsyncClient,
    document_id: str,
    *,
    label: str,
    ordinal: int,
    activate: bool,
    text: str,
) -> str:
    response = await client.post(
        f"/api/documents/{document_id}/versions",
        files={"file": (f"policy-{label}.md", text.encode(), "text/markdown")},
        data={
            "version_label": label,
            "version_ordinal": str(ordinal),
            "revision_number": "1",
            "language": "en",
            "activate": str(activate).lower(),
            "chunk_size": "100",
            "chunk_overlap": "10",
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["document_version"]["document_version_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_versioned_workflow(app: FastAPI, client: AsyncClient) -> None:
    app.state.embedding_provider = E2EEmbeddings()
    app.state.vector_store = E2EVectorStore()
    provider = E2EProvider()
    app.state.llm_provider = provider

    document = await client.post(
        "/api/documents",
        json={"stable_key": "e2e-remote-policy", "title": "Remote Policy"},
    )
    assert document.status_code == 201
    document_id = str(document.json()["logical_document_id"])

    version_2025 = await upload(
        client,
        document_id,
        label="2025",
        ordinal=1,
        activate=False,
        text="# Policy 2025\n\n## Remote Work\n\nEmployees may work remotely one day per week.",
    )
    version_2026 = await upload(
        client,
        document_id,
        label="2026",
        ordinal=2,
        activate=True,
        text="# Policy 2026\n\n## Remote Work\n\nEmployees may work remotely two days per week.",
    )

    for version_id in (version_2025, version_2026):
        indexed = await client.post(f"/api/document-versions/{version_id}/index")
        assert indexed.status_code == 200

    current = await client.post(
        "/api/query",
        json={
            "question": "What is the current remote-work policy?",
            "logical_document_id": document_id,
            "version_selector": {"kind": "current"},
        },
    )
    exact = await client.post(
        "/api/query",
        json={
            "question": "What was the remote-work policy?",
            "version_selector": {
                "kind": "exact",
                "document_version_id": version_2025,
            },
        },
    )
    retrieval_only = await client.post(
        "/api/query",
        json={
            "question": "remote work",
            "version_selector": {
                "kind": "exact",
                "document_version_id": version_2026,
            },
            "retrieval_only": True,
        },
    )
    comparison = await client.post(
        "/api/compare",
        json={
            "logical_document_id": document_id,
            "left_document_version_id": version_2025,
            "right_document_version_id": version_2026,
            "include_unchanged": True,
            "generate_summary": False,
        },
    )

    provider.decisions = [
        AgentDecision(
            tool_call=AgentToolCall(
                "call-1",
                "search_knowledge",
                {
                    "question": "What is the current remote-work policy?",
                    "version_selector": {"kind": "current"},
                },
            )
        ),
        AgentDecision(
            final_answer="Employees may work remotely up to two days per week [S1].",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
    ]
    agent = await client.post(
        "/api/agent",
        json={
            "message": "What is the current remote-work policy?",
            "logical_document_id": document_id,
        },
    )

    assert current.status_code == exact.status_code == retrieval_only.status_code == 200
    assert comparison.status_code == agent.status_code == 200
    assert current.json()["citations"][0]["document_version_id"] == version_2026
    assert exact.json()["citations"][0]["document_version_id"] == version_2025
    assert retrieval_only.json()["answer"] is None
    assert retrieval_only.json()["timing"]["generation_ms"] == 0
    assert comparison.json()["counts"]["modified"] == 1
    assert {item["citation"]["document_version_id"] for item in comparison.json()["citations"]} == {
        version_2025,
        version_2026,
    }
    assert agent.json()["citations"][0]["document_version_id"] == version_2026
    assert all(item["document_version_id"] == version_2025 for item in exact.json()["citations"])
