from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.agent.errors import AgentTimeoutError
from app.agent.models import AgentRequest, AgentResponse, AgentTiming
from app.api.dependencies import get_agent_service


class FakeAgentService:
    def __init__(self) -> None:
        self.request: AgentRequest | None = None

    async def run(self, request: AgentRequest) -> AgentResponse:
        self.request = request
        return AgentResponse(
            answer="grounded response",
            citations=[],
            timing=AgentTiming(
                planning_provider_ms=1,
                tool_execution_ms=2,
                final_generation_ms=1,
                total_ms=4,
            ),
            token_usage=None,
        )


class TimedOutAgentService:
    async def run(self, request: AgentRequest) -> AgentResponse:
        del request
        raise AgentTimeoutError(0.01)


@pytest.mark.asyncio
async def test_agent_api_is_typed(app: FastAPI, client: AsyncClient) -> None:
    service = FakeAgentService()
    app.dependency_overrides[get_agent_service] = lambda: service
    document_id = uuid4()

    response = await client.post(
        "/api/agent",
        json={
            "message": "What is current policy?",
            "logical_document_id": str(document_id),
            "retrieval_mode": "hybrid",
            "max_tool_steps": 3,
            "debug": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "grounded response"
    assert service.request is not None
    assert service.request.logical_document_id == document_id


@pytest.mark.asyncio
async def test_agent_api_validates_request(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[get_agent_service] = lambda: FakeAgentService()
    missing = await client.post("/api/agent", json={})
    too_many = await client.post("/api/agent", json={"message": "question", "max_tool_steps": 11})

    assert missing.status_code == 422
    assert too_many.status_code == 422


@pytest.mark.asyncio
async def test_agent_api_serializes_timeout_error(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[get_agent_service] = lambda: TimedOutAgentService()

    response = await client.post("/api/agent", json={"message": "question"})

    assert response.status_code == 504
    assert response.json()["error"] == {
        "code": "agent_timeout",
        "message": "agent execution exceeded the configured deadline",
        "details": {"timeout_seconds": 0.01},
        "request_id": response.headers["X-Request-ID"],
    }
