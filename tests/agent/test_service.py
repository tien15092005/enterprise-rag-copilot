from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.agent.errors import AgentStepLimitError, AgentTimeoutError
from app.agent.models import AgentRequest, AgentToolResult
from app.agent.provider import (
    AgentDecision,
    AgentMessage,
    AgentToolCall,
    AgentToolDefinition,
)
from app.agent.service import AgentService
from app.core.config import Settings
from app.generation.errors import CitationValidationError
from app.models.query import Citation, TokenUsage
from app.models.retrieval import RetrievalMode
from app.models.versioning import DocumentLanguage


class ScriptedProvider:
    provider_name = "scripted"
    model_name = "deterministic-agent-v1"

    def __init__(self, decisions: list[AgentDecision]) -> None:
        self.decisions = decisions
        self.requests: list[list[AgentMessage]] = []

    async def decide(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentToolDefinition],
        temperature: float,
        max_output_tokens: int,
    ) -> AgentDecision:
        assert "cannot execute shell commands" in system_prompt
        assert tools
        self.requests.append(messages)
        return self.decisions.pop(0)


class RecordingRegistry:
    def __init__(self, results: dict[str, AgentToolResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, Any], UUID | None]] = []

    def definitions(self) -> list[AgentToolDefinition]:
        return [AgentToolDefinition("search_knowledge", "search", {"type": "object"})]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        scope_logical_document_id: UUID | None,
    ) -> AgentToolResult:
        self.calls.append((name, arguments, scope_logical_document_id))
        return self.results[name]


def citation(key: str = "S1") -> Citation:
    return Citation(
        citation_key=key,
        logical_document_id=uuid4(),
        document_version_id=uuid4(),
        version_label="2026",
        chunk_id=uuid4(),
        source_filename="policy.md",
        page=None,
        section_path=["Remote Work"],
        excerpt="Employees may work remotely up to two days per week.",
        rank=1,
        retrieval_score=0.9,
        language=DocumentLanguage.ENGLISH,
        effective_from=None,
        effective_to=None,
    )


def call(name: str, arguments: dict[str, Any], call_id: str = "call-1") -> AgentDecision:
    return AgentDecision(tool_call=AgentToolCall(call_id, name, arguments))


def final(text: str) -> AgentDecision:
    return AgentDecision(
        final_answer=text,
        token_usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selector", "expected_kind"),
    [
        ({"kind": "current"}, "current"),
        ({"kind": "exact", "document_version_id": str(uuid4())}, "exact"),
        ({"kind": "as_of", "as_of": "2026-01-01T00:00:00Z"}, "as_of"),
    ],
)
async def test_search_routing_preserves_version_selector_and_retrieval_mode(
    settings: Settings, selector: dict[str, str], expected_kind: str
) -> None:
    document_id = uuid4()
    result = AgentToolResult(
        result_type="knowledge_answer",
        payload={"answer": "Two days [S1]."},
        citations=[citation()],
    )
    registry = RecordingRegistry({"search_knowledge": result})
    provider = ScriptedProvider(
        [
            call(
                "search_knowledge",
                {
                    "question": "What is the policy?",
                    "logical_document_id": str(document_id),
                    "version_selector": selector,
                },
            ),
            final("Employees may work remotely two days per week [S1]."),
        ]
    )

    response = await AgentService(provider, registry, settings).run(  # type: ignore[arg-type]
        AgentRequest(
            message="What is the policy?",
            logical_document_id=document_id,
            retrieval_mode=RetrievalMode.HYBRID,
            debug=True,
        )
    )

    assert registry.calls[0][1]["version_selector"]["kind"] == expected_kind
    assert registry.calls[0][1]["retrieval_mode"] == "hybrid"
    assert response.citations[0].citation_key == "S1"
    assert response.debug is not None
    assert response.debug.tool_trace[0].arguments["question"].startswith("[redacted:")
    assert response.token_usage is not None and response.token_usage.total_tokens == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "answer"),
    [
        ("compare_versions", "Remote work increased [S1] [S2]."),
        ("list_document_versions", "Versions are 2025 and 2026."),
    ],
)
async def test_comparison_and_version_listing_routes(
    settings: Settings, tool_name: str, answer: str
) -> None:
    citations = [citation("S1"), citation("S2")] if tool_name == "compare_versions" else []
    registry = RecordingRegistry(
        {tool_name: AgentToolResult(result_type=tool_name, payload={}, citations=citations)}
    )
    provider = ScriptedProvider([call(tool_name, {}), final(answer)])

    response = await AgentService(provider, registry, settings).run(  # type: ignore[arg-type]
        AgentRequest(message="request", logical_document_id=uuid4())
    )

    assert registry.calls[0][0] == tool_name
    assert response.answer == answer
    assert len(response.citations) == len(citations)


@pytest.mark.asyncio
async def test_vietnamese_insufficient_evidence_is_returned_without_citations(
    settings: Settings,
) -> None:
    registry = RecordingRegistry(
        {
            "search_knowledge": AgentToolResult(
                result_type="knowledge_answer",
                payload={"answer": "Không có đủ bằng chứng."},
                insufficient_evidence=True,
            )
        }
    )
    provider = ScriptedProvider(
        [
            call("search_knowledge", {"question": "Quy định này là gì?"}),
            final("Không có đủ bằng chứng trong tài liệu đã chọn."),
        ]
    )

    response = await AgentService(provider, registry, settings).run(  # type: ignore[arg-type]
        AgentRequest(message="Quy định này là gì?", logical_document_id=uuid4())
    )

    assert response.citations == []
    assert "Không có đủ bằng chứng" in response.answer


@pytest.mark.asyncio
async def test_step_limit_rejects_another_tool_call(settings: Settings) -> None:
    registry = RecordingRegistry(
        {"search_knowledge": AgentToolResult(result_type="knowledge_answer", payload={})}
    )
    provider = ScriptedProvider(
        [call("search_knowledge", {}), call("search_knowledge", {}, "call-2")]
    )
    limited = settings.model_copy(update={"agent_max_tool_steps": 1})

    with pytest.raises(AgentStepLimitError):
        await AgentService(provider, registry, limited).run(  # type: ignore[arg-type]
            AgentRequest(message="loop", logical_document_id=uuid4())
        )
    assert len(registry.calls) == 1


@pytest.mark.asyncio
async def test_unknown_final_citation_is_rejected(settings: Settings) -> None:
    registry = RecordingRegistry(
        {
            "search_knowledge": AgentToolResult(
                result_type="knowledge_answer", payload={}, citations=[citation()]
            )
        }
    )
    provider = ScriptedProvider([call("search_knowledge", {}), final("Unsupported answer [S99].")])

    with pytest.raises(CitationValidationError):
        await AgentService(provider, registry, settings).run(  # type: ignore[arg-type]
            AgentRequest(message="question", logical_document_id=uuid4())
        )


@pytest.mark.asyncio
async def test_whole_agent_deadline_is_structured(settings: Settings) -> None:
    class SlowProvider(ScriptedProvider):
        async def decide(
            self,
            *,
            system_prompt: str,
            messages: list[AgentMessage],
            tools: list[AgentToolDefinition],
            temperature: float,
            max_output_tokens: int,
        ) -> AgentDecision:
            await asyncio.sleep(0.05)
            return final("late")

    limited = settings.model_copy(update={"agent_timeout_seconds": 0.01})
    provider = SlowProvider([])
    registry = RecordingRegistry({})

    with pytest.raises(AgentTimeoutError) as exc_info:
        await AgentService(provider, registry, limited).run(  # type: ignore[arg-type]
            AgentRequest(message="timeout")
        )

    assert exc_info.value.code == "agent_timeout"
    assert exc_info.value.status_code == 504
