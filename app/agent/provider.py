from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.models.query import TokenUsage


@dataclass(frozen=True, slots=True)
class AgentToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentMessage:
    role: str
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentDecision:
    final_answer: str | None = None
    tool_call: AgentToolCall | None = None
    token_usage: TokenUsage | None = None


@runtime_checkable
class AgentProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def decide(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentToolDefinition],
        temperature: float,
        max_output_tokens: int,
    ) -> AgentDecision: ...
