from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from typing import Any

from app.agent.errors import AgentError, AgentStepLimitError
from app.agent.models import (
    AgentDebug,
    AgentRequest,
    AgentResponse,
    AgentTiming,
    AgentToolResult,
    AgentToolTrace,
)
from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.provider import AgentMessage, AgentProvider
from app.agent.tools import AgentToolRegistry
from app.core.config import Settings
from app.generation.errors import CitationValidationError
from app.models.query import Citation, TokenUsage

logger = logging.getLogger(__name__)
_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


class AgentService:
    def __init__(
        self,
        provider: AgentProvider,
        tools: AgentToolRegistry,
        settings: Settings,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._settings = settings

    async def run(self, request: AgentRequest) -> AgentResponse:
        total_started = perf_counter()
        max_steps = request.max_tool_steps or self._settings.agent_max_tool_steps
        if max_steps > self._settings.agent_max_tool_steps:
            raise AgentError(
                "requested max_tool_steps exceeds the configured ceiling",
                code="agent_step_limit_invalid",
                status_code=422,
                details={"configured_max_tool_steps": self._settings.agent_max_tool_steps},
            )

        messages = [AgentMessage(role="user", content=request.message)]
        traces: list[AgentToolTrace] = []
        citations: dict[str, Citation] = {}
        insufficient_evidence = False
        planning_ms = 0.0
        tool_ms = 0.0
        final_generation_ms = 0.0
        usage = _UsageAccumulator()

        while True:
            provider_started = perf_counter()
            decision = await self._provider.decide(
                system_prompt=AGENT_SYSTEM_PROMPT,
                messages=messages,
                tools=self._tools.definitions(),
                temperature=self._settings.llm_temperature,
                max_output_tokens=self._settings.llm_max_output_tokens,
            )
            provider_ms = (perf_counter() - provider_started) * 1000
            usage.add(decision.token_usage)

            if decision.final_answer is not None and decision.tool_call is None:
                final_generation_ms = provider_ms
                answer_citations = self._validate_final_answer(
                    decision.final_answer,
                    citations,
                    insufficient_evidence=insufficient_evidence,
                )
                total_ms = (perf_counter() - total_started) * 1000
                logger.info(
                    "controlled agent completed",
                    extra={
                        "event": "agent_completed",
                        "tool_steps": len(traces),
                        "tool_names": [trace.tool_name for trace in traces],
                        "planning_provider_latency_ms": round(planning_ms, 3),
                        "tool_execution_latency_ms": round(tool_ms, 3),
                        "final_generation_latency_ms": round(final_generation_ms, 3),
                        "total_latency_ms": round(total_ms, 3),
                    },
                )
                return AgentResponse(
                    answer=decision.final_answer,
                    citations=answer_citations,
                    timing=AgentTiming(
                        planning_provider_ms=planning_ms,
                        tool_execution_ms=tool_ms,
                        final_generation_ms=final_generation_ms,
                        total_ms=total_ms,
                    ),
                    token_usage=usage.response(),
                    debug=(
                        AgentDebug(
                            tool_trace=traces,
                            step_count=len(traces),
                            provider_name=self._provider.provider_name,
                            provider_model=self._provider.model_name,
                        )
                        if request.debug
                        else None
                    ),
                )

            if decision.tool_call is None or decision.final_answer is not None:
                raise AgentError(
                    "agent provider returned an invalid decision",
                    code="agent_invalid_provider_decision",
                    status_code=502,
                )
            planning_ms += provider_ms
            if len(traces) >= max_steps:
                raise AgentStepLimitError(max_steps)

            call = decision.tool_call
            arguments = self._scoped_arguments(call.name, call.arguments, request)
            tool_started = perf_counter()
            result = await self._tools.execute(
                call.name,
                arguments,
                scope_logical_document_id=request.logical_document_id,
            )
            latency_ms = (perf_counter() - tool_started) * 1000
            tool_ms += latency_ms
            usage.add_response(result.token_usage)
            result = self._rekey_result(result, citations)
            insufficient_evidence = insufficient_evidence or result.insufficient_evidence
            traces.append(
                AgentToolTrace(
                    step=len(traces) + 1,
                    tool_name=call.name,
                    arguments=_sanitize(arguments),
                    result_type=result.result_type,
                    latency_ms=latency_ms,
                )
            )
            messages.extend(
                [
                    AgentMessage(
                        role="assistant",
                        content=json.dumps(arguments, default=str, ensure_ascii=False),
                        tool_call_id=call.call_id,
                        tool_name=call.name,
                    ),
                    AgentMessage(
                        role="tool",
                        content=result.model_dump_json(exclude={"token_usage"}),
                        tool_call_id=call.call_id,
                        tool_name=call.name,
                    ),
                ]
            )

    @staticmethod
    def _scoped_arguments(
        tool_name: str, arguments: dict[str, Any], request: AgentRequest
    ) -> dict[str, Any]:
        scoped = dict(arguments)
        if request.logical_document_id is not None and tool_name != "get_document_version":
            scoped.setdefault("logical_document_id", str(request.logical_document_id))
        if tool_name == "search_knowledge":
            scoped["retrieval_mode"] = request.retrieval_mode.value
        return scoped

    @staticmethod
    def _rekey_result(result: AgentToolResult, existing: dict[str, Citation]) -> AgentToolResult:
        key_map: dict[str, str] = {}
        normalized_citations: list[Citation] = []
        for citation in result.citations:
            new_key = f"S{len(existing) + 1}"
            key_map[citation.citation_key] = new_key
            normalized = citation.model_copy(
                update={"citation_key": new_key, "rank": len(existing) + 1}
            )
            existing[new_key] = normalized
            normalized_citations.append(normalized)
        return result.model_copy(
            update={
                "payload": _replace_citation_keys(result.payload, key_map),
                "citations": normalized_citations,
            }
        )

    @staticmethod
    def _validate_final_answer(
        answer: str,
        supplied: dict[str, Citation],
        *,
        insufficient_evidence: bool,
    ) -> list[Citation]:
        referenced = list(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
        unknown = [key for key in referenced if key not in supplied]
        if unknown:
            raise CitationValidationError(
                "agent referenced citation keys outside tool evidence", unknown_keys=unknown
            )
        if supplied and not referenced and not insufficient_evidence:
            raise CitationValidationError("agent answer did not include a supplied citation key")
        return [supplied[key] for key in referenced]


class _UsageAccumulator:
    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0
        self.total = 0
        self.seen = False

    def add(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.seen = True
        self.prompt += usage.prompt_tokens or 0
        self.completion += usage.completion_tokens or 0
        self.total += usage.total_tokens or 0

    def add_response(self, usage: TokenUsage | None) -> None:
        if usage is None:
            return
        self.seen = True
        self.prompt += usage.prompt_tokens or 0
        self.completion += usage.completion_tokens or 0
        self.total += usage.total_tokens or 0

    def response(self) -> TokenUsage | None:
        if not self.seen:
            return None
        return TokenUsage(
            prompt_tokens=self.prompt,
            completion_tokens=self.completion,
            total_tokens=self.total,
        )


def _replace_citation_keys(value: Any, key_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        if value in key_map:
            return key_map[value]
        for old, new in key_map.items():
            value = value.replace(f"[{old}]", f"[{new}]")
        return value
    if isinstance(value, list):
        return [_replace_citation_keys(item, key_map) for item in value]
    if isinstance(value, dict):
        return {key: _replace_citation_keys(item, key_map) for key, item in value.items()}
    return value


def _sanitize(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (f"[redacted:{len(str(value))} chars]" if key in {"question", "message"} else value)
        for key, value in arguments.items()
    }
