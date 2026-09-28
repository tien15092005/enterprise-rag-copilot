from __future__ import annotations

import json
from typing import Any

import httpx

from app.agent.provider import (
    AgentDecision,
    AgentMessage,
    AgentToolCall,
    AgentToolDefinition,
)
from app.generation.errors import LLMProviderError
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage
from app.models.query import TokenUsage


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": request.system_prompt},
                        {"role": "user", "content": request.user_prompt},
                    ],
                    "temperature": request.temperature,
                    "max_tokens": request.max_output_tokens,
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            text = str(payload["choices"][0]["message"]["content"]).strip()
            usage = payload.get("usage") or {}
            return LLMResponse(
                text=text,
                token_usage=LLMTokenUsage(
                    prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                    completion_tokens=_optional_int(usage.get("completion_tokens")),
                    total_tokens=_optional_int(usage.get("total_tokens")),
                ),
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError(
                "OpenAI-compatible generation request failed",
                provider=self.provider_name,
            ) from exc

    async def decide(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentToolDefinition],
        temperature: float,
        max_output_tokens: int,
    ) -> AgentDecision:
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *[_openai_message(message) for message in messages],
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters,
                            },
                        }
                        for tool in tools
                    ],
                    "tool_choice": "auto",
                    "temperature": temperature,
                    "max_tokens": max_output_tokens,
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            message = payload["choices"][0]["message"]
            usage = payload.get("usage") or {}
            token_usage = TokenUsage(
                prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                completion_tokens=_optional_int(usage.get("completion_tokens")),
                total_tokens=_optional_int(usage.get("total_tokens")),
            )
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                call = tool_calls[0]
                function = call["function"]
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be an object")
                return AgentDecision(
                    tool_call=AgentToolCall(
                        call_id=str(call.get("id") or "tool-call"),
                        name=str(function["name"]),
                        arguments=arguments,
                    ),
                    token_usage=token_usage,
                )
            return AgentDecision(
                final_answer=str(message.get("content") or "").strip(),
                token_usage=token_usage,
            )
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "OpenAI-compatible agent request failed", provider=self.provider_name
            ) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _openai_message(message: AgentMessage) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_name and message.tool_call_id:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": message.tool_call_id,
                    "type": "function",
                    "function": {"name": message.tool_name, "arguments": message.content},
                }
            ],
        }
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    return {"role": message.role, "content": message.content}
