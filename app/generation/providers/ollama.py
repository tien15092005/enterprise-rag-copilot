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


class OllamaProvider:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": request.system_prompt},
                        {"role": "user", "content": request.user_prompt},
                    ],
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_output_tokens,
                    },
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            text = str(payload["message"]["content"]).strip()
            prompt_tokens = _optional_int(payload.get("prompt_eval_count"))
            completion_tokens = _optional_int(payload.get("eval_count"))
            total = (
                prompt_tokens + completion_tokens
                if prompt_tokens is not None and completion_tokens is not None
                else None
            )
            return LLMResponse(
                text=text,
                token_usage=LLMTokenUsage(prompt_tokens, completion_tokens, total),
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError(
                "Ollama generation request failed", provider=self.provider_name
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
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *[_ollama_message(message) for message in messages],
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
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_output_tokens,
                    },
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            message = payload["message"]
            prompt_tokens = _optional_int(payload.get("prompt_eval_count"))
            completion_tokens = _optional_int(payload.get("eval_count"))
            usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=(
                    prompt_tokens + completion_tokens
                    if prompt_tokens is not None and completion_tokens is not None
                    else None
                ),
            )
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                function = tool_calls[0]["function"]
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be an object")
                return AgentDecision(
                    tool_call=AgentToolCall(
                        call_id="ollama-tool-call",
                        name=str(function["name"]),
                        arguments=arguments,
                    ),
                    token_usage=usage,
                )
            return AgentDecision(
                final_answer=str(message.get("content") or "").strip(), token_usage=usage
            )
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "Ollama agent request failed", provider=self.provider_name
            ) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _ollama_message(message: AgentMessage) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_name:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": message.tool_name,
                        "arguments": json.loads(message.content),
                    }
                }
            ],
        }
    if message.role == "tool":
        return {"role": "tool", "content": message.content}
    return {"role": message.role, "content": message.content}
