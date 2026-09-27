from __future__ import annotations

from typing import Any

import httpx

from app.generation.errors import LLMProviderError
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage


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

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None
