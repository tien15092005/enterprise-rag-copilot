from __future__ import annotations

from typing import Any

import httpx

from app.generation.errors import LLMProviderError
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage


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

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None
