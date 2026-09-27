from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

ModelLoader = Callable[..., Any]


class SentenceTransformerEmbeddingProvider:
    """Lazy local E5 provider; model loading never occurs at module import time."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None,
        dimension: int,
        batch_size: int,
        device: str,
        normalize: bool = True,
        cache_folder: str | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._model_name = model_name
        self._revision = revision
        self._dimension = dimension
        self._batch_size = batch_size
        self._device = device
        self._normalize = normalize
        self._cache_folder = cache_folder
        self._model_loader = model_loader
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        self.load_latency_ms: float | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._encode([f"query: {text}"])
        return vectors[0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._encode([f"passage: {text}" for text in texts])

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                started = perf_counter()
                loaded_model = await asyncio.to_thread(self._load_model)
                self.load_latency_ms = (perf_counter() - started) * 1000
                dimension_method = getattr(
                    loaded_model,
                    "get_embedding_dimension",
                    loaded_model.get_sentence_embedding_dimension,
                )
                actual_dimension = int(dimension_method())
                if actual_dimension != self._dimension:
                    raise ValueError(
                        f"embedding dimension mismatch: configured {self._dimension}, "
                        f"model reports {actual_dimension}"
                    )
                self._model = loaded_model
                logger.info(
                    "embedding model loaded",
                    extra={
                        "event": "embedding_model_loaded",
                        "embedding_model": self._model_name,
                        "embedding_revision": self._revision,
                        "embedding_dimension": actual_dimension,
                        "load_latency_ms": round(self.load_latency_ms, 3),
                    },
                )
        return self._model

    def _load_model(self) -> Any:
        if self._model_loader is not None:
            return self._model_loader(
                self._model_name,
                revision=self._revision,
                device=self._device,
                cache_folder=self._cache_folder,
            )
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self._model_name,
            revision=self._revision,
            device=self._device,
            cache_folder=self._cache_folder,
        )

    async def _encode(self, inputs: list[str]) -> list[list[float]]:
        model = await self._get_model()
        started = perf_counter()
        raw = await asyncio.to_thread(
            model.encode,
            inputs,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = [self._as_float_vector(vector) for vector in raw]
        for vector in vectors:
            if len(vector) != self._dimension:
                raise ValueError(
                    f"embedding dimension mismatch: expected {self._dimension}, got {len(vector)}"
                )
        logger.info(
            "embedding completed",
            extra={
                "event": "embedding_completed",
                "embedding_model": self._model_name,
                "input_count": len(inputs),
                "embedding_latency_ms": round((perf_counter() - started) * 1000, 3),
            },
        )
        return vectors

    def _as_float_vector(self, vector: Sequence[float]) -> list[float]:
        result = [float(value) for value in vector]
        if not self._normalize:
            return result
        norm = math.sqrt(sum(value * value for value in result))
        if norm == 0:
            raise ValueError("embedding provider returned a zero vector")
        return [value / norm for value in result]
