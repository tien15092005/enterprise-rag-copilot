from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any

from app.models.retrieval import HybridRetrievalCandidate, RerankedRetrievalCandidate
from app.reranking.errors import RerankerError

logger = logging.getLogger(__name__)
ModelLoader = Callable[..., Any]


class CrossEncoderReranker:
    """Lazy local cross-encoder; scores are relevance logits, not probabilities."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None,
        device: str,
        batch_size: int,
        max_length: int,
        cache_folder: str | None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self._model_name = model_name
        self._revision = revision
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._cache_folder = cache_folder
        self._model_loader = model_loader
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def revision(self) -> str | None:
        return self._revision

    async def rerank(
        self, query: str, candidates: list[HybridRetrievalCandidate]
    ) -> list[RerankedRetrievalCandidate]:
        if not candidates:
            return []
        model = await self._get_model()
        pairs = [(query, candidate.text) for candidate in candidates]
        try:
            raw_scores = await asyncio.to_thread(
                model.predict,
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores = self._as_scores(raw_scores)
        except Exception as exc:
            raise RerankerError() from exc
        if len(scores) != len(candidates):
            raise RerankerError("reranker returned an unexpected score count")
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].rrf_rank or 10**9, str(item[0].chunk_id)),
        )
        return [
            RerankedRetrievalCandidate(
                **candidate.model_dump(),
                reranker_score=score,
                reranker_rank=rank,
            )
            for rank, (candidate, score) in enumerate(ordered, start=1)
        ]

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                try:
                    self._model = await asyncio.to_thread(self._load_model)
                except Exception as exc:
                    raise RerankerError("reranker model could not be loaded") from exc
                logger.info(
                    "reranker model loaded",
                    extra={
                        "event": "reranker_model_loaded",
                        "reranker_model": self._model_name,
                        "reranker_revision": self._revision,
                    },
                )
        return self._model

    def _load_model(self) -> Any:
        if self._model_loader is not None:
            return self._model_loader(
                self._model_name,
                revision=self._revision,
                device=self._device,
                max_length=self._max_length,
                cache_folder=self._cache_folder,
            )
        from sentence_transformers import CrossEncoder

        return CrossEncoder(
            self._model_name,
            revision=self._revision,
            device=self._device,
            max_length=self._max_length,
            cache_folder=self._cache_folder,
        )

    @staticmethod
    def _as_scores(values: Sequence[float] | Any) -> list[float]:
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [float(value[0] if isinstance(value, list) else value) for value in values]
