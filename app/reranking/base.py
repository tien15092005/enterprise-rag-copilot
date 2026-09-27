from __future__ import annotations

from typing import Protocol

from app.models.retrieval import HybridRetrievalCandidate, RerankedRetrievalCandidate


class Reranker(Protocol):
    """Replaceable query-passage reranker contract."""

    @property
    def model_name(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    async def rerank(
        self, query: str, candidates: list[HybridRetrievalCandidate]
    ) -> list[RerankedRetrievalCandidate]: ...
