from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Replaceable boundary for query and passage embeddings."""

    @property
    def model_name(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def dimension(self) -> int: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_passages(self, texts: list[str]) -> list[list[float]]: ...
