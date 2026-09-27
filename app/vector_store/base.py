from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID


@dataclass(frozen=True)
class VectorPoint:
    point_id: UUID
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchHit:
    point_id: UUID
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    async def ensure_collection(self, dimension: int) -> None: ...

    async def upsert(self, points: list[VectorPoint]) -> None: ...

    async def search_exact_version(
        self, vector: list[float], document_version_id: UUID, top_k: int
    ) -> list[VectorSearchHit]: ...

    async def search_versions(
        self, vector: list[float], document_version_ids: list[UUID], top_k: int
    ) -> list[VectorSearchHit]: ...

    async def delete_version(self, document_version_id: UUID) -> None: ...

    async def close(self) -> None: ...
