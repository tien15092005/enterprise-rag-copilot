from __future__ import annotations

from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from app.vector_store.base import VectorPoint, VectorSearchHit


class QdrantVectorStore:
    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._client = client or AsyncQdrantClient(
            url=url,
            api_key=api_key,
            timeout=max(1, round(timeout_seconds)),
            check_compatibility=False,
        )

    async def ensure_collection(self, dimension: int) -> None:
        if not await self._client.collection_exists(self.collection_name):
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            return
        info = await self._client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams) or vectors.size != dimension:
            raise ValueError(
                f"Qdrant collection dimension does not match configured dimension {dimension}"
            )

    async def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        await self._client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(point.point_id), vector=point.vector, payload=point.payload
                )
                for point in points
            ],
            wait=True,
        )

    async def search_exact_version(
        self, vector: list[float], document_version_id: UUID, top_k: int
    ) -> list[VectorSearchHit]:
        return await self.search_versions(vector, [document_version_id], top_k)

    async def search_versions(
        self, vector: list[float], document_version_ids: list[UUID], top_k: int
    ) -> list[VectorSearchHit]:
        if not document_version_ids:
            return []
        response = await self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_version_id",
                        match=models.MatchAny(
                            any=[str(version_id) for version_id in document_version_ids]
                        ),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            VectorSearchHit(
                point_id=UUID(str(point.id)),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    async def delete_version(self, document_version_id: UUID) -> None:
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_version_id",
                            match=models.MatchValue(value=str(document_version_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )

    async def close(self) -> None:
        await self._client.close()
