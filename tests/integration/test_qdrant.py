from __future__ import annotations

import os
from uuid import uuid4

import pytest

from app.vector_store import QdrantVectorStore, VectorPoint


@pytest.mark.integration
async def test_live_qdrant_upsert_is_idempotent_and_exact_version_filtered() -> None:
    qdrant_url = os.getenv("TEST_QDRANT_URL")
    if not qdrant_url:
        pytest.skip("TEST_QDRANT_URL is not configured")

    store = QdrantVectorStore(
        url=qdrant_url,
        collection_name=f"phase3_integration_{uuid4().hex}",
    )
    version_a, version_b = uuid4(), uuid4()
    chunk_a, chunk_b = uuid4(), uuid4()
    try:
        await store.ensure_collection(3)
        points = [
            VectorPoint(
                chunk_a,
                [1.0, 0.0, 0.0],
                {"chunk_id": str(chunk_a), "document_version_id": str(version_a)},
            ),
            VectorPoint(
                chunk_b,
                [1.0, 0.0, 0.0],
                {"chunk_id": str(chunk_b), "document_version_id": str(version_b)},
            ),
        ]
        await store.upsert(points)
        await store.upsert(points)

        hits = await store.search_exact_version([1.0, 0.0, 0.0], version_a, 10)

        assert [hit.point_id for hit in hits] == [chunk_a]
        assert all(hit.payload["document_version_id"] == str(version_a) for hit in hits)

        await store.delete_version(version_a)
        assert await store.search_exact_version([1.0, 0.0, 0.0], version_a, 10) == []
        remaining = await store.search_exact_version([1.0, 0.0, 0.0], version_b, 10)
        assert [hit.point_id for hit in remaining] == [chunk_b]
    finally:
        await store.close()
