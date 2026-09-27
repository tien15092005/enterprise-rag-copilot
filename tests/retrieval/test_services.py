from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, DocumentVersion, LogicalDocument
from app.db.session import Database
from app.ingestion.errors import DocumentNotFoundError
from app.models.versioning import (
    DocumentLanguage,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
)
from app.retrieval.errors import (
    RetrievalProvenanceError,
    VectorIndexingError,
    VersionNotIndexedError,
)
from app.retrieval.service import DenseRetrievalService, VectorIndexingService
from app.vector_store.base import VectorPoint, VectorSearchHit


class FakeEmbeddings:
    model_name = "fake-multilingual-e5"
    revision = "test-revision"
    dimension = 3

    async def embed_query(self, text: str) -> list[float]:
        assert text
        return [1.0, 0.0, 0.0]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(index), 0.0] for index, _text in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.points: dict[UUID, VectorPoint] = {}
        self.upsert_calls = 0
        self.fail_upsert = False

    async def ensure_collection(self, dimension: int) -> None:
        assert dimension == 3

    async def upsert(self, points: list[VectorPoint]) -> None:
        if self.fail_upsert:
            raise RuntimeError("qdrant unavailable")
        self.upsert_calls += 1
        self.points.update({point.point_id: point for point in points})

    async def search_exact_version(
        self, _vector: list[float], document_version_id: UUID, top_k: int
    ) -> list[VectorSearchHit]:
        return await self.search_versions(_vector, [document_version_id], top_k)

    async def search_versions(
        self, _vector: list[float], document_version_ids: list[UUID], top_k: int
    ) -> list[VectorSearchHit]:
        matches = [
            point
            for point in self.points.values()
            if point.payload["document_version_id"]
            in {str(version_id) for version_id in document_version_ids}
        ]
        return [
            VectorSearchHit(point.point_id, 1.0 - index * 0.1, point.payload)
            for index, point in enumerate(reversed(matches[:top_k]))
        ]

    async def delete_version(self, document_version_id: UUID) -> None:
        self.points = {
            point_id: point
            for point_id, point in self.points.items()
            if point.payload["document_version_id"] != str(document_version_id)
        }

    async def close(self) -> None:
        return None


class LeakyVectorStore(FakeVectorStore):
    def __init__(self, leaked_point_id: UUID) -> None:
        super().__init__()
        self.leaked_point_id = leaked_point_id

    async def search_versions(
        self, _vector: list[float], _document_version_ids: list[UUID], _top_k: int
    ) -> list[VectorSearchHit]:
        point = self.points[self.leaked_point_id]
        return [VectorSearchHit(point.point_id, 0.99, point.payload)]


async def add_ready_version(
    session: AsyncSession, *, stable_key: str, ordinal: int = 1
) -> tuple[DocumentVersion, list[Chunk]]:
    document = LogicalDocument(stable_key=stable_key, title=stable_key)
    version = DocumentVersion(
        logical_document=document,
        version_label=f"v{ordinal}",
        revision_number=1,
        version_ordinal=ordinal,
        language=DocumentLanguage.ENGLISH,
        source_filename=f"{stable_key}.txt",
        checksum=(str(ordinal) * 64)[:64],
        storage_key=f"{stable_key}/source.txt",
        business_status=VersionBusinessStatus.DRAFT,
        indexing_status=VersionIndexingStatus.READY,
        ingestion_config={"chunk_size": 100, "chunk_overlap": 10},
    )
    chunks = [
        Chunk(
            chunk_id=uuid4(),
            document_version=version,
            ordinal=index,
            text=f"{stable_key} content {index}",
            page=index + 1,
            section_path=["Policy"],
            language=DocumentLanguage.ENGLISH,
        )
        for index in range(2)
    ]
    session.add_all([document, version, *chunks])
    await session.commit()
    return version, chunks


@pytest.mark.asyncio
async def test_indexing_is_idempotent_and_preserves_metadata(app: Any) -> None:
    database: Database = app.state.database
    store = FakeVectorStore()
    async with database.session_factory() as session:
        version, chunks = await add_ready_version(session, stable_key="handbook")
        first_chunk_id = chunks[0].chunk_id
        version_id = version.document_version_id
        logical_document_id = version.logical_document_id
        service = VectorIndexingService(session, FakeEmbeddings(), store)
        first = await service.index_version(version.document_version_id)
        second = await service.index_version(version.document_version_id)
        await session.refresh(version)

    assert first.chunks_indexed == second.chunks_indexed == 2
    assert store.upsert_calls == 2
    assert len(store.points) == 2
    assert version.vector_index_status is VectorIndexStatus.INDEXED
    assert version.embedding_model == "fake-multilingual-e5"
    payload = store.points[first_chunk_id].payload
    assert payload == {
        "chunk_id": str(first_chunk_id),
        "document_version_id": str(version_id),
        "logical_document_id": str(logical_document_id),
        "language": "en",
        "page": 1,
        "section_path": ["Policy"],
        "ordinal": 0,
    }


@pytest.mark.asyncio
async def test_exact_version_retrieval_order_and_no_version_leakage(app: Any) -> None:
    database: Database = app.state.database
    store = FakeVectorStore()
    async with database.session_factory() as session:
        first, first_chunks = await add_ready_version(session, stable_key="policy-a")
        second, second_chunks = await add_ready_version(session, stable_key="policy-b")
        indexer = VectorIndexingService(session, FakeEmbeddings(), store)
        await indexer.index_version(first.document_version_id)
        await indexer.index_version(second.document_version_id)
        result = await DenseRetrievalService(session, FakeEmbeddings(), store).search(
            "benefits", first.document_version_id, 5
        )

    assert [candidate.chunk_id for candidate in result.candidates] == [
        first_chunks[1].chunk_id,
        first_chunks[0].chunk_id,
    ]
    assert [candidate.score for candidate in result.candidates] == [1.0, 0.9]
    assert all(
        candidate.document_version_id == first.document_version_id
        for candidate in result.candidates
    )
    assert not {candidate.chunk_id for candidate in result.candidates}.intersection(
        {chunk.chunk_id for chunk in second_chunks}
    )


@pytest.mark.asyncio
async def test_failed_indexing_records_failed_state(app: Any) -> None:
    database: Database = app.state.database
    store = FakeVectorStore()
    store.fail_upsert = True
    async with database.session_factory() as session:
        version, _chunks = await add_ready_version(session, stable_key="failure")
        with pytest.raises(VectorIndexingError):
            await VectorIndexingService(session, FakeEmbeddings(), store).index_version(
                version.document_version_id
            )
        await session.refresh(version)
        assert version.vector_index_status is VectorIndexStatus.FAILED
        assert version.vector_index_error == "qdrant unavailable"


@pytest.mark.asyncio
async def test_unindexed_version_is_rejected(app: Any) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        version, _chunks = await add_ready_version(session, stable_key="unindexed")
        with pytest.raises(VersionNotIndexedError):
            await DenseRetrievalService(session, FakeEmbeddings(), FakeVectorStore()).search(
                "question", version.document_version_id, 5
            )


@pytest.mark.asyncio
async def test_unknown_version_is_rejected(app: Any) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        with pytest.raises(DocumentNotFoundError):
            await DenseRetrievalService(session, FakeEmbeddings(), FakeVectorStore()).search(
                "question", uuid4(), 5
            )


@pytest.mark.asyncio
async def test_multi_version_retrieval_rejects_store_leakage(app: Any) -> None:
    database: Database = app.state.database
    async with database.session_factory() as session:
        first, _first_chunks = await add_ready_version(session, stable_key="leak-a")
        second, second_chunks = await add_ready_version(session, stable_key="leak-b")
        leaked_chunk_id = second_chunks[0].chunk_id
        first_version_id = first.document_version_id
        second_version_id = second.document_version_id
        store = LeakyVectorStore(leaked_chunk_id)
        indexer = VectorIndexingService(session, FakeEmbeddings(), store)
        await indexer.index_version(first_version_id)
        await indexer.index_version(second_version_id)

        with pytest.raises(RetrievalProvenanceError, match="outside the resolved version scope"):
            await DenseRetrievalService(session, FakeEmbeddings(), store).search_versions(
                "question", [first_version_id], 5
            )
