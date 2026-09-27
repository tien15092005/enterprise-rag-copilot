from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Chunk, DocumentVersion, LogicalDocument
from app.models.versioning import (
    DocumentLanguage,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
)
from app.retrieval.lexical import LexicalRetrievalService


@pytest.mark.integration
async def test_postgres_lexical_english_vietnamese_and_version_filtering() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not configured")

    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            document = LogicalDocument(stable_key=f"fts-{uuid4().hex}", title="FTS")
            english = _version(document, 1, DocumentLanguage.ENGLISH)
            vietnamese = _version(document, 2, DocumentLanguage.VIETNAMESE)
            english_chunk = _chunk(
                english,
                "Employees may work remotely two days each week.",
                DocumentLanguage.ENGLISH,
            )
            vietnamese_chunk = _chunk(
                vietnamese,
                "Nhân viên được hưởng mười hai ngày nghỉ phép có lương mỗi năm.",
                DocumentLanguage.VIETNAMESE,
            )
            session.add_all([document, english, vietnamese, english_chunk, vietnamese_chunk])
            await session.flush()
            for version in (english, vietnamese):
                version.indexing_status = VersionIndexingStatus.READY
                version.vector_index_status = VectorIndexStatus.INDEXED
            await session.flush()

            service = LexicalRetrievalService(session)
            english_result = await service.search_versions(
                "remote work", [english.document_version_id], 5
            )
            vietnamese_result = await service.search_versions(
                "nghỉ phép", [vietnamese.document_version_id], 5
            )
            empty_result = await service.search_versions(
                "cafeteria", [english.document_version_id], 5
            )
            filtered = await service.search_versions("nghỉ phép", [english.document_version_id], 5)

            assert [item.chunk_id for item in english_result.candidates] == [english_chunk.chunk_id]
            assert [item.chunk_id for item in vietnamese_result.candidates] == [
                vietnamese_chunk.chunk_id
            ]
            assert empty_result.candidates == []
            assert filtered.candidates == []
            assert all(
                item.document_version_id == vietnamese.document_version_id
                for item in vietnamese_result.candidates
            )
            await session.rollback()
    finally:
        await engine.dispose()


def _version(
    document: LogicalDocument, ordinal: int, language: DocumentLanguage
) -> DocumentVersion:
    return DocumentVersion(
        logical_document=document,
        version_label=f"v{ordinal}",
        revision_number=1,
        version_ordinal=ordinal,
        language=language,
        source_filename=f"v{ordinal}.txt",
        checksum=uuid4().hex * 2,
        storage_key=f"fts/{uuid4().hex}.txt",
        business_status=VersionBusinessStatus.DRAFT,
        indexing_status=VersionIndexingStatus.PROCESSING,
        vector_index_status=VectorIndexStatus.NOT_INDEXED,
        ingestion_config={},
    )


def _chunk(version: DocumentVersion, text: str, language: DocumentLanguage) -> Chunk:
    return Chunk(
        chunk_id=uuid4(),
        document_version=version,
        ordinal=0,
        text=text,
        page=1,
        section_path=["Policy"],
        language=language,
    )
