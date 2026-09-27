from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from app.core.config import Settings
from app.db.models import Chunk
from app.db.session import Database
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, TextParser
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.ingestion.storage import LocalFilesystemStorage
from app.models.documents import DocumentVersionCreate, LogicalDocumentCreate
from app.models.versioning import DocumentLanguage


async def verify(sample: Path) -> None:
    settings = Settings()
    database = Database(settings)
    storage = LocalFilesystemStorage(
        Path(settings.upload_directory), max_bytes=settings.max_upload_bytes
    )
    parsers = ParserRegistry((PdfParser(), DocxParser(), TextParser(), MarkdownParser()))

    try:
        async with database.session_factory() as session:
            catalog = DocumentCatalogService(session)
            document = await catalog.create(
                LogicalDocumentCreate(
                    stable_key="phase2-verification",
                    title="Phase 2 Verification Document",
                    description="Real-file ingestion verification",
                    owner="Engineering",
                    domain="Knowledge Management",
                )
            )
            ingestion = IngestionService(session, storage, parsers, TokenWindowChunker())
            with sample.open("rb") as source:
                version = await ingestion.ingest(
                    document.logical_document_id,
                    DocumentVersionCreate(
                        version_label="1.0",
                        revision_number=1,
                        version_ordinal=1,
                        language=DocumentLanguage.ENGLISH,
                        activate=True,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                    ),
                    source=source,
                    filename=sample.name,
                )
            chunks = list(
                await session.scalars(
                    select(Chunk)
                    .where(Chunk.document_version_id == version.document_version_id)
                    .order_by(Chunk.ordinal)
                )
            )

            print(
                json.dumps(
                    {
                        "logical_document_id": str(document.logical_document_id),
                        "document_version_id": str(version.document_version_id),
                        "business_status": version.business_status.value,
                        "indexing_status": version.indexing_status.value,
                        "checksum": version.checksum,
                        "storage_key": version.storage_key,
                        "chunk_count": len(chunks),
                        "chunks": [
                            {
                                "ordinal": chunk.ordinal,
                                "page": chunk.page,
                                "section_path": chunk.section_path,
                                "language": chunk.language.value,
                                "text_preview": chunk.text[:100],
                            }
                            for chunk in chunks
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Phase 2 ingestion flow")
    parser.add_argument("sample", type=Path)
    arguments = parser.parse_args()
    asyncio.run(verify(arguments.sample))


if __name__ == "__main__":
    main()
