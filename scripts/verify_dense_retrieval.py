from __future__ import annotations

import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from app.core.config import Settings
from app.db.models import DocumentVersion
from app.db.session import Database
from app.embeddings import SentenceTransformerEmbeddingProvider
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, TextParser
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.ingestion.storage import LocalFilesystemStorage
from app.models.documents import DocumentVersionCreate, LogicalDocumentCreate
from app.models.versioning import DocumentLanguage
from app.retrieval.service import DenseRetrievalService, VectorIndexingService
from app.vector_store import QdrantVectorStore


async def ingest_file(
    catalog: DocumentCatalogService,
    ingestion: IngestionService,
    sample: Path,
    *,
    stable_key: str,
    language: DocumentLanguage,
) -> tuple[str, DocumentVersion]:
    document = await catalog.create(LogicalDocumentCreate(stable_key=stable_key, title=stable_key))
    with sample.open("rb") as source:
        version = await ingestion.ingest(
            document.logical_document_id,
            DocumentVersionCreate(
                version_label="1.0",
                version_ordinal=1,
                language=language,
                chunk_size=35,
                chunk_overlap=5,
            ),
            source=source,
            filename=sample.name,
        )
    return stable_key, version


async def run() -> None:
    settings = Settings()
    database = Database(settings)
    embeddings = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model_name,
        revision=settings.embedding_model_revision,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        normalize=settings.embedding_normalize,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    storage = LocalFilesystemStorage(
        Path(settings.upload_directory), max_bytes=settings.max_upload_bytes
    )
    parsers = ParserRegistry((PdfParser(), DocxParser(), TextParser(), MarkdownParser()))
    suffix = uuid4().hex[:8]
    results: dict[str, object] = {}

    try:
        async with database.session_factory() as session:
            catalog = DocumentCatalogService(session)
            ingestion = IngestionService(session, storage, parsers, TokenWindowChunker())
            _, english = await ingest_file(
                catalog,
                ingestion,
                Path("data/samples/employee_handbook_en.md"),
                stable_key=f"phase3-english-{suffix}",
                language=DocumentLanguage.ENGLISH,
            )
            _, vietnamese = await ingest_file(
                catalog,
                ingestion,
                Path("data/samples/hr_policy_vi.txt"),
                stable_key=f"phase3-vietnamese-{suffix}",
                language=DocumentLanguage.VIETNAMESE,
            )

            versioned_document = await catalog.create(
                LogicalDocumentCreate(
                    stable_key=f"phase3-leakage-{suffix}", title="Leakage verification"
                )
            )
            version_ids = []
            for ordinal, content in (
                (1, b"The travel allowance is 100 dollars per day."),
                (2, b"The travel allowance is 175 dollars per day."),
            ):
                version = await ingestion.ingest(
                    versioned_document.logical_document_id,
                    DocumentVersionCreate(
                        version_label=f"v{ordinal}",
                        version_ordinal=ordinal,
                        language=DocumentLanguage.ENGLISH,
                        chunk_size=30,
                        chunk_overlap=5,
                    ),
                    source=BytesIO(content),
                    filename=f"travel-policy-v{ordinal}.txt",
                )
                version_ids.append(version.document_version_id)

            indexer = VectorIndexingService(session, embeddings, vector_store)
            index_results = []
            for version_id in [
                english.document_version_id,
                vietnamese.document_version_id,
                *version_ids,
            ]:
                index_results.append(
                    (await indexer.index_version(version_id)).model_dump(mode="json")
                )

            retriever = DenseRetrievalService(session, embeddings, vector_store)
            english_result = await retriever.search(
                "How many days per week may employees work remotely?",
                english.document_version_id,
                3,
            )
            vietnamese_result = await retriever.search(
                "Nhân viên được nghỉ phép có lương bao nhiêu ngày mỗi năm?",
                vietnamese.document_version_id,
                3,
            )
            first_version_result = await retriever.search(
                "What is the daily travel allowance?", version_ids[0], 3
            )
            second_version_result = await retriever.search(
                "What is the daily travel allowance?", version_ids[1], 3
            )

            results = {
                "embedding": {
                    "model": embeddings.model_name,
                    "revision": embeddings.revision,
                    "dimension": embeddings.dimension,
                    "model_load_latency_ms": embeddings.load_latency_ms,
                },
                "indexing": index_results,
                "english": english_result.model_dump(mode="json"),
                "vietnamese": vietnamese_result.model_dump(mode="json"),
                "version_leakage": {
                    "requested_version_1": str(version_ids[0]),
                    "returned_version_1_ids": [
                        str(candidate.document_version_id)
                        for candidate in first_version_result.candidates
                    ],
                    "version_1_text": [
                        candidate.text for candidate in first_version_result.candidates
                    ],
                    "requested_version_2": str(version_ids[1]),
                    "returned_version_2_ids": [
                        str(candidate.document_version_id)
                        for candidate in second_version_result.candidates
                    ],
                    "version_2_text": [
                        candidate.text for candidate in second_version_result.candidates
                    ],
                    "passed": all(
                        candidate.document_version_id == version_ids[0]
                        for candidate in first_version_result.candidates
                    )
                    and all(
                        candidate.document_version_id == version_ids[1]
                        for candidate in second_version_result.candidates
                    ),
                },
            }
    finally:
        await vector_store.close()
        await database.close()

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8")
    asyncio.run(run())
