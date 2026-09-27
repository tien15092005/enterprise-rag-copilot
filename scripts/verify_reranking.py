from __future__ import annotations

import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from app.core.config import Settings
from app.db.session import Database
from app.embeddings import create_embedding_provider
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, TextParser
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.ingestion.storage import LocalFilesystemStorage
from app.models.documents import DocumentVersionCreate
from app.models.versioning import DocumentLanguage
from app.reranking import create_reranker
from app.retrieval.hybrid import HybridRetrievalService
from app.retrieval.lexical import LexicalRetrievalService
from app.retrieval.reranked import RerankedRetrievalService
from app.retrieval.service import DenseRetrievalService, VectorIndexingService
from app.vector_store import QdrantVectorStore
from scripts.verify_rag import ingest_sample


def resolved_cache(settings: Settings) -> Path:
    return Path(settings.reranker_cache_directory).resolve()


def summarize(candidates: list[object]) -> list[dict[str, object]]:
    fields = (
        "chunk_id",
        "document_version_id",
        "text",
        "dense_rank",
        "dense_score",
        "lexical_rank",
        "lexical_score",
        "rrf_rank",
        "rrf_score",
        "reranker_rank",
        "reranker_score",
    )
    return [
        {
            field: str(value) if field.endswith("_id") else value
            for field in fields
            if (value := getattr(candidate, field, None)) is not None
        }
        for candidate in candidates
    ]


async def rankings(
    query: str,
    version_id: UUID,
    dense: DenseRetrievalService,
    lexical: LexicalRetrievalService,
    hybrid: HybridRetrievalService,
    reranked: RerankedRetrievalService,
) -> dict[str, object]:
    dense_result = await dense.search(query, version_id, 5)
    lexical_result = await lexical.search(query, version_id, 5)
    hybrid_result = await hybrid.search(query, version_id, 5)
    reranked_result = await reranked.search_versions(query, [version_id], 5)
    return {
        "dense": summarize(list(dense_result.candidates)),
        "lexical": summarize(list(lexical_result.candidates)),
        "rrf": summarize(list(hybrid_result.candidates)),
        "reranked": summarize(list(reranked_result.candidates)),
        "latency_ms": {
            "dense": reranked_result.dense_latency_ms,
            "lexical": reranked_result.lexical_latency_ms,
            "fusion": reranked_result.fusion_latency_ms,
            "reranking": reranked_result.reranking_latency_ms,
            "total": reranked_result.total_latency_ms,
        },
    }


async def run() -> None:
    settings = Settings()
    cache = resolved_cache(settings)
    if cache.drive.upper() != "D:":
        raise RuntimeError(f"reranker cache must resolve to D:, got {cache}")
    database = Database(settings)
    embeddings = create_embedding_provider(settings)
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    storage = LocalFilesystemStorage(
        Path(settings.upload_directory), max_bytes=settings.max_upload_bytes
    )
    parsers = ParserRegistry((PdfParser(), DocxParser(), TextParser(), MarkdownParser()))
    output: dict[str, object] = {}
    try:
        async with database.session_factory() as session:
            catalog = DocumentCatalogService(session)
            ingestion = IngestionService(session, storage, parsers, TokenWindowChunker())
            suffix = uuid4().hex[:8]
            english_document_id, english_version_id = await ingest_sample(
                catalog,
                ingestion,
                path=Path("data/samples/employee_handbook_en.md"),
                stable_key=f"phase7-en-{suffix}",
                language=DocumentLanguage.ENGLISH,
            )
            _, vietnamese_version_id = await ingest_sample(
                catalog,
                ingestion,
                path=Path("data/samples/hr_policy_vi.txt"),
                stable_key=f"phase7-vi-{suffix}",
                language=DocumentLanguage.VIETNAMESE,
            )
            legacy_bytes = await asyncio.to_thread(
                Path("data/samples/employee_handbook_en_2025.md").read_bytes
            )
            legacy = await ingestion.ingest(
                english_document_id,
                DocumentVersionCreate(
                    version_label="2025",
                    version_ordinal=2,
                    language=DocumentLanguage.ENGLISH,
                    activate=False,
                    chunk_size=35,
                    chunk_overlap=5,
                ),
                source=BytesIO(legacy_bytes),
                filename="employee_handbook_en_2025.md",
            )
            indexer = VectorIndexingService(session, embeddings, vector_store)
            for version_id in (
                english_version_id,
                vietnamese_version_id,
                legacy.document_version_id,
            ):
                await indexer.index_version(version_id)
            dense = DenseRetrievalService(session, embeddings, vector_store)
            lexical = LexicalRetrievalService(session)
            hybrid = HybridRetrievalService(
                dense,
                lexical,
                dense_candidate_limit=settings.dense_candidate_limit,
                lexical_candidate_limit=settings.lexical_candidate_limit,
                rrf_k=settings.rrf_k,
            )
            reranked = RerankedRetrievalService(
                hybrid,
                create_reranker(settings),
                candidate_limit=settings.rerank_candidate_limit,
                final_top_k=settings.rerank_final_top_k,
            )
            english = await rankings(
                "remote work",
                english_version_id,
                dense,
                lexical,
                hybrid,
                reranked,
            )
            vietnamese = await rankings(
                "nghỉ phép",
                vietnamese_version_id,
                dense,
                lexical,
                hybrid,
                reranked,
            )
            exact = await reranked.search_versions(
                "How many remote-work days are allowed?",
                [legacy.document_version_id],
                5,
            )
            assert exact.candidates
            assert all(
                item.document_version_id == legacy.document_version_id for item in exact.candidates
            )
            output = {
                "model": {
                    "name": settings.reranker_model_name,
                    "revision": settings.reranker_model_revision,
                    "cache": str(cache),
                },
                "english": english,
                "vietnamese": vietnamese,
                "exact_version_no_leakage": {
                    "expected_document_version_id": str(legacy.document_version_id),
                    "candidates": summarize(list(exact.candidates)),
                    "passed": True,
                },
            }
    finally:
        await vector_store.close()
        await database.close()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8")
    asyncio.run(run())
