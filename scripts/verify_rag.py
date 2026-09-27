from __future__ import annotations

import asyncio
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

from app.core.config import Settings
from app.db.session import Database
from app.embeddings import create_embedding_provider
from app.generation.context import EvidenceContextBuilder
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage
from app.generation.providers.factory import create_llm_provider
from app.generation.service import RagAnswerService
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, TextParser
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.ingestion.storage import LocalFilesystemStorage
from app.models.documents import DocumentVersionCreate, LogicalDocumentCreate
from app.models.query import QueryRequest, QueryVersionSelector
from app.models.retrieval import RetrievalMode
from app.models.versioning import DocumentLanguage, VersionSelectorKind
from app.retrieval.hybrid import HybridRetrievalService, ModeAwareRetrievalService
from app.retrieval.lexical import LexicalRetrievalService
from app.retrieval.service import DenseRetrievalService, VectorIndexingService
from app.retrieval.version_scope import VersionScopeResolver
from app.vector_store import QdrantVectorStore
from app.vector_store.base import VectorPoint, VectorSearchHit, VectorStore


class ControlledBilingualLLM:
    provider_name = "controlled_fake"
    model_name = "controlled-grounded-verifier-v1"

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if "Requested answer language: Vietnamese" in request.user_prompt:
            text = "Nhân viên được hưởng mười hai ngày nghỉ phép có lương mỗi năm [S1]."
        elif "up to one day per week" in request.user_prompt:
            text = "Employees may work remotely up to one day per week [S1]."
        else:
            text = "Employees may work remotely up to two days per week [S1]."
        return LLMResponse(text, LLMTokenUsage(120, 18, 138))

    async def close(self) -> None:
        return None


class EmptySearchVectorStore:
    def __init__(self, delegate: VectorStore) -> None:
        self._delegate = delegate

    async def ensure_collection(self, dimension: int) -> None:
        await self._delegate.ensure_collection(dimension)

    async def upsert(self, points: list[VectorPoint]) -> None:
        await self._delegate.upsert(points)

    async def search_exact_version(
        self, _vector: list[float], _document_version_id: UUID, _top_k: int
    ) -> list[VectorSearchHit]:
        return []

    async def search_versions(
        self, _vector: list[float], _document_version_ids: list[UUID], _top_k: int
    ) -> list[VectorSearchHit]:
        return []

    async def delete_version(self, document_version_id: UUID) -> None:
        await self._delegate.delete_version(document_version_id)

    async def close(self) -> None:
        return None


def cache_details(path_value: str) -> dict[str, object]:
    cache_path = Path(path_value).resolve()
    return {
        "HF_HOME": os.getenv("HF_HOME"),
        "embedding_cache_directory": str(cache_path),
        "drive": cache_path.drive,
        "bytes": sum(item.stat().st_size for item in cache_path.rglob("*") if item.is_file()),
    }


async def ingest_sample(
    catalog: DocumentCatalogService,
    ingestion: IngestionService,
    *,
    path: Path,
    stable_key: str,
    language: DocumentLanguage,
) -> tuple[UUID, UUID]:
    document = await catalog.create(LogicalDocumentCreate(stable_key=stable_key, title=stable_key))
    with path.open("rb") as source:
        version = await ingestion.ingest(
            document.logical_document_id,
            DocumentVersionCreate(
                version_label="2026",
                version_ordinal=1,
                language=language,
                activate=True,
                chunk_size=35,
                chunk_overlap=5,
            ),
            source=source,
            filename=path.name,
        )
    return document.logical_document_id, version.document_version_id


async def run() -> None:
    settings = Settings()
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
    suffix = uuid4().hex[:8]
    output: dict[str, object] = {}

    try:
        async with database.session_factory() as session:
            catalog = DocumentCatalogService(session)
            ingestion = IngestionService(session, storage, parsers, TokenWindowChunker())
            english_document_id, english_version_id = await ingest_sample(
                catalog,
                ingestion,
                path=Path("data/samples/employee_handbook_en.md"),
                stable_key=f"phase4-en-{suffix}",
                language=DocumentLanguage.ENGLISH,
            )
            vietnamese_document_id, vietnamese_version_id = await ingest_sample(
                catalog,
                ingestion,
                path=Path("data/samples/hr_policy_vi.txt"),
                stable_key=f"phase4-vi-{suffix}",
                language=DocumentLanguage.VIETNAMESE,
            )
            indexer = VectorIndexingService(session, embeddings, vector_store)
            await indexer.index_version(english_version_id)
            await indexer.index_version(vietnamese_version_id)

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
            legacy_version_id = legacy.document_version_id
            await indexer.index_version(legacy_version_id)

            llm = ControlledBilingualLLM()
            context_builder = EvidenceContextBuilder(settings.rag_max_context_characters)
            resolver = VersionScopeResolver(session)
            dense = DenseRetrievalService(session, embeddings, vector_store)
            lexical = LexicalRetrievalService(session)
            hybrid = HybridRetrievalService(
                dense,
                lexical,
                dense_candidate_limit=settings.dense_candidate_limit,
                lexical_candidate_limit=settings.lexical_candidate_limit,
                rrf_k=settings.rrf_k,
            )
            retrieval = ModeAwareRetrievalService(dense, hybrid)
            rag = RagAnswerService(
                session,
                resolver,
                retrieval,
                context_builder,
                llm,
                settings,
            )
            english = await rag.answer(
                QueryRequest(
                    question="How many days per week may employees work remotely?",
                    logical_document_id=english_document_id,
                    retrieval_mode=RetrievalMode.HYBRID,
                    debug=True,
                )
            )
            vietnamese = await rag.answer(
                QueryRequest(
                    question="Nhân viên được nghỉ phép có lương bao nhiêu ngày mỗi năm?",
                    logical_document_id=vietnamese_document_id,
                    retrieval_mode=RetrievalMode.HYBRID,
                    debug=True,
                )
            )
            exact = await rag.answer(
                QueryRequest(
                    question="How many days per week may employees work remotely?",
                    version_selector=QueryVersionSelector(
                        kind=VersionSelectorKind.EXACT,
                        document_version_id=legacy_version_id,
                    ),
                    retrieval_mode=RetrievalMode.HYBRID,
                    debug=True,
                )
            )
            assert all(
                citation.document_version_id == legacy_version_id for citation in exact.citations
            )
            retrieval_outputs = {
                "english": {
                    "dense": (await dense.search("remote work", english_version_id, 5)).model_dump(
                        mode="json"
                    ),
                    "lexical": (
                        await lexical.search("remote work", english_version_id, 5)
                    ).model_dump(mode="json"),
                    "hybrid": (
                        await hybrid.search("remote work", english_version_id, 5)
                    ).model_dump(mode="json"),
                },
                "vietnamese": {
                    "dense": (await dense.search("nghỉ phép", vietnamese_version_id, 5)).model_dump(
                        mode="json"
                    ),
                    "lexical": (
                        await lexical.search("nghỉ phép", vietnamese_version_id, 5)
                    ).model_dump(mode="json"),
                    "hybrid": (
                        await hybrid.search("nghỉ phép", vietnamese_version_id, 5)
                    ).model_dump(mode="json"),
                },
            }
            empty_dense = DenseRetrievalService(
                session, embeddings, EmptySearchVectorStore(vector_store)
            )
            empty_hybrid = HybridRetrievalService(
                empty_dense,
                lexical,
                dense_candidate_limit=settings.dense_candidate_limit,
                lexical_candidate_limit=settings.lexical_candidate_limit,
                rrf_k=settings.rrf_k,
            )
            insufficient_rag = RagAnswerService(
                session,
                resolver,
                ModeAwareRetrievalService(empty_dense, empty_hybrid),
                context_builder,
                llm,
                settings,
            )
            insufficient = await insufficient_rag.answer(
                QueryRequest(
                    question="What is the cafeteria menu tomorrow?",
                    logical_document_id=english_document_id,
                    debug=True,
                )
            )
            retrieval_only_rag = RagAnswerService(
                session,
                resolver,
                retrieval,
                context_builder,
                None,
                settings,
            )
            retrieval_only = await retrieval_only_rag.answer(
                QueryRequest(
                    question="What is the remote-work policy?",
                    logical_document_id=english_document_id,
                    retrieval_mode=RetrievalMode.HYBRID,
                    retrieval_only=True,
                    debug=True,
                )
            )

            openai_provider = create_llm_provider(
                Settings(
                    llm_provider="openai_compatible",
                    openai_api_key="verification-placeholder",
                    openai_model="configured-model",
                )
            )
            ollama_provider = create_llm_provider(
                Settings(llm_provider="ollama", ollama_model="configured-model")
            )
            assert openai_provider is not None and ollama_provider is not None
            provider_configuration = {
                "openai_compatible": openai_provider.provider_name,
                "ollama": ollama_provider.provider_name,
                "real_generation_executed": False,
            }
            await openai_provider.close()
            await ollama_provider.close()

            output = {
                "cache": await asyncio.to_thread(cache_details, settings.embedding_cache_directory),
                "providers": provider_configuration,
                "retrieval": retrieval_outputs,
                "english": english.model_dump(mode="json"),
                "vietnamese": vietnamese.model_dump(mode="json"),
                "exact": exact.model_dump(mode="json"),
                "retrieval_only": retrieval_only.model_dump(mode="json"),
                "insufficient_evidence": insufficient.model_dump(mode="json"),
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
