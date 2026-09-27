from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.session import Database
from app.embeddings.base import EmbeddingProvider
from app.generation.context import EvidenceContextBuilder
from app.generation.errors import LLMConfigurationError
from app.generation.providers.base import LLMProvider
from app.generation.service import RagAnswerService
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import ParserRegistry
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.ingestion.storage import DocumentStorage
from app.reranking.base import Reranker
from app.retrieval.hybrid import HybridRetrievalService, ModeAwareRetrievalService
from app.retrieval.lexical import LexicalRetrievalService
from app.retrieval.reranked import RerankedRetrievalService
from app.retrieval.service import DenseRetrievalService, VectorIndexingService
from app.retrieval.version_scope import VersionScopeResolver
from app.vector_store.base import VectorStore


def get_request_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    database = get_database(request)
    async for session in database.session():
        yield session


def get_storage(request: Request) -> DocumentStorage:
    return cast(DocumentStorage, request.app.state.document_storage)


def get_parser_registry(request: Request) -> ParserRegistry:
    return cast(ParserRegistry, request.app.state.parser_registry)


def get_chunker(request: Request) -> TokenWindowChunker:
    return cast(TokenWindowChunker, request.app.state.chunker)


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return cast(EmbeddingProvider, request.app.state.embedding_provider)


def get_vector_store(request: Request) -> VectorStore:
    return cast(VectorStore, request.app.state.vector_store)


def get_reranker(request: Request) -> Reranker:
    return cast(Reranker, request.app.state.reranker)


def get_llm_provider(request: Request) -> LLMProvider:
    provider = cast(LLMProvider | None, request.app.state.llm_provider)
    if provider is None:
        raise LLMConfigurationError("LLM_PROVIDER is not configured")
    return provider


async def get_catalog_service(request: Request) -> AsyncIterator[DocumentCatalogService]:
    database = get_database(request)
    async for session in database.session():
        yield DocumentCatalogService(session)


async def get_ingestion_service(request: Request) -> AsyncIterator[IngestionService]:
    database = get_database(request)
    async for session in database.session():
        yield IngestionService(
            session,
            get_storage(request),
            get_parser_registry(request),
            get_chunker(request),
        )


async def get_vector_indexing_service(
    request: Request,
) -> AsyncIterator[VectorIndexingService]:
    database = get_database(request)
    async for session in database.session():
        yield VectorIndexingService(
            session, get_embedding_provider(request), get_vector_store(request)
        )


async def get_dense_retrieval_service(
    request: Request,
) -> AsyncIterator[DenseRetrievalService]:
    database = get_database(request)
    async for session in database.session():
        yield DenseRetrievalService(
            session, get_embedding_provider(request), get_vector_store(request)
        )


async def get_lexical_retrieval_service(
    request: Request,
) -> AsyncIterator[LexicalRetrievalService]:
    database = get_database(request)
    async for session in database.session():
        yield LexicalRetrievalService(session)


async def get_hybrid_retrieval_service(
    request: Request,
) -> AsyncIterator[HybridRetrievalService]:
    database = get_database(request)
    settings = get_request_settings(request)
    async for session in database.session():
        dense = DenseRetrievalService(
            session, get_embedding_provider(request), get_vector_store(request)
        )
        yield HybridRetrievalService(
            dense,
            LexicalRetrievalService(session),
            dense_candidate_limit=settings.dense_candidate_limit,
            lexical_candidate_limit=settings.lexical_candidate_limit,
            rrf_k=settings.rrf_k,
        )


async def get_rag_service(request: Request) -> AsyncIterator[RagAnswerService]:
    database = get_database(request)
    settings = get_request_settings(request)
    async for session in database.session():
        dense = DenseRetrievalService(
            session, get_embedding_provider(request), get_vector_store(request)
        )
        hybrid = HybridRetrievalService(
            dense,
            LexicalRetrievalService(session),
            dense_candidate_limit=settings.dense_candidate_limit,
            lexical_candidate_limit=settings.lexical_candidate_limit,
            rrf_k=settings.rrf_k,
        )
        reranked = RerankedRetrievalService(
            hybrid,
            get_reranker(request),
            candidate_limit=settings.rerank_candidate_limit,
            final_top_k=settings.rerank_final_top_k,
        )
        yield RagAnswerService(
            session=session,
            scope_resolver=VersionScopeResolver(session),
            retrieval=ModeAwareRetrievalService(dense, hybrid, reranked),
            context_builder=EvidenceContextBuilder(
                get_request_settings(request).rag_max_context_characters,
                get_request_settings(request).citation_excerpt_max_characters,
            ),
            llm=cast(LLMProvider | None, request.app.state.llm_provider),
            settings=get_request_settings(request),
        )
