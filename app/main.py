from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app import __version__
from app.api.error_handlers import register_error_handlers
from app.api.router import api_router
from app.api.routes.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.core.telemetry import RequestContextMiddleware
from app.db.session import Database
from app.embeddings import create_embedding_provider
from app.generation.providers import create_llm_provider
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, TextParser
from app.ingestion.storage import LocalFilesystemStorage
from app.vector_store import QdrantVectorStore


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings)
    database = Database(app_settings)
    storage = LocalFilesystemStorage(
        Path(app_settings.upload_directory),
        max_bytes=app_settings.max_upload_bytes,
    )
    parsers = ParserRegistry((PdfParser(), DocxParser(), TextParser(), MarkdownParser()))
    chunker = TokenWindowChunker()
    embeddings = create_embedding_provider(app_settings)
    llm_provider = create_llm_provider(app_settings)
    vector_store = QdrantVectorStore(
        url=app_settings.qdrant_url,
        collection_name=app_settings.qdrant_collection,
        api_key=(
            app_settings.qdrant_api_key.get_secret_value()
            if app_settings.qdrant_api_key is not None
            else None
        ),
        timeout_seconds=app_settings.qdrant_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if llm_provider is not None:
            await llm_provider.close()
        await vector_store.close()
        await database.close()

    application = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        description="Version-aware enterprise knowledge retrieval API",
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.state.database = database
    application.state.document_storage = storage
    application.state.parser_registry = parsers
    application.state.chunker = chunker
    application.state.embedding_provider = embeddings
    application.state.llm_provider = llm_provider
    application.state.vector_store = vector_store
    application.add_middleware(RequestContextMiddleware)
    register_error_handlers(application)
    application.include_router(health_router)
    application.include_router(api_router, prefix=app_settings.api_prefix)
    return application


app = create_app()
