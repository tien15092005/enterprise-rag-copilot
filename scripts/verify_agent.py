from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.agent.errors import AgentError
from app.agent.models import AgentRequest, AgentResponse
from app.agent.provider import (
    AgentDecision,
    AgentMessage,
    AgentToolCall,
    AgentToolDefinition,
)
from app.agent.service import AgentService
from app.agent.tools import AgentToolRegistry
from app.comparison.service import DocumentComparisonService
from app.core.config import Settings
from app.db.session import Database
from app.embeddings import create_embedding_provider
from app.generation.context import EvidenceContextBuilder
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage
from app.generation.service import RagAnswerService
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, TextParser
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.ingestion.storage import LocalFilesystemStorage
from app.models.query import TokenUsage
from app.models.retrieval import RetrievalMode
from app.models.versioning import DocumentLanguage
from app.reranking import create_reranker
from app.retrieval.hybrid import HybridRetrievalService, ModeAwareRetrievalService
from app.retrieval.lexical import LexicalRetrievalService
from app.retrieval.reranked import RerankedRetrievalService
from app.retrieval.service import DenseRetrievalService, VectorIndexingService
from app.retrieval.version_scope import VersionScopeResolver
from app.vector_store import QdrantVectorStore
from scripts.verify_comparison import ingest_pair
from scripts.verify_rag import EmptySearchVectorStore


class ControlledAgentProvider:
    provider_name = "controlled_fake"
    model_name = "controlled-agent-verifier-v1"

    def __init__(self, decisions: list[AgentDecision]) -> None:
        self._decisions = decisions

    async def decide(
        self,
        *,
        system_prompt: str,
        messages: list[AgentMessage],
        tools: list[AgentToolDefinition],
        temperature: float,
        max_output_tokens: int,
    ) -> AgentDecision:
        return self._decisions.pop(0)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if "Requested answer language: Vietnamese" in request.user_prompt:
            text = "Nhân viên hiện được hưởng 12 ngày nghỉ phép có lương mỗi năm [S1]."
        else:
            text = "Employees may work remotely up to two days per week [S1]."
        return LLMResponse(text, LLMTokenUsage(100, 15, 115))

    async def close(self) -> None:
        return None


def tool(name: str, arguments: dict[str, Any], call_id: str = "call-1") -> AgentDecision:
    return AgentDecision(tool_call=AgentToolCall(call_id, name, arguments))


def final(text: str) -> AgentDecision:
    return AgentDecision(
        final_answer=text,
        token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )


def registry(
    session: Any,
    embeddings: Any,
    vector_store: Any,
    provider: ControlledAgentProvider,
    settings: Settings,
    ingestion: IngestionService,
) -> AgentToolRegistry:
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
    rag = RagAnswerService(
        session,
        VersionScopeResolver(session),
        ModeAwareRetrievalService(dense, hybrid, reranked),
        EvidenceContextBuilder(
            settings.rag_max_context_characters,
            settings.citation_excerpt_max_characters,
        ),
        provider,
        settings,
    )
    comparison = DocumentComparisonService(session, embeddings, provider, settings)
    return AgentToolRegistry(rag, comparison, ingestion)


def compact(response: AgentResponse) -> dict[str, Any]:
    return {
        "answer": response.answer,
        "citations": [citation.model_dump(mode="json") for citation in response.citations],
        "timing": response.timing.model_dump(),
        "token_usage": response.token_usage.model_dump() if response.token_usage else None,
        "debug": response.debug.model_dump(mode="json") if response.debug else None,
    }


async def scenario(
    decisions: list[AgentDecision],
    request: AgentRequest,
    *,
    session: Any,
    embeddings: Any,
    vector_store: Any,
    settings: Settings,
    ingestion: IngestionService,
) -> dict[str, Any]:
    provider = ControlledAgentProvider(decisions)
    service = AgentService(
        provider,
        registry(session, embeddings, vector_store, provider, settings, ingestion),
        settings,
    )
    return compact(await service.run(request))


async def run() -> None:
    settings = Settings()
    cache, uploads = runtime_paths(settings)
    if cache.drive.upper() != "D:" or uploads.drive.upper() != "D:":
        raise RuntimeError(f"agent runtime paths must resolve to D:, got {cache}, {uploads}")
    database = Database(settings)
    embeddings = create_embedding_provider(settings)
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    storage = LocalFilesystemStorage(uploads, max_bytes=settings.max_upload_bytes)
    parsers = ParserRegistry((PdfParser(), DocxParser(), TextParser(), MarkdownParser()))
    output: dict[str, Any] = {}
    try:
        async with database.session_factory() as session:
            catalog = DocumentCatalogService(session)
            ingestion = IngestionService(session, storage, parsers, TokenWindowChunker())
            suffix = uuid4().hex[:8]
            en_doc, en_2025, en_2026 = await ingest_pair(
                catalog,
                ingestion,
                stable_key=f"phase9b-en-{suffix}",
                title="Employee Policy",
                language=DocumentLanguage.ENGLISH,
                left_path=Path("data/comparison/employee_policy_en_2025.md"),
                right_path=Path("data/comparison/employee_policy_en_2026.md"),
            )
            vi_doc, vi_2025, vi_2026 = await ingest_pair(
                catalog,
                ingestion,
                stable_key=f"phase9b-vi-{suffix}",
                title="Chính sách nhân sự",
                language=DocumentLanguage.VIETNAMESE,
                left_path=Path("data/comparison/leave_policy_vi_2025.md"),
                right_path=Path("data/comparison/leave_policy_vi_2026.md"),
            )
            indexer = VectorIndexingService(session, embeddings, vector_store)
            for version_id in (en_2025, en_2026, vi_2025, vi_2026):
                await indexer.index_version(version_id)

            common = {
                "session": session,
                "embeddings": embeddings,
                "vector_store": vector_store,
                "settings": settings,
                "ingestion": ingestion,
            }
            output["english_current"] = await scenario(
                [
                    tool(
                        "search_knowledge",
                        {
                            "question": "What is the current remote-work policy?",
                            "version_selector": {"kind": "current"},
                        },
                    ),
                    final("Employees may work remotely up to two days per week [S1]."),
                ],
                AgentRequest(
                    message="What is the current remote-work policy?",
                    logical_document_id=en_doc,
                    retrieval_mode=RetrievalMode.HYBRID,
                    debug=True,
                ),
                **common,
            )
            output["vietnamese_current"] = await scenario(
                [
                    tool(
                        "search_knowledge",
                        {
                            "question": "Nhân viên hiện tại được nghỉ phép bao nhiêu ngày?",
                            "version_selector": {"kind": "current"},
                        },
                    ),
                    final("Nhân viên hiện được hưởng 12 ngày nghỉ phép có lương mỗi năm [S1]."),
                ],
                AgentRequest(
                    message="Nhân viên hiện tại được nghỉ phép bao nhiêu ngày?",
                    logical_document_id=vi_doc,
                    retrieval_mode=RetrievalMode.HYBRID,
                    debug=True,
                ),
                **common,
            )
            output["english_comparison"] = await scenario(
                [
                    tool(
                        "compare_versions",
                        {
                            "left_document_version_id": str(en_2025),
                            "right_document_version_id": str(en_2026),
                        },
                    ),
                    final("Remote work increased from one day to two days per week [S1] [S2]."),
                ],
                AgentRequest(
                    message="What changed in remote-work policy from 2025 to 2026?",
                    logical_document_id=en_doc,
                    debug=True,
                ),
                **common,
            )
            output["vietnamese_comparison"] = await scenario(
                [
                    tool(
                        "compare_versions",
                        {
                            "left_document_version_id": str(vi_2025),
                            "right_document_version_id": str(vi_2026),
                            "summary_language": "vi",
                        },
                    ),
                    final("Nghỉ phép năm tăng từ 10 lên 12 ngày [S1] [S2]."),
                ],
                AgentRequest(
                    message="So sánh chính sách nghỉ phép năm 2025 và 2026.",
                    logical_document_id=vi_doc,
                    debug=True,
                ),
                **common,
            )
            output["list_versions"] = await scenario(
                [
                    tool("list_document_versions", {}),
                    final("The available versions are 2025 and 2026."),
                ],
                AgentRequest(
                    message="What versions exist?", logical_document_id=en_doc, debug=True
                ),
                **common,
            )

            empty_store = EmptySearchVectorStore(vector_store)
            empty_common = dict(common)
            empty_common["vector_store"] = empty_store
            output["insufficient_evidence"] = await scenario(
                [
                    tool(
                        "search_knowledge",
                        {
                            "question": "What is the lunar office policy?",
                            "version_selector": {"kind": "current"},
                        },
                    ),
                    final(
                        "There is insufficient evidence in the selected document scope "
                        "to answer the question."
                    ),
                ],
                AgentRequest(
                    message="What is the lunar office policy?",
                    logical_document_id=en_doc,
                    retrieval_mode=RetrievalMode.DENSE,
                    debug=True,
                ),
                **empty_common,
            )

            invalid_provider = ControlledAgentProvider([tool("shell", {"command": "dir"})])
            invalid_service = AgentService(
                invalid_provider,
                registry(
                    session,
                    embeddings,
                    vector_store,
                    invalid_provider,
                    settings,
                    ingestion,
                ),
                settings,
            )
            try:
                await invalid_service.run(AgentRequest(message="run a shell command"))
            except AgentError as exc:
                output["invalid_tool"] = {"code": exc.code, "message": exc.message}

            limit_provider = ControlledAgentProvider(
                [tool("list_document_versions", {}), tool("list_document_versions", {}, "call-2")]
            )
            limit_service = AgentService(
                limit_provider,
                registry(
                    session,
                    embeddings,
                    vector_store,
                    limit_provider,
                    settings,
                    ingestion,
                ),
                settings,
            )
            try:
                await limit_service.run(
                    AgentRequest(
                        message="loop",
                        logical_document_id=en_doc,
                        max_tool_steps=1,
                    )
                )
            except AgentError as exc:
                output["step_limit"] = {"code": exc.code, "message": exc.message}

            output["runtime_paths"] = {
                "embedding_cache": str(cache),
                "uploads": str(uploads),
            }
    finally:
        await vector_store.close()
        await database.close()
    print(json.dumps(output, ensure_ascii=False, indent=2))


def runtime_paths(settings: Settings) -> tuple[Path, Path]:
    return (
        Path(settings.embedding_cache_directory).resolve(),
        Path(settings.upload_directory).resolve(),
    )


if __name__ == "__main__":
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8")
    asyncio.run(run())
