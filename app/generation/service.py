from __future__ import annotations

import logging
import re
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import Settings
from app.core.telemetry import request_id_context
from app.db.models import DocumentVersion
from app.generation.context import BuiltContext, EvidenceContextBuilder, EvidenceMetadata
from app.generation.errors import CitationValidationError, LLMConfigurationError
from app.generation.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from app.generation.providers.base import LLMProvider, LLMRequest
from app.models.query import (
    Citation,
    DebugCandidate,
    QueryRequest,
    QueryResponse,
    QueryTiming,
    ResolvedScope,
    TokenUsage,
)
from app.models.retrieval import DenseRetrievalBatch, DenseRetrievalCandidate
from app.models.versioning import DocumentLanguage
from app.retrieval.service import DenseRetrievalService
from app.retrieval.version_scope import VersionScopeResolver

logger = logging.getLogger(__name__)
_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_VIETNAMESE_PATTERN = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯ]|[àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)


class RagAnswerService:
    def __init__(
        self,
        session: AsyncSession,
        scope_resolver: VersionScopeResolver,
        retrieval: DenseRetrievalService,
        context_builder: EvidenceContextBuilder,
        llm: LLMProvider | None,
        settings: Settings,
    ) -> None:
        self._session = session
        self._scope_resolver = scope_resolver
        self._retrieval = retrieval
        self._context_builder = context_builder
        self._llm = llm
        self._settings = settings

    async def answer(self, request: QueryRequest) -> QueryResponse:
        total_started = perf_counter()
        language = detect_question_language(request.question)

        scope_started = perf_counter()
        scope = await self._scope_resolver.resolve(
            request.logical_document_id, request.version_selector
        )
        scope_ms = (perf_counter() - scope_started) * 1000
        version_ids = [version.document_version_id for version in scope.versions]

        retrieval_result = await self._retrieval.search_versions(
            request.question, version_ids, request.top_k
        )
        evidence_metadata = await self._hydrate_evidence(retrieval_result.candidates)
        context_started = perf_counter()
        context = self._context_builder.build(evidence_metadata)
        context_ms = (perf_counter() - context_started) * 1000

        if not context.evidence:
            return self._insufficient_response(
                request=request,
                language=language,
                scope=scope,
                retrieval_result=retrieval_result,
                scope_ms=scope_ms,
                context_ms=context_ms,
                total_started=total_started,
            )
        if request.retrieval_only:
            return self._retrieval_only_response(
                request=request,
                language=language,
                scope=scope,
                retrieval_result=retrieval_result,
                context=context,
                scope_ms=scope_ms,
                context_ms=context_ms,
                total_started=total_started,
            )
        if self._llm is None:
            raise LLMConfigurationError("configure LLM_PROVIDER before requesting RAG generation")

        llm_request = LLMRequest(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(
                question=request.question,
                context=context.text,
                language=language,
                resolved_version_ids=[str(version_id) for version_id in version_ids],
            ),
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_output_tokens,
        )
        generation_started = perf_counter()
        generated = await self._llm.generate(llm_request)
        generation_ms = (perf_counter() - generation_started) * 1000
        citations = self._validated_citations(generated.text, context)
        total_ms = (perf_counter() - total_started) * 1000

        logger.info(
            "RAG answer generated",
            extra={
                "event": "rag_answer_generated",
                "request_id": request_id_context.get(),
                "resolved_version_ids": [str(version_id) for version_id in version_ids],
                "retrieved_candidate_count": len(retrieval_result.candidates),
                "evidence_chunk_count": len(context.evidence),
                "llm_provider": self._llm.provider_name,
                "llm_model": self._llm.model_name,
                "generation_latency_ms": round(generation_ms, 3),
                "total_latency_ms": round(total_ms, 3),
                "token_usage": generated.token_usage,
            },
        )
        return QueryResponse(
            answer=generated.text,
            answer_language=language,
            citations=citations,
            resolved_scope=scope,
            timing=QueryTiming(
                scope_resolution_ms=scope_ms,
                query_embedding_ms=retrieval_result.embedding_latency_ms,
                retrieval_ms=retrieval_result.retrieval_latency_ms,
                context_construction_ms=context_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
            token_usage=(
                TokenUsage(
                    prompt_tokens=generated.token_usage.prompt_tokens,
                    completion_tokens=generated.token_usage.completion_tokens,
                    total_tokens=generated.token_usage.total_tokens,
                )
                if generated.token_usage is not None
                else None
            ),
            prompt_version=PROMPT_VERSION,
            insufficient_evidence=False,
            debug_candidates=(self._debug_candidates(context) if request.debug else None),
        )

    async def _hydrate_evidence(
        self, candidates: list[DenseRetrievalCandidate]
    ) -> list[EvidenceMetadata]:
        version_ids = {candidate.document_version_id for candidate in candidates}
        versions = list(
            await self._session.scalars(
                select(DocumentVersion)
                .where(DocumentVersion.document_version_id.in_(version_ids))
                .options(joinedload(DocumentVersion.logical_document))
            )
        )
        by_id = {version.document_version_id: version for version in versions}
        return [
            EvidenceMetadata(
                candidate=candidate,
                document_title=by_id[candidate.document_version_id].logical_document.title,
                version_label=by_id[candidate.document_version_id].version_label,
                filename=by_id[candidate.document_version_id].source_filename,
            )
            for candidate in candidates
            if candidate.document_version_id in by_id and candidate.text.strip()
        ]

    @staticmethod
    def _validated_citations(text: str, context: BuiltContext) -> list[Citation]:
        evidence_by_key = {item.citation.citation_key: item.citation for item in context.evidence}
        referenced = list(dict.fromkeys(_CITATION_PATTERN.findall(text)))
        unknown = [key for key in referenced if key not in evidence_by_key]
        if unknown:
            raise CitationValidationError(
                "LLM referenced citation keys outside the supplied evidence",
                unknown_keys=unknown,
            )
        if not referenced:
            raise CitationValidationError("LLM answer did not include a supplied citation key")
        return [evidence_by_key[key] for key in referenced]

    @staticmethod
    def _debug_candidates(context: BuiltContext) -> list[DebugCandidate]:
        return [
            DebugCandidate(
                citation_key=item.citation.citation_key,
                chunk_id=item.candidate.chunk_id,
                document_version_id=item.candidate.document_version_id,
                score=item.candidate.score,
                ordinal=item.candidate.ordinal,
                page=item.candidate.page,
                section_path=item.candidate.section_path,
                language=item.candidate.language,
            )
            for item in context.evidence
        ]

    def _insufficient_response(
        self,
        *,
        request: QueryRequest,
        language: DocumentLanguage,
        scope: ResolvedScope,
        retrieval_result: DenseRetrievalBatch,
        scope_ms: float,
        context_ms: float,
        total_started: float,
    ) -> QueryResponse:
        total_ms = (perf_counter() - total_started) * 1000
        answer = (
            "Không có đủ bằng chứng trong phạm vi tài liệu đã chọn để trả lời câu hỏi."
            if language is DocumentLanguage.VIETNAMESE
            else (
                "There is insufficient evidence in the selected document scope "
                "to answer the question."
            )
        )
        logger.info(
            "RAG request refused for insufficient evidence",
            extra={
                "event": "rag_insufficient_evidence",
                "request_id": request_id_context.get(),
                "resolved_version_ids": [
                    str(version.document_version_id) for version in scope.versions
                ],
                "retrieved_candidate_count": len(retrieval_result.candidates),
                "evidence_chunk_count": 0,
                "total_latency_ms": round(total_ms, 3),
            },
        )
        return QueryResponse(
            answer=answer,
            answer_language=language,
            citations=[],
            resolved_scope=scope,
            timing=QueryTiming(
                scope_resolution_ms=scope_ms,
                query_embedding_ms=retrieval_result.embedding_latency_ms,
                retrieval_ms=retrieval_result.retrieval_latency_ms,
                context_construction_ms=context_ms,
                generation_ms=0,
                total_ms=total_ms,
            ),
            token_usage=None,
            prompt_version=PROMPT_VERSION,
            insufficient_evidence=True,
            debug_candidates=[] if request.debug else None,
        )

    def _retrieval_only_response(
        self,
        *,
        request: QueryRequest,
        language: DocumentLanguage,
        scope: ResolvedScope,
        retrieval_result: DenseRetrievalBatch,
        context: BuiltContext,
        scope_ms: float,
        context_ms: float,
        total_started: float,
    ) -> QueryResponse:
        total_ms = (perf_counter() - total_started) * 1000
        logger.info(
            "RAG retrieval-only request completed",
            extra={
                "event": "rag_retrieval_only_completed",
                "request_id": request_id_context.get(),
                "resolved_version_ids": [
                    str(version.document_version_id) for version in scope.versions
                ],
                "retrieved_candidate_count": len(retrieval_result.candidates),
                "evidence_chunk_count": len(context.evidence),
                "total_latency_ms": round(total_ms, 3),
            },
        )
        return QueryResponse(
            answer=None,
            answer_language=language,
            citations=[item.citation for item in context.evidence],
            resolved_scope=scope,
            timing=QueryTiming(
                scope_resolution_ms=scope_ms,
                query_embedding_ms=retrieval_result.embedding_latency_ms,
                retrieval_ms=retrieval_result.retrieval_latency_ms,
                context_construction_ms=context_ms,
                generation_ms=0,
                total_ms=total_ms,
            ),
            token_usage=None,
            prompt_version=PROMPT_VERSION,
            insufficient_evidence=False,
            debug_candidates=(self._debug_candidates(context) if request.debug else None),
        )


def detect_question_language(question: str) -> DocumentLanguage:
    return (
        DocumentLanguage.VIETNAMESE
        if _VIETNAMESE_PATTERN.search(question)
        else DocumentLanguage.ENGLISH
    )
