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
from app.generation.citations import CitationSource, CitationValidationResult, CitationValidator
from app.generation.context import BuiltContext, EvidenceContextBuilder
from app.generation.errors import LLMConfigurationError
from app.generation.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from app.generation.providers.base import LLMProvider, LLMRequest
from app.models.query import (
    CitationValidationDebug,
    QueryDebug,
    QueryRequest,
    QueryResponse,
    QueryTiming,
    ResolvedScope,
    TokenUsage,
)
from app.models.retrieval import (
    HybridRetrievalBatch,
    RerankedRetrievalBatch,
    RetrievalBatch,
    RetrievalCandidate,
)
from app.models.versioning import DocumentLanguage
from app.retrieval.hybrid import ModeAwareRetrievalService
from app.retrieval.version_scope import VersionScopeResolver

logger = logging.getLogger(__name__)
_VIETNAMESE_PATTERN = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯ]|[àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)


class RagAnswerService:
    def __init__(
        self,
        session: AsyncSession,
        scope_resolver: VersionScopeResolver,
        retrieval: ModeAwareRetrievalService,
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
        self._citation_validator = CitationValidator()

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
            request.question,
            version_ids,
            request.top_k,
            request.retrieval_mode,
        )
        retrieved_candidates: list[RetrievalCandidate] = [
            candidate for candidate in retrieval_result.candidates
        ]
        evidence_metadata = await self._hydrate_evidence(retrieved_candidates)
        context_started = perf_counter()
        context = self._context_builder.build(
            evidence_metadata,
            allowed_version_ids=set(version_ids),
        )
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
        validation = self._citation_validator.validate(
            generated.text,
            tuple(item.citation for item in context.evidence),
        )
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
            citations=[citation.to_response() for citation in validation.citations],
            resolved_scope=scope,
            timing=QueryTiming(
                scope_resolution_ms=scope_ms,
                query_embedding_ms=retrieval_result.embedding_latency_ms,
                retrieval_ms=self._retrieval_latency(retrieval_result),
                lexical_retrieval_ms=self._lexical_latency(retrieval_result),
                fusion_ms=self._fusion_latency(retrieval_result),
                reranking_ms=self._reranking_latency(retrieval_result),
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
            debug=(self._debug(context, validation) if request.debug else None),
        )

    async def _hydrate_evidence(self, candidates: list[RetrievalCandidate]) -> list[CitationSource]:
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
            CitationSource(
                candidate=candidate,
                document_title=by_id[candidate.document_version_id].logical_document.title,
                version_label=by_id[candidate.document_version_id].version_label,
                source_filename=by_id[candidate.document_version_id].source_filename,
                effective_from=by_id[candidate.document_version_id].effective_from,
                effective_to=by_id[candidate.document_version_id].effective_to,
            )
            for candidate in candidates
            if candidate.document_version_id in by_id and candidate.text.strip()
        ]

    @staticmethod
    def _debug(context: BuiltContext, validation: CitationValidationResult) -> QueryDebug:
        return QueryDebug(
            evidence=[item.citation.to_response() for item in context.evidence],
            citation_validation=CitationValidationDebug(
                status="validated",
                referenced_keys=list(validation.referenced_keys),
                normalized_keys=list(validation.normalized_keys),
                duplicate_keys=list(validation.duplicate_keys),
                omitted_evidence_keys=list(validation.omitted_evidence_keys),
            ),
        )

    @staticmethod
    def _not_applicable_debug(context: BuiltContext) -> QueryDebug:
        return QueryDebug(
            evidence=[item.citation.to_response() for item in context.evidence],
            citation_validation=CitationValidationDebug(
                status="not_applicable",
                referenced_keys=[],
                normalized_keys=[],
                duplicate_keys=[],
                omitted_evidence_keys=[],
            ),
        )

    def _insufficient_response(
        self,
        *,
        request: QueryRequest,
        language: DocumentLanguage,
        scope: ResolvedScope,
        retrieval_result: RetrievalBatch,
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
                retrieval_ms=self._retrieval_latency(retrieval_result),
                lexical_retrieval_ms=self._lexical_latency(retrieval_result),
                fusion_ms=self._fusion_latency(retrieval_result),
                reranking_ms=self._reranking_latency(retrieval_result),
                context_construction_ms=context_ms,
                generation_ms=0,
                total_ms=total_ms,
            ),
            token_usage=None,
            prompt_version=PROMPT_VERSION,
            insufficient_evidence=True,
            debug=(
                self._not_applicable_debug(BuiltContext("", (), False)) if request.debug else None
            ),
        )

    def _retrieval_only_response(
        self,
        *,
        request: QueryRequest,
        language: DocumentLanguage,
        scope: ResolvedScope,
        retrieval_result: RetrievalBatch,
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
            citations=[item.citation.to_response() for item in context.evidence],
            resolved_scope=scope,
            timing=QueryTiming(
                scope_resolution_ms=scope_ms,
                query_embedding_ms=retrieval_result.embedding_latency_ms,
                retrieval_ms=self._retrieval_latency(retrieval_result),
                lexical_retrieval_ms=self._lexical_latency(retrieval_result),
                fusion_ms=self._fusion_latency(retrieval_result),
                reranking_ms=self._reranking_latency(retrieval_result),
                context_construction_ms=context_ms,
                generation_ms=0,
                total_ms=total_ms,
            ),
            token_usage=None,
            prompt_version=PROMPT_VERSION,
            insufficient_evidence=False,
            debug=(self._not_applicable_debug(context) if request.debug else None),
        )

    @staticmethod
    def _retrieval_latency(result: RetrievalBatch) -> float:
        if isinstance(result, (HybridRetrievalBatch, RerankedRetrievalBatch)):
            return result.total_latency_ms
        return result.retrieval_latency_ms

    @staticmethod
    def _lexical_latency(result: RetrievalBatch) -> float:
        return (
            result.lexical_latency_ms
            if isinstance(result, (HybridRetrievalBatch, RerankedRetrievalBatch))
            else 0
        )

    @staticmethod
    def _fusion_latency(result: RetrievalBatch) -> float:
        return (
            result.fusion_latency_ms
            if isinstance(result, (HybridRetrievalBatch, RerankedRetrievalBatch))
            else 0
        )

    @staticmethod
    def _reranking_latency(result: RetrievalBatch) -> float:
        return result.reranking_latency_ms if isinstance(result, RerankedRetrievalBatch) else 0


def detect_question_language(question: str) -> DocumentLanguage:
    return (
        DocumentLanguage.VIETNAMESE
        if _VIETNAMESE_PATTERN.search(question)
        else DocumentLanguage.ENGLISH
    )
