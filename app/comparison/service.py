from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from time import perf_counter
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.comparison.alignment import (
    AlignedSection,
    SectionAligner,
    SectionSnapshot,
    normalize_text,
)
from app.comparison.errors import ComparisonError, ComparisonVersionNotFoundError
from app.comparison.models import (
    ChangeType,
    ComparisonCitation,
    ComparisonCounts,
    ComparisonDebug,
    ComparisonItem,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonSide,
    ComparisonTiming,
    ComparisonVersion,
    SentenceDiff,
)
from app.core.config import Settings
from app.db.models import Chunk, DocumentVersion
from app.embeddings.base import EmbeddingProvider
from app.generation.citations import (
    CitationFactory,
    CitationRecord,
    CitationSource,
    CitationValidator,
)
from app.generation.errors import LLMConfigurationError
from app.generation.providers.base import LLMProvider, LLMRequest
from app.models.query import CitationValidationDebug, TokenUsage
from app.models.retrieval import DenseRetrievalCandidate
from app.models.versioning import VersionIndexingStatus

logger = logging.getLogger(__name__)
COMPARISON_PROMPT_VERSION = "version_comparison_v1"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True, slots=True)
class _CitationBinding:
    record: CitationRecord
    side: ComparisonSide
    item_order: int


class DocumentComparisonService:
    def __init__(
        self,
        session: AsyncSession,
        embeddings: EmbeddingProvider,
        llm: LLMProvider | None,
        settings: Settings,
    ) -> None:
        self._session = session
        self._llm = llm
        self._settings = settings
        self._aligner = SectionAligner(
            embeddings,
            content_threshold=settings.comparison_content_similarity_threshold,
            semantic_threshold=settings.comparison_semantic_similarity_threshold,
        )
        self._citation_factory = CitationFactory(settings.citation_excerpt_max_characters)
        self._citation_validator = CitationValidator()

    async def compare(self, request: ComparisonRequest) -> ComparisonResponse:
        total_started = perf_counter()
        load_started = perf_counter()
        left_version = await self._load_version(request.left_document_version_id)
        right_version = await self._load_version(request.right_document_version_id)
        self._validate_pair(request, left_version, right_version)
        left_sections = await self._load_sections(left_version)
        right_sections = await self._load_sections(right_version)
        version_load_ms = (perf_counter() - load_started) * 1000

        alignment_started = perf_counter()
        aligned = await self._aligner.align(left_sections, right_sections)
        alignment_ms = (perf_counter() - alignment_started) * 1000

        classification_started = perf_counter()
        all_items = [self._classify(pair, index) for index, pair in enumerate(aligned.pairs, 1)]
        counts = ComparisonCounts(**Counter(item.change_type.value for item in all_items))
        visible_items = [
            item
            for item in all_items
            if request.include_unchanged or item.change_type is not ChangeType.UNCHANGED
        ]
        bindings, visible_items = self._build_citations(
            visible_items,
            left_version,
            right_version,
            left_sections,
            right_sections,
        )
        classification_ms = (perf_counter() - classification_started) * 1000

        language = request.summary_language or left_version.language
        summary: str | None = None
        generation_ms = 0.0
        token_usage: TokenUsage | None = None
        validation_debug = CitationValidationDebug(
            status="not_applicable",
            referenced_keys=[],
            normalized_keys=[],
            duplicate_keys=[],
            omitted_evidence_keys=[],
        )
        response_bindings = bindings
        if request.generate_summary:
            if self._llm is None:
                raise LLMConfigurationError(
                    "configure LLM_PROVIDER before requesting a comparison summary"
                )
            generation_started = perf_counter()
            generated = await self._llm.generate(
                LLMRequest(
                    system_prompt=self._system_prompt(),
                    user_prompt=self._user_prompt(
                        visible_items, bindings, left_version, right_version, language.value
                    ),
                    temperature=self._settings.llm_temperature,
                    max_output_tokens=self._settings.llm_max_output_tokens,
                )
            )
            generation_ms = (perf_counter() - generation_started) * 1000
            validation = self._citation_validator.validate(
                generated.text, tuple(binding.record for binding in bindings)
            )
            cited_keys = {item.citation_key for item in validation.citations}
            response_bindings = [
                binding for binding in bindings if binding.record.citation_key in cited_keys
            ]
            validation_debug = CitationValidationDebug(
                status="validated",
                referenced_keys=list(validation.referenced_keys),
                normalized_keys=list(validation.normalized_keys),
                duplicate_keys=list(validation.duplicate_keys),
                omitted_evidence_keys=list(validation.omitted_evidence_keys),
            )
            summary = generated.text
            if generated.token_usage is not None:
                token_usage = TokenUsage(
                    prompt_tokens=generated.token_usage.prompt_tokens,
                    completion_tokens=generated.token_usage.completion_tokens,
                    total_tokens=generated.token_usage.total_tokens,
                )

        total_ms = (perf_counter() - total_started) * 1000
        method_counts = Counter(item.alignment_method for item in all_items)
        logger.info(
            "document comparison completed",
            extra={
                "event": "document_comparison_completed",
                "logical_document_id": str(request.logical_document_id),
                "left_document_version_id": str(left_version.document_version_id),
                "right_document_version_id": str(right_version.document_version_id),
                "comparison_counts": counts.model_dump(),
                "alignment_latency_ms": round(alignment_ms, 3),
                "semantic_fallback_latency_ms": round(aligned.semantic_latency_ms, 3),
                "generation_latency_ms": round(generation_ms, 3),
                "total_latency_ms": round(total_ms, 3),
            },
        )
        return ComparisonResponse(
            left_version=self._version_response(left_version),
            right_version=self._version_response(right_version),
            items=visible_items,
            counts=counts,
            summary=summary,
            summary_language=language,
            citations=[
                ComparisonCitation(
                    side=binding.side,
                    item_order=binding.item_order,
                    citation=binding.record.to_response(),
                )
                for binding in response_bindings
            ],
            timing=ComparisonTiming(
                version_load_ms=version_load_ms,
                alignment_ms=alignment_ms,
                semantic_fallback_ms=aligned.semantic_latency_ms,
                classification_diff_ms=classification_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
            token_usage=token_usage,
            prompt_version=COMPARISON_PROMPT_VERSION,
            debug=(
                ComparisonDebug(
                    alignment_method_counts=dict(method_counts),
                    unmatched_left_count=sum(item.right_text is None for item in all_items),
                    unmatched_right_count=sum(item.left_text is None for item in all_items),
                    citation_validation=validation_debug,
                )
                if request.debug
                else None
            ),
        )

    async def _load_version(self, version_id: UUID) -> DocumentVersion:
        version = await self._session.scalar(
            select(DocumentVersion)
            .where(DocumentVersion.document_version_id == version_id)
            .options(joinedload(DocumentVersion.logical_document))
        )
        if version is None:
            raise ComparisonVersionNotFoundError(version_id)
        return version

    @staticmethod
    def _validate_pair(
        request: ComparisonRequest, left: DocumentVersion, right: DocumentVersion
    ) -> None:
        if left.document_version_id == right.document_version_id:
            raise ComparisonError("comparison versions must be different")
        if left.logical_document_id != right.logical_document_id:
            raise ComparisonError(
                "comparison versions must belong to the same logical document",
                code="comparison_logical_document_mismatch",
            )
        if left.logical_document_id != request.logical_document_id:
            raise ComparisonError(
                "comparison versions do not belong to logical_document_id",
                code="comparison_logical_document_mismatch",
            )
        if left.language is not right.language:
            raise ComparisonError(
                "cross-language version comparison is not supported",
                code="comparison_language_mismatch",
            )
        if (
            left.indexing_status is not VersionIndexingStatus.READY
            or right.indexing_status is not VersionIndexingStatus.READY
        ):
            raise ComparisonError("both versions must be ingestion-ready")

    async def _load_sections(self, version: DocumentVersion) -> list[SectionSnapshot]:
        chunks = list(
            await self._session.scalars(
                select(Chunk)
                .where(Chunk.document_version_id == version.document_version_id)
                .order_by(Chunk.ordinal)
            )
        )
        groups: dict[tuple[str, ...], list[Chunk]] = {}
        for chunk in chunks:
            key = tuple(chunk.section_path or [f"__chunk_{chunk.ordinal}"])
            groups.setdefault(key, []).append(chunk)
        if len(groups) > self._settings.comparison_max_sections:
            raise ComparisonError("document exceeds configured comparison section limit")
        return [
            SectionSnapshot(
                document_version_id=version.document_version_id,
                logical_document_id=version.logical_document_id,
                chunk_ids=tuple(chunk.chunk_id for chunk in section_chunks),
                citation_chunk_id=section_chunks[0].chunk_id,
                citation_text=section_chunks[0].text,
                text="\n".join(chunk.text for chunk in section_chunks),
                section_path=(None if key[0].startswith("__chunk_") else key),
                section_fingerprint=section_chunks[0].section_fingerprint,
                ordinal=section_chunks[0].ordinal,
                page=section_chunks[0].page,
                language=section_chunks[0].language,
            )
            for key, section_chunks in groups.items()
        ]

    @staticmethod
    def _classify(pair: AlignedSection, order: int) -> ComparisonItem:
        if pair.left is None:
            change_type = ChangeType.ADDED
        elif pair.right is None:
            change_type = ChangeType.REMOVED
        elif normalize_text(pair.left.text) == normalize_text(pair.right.text):
            change_type = ChangeType.UNCHANGED
        else:
            change_type = ChangeType.MODIFIED
        return ComparisonItem(
            order=order,
            change_type=change_type,
            left_chunk_ids=list(pair.left.chunk_ids) if pair.left else [],
            right_chunk_ids=list(pair.right.chunk_ids) if pair.right else [],
            left_section_path=list(pair.left.section_path)
            if pair.left and pair.left.section_path
            else None,
            right_section_path=(
                list(pair.right.section_path) if pair.right and pair.right.section_path else None
            ),
            left_text=pair.left.text if pair.left else None,
            right_text=pair.right.text if pair.right else None,
            alignment_method=pair.method,
            alignment_score=pair.score,
            diff=(
                sentence_diff(pair.left.text, pair.right.text)
                if change_type is ChangeType.MODIFIED and pair.left and pair.right
                else None
            ),
            citation_keys=[],
        )

    def _build_citations(
        self,
        items: list[ComparisonItem],
        left_version: DocumentVersion,
        right_version: DocumentVersion,
        left_sections: list[SectionSnapshot],
        right_sections: list[SectionSnapshot],
    ) -> tuple[list[_CitationBinding], list[ComparisonItem]]:
        bindings: list[_CitationBinding] = []
        updated: list[ComparisonItem] = []
        evidence_by_chunk_id = {
            section.citation_chunk_id: (section.citation_text, section.page)
            for section in (*left_sections, *right_sections)
        }
        rank = 1
        for item in items:
            keys: list[str] = []
            for side, version, chunk_ids, text, section_path in (
                (
                    ComparisonSide.LEFT,
                    left_version,
                    item.left_chunk_ids,
                    item.left_text,
                    item.left_section_path,
                ),
                (
                    ComparisonSide.RIGHT,
                    right_version,
                    item.right_chunk_ids,
                    item.right_text,
                    item.right_section_path,
                ),
            ):
                if not chunk_ids or text is None:
                    continue
                citation_text, citation_page = evidence_by_chunk_id[chunk_ids[0]]
                candidate = DenseRetrievalCandidate(
                    chunk_id=chunk_ids[0],
                    document_version_id=version.document_version_id,
                    logical_document_id=version.logical_document_id,
                    text=citation_text,
                    score=item.alignment_score or 1.0,
                    ordinal=item.order,
                    page=citation_page,
                    section_path=section_path,
                    language=version.language,
                )
                record = replace(
                    self._citation_factory.create(
                        CitationSource(
                            candidate=candidate,
                            document_title=version.logical_document.title,
                            version_label=version.version_label,
                            source_filename=version.source_filename,
                            effective_from=version.effective_from,
                            effective_to=version.effective_to,
                        ),
                        rank=rank,
                        supplied_text=citation_text,
                        allowed_version_ids={
                            left_version.document_version_id,
                            right_version.document_version_id,
                        },
                    ),
                    dense_rank=None,
                    dense_score=None,
                )
                bindings.append(_CitationBinding(record, side, item.order))
                keys.append(record.citation_key)
                rank += 1
            updated.append(item.model_copy(update={"citation_keys": keys}))
        return bindings, updated

    @staticmethod
    def _version_response(version: DocumentVersion) -> ComparisonVersion:
        return ComparisonVersion(
            document_version_id=version.document_version_id,
            logical_document_id=version.logical_document_id,
            version_label=version.version_label,
            language=version.language,
            source_filename=version.source_filename,
            effective_from=version.effective_from,
            effective_to=version.effective_to,
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Summarize only the supplied deterministic comparison items. "
            "Do not invent changes. Cite factual change claims using only supplied [S#] keys. "
            "Modified claims should cite both left and right evidence."
        )

    @staticmethod
    def _user_prompt(
        items: list[ComparisonItem],
        bindings: list[_CitationBinding],
        left: DocumentVersion,
        right: DocumentVersion,
        language: str,
    ) -> str:
        by_item: dict[int, list[_CitationBinding]] = {}
        for binding in bindings:
            by_item.setdefault(binding.item_order, []).append(binding)
        blocks = []
        for item in items:
            citations = " ".join(
                f"{binding.side.value}=[{binding.record.citation_key}]"
                for binding in by_item.get(item.order, [])
            )
            blocks.append(
                f"Item {item.order}: {item.change_type.value}; {citations}\n"
                f"Left: {item.left_text or '<missing>'}\n"
                f"Right: {item.right_text or '<missing>'}"
            )
        return (
            f"Compare version {left.version_label} with {right.version_label}.\n"
            f"Requested summary language: {language}.\n\n" + "\n\n".join(blocks)
        )


def sentence_diff(left: str, right: str) -> SentenceDiff:
    left_sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(left) if item.strip()]
    right_sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(right) if item.strip()]
    removed: list[str] = []
    added: list[str] = []
    matcher = SequenceMatcher(None, left_sentences, right_sentences, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            removed.extend(left_sentences[left_start:left_end])
        if tag in {"insert", "replace"}:
            added.extend(right_sentences[right_start:right_end])
    return SentenceDiff(removed=removed, added=added)
