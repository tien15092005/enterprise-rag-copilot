from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from app.core.exceptions import AppError
from app.generation.errors import CitationValidationError
from app.models.query import Citation
from app.models.retrieval import DenseRetrievalCandidate
from app.models.versioning import DocumentLanguage

_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")


class CitationScopeError(AppError):
    def __init__(self, document_version_id: UUID) -> None:
        super().__init__(
            "retrieved evidence falls outside the resolved document-version scope",
            code="citation_scope_violation",
            status_code=500,
            details={"document_version_id": str(document_version_id)},
        )


@dataclass(frozen=True, slots=True)
class CitationSource:
    candidate: DenseRetrievalCandidate
    document_title: str
    version_label: str
    source_filename: str
    effective_from: datetime | None = None
    effective_to: datetime | None = None


@dataclass(frozen=True, slots=True)
class CitationRecord:
    citation_key: str
    logical_document_id: UUID
    document_version_id: UUID
    version_label: str
    chunk_id: UUID
    source_filename: str
    page: int | None
    section_path: list[str] | None
    excerpt: str
    rank: int
    retrieval_score: float
    language: DocumentLanguage
    effective_from: datetime | None
    effective_to: datetime | None

    def to_response(self) -> Citation:
        return Citation(
            citation_key=self.citation_key,
            logical_document_id=self.logical_document_id,
            document_version_id=self.document_version_id,
            version_label=self.version_label,
            chunk_id=self.chunk_id,
            source_filename=self.source_filename,
            page=self.page,
            section_path=self.section_path,
            excerpt=self.excerpt,
            rank=self.rank,
            retrieval_score=self.retrieval_score,
            language=self.language,
            effective_from=self.effective_from,
            effective_to=self.effective_to,
        )


@dataclass(frozen=True, slots=True)
class CitationValidationResult:
    citations: tuple[CitationRecord, ...]
    referenced_keys: tuple[str, ...]
    normalized_keys: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    omitted_evidence_keys: tuple[str, ...]


class CitationFactory:
    """Create immutable citation records from evidence supplied to generation."""

    def __init__(self, excerpt_max_characters: int = 500) -> None:
        if excerpt_max_characters < 1:
            raise ValueError("excerpt_max_characters must be positive")
        self._excerpt_max_characters = excerpt_max_characters

    def create(
        self,
        source: CitationSource,
        *,
        rank: int,
        supplied_text: str,
        allowed_version_ids: set[UUID] | None = None,
    ) -> CitationRecord:
        if rank < 1:
            raise ValueError("citation rank must be positive")
        if (
            allowed_version_ids is not None
            and source.candidate.document_version_id not in allowed_version_ids
        ):
            raise CitationScopeError(source.candidate.document_version_id)
        return CitationRecord(
            citation_key=citation_key_for_rank(rank),
            logical_document_id=source.candidate.logical_document_id,
            document_version_id=source.candidate.document_version_id,
            version_label=source.version_label,
            chunk_id=source.candidate.chunk_id,
            source_filename=source.source_filename,
            page=source.candidate.page,
            section_path=source.candidate.section_path,
            excerpt=supplied_text[: self._excerpt_max_characters],
            rank=rank,
            retrieval_score=source.candidate.score,
            language=source.candidate.language,
            effective_from=source.effective_from,
            effective_to=source.effective_to,
        )


class CitationValidator:
    """Resolve model-produced keys exclusively against authoritative supplied evidence."""

    def validate(
        self, answer: str, supplied_citations: tuple[CitationRecord, ...]
    ) -> CitationValidationResult:
        by_key = {citation.citation_key: citation for citation in supplied_citations}
        referenced = tuple(_CITATION_PATTERN.findall(answer))
        normalized = tuple(dict.fromkeys(referenced))
        unknown = [key for key in normalized if key not in by_key]
        if unknown:
            raise CitationValidationError(
                "LLM referenced citation keys outside the supplied evidence",
                unknown_keys=unknown,
            )
        if not normalized:
            raise CitationValidationError("LLM answer did not include a supplied citation key")

        counts = Counter(referenced)
        duplicate_keys = tuple(key for key in normalized if counts[key] > 1)
        omitted = tuple(key for key in by_key if key not in normalized)
        return CitationValidationResult(
            citations=tuple(by_key[key] for key in normalized),
            referenced_keys=referenced,
            normalized_keys=normalized,
            duplicate_keys=duplicate_keys,
            omitted_evidence_keys=omitted,
        )


def citation_key_for_rank(rank: int) -> str:
    if rank < 1:
        raise ValueError("citation rank must be positive")
    return f"S{rank}"
