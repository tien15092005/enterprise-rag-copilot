from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.models.query import Citation, CitationValidationDebug, TokenUsage
from app.models.versioning import DocumentLanguage


class ChangeType(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class AlignmentMethod(StrEnum):
    EXACT_PATH = "exact_path"
    NORMALIZED_PATH = "normalized_path"
    EXACT_HEADING = "exact_heading"
    NORMALIZED_HEADING = "normalized_heading"
    SECTION_FINGERPRINT = "section_fingerprint"
    CONTENT_SIMILARITY = "content_similarity"
    SEMANTIC = "semantic"
    UNALIGNED = "unaligned"


class ComparisonSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class ComparisonRequest(BaseModel):
    logical_document_id: UUID
    left_document_version_id: UUID
    right_document_version_id: UUID
    include_unchanged: bool = False
    generate_summary: bool = False
    summary_language: DocumentLanguage | None = None
    debug: bool = False

    @model_validator(mode="after")
    def validate_pair(self) -> ComparisonRequest:
        if self.left_document_version_id == self.right_document_version_id:
            raise ValueError("comparison versions must be different")
        return self


class ComparisonVersion(BaseModel):
    document_version_id: UUID
    logical_document_id: UUID
    version_label: str
    language: DocumentLanguage
    source_filename: str
    effective_from: datetime | None
    effective_to: datetime | None


class SentenceDiff(BaseModel):
    removed: list[str]
    added: list[str]


class ComparisonItem(BaseModel):
    order: int = Field(ge=1)
    change_type: ChangeType
    left_chunk_ids: list[UUID]
    right_chunk_ids: list[UUID]
    left_section_path: list[str] | None
    right_section_path: list[str] | None
    left_text: str | None
    right_text: str | None
    alignment_method: AlignmentMethod
    alignment_score: float | None
    diff: SentenceDiff | None
    citation_keys: list[str]


class ComparisonCounts(BaseModel):
    added: int = 0
    removed: int = 0
    modified: int = 0
    unchanged: int = 0


class ComparisonCitation(BaseModel):
    side: ComparisonSide
    item_order: int
    citation: Citation


class ComparisonTiming(BaseModel):
    version_load_ms: float
    alignment_ms: float
    semantic_fallback_ms: float
    classification_diff_ms: float
    generation_ms: float
    total_ms: float


class ComparisonDebug(BaseModel):
    alignment_method_counts: dict[AlignmentMethod, int]
    unmatched_left_count: int
    unmatched_right_count: int
    citation_validation: CitationValidationDebug


class ComparisonResponse(BaseModel):
    left_version: ComparisonVersion
    right_version: ComparisonVersion
    items: list[ComparisonItem]
    counts: ComparisonCounts
    summary: str | None
    summary_language: DocumentLanguage
    citations: list[ComparisonCitation]
    timing: ComparisonTiming
    token_usage: TokenUsage | None
    prompt_version: str
    debug: ComparisonDebug | None = None


CitationValidationStatus = Literal["validated", "not_applicable"]
