from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.models.retrieval import RetrievalMode
from app.models.versioning import DocumentLanguage, VersionSelectorKind


class QueryVersionSelector(BaseModel):
    kind: VersionSelectorKind = VersionSelectorKind.CURRENT
    document_version_id: UUID | None = None
    as_of: datetime | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> QueryVersionSelector:
        if self.kind is VersionSelectorKind.EXACT and self.document_version_id is None:
            raise ValueError("exact selector requires document_version_id")
        if self.kind is not VersionSelectorKind.EXACT and self.document_version_id is not None:
            raise ValueError("document_version_id is valid only for exact selector")
        if self.kind is VersionSelectorKind.AS_OF and self.as_of is None:
            raise ValueError("as_of selector requires as_of")
        if self.kind is not VersionSelectorKind.AS_OF and self.as_of is not None:
            raise ValueError("as_of is valid only for as_of selector")
        return self


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    logical_document_id: UUID | None = None
    version_selector: QueryVersionSelector = Field(default_factory=QueryVersionSelector)
    top_k: int = Field(default=5, ge=1, le=100)
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE
    retrieval_only: bool = False
    debug: bool = False


class ResolvedVersion(BaseModel):
    document_version_id: UUID
    version_label: str
    effective_from: datetime | None
    effective_to: datetime | None


class ResolvedScope(BaseModel):
    kind: VersionSelectorKind
    logical_document_id: UUID
    versions: list[ResolvedVersion]


class Citation(BaseModel):
    citation_key: str
    logical_document_id: UUID
    document_version_id: UUID
    version_label: str
    chunk_id: UUID
    source_filename: str
    page: int | None
    section_path: list[str] | None
    excerpt: str
    rank: int = Field(ge=1)
    retrieval_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    language: DocumentLanguage
    effective_from: datetime | None
    effective_to: datetime | None


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class QueryTiming(BaseModel):
    scope_resolution_ms: float
    query_embedding_ms: float
    retrieval_ms: float
    lexical_retrieval_ms: float = 0
    fusion_ms: float = 0
    context_construction_ms: float
    generation_ms: float
    total_ms: float


class CitationValidationDebug(BaseModel):
    status: Literal["validated", "not_applicable"]
    referenced_keys: list[str]
    normalized_keys: list[str]
    duplicate_keys: list[str]
    omitted_evidence_keys: list[str]


class QueryDebug(BaseModel):
    evidence: list[Citation]
    citation_validation: CitationValidationDebug


class QueryResponse(BaseModel):
    answer: str | None
    answer_language: DocumentLanguage
    citations: list[Citation]
    resolved_scope: ResolvedScope
    timing: QueryTiming
    token_usage: TokenUsage | None
    prompt_version: str
    insufficient_evidence: bool
    debug: QueryDebug | None = None
