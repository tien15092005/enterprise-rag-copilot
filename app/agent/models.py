from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.query import Citation, QueryVersionSelector, TokenUsage
from app.models.retrieval import RetrievalMode
from app.models.versioning import (
    DocumentLanguage,
    VectorIndexStatus,
    VersionBusinessStatus,
    VersionIndexingStatus,
)


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    logical_document_id: UUID | None = None
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE
    max_tool_steps: int | None = Field(default=None, ge=1, le=10)
    debug: bool = False


class SearchKnowledgeInput(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    logical_document_id: UUID
    version_selector: QueryVersionSelector = Field(default_factory=QueryVersionSelector)
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE
    top_k: int = Field(default=5, ge=1, le=100)


class CompareVersionsInput(BaseModel):
    logical_document_id: UUID
    left_document_version_id: UUID
    right_document_version_id: UUID
    include_unchanged: bool = False
    summary_language: DocumentLanguage | None = None


class ListDocumentVersionsInput(BaseModel):
    logical_document_id: UUID


class GetDocumentVersionInput(BaseModel):
    document_version_id: UUID


class AgentVersion(BaseModel):
    document_version_id: UUID
    logical_document_id: UUID
    version_label: str
    revision_number: int
    version_ordinal: int
    language: DocumentLanguage
    source_filename: str
    business_status: VersionBusinessStatus
    indexing_status: VersionIndexingStatus
    vector_index_status: VectorIndexStatus
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    published_at: datetime | None = None
    is_current: bool


class AgentToolResult(BaseModel):
    result_type: str
    payload: dict[str, Any]
    citations: list[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False
    token_usage: TokenUsage | None = None


class AgentToolTrace(BaseModel):
    step: int = Field(ge=1)
    tool_name: str
    arguments: dict[str, Any]
    result_type: str
    latency_ms: float


class AgentTiming(BaseModel):
    planning_provider_ms: float
    tool_execution_ms: float
    final_generation_ms: float
    total_ms: float


class AgentDebug(BaseModel):
    tool_trace: list[AgentToolTrace]
    step_count: int
    provider_name: str
    provider_model: str


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation]
    timing: AgentTiming
    token_usage: TokenUsage | None
    debug: AgentDebug | None = None
