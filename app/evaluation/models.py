from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.models.retrieval import RetrievalMode
from app.models.versioning import DocumentLanguage, VersionSelectorKind


class EvaluationChunking(BaseModel):
    chunk_size: int = Field(ge=1)
    chunk_overlap: int = Field(ge=0)


class EvaluationVersion(BaseModel):
    label: str
    ordinal: int = Field(ge=1)
    path: str
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    active: bool = False


class EvaluationDocument(BaseModel):
    stable_id: str
    title: str
    language: DocumentLanguage
    versions: list[EvaluationVersion] = Field(min_length=1)


class EvidenceLabel(BaseModel):
    id: str
    document: str
    version: str
    text_contains: str = Field(min_length=1)


class EvaluationSelector(BaseModel):
    kind: VersionSelectorKind
    version: str | None = None
    as_of: datetime | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> EvaluationSelector:
        if self.kind is VersionSelectorKind.EXACT and self.version is None:
            raise ValueError("exact evaluation selector requires version")
        if self.kind is not VersionSelectorKind.EXACT and self.version is not None:
            raise ValueError("version is valid only for exact selectors")
        if self.kind is VersionSelectorKind.AS_OF and self.as_of is None:
            raise ValueError("as_of evaluation selector requires a timestamp")
        if self.kind is not VersionSelectorKind.AS_OF and self.as_of is not None:
            raise ValueError("as_of is valid only for as_of selectors")
        return self


class EvaluationCase(BaseModel):
    id: str
    query: str = Field(min_length=1)
    language: DocumentLanguage
    document: str
    selector: EvaluationSelector
    relevant: list[str]
    expected_answer: str | None = None
    tags: list[str]


class EvaluationDataset(BaseModel):
    dataset_id: str
    revision: str
    description: str
    chunking: EvaluationChunking
    documents: list[EvaluationDocument] = Field(min_length=1)
    evidence: list[EvidenceLabel] = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> EvaluationDataset:
        documents = {item.stable_id: item for item in self.documents}
        evidence_ids = {item.id for item in self.evidence}
        if len(documents) != len(self.documents):
            raise ValueError("document stable IDs must be unique")
        if len(evidence_ids) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        for evidence in self.evidence:
            document = documents.get(evidence.document)
            if document is None:
                raise ValueError(f"unknown evidence document: {evidence.document}")
            if evidence.version not in {version.label for version in document.versions}:
                raise ValueError(f"unknown evidence version: {evidence.id}")
        case_ids: set[str] = set()
        for case in self.cases:
            if case.id in case_ids:
                raise ValueError(f"duplicate case ID: {case.id}")
            case_ids.add(case.id)
            if case.document not in documents:
                raise ValueError(f"unknown case document: {case.document}")
            unknown = set(case.relevant) - evidence_ids
            if unknown:
                raise ValueError(f"case references unknown evidence: {sorted(unknown)}")
        return self


class RankingScores(BaseModel):
    hit_rate: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]
    reciprocal_rank: float


class EvaluationLatency(BaseModel):
    dense_ms: float = 0
    lexical_ms: float = 0
    fusion_ms: float = 0
    reranking_ms: float = 0
    total_ms: float = 0


class CaseOutcome(BaseModel):
    case_id: str
    language: DocumentLanguage
    selector_kind: VersionSelectorKind
    tags: list[str]
    relevant_evidence_ids: list[str]
    retrieved_evidence_ids: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_version_ids: list[str]
    allowed_version_ids: list[str]
    scores: RankingScores | None
    insufficient_evidence: bool
    leakage_count: int
    latency: EvaluationLatency


class MetricSummary(BaseModel):
    evaluated_query_count: int
    insufficient_query_count: int
    hit_rate: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]
    mrr: float
    leakage_count: int
    retrieved_result_count: int
    leakage_rate: float
    scope_violations: dict[str, int]
    insufficient_empty_result_rate: float


class LatencySummary(BaseModel):
    samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float | None


class ExperimentResult(BaseModel):
    retrieval_mode: RetrievalMode
    profile: str
    configuration: dict[str, int]
    outcomes: list[CaseOutcome]
    metrics: dict[Literal["en", "vi", "combined"], MetricSummary]
    latency: dict[str, LatencySummary]
