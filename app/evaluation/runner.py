from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.evaluation.metrics import aggregate_outcomes, score_ranking, summarize_latency
from app.evaluation.models import (
    CaseOutcome,
    EvaluationCase,
    EvaluationDataset,
    EvaluationLatency,
    ExperimentResult,
)
from app.models.query import QueryVersionSelector, ResolvedScope
from app.models.retrieval import (
    DenseRetrievalBatch,
    HybridRetrievalBatch,
    RerankedRetrievalBatch,
    RetrievalBatch,
    RetrievalMode,
)
from app.models.versioning import DocumentLanguage, VersionSelectorKind
from app.retrieval.hybrid import ModeAwareRetrievalService
from app.retrieval.version_scope import VersionScopeResolver


@dataclass(frozen=True, slots=True)
class EvaluationCatalog:
    document_ids: dict[str, UUID]
    version_ids: dict[tuple[str, str], UUID]
    evidence_chunk_ids: dict[str, UUID]


class RetrievalEvaluationRunner:
    def __init__(
        self,
        dataset: EvaluationDataset,
        catalog: EvaluationCatalog,
        scope_resolver: VersionScopeResolver,
        retrieval: ModeAwareRetrievalService,
        *,
        ks: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        self._dataset = dataset
        self._catalog = catalog
        self._scope_resolver = scope_resolver
        self._retrieval = retrieval
        self._ks = ks
        self._evidence_by_chunk = {
            chunk_id: evidence_id for evidence_id, chunk_id in catalog.evidence_chunk_ids.items()
        }

    async def run(
        self,
        mode: RetrievalMode,
        *,
        profile: str,
        configuration: dict[str, int],
        top_k: int,
    ) -> ExperimentResult:
        outcomes = [await self._run_case(case, mode, top_k) for case in self._dataset.cases]
        metrics = {
            "en": aggregate_outcomes(
                (item for item in outcomes if item.language is DocumentLanguage.ENGLISH),
                ks=self._ks,
            ),
            "vi": aggregate_outcomes(
                (item for item in outcomes if item.language is DocumentLanguage.VIETNAMESE),
                ks=self._ks,
            ),
            "combined": aggregate_outcomes(outcomes, ks=self._ks),
        }
        latency = {
            component: summarize_latency(
                [getattr(outcome.latency, f"{component}_ms") for outcome in outcomes]
            )
            for component in ("dense", "lexical", "fusion", "reranking", "total")
        }
        return ExperimentResult(
            retrieval_mode=mode,
            profile=profile,
            configuration=configuration,
            outcomes=outcomes,
            metrics=metrics,
            latency=latency,
        )

    async def cold_probe(self, case: EvaluationCase, mode: RetrievalMode, top_k: int) -> float:
        scope = await self._scope(case)
        result = await self._retrieval.search_versions(
            case.query,
            [version.document_version_id for version in scope.versions],
            top_k,
            mode,
        )
        return result.total_latency_ms

    async def _run_case(self, case: EvaluationCase, mode: RetrievalMode, top_k: int) -> CaseOutcome:
        scope = await self._scope(case)
        allowed = [version.document_version_id for version in scope.versions]
        result = await self._retrieval.search_versions(case.query, allowed, top_k, mode)
        retrieved_identities = [
            self._evidence_by_chunk.get(candidate.chunk_id, f"chunk:{candidate.chunk_id}")
            for candidate in result.candidates
        ]
        retrieved_versions = [candidate.document_version_id for candidate in result.candidates]
        allowed_set = set(allowed)
        leakage_count = sum(version_id not in allowed_set for version_id in retrieved_versions)
        return CaseOutcome(
            case_id=case.id,
            language=case.language,
            selector_kind=case.selector.kind,
            tags=case.tags,
            relevant_evidence_ids=case.relevant,
            retrieved_evidence_ids=retrieved_identities,
            retrieved_chunk_ids=[str(candidate.chunk_id) for candidate in result.candidates],
            retrieved_version_ids=[str(item) for item in retrieved_versions],
            allowed_version_ids=[str(item) for item in allowed],
            scores=score_ranking(retrieved_identities, set(case.relevant), ks=self._ks),
            insufficient_evidence=not case.relevant,
            leakage_count=leakage_count,
            latency=self._latency(result),
        )

    async def _scope(self, case: EvaluationCase) -> ResolvedScope:
        document_id = self._catalog.document_ids[case.document]
        selector = case.selector
        request_selector = QueryVersionSelector(
            kind=selector.kind,
            document_version_id=(
                self._catalog.version_ids[(case.document, selector.version)]
                if selector.kind is VersionSelectorKind.EXACT and selector.version is not None
                else None
            ),
            as_of=selector.as_of,
        )
        return await self._scope_resolver.resolve(document_id, request_selector)

    @staticmethod
    def _latency(result: RetrievalBatch) -> EvaluationLatency:
        if isinstance(result, RerankedRetrievalBatch):
            return EvaluationLatency(
                dense_ms=result.dense_latency_ms,
                lexical_ms=result.lexical_latency_ms,
                fusion_ms=result.fusion_latency_ms,
                reranking_ms=result.reranking_latency_ms,
                total_ms=result.total_latency_ms,
            )
        if isinstance(result, HybridRetrievalBatch):
            return EvaluationLatency(
                dense_ms=result.dense_latency_ms,
                lexical_ms=result.lexical_latency_ms,
                fusion_ms=result.fusion_latency_ms,
                total_ms=result.total_latency_ms,
            )
        assert isinstance(result, DenseRetrievalBatch)
        return EvaluationLatency(
            dense_ms=result.total_latency_ms,
            total_ms=result.total_latency_ms,
        )
