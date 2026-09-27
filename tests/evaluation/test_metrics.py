from __future__ import annotations

from math import log2

import pytest

from app.evaluation.metrics import aggregate_outcomes, score_ranking, summarize_latency
from app.evaluation.models import CaseOutcome, EvaluationLatency
from app.models.versioning import DocumentLanguage, VersionSelectorKind


def outcome(
    case_id: str,
    retrieved: list[str],
    relevant: set[str],
    *,
    language: DocumentLanguage = DocumentLanguage.ENGLISH,
    selector: VersionSelectorKind = VersionSelectorKind.CURRENT,
    leaked: int = 0,
) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id,
        language=language,
        selector_kind=selector,
        tags=[],
        relevant_evidence_ids=sorted(relevant),
        retrieved_evidence_ids=retrieved,
        retrieved_chunk_ids=[f"chunk-{index}" for index, _ in enumerate(retrieved)],
        retrieved_version_ids=[
            "wrong" if index < leaked else "allowed" for index, _ in enumerate(retrieved)
        ],
        allowed_version_ids=["allowed"],
        scores=score_ranking(retrieved, relevant),
        insufficient_evidence=not relevant,
        leakage_count=leaked,
        latency=EvaluationLatency(total_ms=10),
    )


def test_ranking_metrics_match_hand_calculation() -> None:
    scores = score_ranking(["noise", "a", "b"], {"a", "b"})

    assert scores is not None
    assert scores.hit_rate == {1: 0, 3: 1, 5: 1}
    assert scores.recall == {1: 0, 3: 1, 5: 1}
    assert scores.reciprocal_rank == 0.5
    assert scores.ndcg[1] == 0
    assert scores.ndcg[3] == pytest.approx((1 / log2(3) + 1 / log2(4)) / (1 + 1 / log2(3)))


def test_aggregate_separates_insufficient_cases_and_detects_scope_leakage() -> None:
    items = [
        outcome("hit", ["a"], {"a"}),
        outcome(
            "miss",
            ["noise"],
            {"b"},
            selector=VersionSelectorKind.EXACT,
            leaked=1,
        ),
        outcome("unknown", ["noise"], set()),
    ]

    metrics = aggregate_outcomes(items)

    assert metrics.evaluated_query_count == 2
    assert metrics.insufficient_query_count == 1
    assert metrics.hit_rate[1] == 0.5
    assert metrics.recall[5] == 0.5
    assert metrics.mrr == 0.5
    assert metrics.leakage_count == 1
    assert metrics.leakage_rate == pytest.approx(1 / 3)
    assert metrics.scope_violations["exact"] == 1
    assert metrics.insufficient_empty_result_rate == 0


def test_latency_summary_reports_mean_p50_and_nearest_rank_p95() -> None:
    summary = summarize_latency([1, 2, 3, 4, 100])

    assert summary.mean_ms == 22
    assert summary.p50_ms == 3
    assert summary.p95_ms == 100
    assert summarize_latency([1, 2]).p95_ms is None
