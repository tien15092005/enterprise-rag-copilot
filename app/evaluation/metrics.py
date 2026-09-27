from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import mean, median

from app.evaluation.models import (
    CaseOutcome,
    LatencySummary,
    MetricSummary,
    RankingScores,
)
from app.models.versioning import VersionSelectorKind


def score_ranking(
    retrieved: list[str], relevant: set[str], *, ks: tuple[int, ...] = (1, 3, 5)
) -> RankingScores | None:
    """Score final evidence identities; score magnitudes are intentionally ignored."""

    if not relevant:
        return None
    first_rank = next(
        (rank for rank, evidence_id in enumerate(retrieved, start=1) if evidence_id in relevant),
        None,
    )
    hit_rate: dict[int, float] = {}
    recall: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    for k in ks:
        prefix = retrieved[:k]
        relevant_count = sum(item in relevant for item in prefix)
        hit_rate[k] = float(relevant_count > 0)
        recall[k] = relevant_count / len(relevant)
        dcg = sum(
            1 / math.log2(rank + 1) for rank, item in enumerate(prefix, start=1) if item in relevant
        )
        ideal_count = min(k, len(relevant))
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg[k] = dcg / ideal if ideal else 0
    return RankingScores(
        hit_rate=hit_rate,
        recall=recall,
        ndcg=ndcg,
        reciprocal_rank=0 if first_rank is None else 1 / first_rank,
    )


def aggregate_outcomes(
    outcomes: Iterable[CaseOutcome], *, ks: tuple[int, ...] = (1, 3, 5)
) -> MetricSummary:
    items = list(outcomes)
    scored = [item.scores for item in items if item.scores is not None]
    insufficient = [item for item in items if item.insufficient_evidence]
    retrieved_count = sum(len(item.retrieved_chunk_ids) for item in items)
    leakage_count = sum(item.leakage_count for item in items)
    scope_violations = {
        kind.value: sum(item.leakage_count for item in items if item.selector_kind is kind)
        for kind in (
            VersionSelectorKind.EXACT,
            VersionSelectorKind.CURRENT,
            VersionSelectorKind.AS_OF,
        )
    }
    return MetricSummary(
        evaluated_query_count=len(scored),
        insufficient_query_count=len(insufficient),
        hit_rate={k: mean(scores.hit_rate[k] for scores in scored) if scored else 0 for k in ks},
        recall={k: mean(scores.recall[k] for scores in scored) if scored else 0 for k in ks},
        ndcg={k: mean(scores.ndcg[k] for scores in scored) if scored else 0 for k in ks},
        mrr=mean(scores.reciprocal_rank for scores in scored) if scored else 0,
        leakage_count=leakage_count,
        retrieved_result_count=retrieved_count,
        leakage_rate=leakage_count / retrieved_count if retrieved_count else 0,
        scope_violations=scope_violations,
        insufficient_empty_result_rate=(
            mean(not item.retrieved_chunk_ids for item in insufficient) if insufficient else 0
        ),
    )


def summarize_latency(values: list[float]) -> LatencySummary:
    if not values:
        return LatencySummary(samples=0, mean_ms=0, p50_ms=0, p95_ms=None)
    ordered = sorted(values)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1] if len(ordered) >= 5 else None
    return LatencySummary(
        samples=len(values),
        mean_ms=mean(values),
        p50_ms=median(values),
        p95_ms=p95,
    )
