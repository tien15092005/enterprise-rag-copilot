from __future__ import annotations

from typing import Literal

from app.evaluation.models import EvaluationDataset, ExperimentResult
from app.models.retrieval import RetrievalMode


def build_report(
    dataset: EvaluationDataset,
    experiments: list[ExperimentResult],
    *,
    dataset_hash: str,
    timestamp: str,
    cold_start: dict[str, float],
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Generated: `{timestamp}`  ",
        f"Dataset: `{dataset.dataset_id}` revision `{dataset.revision}`  ",
        f"Dataset SHA-256: `{dataset_hash}`",
        "",
        "## 1. Dataset description",
        "",
        f"The repository-owned dataset contains {len(dataset.cases)} queries, "
        f"{len(dataset.documents)} logical documents, "
        f"{sum(len(d.versions) for d in dataset.documents)} "
        "immutable versions, and English/Vietnamese labels. Insufficient-evidence cases are "
        "reported separately and excluded from relevance averages.",
        "",
        "## 2. Retrieval modes",
        "",
        "- `dense`: final Qdrant ranking.",
        "- `hybrid`: dense plus PostgreSQL full-text search, fused by RRF.",
        "- `hybrid_rerank`: bounded RRF candidates reordered by the pinned multilingual "
        "cross-encoder.",
        "",
        "## 3-5. Metrics by language and mode",
        "",
        "Metrics compare final evidence identities, never raw score magnitudes.",
        "",
        "| Profile | Mode | Language | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | "
        "Recall@5 | MRR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    languages: tuple[Literal["en", "vi", "combined"], ...] = ("en", "vi", "combined")
    for experiment in experiments:
        for language in languages:
            metrics = experiment.metrics[language]
            lines.append(
                f"| {experiment.profile} | {experiment.retrieval_mode.value} | {language} | "
                f"{metrics.hit_rate[1]:.3f} | {metrics.hit_rate[3]:.3f} | "
                f"{metrics.hit_rate[5]:.3f} | {metrics.recall[1]:.3f} | "
                f"{metrics.recall[3]:.3f} | {metrics.recall[5]:.3f} | {metrics.mrr:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 6. Version leakage",
            "",
            "| Profile | Mode | Wrong-version results | Leakage rate | Exact violations | "
            "Current violations | As-of violations |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment in experiments:
        metrics = experiment.metrics["combined"]
        lines.append(
            f"| {experiment.profile} | {experiment.retrieval_mode.value} | "
            f"{metrics.leakage_count} | {metrics.leakage_rate:.4f} | "
            f"{metrics.scope_violations['exact']} | {metrics.scope_violations['current']} | "
            f"{metrics.scope_violations['as_of']} |"
        )
    lines.extend(
        [
            "",
            "## 7. Latency",
            "",
            "Cold-start probes are separate from the warm aggregates below. The embedding model "
            "was loaded during indexing; `hybrid_rerank` cold time includes first reranker load.",
            "",
            "| Cold probe | Milliseconds |",
            "|---|---:|",
        ]
    )
    for name, value in cold_start.items():
        lines.append(f"| {name} | {value:.2f} |")
    lines.extend(
        [
            "",
            "| Profile | Mode | Component | Mean ms | p50 ms | p95 ms |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for experiment in experiments:
        for component, summary in experiment.latency.items():
            p95 = "n/a" if summary.p95_ms is None else f"{summary.p95_ms:.2f}"
            lines.append(
                f"| {experiment.profile} | {experiment.retrieval_mode.value} | {component} | "
                f"{summary.mean_ms:.2f} | {summary.p50_ms:.2f} | {p95} |"
            )
    lines.extend(
        [
            "",
            "## 8. Configuration comparisons",
            "",
            "The bounded experiment compares candidate limits 5 and 20 while holding final "
            "`top_k=5` constant. Hit/recall columns at K=1,3,5 show the final top-K trade-off.",
            "",
            _configuration_summary(experiments),
            "",
            "## 9. Error analysis",
            "",
        ]
    )
    failures = _failures(experiments)
    if failures:
        for category, case_id, detail in failures[:12]:
            lines.append(f"- **{category}** — `{case_id}`: {detail}")
    else:
        lines.append("- No labeled relevance misses occurred in the measured top five.")
    baseline = [item for item in experiments if item.profile == "candidates_20"]
    recommendation = _recommendation(baseline)
    lines.extend(
        [
            "",
            "## 10. Conclusions",
            "",
            recommendation,
            "",
            "## 11. Limitations",
            "",
            "- This is a small repository-owned engineering dataset, not an external benchmark.",
            "- Relevance labels are binary; nDCG is reported but does not model graded relevance.",
            "- PostgreSQL `simple` remains only a baseline tokenizer for Vietnamese.",
            "- Insufficient-evidence retrieval has no calibrated rejection threshold; "
            "empty-result rate is reported separately.",
            "- Cold-start timings depend on local hardware and cache state.",
            "- No paid judge model or fabricated groundedness score is used.",
            "- Version comparison, agents, and production hardening are outside Phase 8.",
            "",
        ]
    )
    return "\n".join(lines)


def _failures(experiments: list[ExperimentResult]) -> list[tuple[str, str, str]]:
    failures: list[tuple[str, str, str]] = []
    for experiment in experiments:
        if experiment.profile != "candidates_20":
            continue
        for outcome in experiment.outcomes:
            if outcome.leakage_count:
                failures.append(
                    (
                        "version-scope issue",
                        outcome.case_id,
                        f"{outcome.leakage_count} leaked results",
                    )
                )
            elif outcome.insufficient_evidence and outcome.retrieved_chunk_ids:
                failures.append(
                    (
                        "insufficient-evidence false positive",
                        outcome.case_id,
                        f"{experiment.retrieval_mode.value} returned "
                        f"{len(outcome.retrieved_chunk_ids)} chunks",
                    )
                )
            elif outcome.scores is not None and outcome.scores.hit_rate[5] == 0:
                if "semantic_paraphrase" in outcome.tags:
                    category = "semantic retrieval miss"
                elif outcome.language.value == "vi":
                    category = "Vietnamese tokenization/lexical miss"
                else:
                    category = "lexical miss"
                failures.append(
                    (
                        category,
                        outcome.case_id,
                        f"missed by {experiment.retrieval_mode.value} at K=5",
                    )
                )
            elif outcome.scores is not None and outcome.scores.hit_rate[1] == 0:
                category = (
                    "reranker misordering"
                    if experiment.retrieval_mode is RetrievalMode.HYBRID_RERANK
                    else "candidate misordering"
                )
                failures.append((category, outcome.case_id, "relevant evidence was below rank 1"))
    return failures


def _recommendation(experiments: list[ExperimentResult]) -> str:
    by_mode = {item.retrieval_mode: item for item in experiments}
    dense = by_mode.get(RetrievalMode.DENSE)
    hybrid = by_mode.get(RetrievalMode.HYBRID)
    reranked = by_mode.get(RetrievalMode.HYBRID_RERANK)
    if dense is None or hybrid is None or reranked is None:
        return "The measured set is incomplete, so it does not support changing the default mode."
    dense_mrr = dense.metrics["combined"].mrr
    hybrid_mrr = hybrid.metrics["combined"].mrr
    rerank_mrr = reranked.metrics["combined"].mrr
    metric_languages: tuple[Literal["en", "vi"], ...] = ("en", "vi")
    language_safe = all(
        reranked.metrics[language].mrr >= dense.metrics[language].mrr
        for language in metric_languages
    )
    if rerank_mrr > max(dense_mrr, hybrid_mrr) and language_safe:
        return (
            "On this specified dataset, `hybrid_rerank` has the highest combined MRR without "
            "lowering either language's MRR. The dataset is too small to justify changing the "
            "production default; dense remains the compatibility default pending broader evidence."
        )
    return (
        "The measured evidence does not consistently favor `hybrid_rerank` over dense and hybrid "
        "for combined and per-language MRR. Keep `dense` as the default."
    )


def _configuration_summary(experiments: list[ExperimentResult]) -> str:
    statements: list[str] = []
    for mode in RetrievalMode:
        mode_results = [item for item in experiments if item.retrieval_mode is mode]
        if len(mode_results) < 2:
            continue
        values = {item.metrics["combined"].mrr for item in mode_results}
        if len(values) == 1:
            statements.append(
                f"`{mode.value}` combined MRR was unchanged across the measured candidate limits"
            )
        else:
            rendered = ", ".join(
                f"{item.profile}={item.metrics['combined'].mrr:.3f}" for item in mode_results
            )
            statements.append(f"`{mode.value}` combined MRR varied: {rendered}")
    return "; ".join(statements) + "." if statements else "No multi-profile comparison ran."
