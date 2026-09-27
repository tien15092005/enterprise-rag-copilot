# Evaluation Report

Generated: `2026-09-27T15:06:46.180269+00:00`  
Dataset: `enterprise-rag-bilingual-v1` revision `1.0.0`  
Dataset SHA-256: `005ce2b8da158f5a295b6e262f6b507063d4a3e2bcc9459d8e422130e4250db5`

## 1. Dataset description

The repository-owned dataset contains 14 queries, 4 logical documents, 6 immutable versions, and English/Vietnamese labels. Insufficient-evidence cases are reported separately and excluded from relevance averages.

## 2. Retrieval modes

- `dense`: final Qdrant ranking.
- `hybrid`: dense plus PostgreSQL full-text search, fused by RRF.
- `hybrid_rerank`: bounded RRF candidates reordered by the pinned multilingual cross-encoder.

## 3-5. Metrics by language and mode

Metrics compare final evidence identities, never raw score magnitudes.

| Profile | Mode | Language | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| candidates_5 | dense | en | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| candidates_5 | dense | vi | 0.833 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.917 |
| candidates_5 | dense | combined | 0.917 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 0.958 |
| candidates_5 | hybrid | en | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| candidates_5 | hybrid | vi | 0.833 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.917 |
| candidates_5 | hybrid | combined | 0.917 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 0.958 |
| candidates_5 | hybrid_rerank | en | 0.833 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.917 |
| candidates_5 | hybrid_rerank | vi | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| candidates_5 | hybrid_rerank | combined | 0.917 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 0.958 |
| candidates_20 | dense | en | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| candidates_20 | dense | vi | 0.833 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.917 |
| candidates_20 | dense | combined | 0.917 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 0.958 |
| candidates_20 | hybrid | en | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| candidates_20 | hybrid | vi | 0.833 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.917 |
| candidates_20 | hybrid | combined | 0.917 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 0.958 |
| candidates_20 | hybrid_rerank | en | 0.833 | 1.000 | 1.000 | 0.750 | 1.000 | 1.000 | 0.917 |
| candidates_20 | hybrid_rerank | vi | 1.000 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 | 1.000 |
| candidates_20 | hybrid_rerank | combined | 0.917 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 0.958 |

## 6. Version leakage

| Profile | Mode | Wrong-version results | Leakage rate | Exact violations | Current violations | As-of violations |
|---|---|---:|---:|---:|---:|---:|
| candidates_5 | dense | 0 | 0.0000 | 0 | 0 | 0 |
| candidates_5 | hybrid | 0 | 0.0000 | 0 | 0 | 0 |
| candidates_5 | hybrid_rerank | 0 | 0.0000 | 0 | 0 | 0 |
| candidates_20 | dense | 0 | 0.0000 | 0 | 0 | 0 |
| candidates_20 | hybrid | 0 | 0.0000 | 0 | 0 | 0 |
| candidates_20 | hybrid_rerank | 0 | 0.0000 | 0 | 0 | 0 |

## 7. Latency

Cold-start probes are separate from the warm aggregates below. The embedding model was loaded during indexing; `hybrid_rerank` cold time includes first reranker load.

| Cold probe | Milliseconds |
|---|---:|
| dense | 44.73 |
| hybrid | 47.41 |
| hybrid_rerank | 6614.19 |

| Profile | Mode | Component | Mean ms | p50 ms | p95 ms |
|---|---|---|---:|---:|---:|
| candidates_5 | dense | dense | 36.07 | 36.14 | 43.96 |
| candidates_5 | dense | lexical | 0.00 | 0.00 | 0.00 |
| candidates_5 | dense | fusion | 0.00 | 0.00 | 0.00 |
| candidates_5 | dense | reranking | 0.00 | 0.00 | 0.00 |
| candidates_5 | dense | total | 36.07 | 36.14 | 43.96 |
| candidates_5 | hybrid | dense | 34.98 | 34.70 | 44.79 |
| candidates_5 | hybrid | lexical | 5.82 | 5.64 | 9.28 |
| candidates_5 | hybrid | fusion | 0.08 | 0.08 | 0.10 |
| candidates_5 | hybrid | reranking | 0.00 | 0.00 | 0.00 |
| candidates_5 | hybrid | total | 40.96 | 40.99 | 52.51 |
| candidates_5 | hybrid_rerank | dense | 34.26 | 33.09 | 44.88 |
| candidates_5 | hybrid_rerank | lexical | 4.96 | 5.15 | 6.12 |
| candidates_5 | hybrid_rerank | fusion | 0.08 | 0.07 | 0.13 |
| candidates_5 | hybrid_rerank | reranking | 33.41 | 32.02 | 48.44 |
| candidates_5 | hybrid_rerank | total | 72.85 | 71.26 | 89.95 |
| candidates_20 | dense | dense | 33.93 | 34.28 | 37.51 |
| candidates_20 | dense | lexical | 0.00 | 0.00 | 0.00 |
| candidates_20 | dense | fusion | 0.00 | 0.00 | 0.00 |
| candidates_20 | dense | reranking | 0.00 | 0.00 | 0.00 |
| candidates_20 | dense | total | 33.93 | 34.28 | 37.51 |
| candidates_20 | hybrid | dense | 33.21 | 33.27 | 37.37 |
| candidates_20 | hybrid | lexical | 4.79 | 4.77 | 5.73 |
| candidates_20 | hybrid | fusion | 0.07 | 0.08 | 0.09 |
| candidates_20 | hybrid | reranking | 0.00 | 0.00 | 0.00 |
| candidates_20 | hybrid | total | 38.14 | 38.22 | 42.49 |
| candidates_20 | hybrid_rerank | dense | 32.27 | 32.66 | 37.53 |
| candidates_20 | hybrid_rerank | lexical | 4.65 | 4.41 | 8.19 |
| candidates_20 | hybrid_rerank | fusion | 0.08 | 0.06 | 0.14 |
| candidates_20 | hybrid_rerank | reranking | 33.38 | 31.97 | 45.55 |
| candidates_20 | hybrid_rerank | total | 70.51 | 69.67 | 81.73 |

## 8. Configuration comparisons

The bounded experiment compares candidate limits 5 and 20 while holding final `top_k=5` constant. Hit/recall columns at K=1,3,5 show the final top-K trade-off.

`dense` combined MRR was unchanged across the measured candidate limits; `hybrid` combined MRR was unchanged across the measured candidate limits; `hybrid_rerank` combined MRR was unchanged across the measured candidate limits.

## 9. Error analysis

- **insufficient-evidence false positive** — `en-insufficient`: dense returned 3 chunks
- **candidate misordering** — `vi-asof-code`: relevant evidence was below rank 1
- **insufficient-evidence false positive** — `vi-insufficient`: dense returned 3 chunks
- **insufficient-evidence false positive** — `en-insufficient`: hybrid returned 3 chunks
- **candidate misordering** — `vi-asof-code`: relevant evidence was below rank 1
- **insufficient-evidence false positive** — `vi-insufficient`: hybrid returned 3 chunks
- **reranker misordering** — `en-current-remote`: relevant evidence was below rank 1
- **insufficient-evidence false positive** — `en-insufficient`: hybrid_rerank returned 3 chunks
- **insufficient-evidence false positive** — `vi-insufficient`: hybrid_rerank returned 3 chunks

## 10. Conclusions

The measured evidence does not consistently favor `hybrid_rerank` over dense and hybrid for combined and per-language MRR. Keep `dense` as the default.

## 11. Limitations

- This is a small repository-owned engineering dataset, not an external benchmark.
- Relevance labels are binary; nDCG is reported but does not model graded relevance.
- PostgreSQL `simple` remains only a baseline tokenizer for Vietnamese.
- Insufficient-evidence retrieval has no calibrated rejection threshold; empty-result rate is reported separately.
- Cold-start timings depend on local hardware and cache state.
- No paid judge model or fabricated groundedness score is used.
- Version comparison, agents, and production hardening are outside Phase 8.
