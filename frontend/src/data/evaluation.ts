export const evaluationSummary = {
  dataset: "Enterprise RAG bilingual v1",
  queries: 14,
  modes: [
    { id: "dense", label: "Dense", mrr: 0.958, warmP50Ms: 36.14, warmP95Ms: 43.96 },
    { id: "hybrid", label: "Hybrid", mrr: 0.958, warmP50Ms: 40.99, warmP95Ms: 52.51 },
    {
      id: "hybrid_rerank",
      label: "Hybrid + rerank",
      mrr: 0.958,
      warmP50Ms: 71.26,
      warmP95Ms: 89.95,
    },
  ],
  hitAt3: 1,
  hitAt5: 1,
  versionLeakage: 0,
  coldStartMs: { dense: 44.73, hybrid: 47.41, hybrid_rerank: 6614.19 },
  source: "EVALUATION_REPORT.md · candidates_5 profile",
} as const;
