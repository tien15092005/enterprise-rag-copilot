export type Language = "en" | "vi";
export type RetrievalMode = "dense" | "hybrid" | "hybrid_rerank";
export type ScopeKind = "current" | "exact" | "all" | "as_of";
export type ChangeType = "added" | "removed" | "modified" | "unchanged";

export interface LogicalDocument {
  logical_document_id: string;
  stable_key: string;
  title: string;
  description: string | null;
  owner: string | null;
  domain: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentVersion {
  document_version_id: string;
  logical_document_id: string;
  version_label: string;
  revision_number: number;
  version_ordinal: number;
  language: Language;
  source_filename: string;
  checksum: string;
  business_status: "draft" | "active" | "superseded" | "withdrawn";
  indexing_status: "pending" | "processing" | "ready" | "failed" | "deleting";
  vector_index_status: "not_indexed" | "indexing" | "indexed" | "failed";
  created_at: string;
  is_current: boolean;
}

export interface Citation {
  citation_key: string;
  logical_document_id: string;
  document_version_id: string;
  version_label: string;
  chunk_id: string;
  source_filename: string;
  page: number | null;
  section_path: string[] | null;
  excerpt: string;
  rank: number;
  retrieval_score: number;
  dense_rank: number | null;
  dense_score: number | null;
  lexical_rank: number | null;
  lexical_score: number | null;
  rrf_score: number | null;
  rrf_rank: number | null;
  reranker_score: number | null;
  reranker_rank: number | null;
  language: Language;
  effective_from: string | null;
  effective_to: string | null;
}

export interface QueryTiming extends Record<string, number> {
  scope_resolution_ms: number;
  query_embedding_ms: number;
  retrieval_ms: number;
  lexical_retrieval_ms: number;
  fusion_ms: number;
  reranking_ms: number;
  context_construction_ms: number;
  generation_ms: number;
  total_ms: number;
}

export interface QueryRequest {
  question: string;
  logical_document_id?: string;
  version_selector: {
    kind: ScopeKind;
    document_version_id?: string;
    as_of?: string;
  };
  top_k: number;
  retrieval_mode: RetrievalMode;
  retrieval_only: boolean;
  debug: boolean;
}

export interface QueryResponse {
  answer: string | null;
  answer_language: Language;
  citations: Citation[];
  resolved_scope: {
    kind: ScopeKind;
    logical_document_id: string;
    versions: Array<{
      document_version_id: string;
      version_label: string;
      effective_from: string | null;
      effective_to: string | null;
    }>;
  };
  timing: QueryTiming;
  token_usage: TokenUsage | null;
  prompt_version: string;
  insufficient_evidence: boolean;
  debug: {
    evidence: Citation[];
    citation_validation: Record<string, unknown>;
  } | null;
}

export interface TokenUsage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
}

export interface ComparisonItem {
  order: number;
  change_type: ChangeType;
  left_chunk_ids: string[];
  right_chunk_ids: string[];
  left_section_path: string[] | null;
  right_section_path: string[] | null;
  left_text: string | null;
  right_text: string | null;
  alignment_method: string;
  alignment_score: number | null;
  diff: { removed: string[]; added: string[] } | null;
  citation_keys: string[];
}

export interface ComparisonResponse {
  left_version: ComparisonVersion;
  right_version: ComparisonVersion;
  items: ComparisonItem[];
  counts: Record<ChangeType, number>;
  summary: string | null;
  summary_language: Language;
  citations: Array<{ side: "left" | "right"; item_order: number; citation: Citation }>;
  timing: Record<string, number>;
  token_usage: TokenUsage | null;
  prompt_version: string;
  debug: Record<string, unknown> | null;
}

export interface ComparisonVersion {
  document_version_id: string;
  logical_document_id: string;
  version_label: string;
  language: Language;
  source_filename: string;
  effective_from: string | null;
  effective_to: string | null;
}

export interface AgentResponse {
  answer: string;
  citations: Citation[];
  timing: {
    planning_provider_ms: number;
    tool_execution_ms: number;
    final_generation_ms: number;
    total_ms: number;
  };
  token_usage: TokenUsage | null;
  debug: {
    tool_trace: Array<{
      step: number;
      tool_name: string;
      arguments: Record<string, unknown>;
      result_type: string;
      latency_ms: number;
    }>;
    step_count: number;
    provider_name: string;
    provider_model: string;
  } | null;
}

export interface ApiErrorBody {
  error?: {
    code?: string;
    message?: string;
    details?: unknown;
    request_id?: string | null;
  };
  detail?: unknown;
}
