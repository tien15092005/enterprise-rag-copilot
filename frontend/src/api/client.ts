import type {
  AgentResponse,
  ApiErrorBody,
  ComparisonResponse,
  DocumentVersion,
  LogicalDocument,
  QueryRequest,
  QueryResponse,
  RetrievalMode,
} from "../types/api";

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
export const API_BASE_URL = (configuredBaseUrl || "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code?: string,
    readonly requestId?: string | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function validationMessage(detail: unknown): string | undefined {
  if (!Array.isArray(detail)) return undefined;
  const first = detail[0] as { msg?: unknown; loc?: unknown } | undefined;
  if (!first || typeof first.msg !== "string") return undefined;
  const location = Array.isArray(first.loc) ? first.loc.slice(1).join(" → ") : "request";
  return `${location || "request"}: ${first.msg}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: init?.body instanceof FormData
        ? init.headers
        : { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE_URL}. Check that it is running and CORS allows this origin.`,
      0,
      "backend_unavailable",
    );
  }

  if (!response.ok) {
    let body: ApiErrorBody = {};
    try {
      body = (await response.json()) as ApiErrorBody;
    } catch {
      // The UI deliberately does not expose an unstructured backend response.
    }
    const message = body.error?.message
      ?? validationMessage(body.detail)
      ?? `Request failed with status ${response.status}.`;
    throw new ApiError(message, response.status, body.error?.code, body.error?.request_id);
  }

  return response.json() as Promise<T>;
}

export const api = {
  listDocuments: () => request<LogicalDocument[]>("/api/documents"),
  listVersions: (documentId: string) =>
    request<DocumentVersion[]>(`/api/documents/${documentId}/versions`),
  createDocument: (input: {
    stable_key: string;
    title: string;
    description?: string;
    owner?: string;
    domain?: string;
  }) => request<LogicalDocument>("/api/documents", { method: "POST", body: JSON.stringify(input) }),
  uploadVersion: (documentId: string, form: FormData) =>
    request<{ document_version: DocumentVersion; message: string }>(
      `/api/documents/${documentId}/versions`,
      { method: "POST", body: form },
    ),
  indexVersion: (versionId: string) =>
    request<{ status: string; chunks_indexed: number; total_latency_ms: number }>(
      `/api/document-versions/${versionId}/index`,
      { method: "POST" },
    ),
  query: (input: QueryRequest) =>
    request<QueryResponse>("/api/query", { method: "POST", body: JSON.stringify(input) }),
  compare: (input: {
    logical_document_id: string;
    left_document_version_id: string;
    right_document_version_id: string;
    include_unchanged: boolean;
    generate_summary: boolean;
    debug: boolean;
  }) => request<ComparisonResponse>("/api/compare", { method: "POST", body: JSON.stringify(input) }),
  agent: (input: {
    message: string;
    logical_document_id?: string;
    retrieval_mode: RetrievalMode;
    debug: boolean;
  }) => request<AgentResponse>("/api/agent", { method: "POST", body: JSON.stringify(input) }),
};
