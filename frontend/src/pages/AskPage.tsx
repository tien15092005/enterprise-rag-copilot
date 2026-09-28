import { type FormEvent, useState } from "react";

import { api } from "../api/client";
import { CitationList } from "../components/CitationList";
import { DocumentSelect } from "../components/DocumentSelect";
import { ErrorState, LoadingState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { TimingStrip } from "../components/TimingStrip";
import { useDocuments } from "../hooks/useDocuments";
import type { QueryResponse, RetrievalMode, ScopeKind } from "../types/api";

export function AskPage() {
  const { documents, versions, loading: docsLoading, error: docsError } = useDocuments();
  const [documentId, setDocumentId] = useState("");
  const [scope, setScope] = useState<ScopeKind>("current");
  const [exactVersionId, setExactVersionId] = useState("");
  const [asOf, setAsOf] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("dense");
  const [retrievalOnly, setRetrievalOnly] = useState(false);
  const [debug, setDebug] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);

  const availableVersions = versions[documentId] ?? [];

  function changeDocument(value: string) {
    setDocumentId(value);
    setExactVersionId("");
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const question = String(form.get("question"));
    setLoading(true); setError(null); setResult(null);
    try {
      const selector = scope === "exact"
        ? { kind: scope, document_version_id: exactVersionId }
        : scope === "as_of"
          ? { kind: scope, as_of: new Date(asOf).toISOString() }
          : { kind: scope };
      setResult(await api.query({
        question,
        logical_document_id: documentId,
        version_selector: selector,
        top_k: 5,
        retrieval_mode: mode,
        retrieval_only: retrievalOnly,
        debug,
      }));
    } catch (cause) {
      setError(cause instanceof Error ? cause : new Error("Query failed."));
    } finally { setLoading(false); }
  }

  return (
    <div>
      <PageHeader eyebrow="Version-aware RAG" title="Ask the knowledge base"
        description="Resolve a precise version scope, retrieve authoritative evidence, and return only validated citations." />
      {docsError && <ErrorState error={docsError} />}
      <div className="workspace-layout">
        <form className="panel control-panel" onSubmit={submit}>
          <div className="control-panel__section"><span className="step-index">01</span><h3>Question</h3></div>
          <label>Logical document<DocumentSelect documents={documents} value={documentId} onChange={changeDocument} /></label>
          <label>Ask a question<textarea name="question" required rows={5} placeholder="What is the current remote-work policy?" /></label>

          <div className="control-panel__section"><span className="step-index">02</span><h3>Retrieval controls</h3></div>
          <div className="segmented" role="group" aria-label="Version scope">
            {(["current", "exact", "all", "as_of"] as ScopeKind[]).map((value) => (
              <button type="button" className={scope === value ? "is-active" : ""} onClick={() => setScope(value)} key={value}>{value.replace("_", " ")}</button>
            ))}
          </div>
          {scope === "exact" && (
            <label>Exact version<select required value={exactVersionId} onChange={(event) => setExactVersionId(event.target.value)}><option value="">Select immutable version</option>{availableVersions.map((version) => <option value={version.document_version_id} key={version.document_version_id}>{version.version_label} · {version.source_filename}</option>)}</select></label>
          )}
          {scope === "as_of" && <label>Effective at<input type="datetime-local" required value={asOf} onChange={(event) => setAsOf(event.target.value)} /></label>}
          <label>Retrieval mode<select value={mode} onChange={(event) => setMode(event.target.value as RetrievalMode)}><option value="dense">Dense semantic</option><option value="hybrid">Hybrid · dense + lexical + RRF</option><option value="hybrid_rerank">Hybrid + multilingual reranker</option></select></label>
          <div className="switch-stack">
            <label className="switch-row"><span><strong>Retrieval only</strong><small>Return evidence without calling the LLM</small></span><input type="checkbox" checked={retrievalOnly} onChange={(event) => setRetrievalOnly(event.target.checked)} /></label>
            <label className="switch-row"><span><strong>Debug metadata</strong><small>Show ranks, scores, and validation</small></span><input type="checkbox" checked={debug} onChange={(event) => setDebug(event.target.checked)} /></label>
          </div>
          <button className="button button--wide" disabled={loading || docsLoading || !documentId}>{loading ? "Searching…" : retrievalOnly ? "Retrieve evidence" : "Ask copilot"}</button>
        </form>

        <section className="result-column">
          {loading && <LoadingState label="Resolving scope and retrieving evidence" />}
          {error && <ErrorState error={error} />}
          {!loading && !error && !result && (
            <div className="answer-placeholder"><div className="answer-placeholder__glyph">K</div><h2>Your grounded answer will appear here</h2><p>Choose a document and version scope. The backend—not the browser—controls retrieval, provenance, and citation validation.</p></div>
          )}
          {result && (
            <>
              <article className={`answer-card ${result.insufficient_evidence ? "answer-card--warning" : ""}`}>
                <div className="answer-card__header"><div><span className="eyebrow">{retrievalOnly ? "Evidence response" : "Grounded answer"}</span><h2>{result.insufficient_evidence ? "Insufficient evidence" : retrievalOnly ? "Retrieved evidence" : "Copilot response"}</h2></div><span className="language-chip">{result.answer_language.toUpperCase()}</span></div>
                {result.answer ? <p className="answer-text">{result.answer}</p> : <p className="muted">Generation skipped. Review the structured evidence below.</p>}
                <div className="scope-summary"><b>{result.resolved_scope.kind.replace("_", " ")} scope</b>{result.resolved_scope.versions.map((version) => <span key={version.document_version_id}>v{version.version_label}</span>)}</div>
                <TimingStrip values={result.timing} />
              </article>
              <CitationList citations={result.citations} debug={debug} />
              {debug && result.debug && <details className="debug-panel"><summary>Validation debug</summary><pre>{JSON.stringify(result.debug.citation_validation, null, 2)}</pre></details>}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
