import { type FormEvent, useMemo, useState } from "react";

import { api } from "../api/client";
import { CitationList } from "../components/CitationList";
import { DocumentSelect } from "../components/DocumentSelect";
import { ErrorState, LoadingState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { TimingStrip } from "../components/TimingStrip";
import { useDocuments } from "../hooks/useDocuments";
import type { ChangeType, ComparisonItem, ComparisonResponse } from "../types/api";

const groups: Array<{ type: ChangeType; label: string }> = [
  { type: "modified", label: "Modified" },
  { type: "added", label: "Added" },
  { type: "removed", label: "Removed" },
  { type: "unchanged", label: "Unchanged" },
];

function ChangeItem({ item, response }: { item: ComparisonItem; response: ComparisonResponse }) {
  const itemCitations = response.citations.filter((citation) => citation.item_order === item.order).map((entry) => entry.citation);
  return (
    <article className={`change-card change-card--${item.change_type}`}>
      <div className="change-card__header"><StatusBadge value={item.change_type} tone={item.change_type === "added" ? "success" : item.change_type === "removed" ? "danger" : item.change_type === "modified" ? "warning" : "neutral"} /><strong>{item.left_section_path?.join(" / ") || item.right_section_path?.join(" / ") || `Section ${item.order}`}</strong><span>{item.alignment_method.replaceAll("_", " ")}{item.alignment_score !== null ? ` · ${item.alignment_score.toFixed(3)}` : ""}</span></div>
      <div className={`diff-grid ${item.change_type === "modified" || item.change_type === "unchanged" ? "" : "diff-grid--single"}`}>
        {item.left_text !== null && <div className="diff-side diff-side--left"><div className="diff-label"><span>LEFT</span>Version {response.left_version.version_label}</div><p>{item.left_text}</p></div>}
        {item.right_text !== null && <div className="diff-side diff-side--right"><div className="diff-label"><span>RIGHT</span>Version {response.right_version.version_label}</div><p>{item.right_text}</p></div>}
      </div>
      {item.diff && (item.diff.removed.length > 0 || item.diff.added.length > 0) && (
        <div className="sentence-diff">{item.diff.removed.map((text) => <span className="sentence-diff__removed" key={`r-${text}`}>− {text}</span>)}{item.diff.added.map((text) => <span className="sentence-diff__added" key={`a-${text}`}>+ {text}</span>)}</div>
      )}
      {itemCitations.length > 0 && <div className="inline-citations">{itemCitations.map((citation) => <span key={`${citation.citation_key}-${citation.document_version_id}`}>{citation.citation_key} · v{citation.version_label}</span>)}</div>}
    </article>
  );
}

export function ComparePage() {
  const { documents, versions, error: docsError } = useDocuments();
  const [documentId, setDocumentId] = useState("");
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [includeUnchanged, setIncludeUnchanged] = useState(false);
  const [generateSummary, setGenerateSummary] = useState(false);
  const [debug, setDebug] = useState(false);
  const [result, setResult] = useState<ComparisonResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  const documentVersions = versions[documentId] ?? [];
  const grouped = useMemo(() => groups.map((group) => ({ ...group, items: result?.items.filter((item) => item.change_type === group.type) ?? [] })), [result]);

  function chooseDocument(value: string) { setDocumentId(value); setLeftId(""); setRightId(""); setResult(null); }
  async function submit(event: FormEvent) {
    event.preventDefault(); setLoading(true); setError(null); setResult(null);
    try {
      setResult(await api.compare({ logical_document_id: documentId, left_document_version_id: leftId, right_document_version_id: rightId, include_unchanged: includeUnchanged, generate_summary: generateSummary, debug }));
    } catch (cause) { setError(cause instanceof Error ? cause : new Error("Comparison failed.")); }
    finally { setLoading(false); }
  }

  return (
    <div>
      <PageHeader eyebrow="Deterministic change analysis" title="Compare immutable versions" description="Align sections, classify structural changes, and preserve citations from both selected source versions." />
      {docsError && <ErrorState error={docsError} />}
      <form className="panel compare-controls" onSubmit={submit}>
        <label>Logical document<DocumentSelect documents={documents} value={documentId} onChange={chooseDocument} /></label>
        <label>Left / baseline version<select required value={leftId} onChange={(event) => setLeftId(event.target.value)}><option value="">Select version</option>{documentVersions.map((version) => <option value={version.document_version_id} key={version.document_version_id}>v{version.version_label} · {version.source_filename}</option>)}</select></label>
        <div className="compare-arrow">→</div>
        <label>Right / target version<select required value={rightId} onChange={(event) => setRightId(event.target.value)}><option value="">Select version</option>{documentVersions.map((version) => <option value={version.document_version_id} key={version.document_version_id} disabled={version.document_version_id === leftId}>v{version.version_label} · {version.source_filename}</option>)}</select></label>
        <div className="compare-options"><label className="check-row"><input type="checkbox" checked={includeUnchanged} onChange={(event) => setIncludeUnchanged(event.target.checked)} /> Include unchanged</label><label className="check-row"><input type="checkbox" checked={generateSummary} onChange={(event) => setGenerateSummary(event.target.checked)} /> Generate summary</label><label className="check-row"><input type="checkbox" checked={debug} onChange={(event) => setDebug(event.target.checked)} /> Debug</label></div>
        <button className="button" disabled={loading || !documentId || !leftId || !rightId || leftId === rightId}>{loading ? "Comparing…" : "Compare versions"}</button>
      </form>
      {loading && <LoadingState label="Loading versions and aligning sections" />}
      {error && <ErrorState error={error} />}
      {result && <div className="comparison-results">
        <section className="comparison-summary panel"><div><span className="eyebrow">Comparison result</span><h2>Version {result.left_version.version_label} <i>→</i> Version {result.right_version.version_label}</h2><p>{result.summary || "Deterministic comparison complete. Generated summary was not requested."}</p></div><div className="count-grid">{groups.map((group) => <div key={group.type}><b>{result.counts[group.type]}</b><span>{group.label}</span></div>)}</div><TimingStrip values={result.timing} /></section>
        {grouped.map((group) => group.items.length > 0 && <section className="change-group" key={group.type}><div className="section-heading"><div><span className="eyebrow">{group.type}</span><h2>{group.label} sections</h2></div><span className="count-bubble">{group.items.length}</span></div>{group.items.map((item) => <ChangeItem item={item} response={result} key={item.order} />)}</section>)}
        <CitationList citations={result.citations.map((entry) => entry.citation)} debug={debug} />
        {debug && result.debug && <details className="debug-panel"><summary>Alignment debug</summary><pre>{JSON.stringify(result.debug, null, 2)}</pre></details>}
      </div>}
    </div>
  );
}
