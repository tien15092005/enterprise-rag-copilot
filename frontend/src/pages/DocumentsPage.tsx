import { type FormEvent, useState } from "react";

import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusBadge } from "../components/StatusBadge";
import { useDocuments } from "../hooks/useDocuments";

export function DocumentsPage() {
  const { documents, versions, loading, error, refresh } = useDocuments();
  const [selectedId, setSelectedId] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<Error | null>(null);
  const [notice, setNotice] = useState("");

  const activeId = selectedId || documents[0]?.logical_document_id || "";
  const activeDocument = documents.find((doc) => doc.logical_document_id === activeId);
  const activeVersions = versions[activeId] ?? [];

  async function createDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true); setActionError(null); setNotice("");
    const form = new FormData(event.currentTarget);
    try {
      const document = await api.createDocument({
        stable_key: String(form.get("stable_key")),
        title: String(form.get("title")),
        description: String(form.get("description") || "") || undefined,
        domain: String(form.get("domain") || "") || undefined,
      });
      await refresh();
      setSelectedId(document.logical_document_id);
      setShowCreate(false);
      setNotice(`Created “${document.title}”.`);
    } catch (cause) {
      setActionError(cause instanceof Error ? cause : new Error("Document creation failed."));
    } finally { setBusy(false); }
  }

  async function uploadVersion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeId) return;
    setBusy(true); setActionError(null); setNotice("");
    const form = new FormData(event.currentTarget);
    try {
      const result = await api.uploadVersion(activeId, form);
      setNotice(`Version ${result.document_version.version_label} ingested. Index it to make it queryable.`);
      setShowUpload(false);
      await refresh();
    } catch (cause) {
      setActionError(cause instanceof Error ? cause : new Error("Version upload failed."));
    } finally { setBusy(false); }
  }

  async function indexVersion(versionId: string) {
    setBusy(true); setActionError(null); setNotice("");
    try {
      const result = await api.indexVersion(versionId);
      setNotice(`Indexed ${result.chunks_indexed} chunks in ${result.total_latency_ms.toFixed(0)} ms.`);
      await refresh();
    } catch (cause) {
      setActionError(cause instanceof Error ? cause : new Error("Indexing failed."));
    } finally { setBusy(false); }
  }

  return (
    <div>
      <PageHeader eyebrow="Knowledge catalog" title="Documents & versions"
        description="Inspect logical documents and their immutable source snapshots. Current status is explicit—never inferred from filenames or upload order."
        action={<button className="button" onClick={() => setShowCreate((value) => !value)}>+ New document</button>} />

      {showCreate && (
        <form className="panel form-panel" onSubmit={createDocument}>
          <div className="form-panel__heading"><h2>Create logical document</h2><button type="button" className="icon-button" onClick={() => setShowCreate(false)}>×</button></div>
          <div className="form-grid form-grid--2">
            <label>Stable key<input name="stable_key" required pattern="[A-Za-z0-9][A-Za-z0-9._-]*" placeholder="hr-policy" /></label>
            <label>Title<input name="title" required placeholder="HR Policy" /></label>
            <label>Domain<input name="domain" placeholder="Human Resources" /></label>
            <label>Description<input name="description" placeholder="Policies for all employees" /></label>
          </div>
          <button className="button" disabled={busy}>{busy ? "Creating…" : "Create document"}</button>
        </form>
      )}

      {actionError && <ErrorState error={actionError} />}
      {notice && <div className="notice" role="status">✓ {notice}</div>}
      {loading && <LoadingState label="Loading document catalog" />}
      {error && <ErrorState error={error} onRetry={() => void refresh()} />}

      {!loading && !error && documents.length === 0 && (
        <EmptyState title="No documents yet" detail="Create a logical document, then upload its first immutable version." />
      )}

      {documents.length > 0 && (
        <div className="catalog-layout">
          <aside className="document-list">
            <div className="list-caption">{documents.length} logical document{documents.length === 1 ? "" : "s"}</div>
            {documents.map((document) => (
              <button className={`document-row ${activeId === document.logical_document_id ? "is-active" : ""}`}
                onClick={() => setSelectedId(document.logical_document_id)} key={document.logical_document_id}>
                <span className="document-row__icon">{document.title.slice(0, 1).toUpperCase()}</span>
                <span><strong>{document.title}</strong><small>{document.domain || document.stable_key}</small></span>
                <b>{versions[document.logical_document_id]?.length ?? 0}</b>
              </button>
            ))}
          </aside>

          <section className="panel document-detail">
            <div className="document-detail__header">
              <div><span className="mono-label">{activeDocument?.stable_key}</span><h2>{activeDocument?.title}</h2><p>{activeDocument?.description || "No description provided."}</p></div>
              <button className="button button--secondary" onClick={() => setShowUpload((value) => !value)}>Upload version</button>
            </div>
            {showUpload && (
              <form className="upload-form" onSubmit={uploadVersion}>
                <div className="form-grid form-grid--3">
                  <label className="file-field">Source file<input name="file" type="file" accept=".pdf,.docx,.txt,.md" required /></label>
                  <label>Version label<input name="version_label" placeholder="2026" required /></label>
                  <label>Version ordinal<input name="version_ordinal" type="number" min="1" defaultValue={activeVersions.length + 1} required /></label>
                  <label>Language<select name="language" defaultValue="en"><option value="en">English</option><option value="vi">Vietnamese</option></select></label>
                  <label>Revision<input name="revision_number" type="number" min="1" defaultValue="1" /></label>
                  <label>Effective from<input name="effective_from" type="datetime-local" /></label>
                </div>
                <label className="check-row"><input name="activate" type="checkbox" value="true" /> Mark active/current after ingestion</label>
                <button className="button" disabled={busy}>{busy ? "Uploading…" : "Ingest version"}</button>
              </form>
            )}
            <div className="version-list">
              {activeVersions.length === 0 && <EmptyState title="No versions" detail="Upload a PDF, DOCX, UTF-8 text, or Markdown source." />}
              {activeVersions.map((version) => (
                <article className="version-row" key={version.document_version_id}>
                  <div className="version-row__identity"><span className="version-number">v{version.version_label}</span><div><strong>{version.source_filename}</strong><small>Revision {version.revision_number} · {version.language.toUpperCase()} · ordinal {version.version_ordinal}</small></div></div>
                  <div className="version-row__states">
                    {version.is_current && <StatusBadge value="current" />}
                    <StatusBadge value={version.business_status} />
                    <StatusBadge value={version.indexing_status} />
                    <StatusBadge value={version.vector_index_status} />
                  </div>
                  {version.indexing_status === "ready" && version.vector_index_status !== "indexed" && (
                    <button className="button button--small button--secondary" disabled={busy} onClick={() => void indexVersion(version.document_version_id)}>Index</button>
                  )}
                </article>
              ))}
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
