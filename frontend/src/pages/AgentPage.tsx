import { type FormEvent, useState } from "react";

import { ApiError, api } from "../api/client";
import { CitationList } from "../components/CitationList";
import { DocumentSelect } from "../components/DocumentSelect";
import { ErrorState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { TimingStrip } from "../components/TimingStrip";
import { useDocuments } from "../hooks/useDocuments";
import type { AgentResponse, RetrievalMode } from "../types/api";

interface Exchange { id: number; question: string; response?: AgentResponse; error?: Error }

export function AgentPage() {
  const { documents } = useDocuments();
  const [documentId, setDocumentId] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("dense");
  const [debug, setDebug] = useState(false);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const messageInput = form.elements.namedItem("message") as HTMLTextAreaElement;
    const message = messageInput.value.trim();
    if (!message) return;
    const id = Date.now();
    setExchanges((current) => [...current, { id, question: message }]);
    setLoading(true); messageInput.value = "";
    try {
      const response = await api.agent({ message, logical_document_id: documentId || undefined, retrieval_mode: mode, debug });
      setExchanges((current) => current.map((exchange) => exchange.id === id ? { ...exchange, response } : exchange));
    } catch (cause) {
      let error = cause instanceof Error ? cause : new Error("Agent request failed.");
      if (cause instanceof ApiError && cause.code === "agent_timeout") error = new Error("The controlled agent reached its execution deadline. Try a narrower request.");
      setExchanges((current) => current.map((exchange) => exchange.id === id ? { ...exchange, error } : exchange));
    } finally { setLoading(false); }
  }

  return (
    <div>
      <PageHeader eyebrow="Controlled orchestration" title="Knowledge agent" description="Ask naturally. The agent can call only approved application tools and must return grounded, citation-validated evidence." />
      <div className="agent-shell">
        <aside className="panel agent-settings">
          <h3>Agent context</h3>
          <label>Logical document<DocumentSelect documents={documents} value={documentId} onChange={setDocumentId} optional /></label>
          <label>Search mode<select value={mode} onChange={(event) => setMode(event.target.value as RetrievalMode)}><option value="dense">Dense</option><option value="hybrid">Hybrid</option><option value="hybrid_rerank">Hybrid + rerank</option></select></label>
          <label className="switch-row"><span><strong>Safe tool trace</strong><small>Names, arguments, result type, latency</small></span><input type="checkbox" checked={debug} onChange={(event) => setDebug(event.target.checked)} /></label>
          <div className="allowlist"><span className="eyebrow">Approved tools</span>{["search_knowledge", "compare_versions", "list_document_versions", "get_document_version"].map((tool) => <code key={tool}>{tool}</code>)}</div>
          <p className="safety-note">No shell, arbitrary SQL, filesystem, network, or hidden chain-of-thought access.</p>
        </aside>
        <section className="panel chat-panel">
          <div className="chat-stream">
            {exchanges.length === 0 && <div className="chat-welcome"><div className="answer-placeholder__glyph">A</div><h2>Ask across knowledge and history</h2><p>Try “What is the current remote-work policy?” or “Compare the 2025 and 2026 policy.”</p></div>}
            {exchanges.map((exchange) => <div className="exchange" key={exchange.id}>
              <div className="message message--user"><span>You</span><p>{exchange.question}</p></div>
              {exchange.error && <ErrorState error={exchange.error} />}
              {exchange.response && <div className="message message--agent"><span>Agent</span><p>{exchange.response.answer}</p><TimingStrip values={exchange.response.timing} /><CitationList citations={exchange.response.citations} debug={false} />{debug && exchange.response.debug && <div className="tool-trace"><div className="tool-trace__header"><strong>Tool trace</strong><span>{exchange.response.debug.step_count} step{exchange.response.debug.step_count === 1 ? "" : "s"}</span></div>{exchange.response.debug.tool_trace.map((step) => <div className="tool-step" key={step.step}><span>{step.step}</span><div><strong>{step.tool_name}</strong><small>{step.result_type} · {step.latency_ms.toFixed(1)} ms</small><code>{JSON.stringify(step.arguments)}</code></div></div>)}</div>}</div>}
              {!exchange.response && !exchange.error && <div className="message-loading"><span className="spinner" /> Planning within approved tools…</div>}
            </div>)}
          </div>
          <form className="composer" onSubmit={submit}><textarea name="message" rows={2} required placeholder="Ask about a policy, version, or change…" disabled={loading} /><button className="button" disabled={loading}>{loading ? "Working…" : "Send"}</button></form>
        </section>
      </div>
    </div>
  );
}
