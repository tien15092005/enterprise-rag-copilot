import { API_BASE_URL } from "../api/client";
import { evaluationSummary } from "../data/evaluation";
import { PageHeader } from "../components/PageHeader";

export function SystemPage() {
  return (
    <div>
      <PageHeader eyebrow="Measured engineering evidence" title="Evaluation & system info" description="Repository-owned benchmark results and the operating boundaries behind this demo." />
      <div className="benchmark-banner"><div><span className="eyebrow">Portfolio evaluation dataset</span><h2>14 bilingual, multi-version queries</h2></div><p>Small project-owned benchmark—not external or universal evidence. Dense remains the default because combined MRR did not justify changing it.</p></div>
      <section className="metric-cards">
        {evaluationSummary.modes.map((mode) => <article className="metric-card" key={mode.id}><span>{mode.label}</span><strong>{mode.mrr.toFixed(3)}</strong><small>Combined MRR</small><div className="latency-row"><span>Warm p50 <b>{mode.warmP50Ms.toFixed(1)} ms</b></span><span>p95 <b>{mode.warmP95Ms.toFixed(1)} ms</b></span></div></article>)}
      </section>
      <section className="quality-strip"><div><strong>{evaluationSummary.hitAt3.toFixed(1)}</strong><span>Hit@3</span></div><div><strong>{evaluationSummary.hitAt5.toFixed(1)}</strong><span>Hit@5</span></div><div><strong>{evaluationSummary.versionLeakage}</strong><span>Version leakage</span></div><div><strong>EN + VI</strong><span>Reported separately</span></div></section>
      <div className="system-grid">
        <section className="panel"><div className="section-heading section-heading--compact"><div><span className="eyebrow">Latency context</span><h2>Cold vs warm</h2></div></div><p className="muted">Cold probes include lazy model initialization. Warm latency uses the five-candidate profile.</p><div className="data-table"><div className="data-table__header"><span>Mode</span><span>Cold start</span><span>Warm p50</span></div>{evaluationSummary.modes.map((mode) => <div key={mode.id}><span>{mode.label}</span><span>{evaluationSummary.coldStartMs[mode.id]} ms</span><span>{mode.warmP50Ms} ms</span></div>)}</div><small className="source-note">Source: {evaluationSummary.source}</small></section>
        <section className="panel"><div className="section-heading section-heading--compact"><div><span className="eyebrow">Runtime</span><h2>System boundaries</h2></div></div><dl className="system-list"><div><dt>API base URL</dt><dd><code>{API_BASE_URL}</code></dd></div><div><dt>Default retrieval</dt><dd>Dense semantic</dd></div><div><dt>Source of truth</dt><dd>PostgreSQL</dd></div><div><dt>Vector index</dt><dd>Qdrant</dd></div><div><dt>Agent tools</dt><dd>4 allowlisted</dd></div><div><dt>Authentication</dt><dd>Not implemented</dd></div></dl></section>
      </div>
      <section className="limitations-panel"><div><span className="eyebrow">Interpret carefully</span><h2>What these numbers do not prove</h2></div><ul><li>They do not establish a universally best retrieval mode.</li><li>Vietnamese PostgreSQL search uses the baseline <code>simple</code> configuration.</li><li>Insufficient-evidence thresholds remain experimental and disabled by default.</li><li>Latency depends on local CPU, cache state, and candidate configuration.</li></ul></section>
    </div>
  );
}
