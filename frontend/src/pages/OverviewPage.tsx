const capabilities = [
  ["EN · VI", "Bilingual retrieval", "English and Vietnamese evidence stays in its source language."],
  ["v1 → vN", "Immutable versions", "Current, exact, all-version, and historical as-of scopes."],
  ["[S1]", "Grounded citations", "Every citation maps to evidence supplied before generation."],
  ["RRF", "Hybrid retrieval", "Semantic and lexical candidates fused with transparent ranks."],
  ["Δ", "Version comparison", "Deterministic added, removed, modified, and unchanged sections."],
  ["4 tools", "Controlled agent", "Bounded orchestration over an explicit internal allowlist."],
];

export function OverviewPage({ navigate }: { navigate: (route: string) => void }) {
  return (
    <div className="overview">
      <section className="hero">
        <div className="hero__copy">
          <span className="eyebrow">Portfolio system · Backend phases 1–10 complete</span>
          <h1>Enterprise knowledge,<br /><em>with its history intact.</em></h1>
          <p>
            A bilingual, version-aware RAG system for enterprise documents—built around immutable
            provenance, measurable retrieval, and controlled AI orchestration.
          </p>
          <div className="hero__actions">
            <button className="button" onClick={() => navigate("ask")}>Ask the knowledge base</button>
            <button className="button button--secondary" onClick={() => navigate("documents")}>Browse documents</button>
          </div>
        </div>
        <div className="flow-card" aria-label="System flow">
          <span className="flow-card__label">GROUNDING PIPELINE</span>
          {[
            ["01", "Resolve", "Explicit version scope"],
            ["02", "Retrieve", "Dense + lexical + RRF"],
            ["03", "Rerank", "Bounded multilingual model"],
            ["04", "Ground", "Structured citations"],
          ].map(([number, title, detail]) => (
            <div className="flow-step" key={number}>
              <span>{number}</span><div><strong>{title}</strong><small>{detail}</small></div>
            </div>
          ))}
        </div>
      </section>
      <section className="capability-grid">
        {capabilities.map(([mark, title, detail]) => (
          <article className="capability-card" key={title}>
            <span className="capability-card__mark">{mark}</span>
            <h3>{title}</h3>
            <p>{detail}</p>
          </article>
        ))}
      </section>
      <section className="architecture-band">
        <span>Upload</span><i>→</i><span>Parse & chunk</span><i>→</i><span>PostgreSQL + Qdrant</span>
        <i>→</i><span>Version-safe retrieval</span><i>→</i><span>Citations</span><i>→</i><span>Answer</span>
      </section>
    </div>
  );
}
