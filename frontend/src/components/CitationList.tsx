import type { Citation } from "../types/api";

function score(value: number | null): string {
  return value === null ? "—" : value.toFixed(4);
}

export function CitationList({ citations, debug = false }: { citations: Citation[]; debug?: boolean }) {
  if (citations.length === 0) return null;
  return (
    <section className="citations" aria-label="Citations">
      <div className="section-heading section-heading--compact">
        <div>
          <span className="eyebrow">Grounded evidence</span>
          <h3>{citations.length} citation{citations.length === 1 ? "" : "s"}</h3>
        </div>
      </div>
      <div className="citation-grid">
        {citations.map((citation) => (
          <article className="citation-card" key={`${citation.citation_key}-${citation.chunk_id}`}>
            <div className="citation-card__topline">
              <span className="citation-key">{citation.citation_key}</span>
              <span className="version-pill">Version {citation.version_label}</span>
            </div>
            <strong className="citation-card__file">{citation.source_filename}</strong>
            <div className="meta-line">
              {citation.section_path?.length ? citation.section_path.join(" / ") : "Section unavailable"}
              {citation.page ? ` · Page ${citation.page}` : ""}
            </div>
            <blockquote>{citation.excerpt}</blockquote>
            {debug && (
              <div className="score-grid">
                <span>Dense <b>#{citation.dense_rank ?? "—"}</b> · {score(citation.dense_score)}</span>
                <span>Lexical <b>#{citation.lexical_rank ?? "—"}</b> · {score(citation.lexical_score)}</span>
                <span>RRF <b>#{citation.rrf_rank ?? "—"}</b> · {score(citation.rrf_score)}</span>
                <span>Reranker <b>#{citation.reranker_rank ?? "—"}</b> · {score(citation.reranker_score)}</span>
              </div>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
