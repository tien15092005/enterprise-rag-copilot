# Enterprise Knowledge Copilot Architecture

## 1. Purpose and scope

Enterprise Knowledge Copilot is a production-oriented, version-aware Retrieval-Augmented Generation (RAG) backend for internal enterprise documents. Its distinguishing capability is treating successive files such as `HR_Policy_2025.pdf` and `HR_Policy_2026.pdf` as immutable versions of one logical document. The system is designed to retrieve from an exact version or an intentional version scope, compare versions, explain changes, and cite evidence from both sides of a comparison.

The initial system supports English and Vietnamese PDF, DOCX, TXT, and Markdown ingestion; dense and hybrid retrieval; optional reranking; cited answers; reproducible evaluation; and a small, controlled tool-calling layer. Version comparison is an explicitly planned application capability, but its comparison engine is not part of Phase 0 or Phase 1.

The initial deployment is a modular monolith. FastAPI exposes the REST API, while application services keep parsing, chunking, embedding, retrieval, reranking, generation, and evaluation separate. This preserves understandable source code and permits individual components to be replaced without introducing distributed-system complexity prematurely.

## 2. System context

```text
API client
    |
    v
FastAPI application
    |-- document/version catalog --> immutable document versions
    |-- ingestion service --> parser --> chunker --> embedding provider
    |                                           |        |
    |                                           |        +--> Qdrant
    |                                           +----------> PostgreSQL
    |
    |-- query service --> dense retrieval --------> Qdrant
    |                 --> lexical retrieval ------> PostgreSQL FTS
    |                 --> score fusion --> reranker --> context builder
    |                 --> LLM provider --> citation validator --> response
    |
    |-- comparison service --> version-scoped retrieval x 2
    |                      --> section alignment/change analysis
    |                      --> dual-version cited response
    |
    |-- evaluation service --> retrieval and RAG evaluators --> JSON/CSV results
    |
    +-- optional agent service --> small allow-listed tool registry

Local document volume (initial file store) <--> document service
```

## 3. Proposed repository structure

This is the target structure, not a structure created during Phase 0.

```text
enterprise-rag-copilot/
|-- app/
|   |-- api/
|   |   |-- dependencies.py
|   |   |-- error_handlers.py
|   |   |-- router.py
|   |   `-- routes/
|   |       |-- documents.py
|   |       |-- evaluation.py
|   |       |-- health.py
|   |       |-- comparisons.py
|   |       |-- versions.py
|   |       `-- query.py
|   |-- agents/
|   |   |-- orchestrator.py
|   |   `-- tools.py
|   |-- core/
|   |   |-- config.py
|   |   |-- exceptions.py
|   |   |-- logging.py
|   |   `-- telemetry.py
|   |-- comparison/
|   |   |-- alignment.py
|   |   |-- models.py
|   |   `-- service.py
|   |-- db/
|   |   |-- migrations/
|   |   |-- repositories/
|   |   |-- models.py
|   |   `-- session.py
|   |-- embeddings/
|   |   |-- base.py
|   |   |-- factory.py
|   |   `-- sentence_transformers.py
|   |-- evaluation/
|   |   |-- datasets.py
|   |   |-- generation_metrics.py
|   |   |-- retrieval_metrics.py
|   |   `-- runner.py
|   |-- generation/
|   |   |-- citations.py
|   |   |-- context.py
|   |   |-- prompts.py
|   |   |-- service.py
|   |   `-- providers/
|   |       |-- base.py
|   |       |-- ollama.py
|   |       `-- openai_compatible.py
|   |-- ingestion/
|   |   |-- chunker.py
|   |   |-- parsers/
|   |   |   |-- base.py
|   |   |   |-- docx.py
|   |   |   |-- markdown.py
|   |   |   |-- pdf.py
|   |   |   `-- text.py
|   |   |-- service.py
|   |   `-- storage.py
|   |-- models/
|   |   |-- documents.py
|   |   |-- evaluation.py
|   |   |-- query.py
|   |   |-- retrieval.py
|   |   `-- versions.py
|   |-- reranking/
|   |   |-- base.py
|   |   `-- cross_encoder.py
|   |-- retrieval/
|   |   |-- dense.py
|   |   |-- fusion.py
|   |   |-- hybrid.py
|   |   |-- lexical.py
|   |   `-- service.py
|   `-- main.py
|-- data/
|   |-- evaluation/
|   |   |-- corpus/
|   |   `-- questions.jsonl
|   `-- samples/
|-- docs/
|   |-- decisions/
|   `-- evaluation.md
|-- scripts/
|   |-- ingest_samples.py
|   `-- run_evaluation.py
|-- tests/
|   |-- api/
|   |-- evaluation/
|   |-- generation/
|   |-- ingestion/
|   |-- integration/
|   |-- retrieval/
|   `-- conftest.py
|-- .env.example
|-- .gitignore
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- ARCHITECTURE.md
|-- IMPLEMENTATION_PLAN.md
`-- README.md
```

## 4. Components and responsibilities

### API layer

FastAPI routes validate transport schemas, resolve dependencies, call application services, and translate domain errors into stable HTTP responses. Routes do not implement parsing, retrieval, or generation logic.

Planned endpoints:

- `GET /health`: application status and dependency readiness.
- `POST /documents`: create a logical document and its first version, or add a version to an explicitly selected logical document.
- `GET /documents`: list logical documents with current/latest-version summaries.
- `GET /documents/{logical_document_id}/versions`: list immutable versions and lifecycle metadata.
- `DELETE /document-versions/{document_version_id}`: remove one version, its chunks, and vectors subject to retention rules.
- `POST /query`: retrieve evidence from a current, exact, all-version, or as-of version scope and optionally generate a cited answer.
- `POST /compare`: compare two explicit versions of the same logical document; introduced only when the comparison service is implemented.
- `POST /evaluation/run`: run a named, versioned evaluation dataset.
- `GET /evaluation/results`: list or retrieve persisted experiment results.

### Ingestion

The ingestion service validates file type and size, computes a content hash, stores the original file, associates it with a logical document, creates an immutable document-version record, selects a parser, normalizes extracted blocks, chunks them, and writes authoritative application records. A separate retryable indexing service loads ready chunks, creates embeddings, and writes derived vector records. Parsers return structured blocks rather than raw strings so page and section information survive chunking.

Version grouping is explicit. An upload either creates a new logical document or supplies an existing `logical_document_id`. Filename parsing may later suggest a group or version label, but it must never silently decide document identity. This prevents unrelated documents with similar names from being merged. Replacing file bytes creates a new version; indexed version content and source-semantic metadata are not mutated in place.

`DocumentVersion` is an immutable source snapshot after successful ingestion. Source bytes, checksum, extracted text, source filename, version label/order, language, effective dates, predecessor relationship, parser/chunking configuration, and derived chunks cannot be silently replaced on a ready version. A source or source-semantic change creates a new `DocumentVersion` (including a revision when the business-facing label is unchanged). Operational metadata such as retry diagnostics, storage verification state, indexing timestamps, and lifecycle bookkeeping may be updated when it does not change the historical meaning or retrieved evidence.

The chunker is configurable through settings and explicit experiment overrides. `chunk_size` and `chunk_overlap` are measured in tokenizer tokens, not characters. Chunk boundaries prefer headings and paragraphs before applying token windows. Configuration used for every ingestion is stored with the document/index record so results are reproducible.

### Embeddings

An `EmbeddingProvider` interface exposes model identity, vector dimension, document embedding, and query embedding operations. Retrieval code depends on this interface, not on a vendor SDK.

Proposed initial implementation: `sentence-transformers` using `intfloat/multilingual-e5-small`. It supports multilingual retrieval, including English and Vietnamese through its multilingual XLM-R vocabulary, produces 384-dimensional vectors, and remains practical for CPU-based portfolio development. Its model weight is about 471 MB, compared with about 133 MB for `BAAI/bge-small-en-v1.5`; it therefore uses more memory and is expected to be slower, but the English-only BGE model is not an acceptable default for the bilingual requirement. `BAAI/bge-m3` is a stronger long-context multilingual future experiment, but its 1024-dimensional vectors and substantially larger model make it a poor lightweight baseline.

E5 input conventions are part of the provider implementation: queries receive the `query: ` prefix and chunks receive the `passage: ` prefix, including for non-English text. The configured model repository and revision/commit are recorded and may be pinned for reproducible evaluations. Model name, revision, batch size, device, normalization, and prefix strategy are configuration values. Changing model or vector dimension requires a new Qdrant collection/index version rather than mixing incompatible vectors. English and Vietnamese retrieval quality must be measured separately; multilingual support is a capability target, not an assumed score.

### Retrieval

Retrieval remains explicit and staged:

1. A version-scope resolver converts `current`, `exact`, `all`, or `as_of` selection into allowed immutable `document_version_id` values.
2. Dense retriever embeds the query and searches Qdrant with logical-document and version filters.
3. Lexical retriever searches PostgreSQL full-text indexes over the same allowed version set.
4. Fusion normalizes and combines result ranks using Reciprocal Rank Fusion (RRF), avoiding invalid direct comparison of cosine and lexical scores.
5. Optional reranker scores the fused candidates and returns the configured `rerank_top_n`.
6. Context builder enforces a context budget and formats stable citation identifiers that include version provenance.

Phase A implements step 1. Phase B adds steps 2 and 3. Phase C adds step 4. Query settings include `top_k`, retrieval mode, RRF parameters, and `rerank_top_n`, with server-side bounds to prevent excessive workloads.

PostgreSQL full-text search is the proposed lexical implementation. It avoids operating Elasticsearch/OpenSearch for the portfolio baseline. PostgreSQL's native ranking is not a textbook BM25 implementation, so the code and documentation will call it lexical/full-text retrieval unless a true BM25 extension or in-process BM25 index is later selected.

For English, PostgreSQL's `english` text-search configuration can provide stemming. PostgreSQL has no built-in Vietnamese linguistic stemmer, so Vietnamese initially uses the `simple` configuration with Unicode-preserving normalization. Dense multilingual retrieval carries more of the semantic burden for Vietnamese. Lexical quality, diacritic handling, and mixed-language queries must be covered by bilingual evaluation rather than assumed.

### Reranking

A `Reranker` interface accepts a query and retrieved candidates. The selected default must support both English and Vietnamese; a multilingual cross-encoder will be chosen and measured during Phase 7 rather than fixed during Phase 0. A no-op implementation keeps reranking optional and tests deterministic. Reranker scores are retained separately from original retrieval scores.

### Generation and citations

An `LLMProvider` interface exposes generation and structured tool-calling capabilities plus optional token-usage metadata. Initial implementations are:

- An OpenAI-compatible cloud provider configured only through environment variables.
- An optional Ollama provider configured by base URL and model name.
- A deterministic fake used by tests; it is not exposed as a production answer generator.

The application does not require Ollama. If no usable LLM provider is configured, ingestion, retrieval, and retrieval evaluation remain available; generation endpoints return a clear configuration error or can return retrieval-only results when explicitly requested.

The dedicated citation subsystem constructs immutable evidence records before generation. It
deduplicates chunks in final evidence order, assigns response-local keys such as `[S1]`, and
captures logical-document ID, document-version ID, version label/effective interval, source
filename, chunk ID, page/section, rank, retrieval score, language, and a bounded source excerpt.
The prompt requires factual claims to cite those keys and to state that evidence is insufficient
when context does not support a claim. After generation, the validator resolves only emitted keys
that exist in the supplied evidence set, rejects unknown or missing keys, and normalizes repeats.
Evidence outside the resolved immutable version scope is rejected during citation construction.
The API returns cited records by default and full evidence plus validation metadata only in debug
mode. Excerpts are deterministic Unicode-safe prefixes of supplied source chunk text and are never
LLM-generated. Citation validation proves key-to-evidence provenance, not semantic faithfulness;
automatic claim/evidence verification remains future evaluation work.

### Evaluation

The repository contains a small, versioned bilingual corpus and JSONL question set. It includes English and Vietnamese questions, multiple versions of at least one logical document, exact-version questions, current-version questions, and version-difference questions. Each record includes a stable question ID, language, version scope, relevant logical-document/version/chunk identifiers, and reference answer or change annotations when available.

Retrieval evaluation computes Recall@K, Hit Rate@K, and MRR from labeled relevance data. Generation evaluation computes citation accuracy using deterministic citation/reference checks. Groundedness/faithfulness and answer relevance can use a configured judge provider; when no judge is configured, their status is recorded as `not_executed` with a reason, never as a fabricated score.

Each run records dataset version/hash, timestamp, Git commit when available, provider/model identities, chunking and retrieval configuration, per-example outputs, aggregate metrics, latency summaries, and errors. Results are stored in PostgreSQL for API access and exported to versionable JSON/CSV artifacts under a configurable results directory.

### Controlled agent layer

The agent layer is added only after the direct RAG pipeline works. It uses a small allow-list of typed tools:

- `search_knowledge_base`
- `get_document_metadata`
- `summarize_document`
- `compare_documents`

The orchestrator limits tool names, argument schemas, call count, and time budget. Version-aware arguments are typed, and `compare_documents` requires two explicit version IDs. Tools call the same application services as REST endpoints; the comparison application service exists before the agent exposes it. Direct `/query` remains the primary path and does not depend on the agent.

### Observability

Structured JSON logs include request/correlation ID, operation, document or query identifiers where safe, duration, provider/model, candidate counts, and outcome. Timers cover parsing, chunking, embedding, dense retrieval, lexical retrieval, fusion, reranking, LLM generation, and total request latency. Provider token usage is logged and returned to internal metrics only when supplied by the provider. Raw document text, prompts, API keys, and full user queries are not logged by default.

## 5. Core interfaces

The exact Python types will be refined during implementation, while these boundaries should remain stable.

```python
class DocumentParser(Protocol):
    def parse(self, source: StoredDocument) -> ParsedDocument: ...

class ChunkingStrategy(Protocol):
    def chunk(self, document: ParsedDocument, config: ChunkingConfig) -> list[Chunk]: ...

class VersionScopeResolver(Protocol):
    async def resolve(self, selector: VersionSelector) -> ResolvedVersionScope: ...

class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...

class DenseRetriever(Protocol):
    async def search(self, query: str, options: RetrievalOptions) -> list[Candidate]: ...

class LexicalRetriever(Protocol):
    async def search(self, query: str, options: RetrievalOptions) -> list[Candidate]: ...

class Reranker(Protocol):
    async def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]: ...

class LLMProvider(Protocol):
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

class VersionComparisonService(Protocol):
    async def compare(self, request: VersionComparisonRequest) -> VersionComparisonResult: ...
```

Provider factories select implementations from validated settings. Domain models cross module boundaries; vendor response objects do not.

## 6. Data model and ownership

### Version-aware domain model

```text
LogicalDocument 1 ─────── * DocumentVersion 1 ─────── * Chunk
      |                         |
      |                         +── optional supersedes_version_id
      +── stable identity       +── immutable source snapshot
```

| Entity | Important fields | Invariants and purpose |
|---|---|---|
| `logical_documents` | `logical_document_id`, stable `key`, title, description, owner/domain, created/updated timestamps | Represents the continuing policy/manual/contract independent of any file version. `key` is unique within the future tenant/scope boundary. |
| `document_versions` | `document_version_id`, `logical_document_id`, `version_label`, `revision_number`, `version_ordinal`, `effective_from`, `effective_to`, `published_at`, `supersedes_version_id`, `language`, source filename, checksum, storage key, indexing status, business status, ingestion/index config | Represents one immutable source snapshot. `(logical_document_id, version_label, revision_number)` and `(logical_document_id, version_ordinal)` are unique. A revision number permits a corrected source snapshot to retain its business-facing version label without mutation. A self-reference may describe the predecessor without making labels machine-sortable. |
| `chunks` | `chunk_id`, `document_version_id`, ordinal, text, token count, page, section path, section fingerprint, language, search vector | Belongs to exactly one version. Section path/fingerprint will later help comparison alignment, but does not claim that matching sections are identical. |

`version_ordinal` is an application-assigned monotonic integer used only for explicit ordering and comparison navigation within a logical document. It never determines the current version. `version_label` remains a human label such as `2026`, `v2.1`, or `Approved 2026-01`; code must not guess ordering or current status from that label. `effective_from` supports as-of retrieval but also does not imply current status. Business status (`draft`, `active`, `superseded`, `withdrawn`) is separate from indexing status (`pending`, `processing`, `ready`, `failed`, `deleting`). At most one version is `active` for a logical document, enforced by a PostgreSQL partial unique constraint unless a later domain requirement explicitly allows parallel active variants. Activation is allowed only for a `ready` version and must explicitly supersede/deactivate any prior active version in the same transaction.

The default query policy is `current`: resolve the single version whose business status is explicitly `active` and whose indexing status is `ready`. There is no fallback to upload time, filename, version label, publication/effective date, or greatest version ordinal. If no version is both active and ready, current-version retrieval fails with an explicit domain error; it must not silently retrieve a superseded or draft version. Callers can instead request one or more exact `document_version_id` values, `all` versions, or an `as_of` timestamp. The resolved IDs are recorded in debug metadata so version selection is observable.

### PostgreSQL: system of record for application data

PostgreSQL owns:

- Logical-document identity and the relationships between immutable versions.
- Per-version label/order, language, effective dates, predecessor, filename, media type, checksum, storage key, lifecycle status, and timestamps.
- Index version and the parser, chunking, embedding, and schema configuration used.
- Chunk identity, ordered text, token count, page, section path/fingerprint, and document-version relationship.
- Searchable `tsvector` data and lexical indexes.
- Evaluation run configuration, status, per-example results, and aggregates.

Suggested indexing lifecycle states are `pending`, `processing`, `ready`, `failed`, and `deleting`. A version becomes queryable only when indexing reaches `ready` and its business status permits the selected query policy.

### Qdrant: derived vector index

Qdrant owns embeddings and the minimal payload required for filtered vector search: `chunk_id`, `logical_document_id`, `document_version_id`, chunk ordinal, language, page, and section path. Model, revision, dimension, and indexing state are recorded on the authoritative version row in PostgreSQL. Full chunk text and authoritative metadata remain in PostgreSQL. Qdrant records are reconstructable from PostgreSQL plus the embedding provider.

### File storage: original document bytes

The initial default is a bind-mounted local file store behind a `DocumentStore` interface. PostgreSQL stores a storage key, not an absolute host path. This keeps local Docker setup simple and leaves a future S3-compatible implementation possible. File writes use generated document IDs and never trust upload filenames as paths.

### Identifiers and metadata

- `logical_document_id`: application-generated UUID for the stable conceptual document.
- `document_version_id`: application-generated UUID for one immutable uploaded version.
- `chunk_id`: deterministic from document-version ID, index version, and chunk ordinal, allowing safe re-indexing.
- `version_label`: human-readable and not used alone for ordering.
- `version_ordinal`: explicit ordering within a logical document.
- `language`: validated BCP 47-style code, initially `en` or `vi`; explicit metadata may override conservative detection.
- `filename`: sanitized display metadata; not an identifier.
- `page`: one-based page number when the parser can establish it; otherwise null.
- `section`: nearest available heading/title; otherwise null.

## 7. Main flows

### Ingestion flow

1. API validates content type, extension, and size, streams bytes to the document store, and calculates SHA-256.
2. The caller explicitly creates/selects a logical document and supplies a version label, language, and optional effective/predecessor metadata.
3. Service validates uniqueness/order and creates an immutable document-version row in `pending`; duplicate policy is applied using checksum within the logical document and future tenant/scope.
4. Status changes to `processing`; the selected parser emits ordered text blocks with source metadata.
5. Chunker produces chunks using the requested/default configuration and preserves page/section provenance plus version identity.
6. Chunks are stored in PostgreSQL and the immutable source snapshot is marked ready.
7. The separately invokable indexing service sends ready chunks to the multilingual embedding provider in bounded batches.
8. Vectors are upserted into a versioned Qdrant collection using deterministic point IDs and version-filter payloads; operational vector-index state is marked indexed or failed.
8. The document version is marked `ready` only after both stores succeed. At that boundary its source snapshot becomes immutable. Activation/supersession is a separate, explicit transactional catalog operation and is never inferred from ingestion order. Failures are recorded as `failed`, and a retry performs idempotent upserts without changing the source identity.

For the first implementation, this work may execute in-process. The service boundary and lifecycle states allow a background worker to be introduced later without changing API contracts.

### Query flow

1. API validates the question, language hint, logical-document filters, version selector, retrieval options, debug flag, and generation mode.
2. Version-scope resolver produces an explicit set of allowed ready version IDs. A `current` selector requires an explicitly active and ready version; missing, ambiguous, or nonexistent selectors fail rather than silently broadening scope or choosing a version heuristically.
3. Retrieval service runs dense search and, when enabled, language-appropriate lexical search using the same version filters.
4. Fusion deduplicates candidates by `chunk_id` and computes RRF ranks.
5. Optional multilingual reranker orders the candidate pool and selects `rerank_top_n`.
6. Context builder selects chunks within the context budget and assigns citation keys containing version provenance.
7. LLM provider receives the evidence-only prompt in the requested answer language.
8. Citation validator resolves cited keys and flags malformed/unknown or out-of-scope citations.
9. Response returns answer, structured version-aware citations, timing metadata, and debug scores/resolved version scope only when requested.

### Version comparison flow (future Phase 9A)

1. API/service requires one logical document and two explicit ready version IDs belonging to it; `base` and `target` order is never inferred.
2. Candidate sections are retrieved independently inside each version. Headings, normalized section paths, fingerprints, and semantic similarity provide alignment signals.
3. Alignment classifies section pairs as added, removed, modified, or unchanged and records confidence. It must preserve unmatched sections rather than forcing a pair.
4. Relevant changed pairs are selected for the user's question, or all changes are summarized within bounded limits.
5. Generation receives labeled base/target evidence and must describe only supported changes.
6. Response returns structured change items and citations from both versions. One-sided citations are valid for additions/removals; modified claims should normally cite both sides.

Comparison is not a plain global text diff: parsing and formatting can shift chunk boundaries. The first implementation favors interpretable heading/path alignment plus semantic candidate matching. Exact text diff may be applied only after sections are aligned. Alignment quality receives its own evaluation fixtures.

### Deletion flow

1. Mark the selected document version `deleting` so queries exclude it, and prevent deletion if retention rules or active-version constraints require it.
2. Delete Qdrant points filtered by document-version/index identity.
3. Delete chunk/application rows transactionally.
4. Delete original bytes.

Deleting a logical document is a separate explicit operation and cascades through all versions only after the caller confirms that broader scope. Removing a predecessor does not rewrite historical comparison records.

Cross-store deletion cannot be one atomic transaction. Operations are idempotent, states are explicit, and incomplete deletion is retryable. A later reconciliation command can compare PostgreSQL records with Qdrant points and stored files.

## 8. Configuration and security

Pydantic Settings loads environment variables and optional local `.env` files. `.env.example` contains names and safe placeholders only. Secrets are never committed or logged.

Configuration groups include:

- PostgreSQL and Qdrant connection details.
- File size/type limits and storage root.
- Chunk size, overlap, multilingual tokenizer, language handling, and parsing options.
- Embedding provider, model, dimension, batch size, and device.
- Dense/lexical candidate counts, `top_k`, RRF constant, and `rerank_top_n`.
- Reranker provider and model.
- LLM provider, endpoint, model, API key, timeout, and retry limits.
- Debug response policy and evaluation result path.

Production hardening such as authentication, authorization, malware scanning, tenant isolation, rate limiting, and secret management is intentionally documented as required future work rather than falsely implied by the portfolio baseline.

## 9. Consistency, failure handling, and reproducibility

- Database transactions protect PostgreSQL writes, but PostgreSQL, Qdrant, and file storage do not share a transaction.
- Deterministic IDs and idempotent upserts make ingestion retryable.
- Lifecycle states ensure partially indexed documents are not searchable.
- Timeouts, bounded retries with backoff, and explicit failure records cover external providers.
- A content checksum supports duplicate detection and data-integrity checks.
- Index versions prevent vectors from different embedding dimensions/models being mixed.
- Immutable version IDs and explicit version scopes prevent accidental mixing of current and superseded evidence.
- Database constraints and service rules make current-version activation explicit and prevent heuristic fallback.
- Ready source snapshots are immutable; changed source content creates a new version/revision so historical citations and evaluations remain reproducible.
- Model repository revisions and embedding prefix strategy are recorded for reproducibility.
- Every evaluation result captures enough configuration to reproduce the run.

## 10. Important technology choices and trade-offs

| Area | Proposed choice | Reason | Trade-off |
|---|---|---|---|
| Application shape | Modular FastAPI monolith | Easy to understand, test, and run | Long ingestion can eventually need a worker |
| Metadata access | SQLAlchemy 2.x + Alembic + PostgreSQL | Explicit models, migrations, mature async support | More code than a high-level RAG framework |
| Vector index | Qdrant | Required, strong filtering and Docker support | Cross-store consistency must be managed |
| Document versioning | Stable logical document + immutable document versions | Supports current/exact/as-of retrieval and reliable citations | Requires explicit grouping and lifecycle semantics |
| Default embeddings | `intfloat/multilingual-e5-small` via `sentence-transformers` | 384 dimensions, English/Vietnamese support, local and reproducible | About 471 MB weights; slower/larger than the 133 MB English-only BGE baseline |
| Lexical retrieval | PostgreSQL full-text search with `english`/`simple` configurations | Reuses required database and preserves transparent logic | No native Vietnamese stemming; native ranking is not true BM25 |
| Hybrid fusion | Reciprocal Rank Fusion | Transparent and score-scale independent | RRF constant/candidate sizes need evaluation |
| Reranker | Replaceable local cross-encoder | Improves precision without coupling pipeline | Adds latency and model download |
| Cloud LLM | OpenAI-compatible provider | Broad compatibility through one interface | Vendor-specific features may need adapters later |
| Local LLM | Optional Ollama provider | Local experimentation | Hardware-dependent and never required |
| Original files | Local mounted volume behind interface | Simple reproducible baseline | Not horizontally scalable like object storage |
| Background jobs | In-process initially, explicit lifecycle | Keeps initial architecture small | Upload request latency and restart behavior are limited |

## 11. Decisions requiring approval before relevant implementation

These are reversible defaults, but they affect dependencies and data/index design and should be confirmed before their phases:

1. Use `intfloat/multilingual-e5-small` through `sentence-transformers` as the default embedding provider, pinning/recording a model revision and applying required E5 query/passage prefixes.
2. Use PostgreSQL full-text search for Phase B lexical retrieval and describe it accurately as lexical search, not claim exact BM25. If exact BM25 is required, select a PostgreSQL extension or a separate maintained index.
3. Use an OpenAI-compatible cloud provider as the first cloud LLM adapter, with Ollama as optional.
4. Store original uploaded files in a Docker-mounted local volume for the initial version, behind a storage interface.
5. Run ingestion synchronously/in-process initially while retaining lifecycle states and service boundaries for a future worker.
6. Support English and Vietnamese from the first retrieval/evaluation baseline, with language-tagged fixtures and separate per-language metric reporting.
7. Model document identity as a stable logical document with immutable version records; require explicit grouping and explicit base/target selection rather than inferring identity or order from filenames.
8. Default ordinary queries only to the explicitly active and ready version, while supporting exact-version, all-version, and as-of selectors. Never infer current from timestamps, filenames, labels, ordinals, or ingestion order, and never silently fall back from a requested version to another version.
9. Make a ready `DocumentVersion` an immutable source snapshot. Any content or source-semantic metadata change creates a new version/revision; only non-semantic operational metadata may be updated in place.

No irreversible infrastructure decision is intended here. Database migrations, versioned vector collections, and provider interfaces are specifically used to keep these choices replaceable.

## 12. Embedding decision references

- [Multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-small): multilingual support, 384-dimensional output, usage prefixes, model files, and limitations.
- [Multilingual E5 technical report](https://arxiv.org/abs/2402.05672): training methodology and the small/base/large quality-efficiency trade-off.
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3): multilingual 1024-dimensional, 8192-token alternative for a future higher-resource experiment.
- [BGE small English model file](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/main/model.safetensors): confirms the existing candidate is English-specific and has a roughly 133 MB weight file.
