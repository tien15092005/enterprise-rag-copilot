# Enterprise Knowledge Copilot

A production-oriented, version-aware Retrieval-Augmented Generation backend for enterprise
knowledge. The defining product requirement is explicit retrieval and comparison across immutable
versions of the same logical document.

## Current status

Phase 6 implements bilingual, version-aware hybrid retrieval on top of the evidence-grounded RAG
baseline:

- Immutable document-version records grouped under stable logical documents.
- Local filesystem source storage behind a replaceable storage protocol.
- PDF, DOCX, UTF-8 TXT, and Markdown parsing without OCR.
- Configurable deterministic token-window chunking.
- PostgreSQL persistence for logical documents, versions, chunks, and ingestion jobs.
- Explicit failed-ingestion state and structured API errors.
- SHA-256 duplicate-source detection within a logical document.
- Explicit activation: only a version marked `active` and indexed as `ready` is current.
- Replaceable embedding and vector-store interfaces.
- Lazy, process-cached `intfloat/multilingual-e5-small` embeddings pinned to a model revision.
- Idempotent Qdrant upserts with deterministic chunk UUIDs and version-filter payloads.
- Explicit vector indexing states (`not_indexed`, `indexing`, `indexed`, `failed`).
- Dense retrieval scoped to exactly one caller-supplied document version.
- PostgreSQL full-text lexical retrieval using `english` for English and `simple` for Vietnamese.
- Explicit Reciprocal Rank Fusion (RRF) with per-branch rank and score provenance.
- Dense and lexical branches filtered by the same resolved immutable version IDs.
- PostgreSQL provenance validation for every Qdrant search result.
- Explicit exact, current, all-version, and as-of scope resolution.
- Bounded ranked evidence contexts with stable citation keys.
- Replaceable OpenAI-compatible and optional Ollama generation providers.
- English/Vietnamese answer-language behavior and deterministic no-evidence refusal.
- Structured citations resolved only from evidence actually supplied to the LLM.

Reranking, version comparison, agents, and final evaluation benchmarking are intentionally not
implemented yet.

See [ARCHITECTURE.md](ARCHITECTURE.md) and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the
approved design and phased delivery plan.

## Local setup

Requirements:

- Python 3.12
- Docker with Docker Compose for PostgreSQL and Qdrant integration work

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Run the schema migration before starting the API:

```powershell
.venv\Scripts\python -m alembic upgrade head
.venv\Scripts\python -m uvicorn app.main:app --reload
```

The application exposes:

- `GET /health` for liveness.
- `GET /health/ready` for PostgreSQL readiness.
- `POST /api/documents` and `GET /api/documents` for the logical-document catalog.
- `GET /api/documents/{logical_document_id}` for document and version summaries.
- `POST /api/documents/{logical_document_id}/versions` for multipart ingestion.
- `GET /api/documents/{logical_document_id}/versions` for version history.
- `GET /api/document-versions/{document_version_id}` for ingestion and chunk status.
- `POST /api/document-versions/{document_version_id}/index` for explicit vector indexing.
- `POST /api/retrieval/dense` for engineering/debug exact-version retrieval.
- `POST /api/retrieval/lexical` for engineering/debug exact-version PostgreSQL FTS retrieval.
- `POST /api/retrieval/hybrid` for engineering/debug exact-version RRF retrieval.
- `POST /api/query` for version-scoped RAG answering with structured citations.
- `GET /docs` for generated OpenAPI documentation.

## Ingestion architecture

```text
multipart request
  -> validate logical document and version metadata
  -> stream original bytes to DocumentStorage
  -> calculate SHA-256 checksum
  -> persist processing version + ingestion job
  -> select PDF/DOCX/TXT/Markdown parser
  -> produce page/section-aware text blocks
  -> deterministic configurable chunking
  -> persist chunks
  -> mark version ready or explicitly failed
  -> optionally activate the ready version
```

Original binaries are stored under `data/uploads/` by default. PostgreSQL stores the opaque
storage key, original filename, and checksum; it does not store source binaries. Ingestion code
depends on the `DocumentStorage` protocol rather than filesystem paths, leaving an S3-compatible
implementation possible later.

## Document and version model

A `LogicalDocument` is the stable identity of a policy, manual, or similar knowledge asset. Each
`DocumentVersion` is an immutable source snapshot with an explicit version label, revision number,
ordinal, language, effective dates, checksum, and lifecycle statuses. Chunks belong to exactly one
version.

The current version is never inferred from filename, timestamp, label, ordinal, or upload order.
It is only the single version explicitly marked `active` while its indexing status is `ready`. A
new active version explicitly supersedes the previous active version. Changing ready source
content or source-semantic metadata requires creating a new version/revision.

## Dense, lexical, hybrid, and reranked retrieval

The default local provider is `intfloat/multilingual-e5-small` (384 dimensions), selected as a
lightweight multilingual baseline for English and Vietnamese. Its repository revision is pinned
in configuration. The provider applies E5's required conventions before encoding:

- User search text: `query: <text>`
- Stored chunk text: `passage: <text>`

Both query and passage vectors are L2-normalized. The model loads lazily on first indexing or
retrieval use and is reused for later requests; application import and unrelated tests do not load
or download model weights. Tests inject a deterministic fake through the `EmbeddingProvider`
boundary.

```text
ready DocumentVersion -> PostgreSQL chunks -> passage embeddings
  -> deterministic Qdrant upsert -> vector_index_status=indexed

query + explicit document_version_id -> query embedding
  -> Qdrant cosine search with mandatory version filter
  -> PostgreSQL chunk hydration and provenance validation -> ranked candidates
```

PostgreSQL is authoritative for source snapshots, chunks, text, lifecycle state, and provenance.
Qdrant owns only reconstructable embeddings plus the minimal filter/citation payload. Re-indexing
uses each chunk UUID as its point ID, so it replaces the same points instead of creating duplicates.
An indexing failure is recorded explicitly and never marks the version indexed.

Lexical retrieval searches authoritative PostgreSQL chunk text. English chunks use the `english`
text-search configuration, which provides English stemming. Vietnamese chunks use `simple`, which
preserves tokens and diacritics but provides no Vietnamese-specific stemming or sophisticated
linguistic tokenization. Expression GIN indexes support both configurations.

Hybrid mode retrieves bounded candidate lists from Qdrant and PostgreSQL using the exact same
resolved version IDs, then fuses their ranks with `1 / (RRF_K + rank)`. The default `RRF_K=60` is
a conventional starting point, not a claim of global optimality. Duplicate chunks are merged by
`chunk_id`; debug citation metadata retains dense rank/score, lexical rank/score, and final RRF
score. Raw cosine and PostgreSQL scores are never averaged. `/api/query` accepts
`retrieval_mode: "dense" | "hybrid" | "hybrid_rerank"`; dense remains the default for
compatibility until evaluation justifies changing it.

`hybrid_rerank` takes the bounded post-RRF pool, scores query/passage pairs with the local
cross-encoder, and returns only the configured final top K. Reranking can reorder or remove those
candidates but cannot introduce chunks, change immutable version provenance, or discard the dense,
lexical, or RRF scores. The response timing separates dense, lexical, fusion, and reranking work.
Debug citations expose dense rank/score, lexical rank/score, RRF rank/score, and reranker
rank/score; raw reranker outputs are relevance scores and are not presented as probabilities.

The selected baseline is
[`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1),
pinned to revision `1427fd652930e4ba29e8149678df786c240d8825`. It is an Apache-2.0,
118M-parameter multilingual MiniLM cross-encoder trained on mMARCO; its declared language set
includes English and Vietnamese. It is expected to be usable on CPU for a small candidate pool,
but each query/passage pair requires a joint forward pass, so it is slower and uses more memory
than rank fusion. The model loads lazily once per process and standard tests inject a fake.

Alternatives considered were
[`jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)
(278M parameters, longer context and strong multilingual results, but a larger CPU footprint and
CC-BY-NC-4.0 weights) and
[`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
(568M parameters, Apache-2.0 and multilingual, but materially heavier for this CPU-oriented
portfolio baseline). Model support does not establish quality improvement; Phase 8 evaluation is
required before making that claim.

## End-to-end RAG flow

```text
question + selector
  -> explicit version-scope resolution
  -> dense Qdrant + optional PostgreSQL lexical retrieval with identical version filters
  -> optional RRF fusion -> optional bounded multilingual reranking
  -> PostgreSQL provenance validation and metadata hydration
  -> bounded ranked evidence context with [S1], [S2], ... keys
  -> configured LLM provider
  -> citation-key validation
  -> grounded answer, structured citations, resolved scope, and timings
```

Supported selectors are:

- `exact`: one explicit immutable `document_version_id`.
- `current`: only the single version explicitly marked `active`, ingestion-ready, and
  vector-indexed. There is no timestamp, filename, label, ordinal, or upload-order fallback.
- `all`: all ingestion-ready and vector-indexed versions of the selected logical document.
- `as_of`: the unique ready/indexed version whose explicit effective interval contains the
  requested timestamp. Missing and overlapping intervals return domain errors.

The context builder preserves dense ranking and stops at `RAG_MAX_CONTEXT_CHARACTERS`, retaining
higher-ranked evidence first. The dedicated citation subsystem then deduplicates repeated chunk
IDs, assigns response-local keys in final evidence order (`[S1]`, `[S2]`, ...), and creates the
authoritative typed citation records before any LLM call. Each record preserves logical document,
immutable version, version label/effective interval, chunk ID, source filename, page/section,
rank, dense retrieval score, language, and a source-derived excerpt.

The LLM can produce only citation keys, never citation metadata. Its references are validated
against the evidence set: unknown keys and answers with no supplied key are rejected; repeated
keys are normalized in first-reference order. By default the response returns only cited records;
`debug: true` exposes full supplied evidence and validation metadata. Citation keys are stable
within one response/context build, not permanent identifiers across queries.

Excerpts are never model-generated or paraphrased. They are Unicode-safe prefixes of the actual
chunk text supplied to the LLM, bounded by `CITATION_EXCERPT_MAX_CHARACTERS` (default 500) and,
when context is tighter, by the text that actually fit in the context. Version-scope enforcement
rejects any evidence whose immutable version is outside the resolved scope.

Questions containing Vietnamese orthographic markers are answered in Vietnamese; other questions
default to English. Source excerpts remain in their original language. If retrieval produces no
usable evidence, the service returns a deterministic language-appropriate insufficient-evidence
response without calling the LLM. No unvalidated similarity threshold is used in this phase.

Set `retrieval_only` to `true` on `/api/query` to resolve any supported version selector and return
the bounded evidence in `citations` without calling an LLM. In this mode `answer` is `null`,
generation latency is zero, and no LLM provider needs to be configured. Debug validation is marked
`not_applicable` because no model-produced keys exist. When `retrieval_mode` is `hybrid_rerank`,
the complete retrieval and reranking stages still execute.

## LLM providers

Set `LLM_PROVIDER=openai_compatible` with `OPENAI_API_KEY` and `OPENAI_MODEL`, or set
`LLM_PROVIDER=ollama` with `OLLAMA_MODEL`. Ollama remains optional and no model is downloaded
automatically. Provider calls are isolated behind `LLMProvider`; tests and controlled verification
use a deterministic fake. Generation latency and provider token usage are returned when available.

Embedding and reranker caches default to `data/model_cache`, which resolves beneath this project's
`D:` directory. On Windows, `EMBEDDING_CACHE_DIRECTORY`, `RERANKER_CACHE_DIRECTORY`, and optionally
`HF_HOME` may instead point to `D:\AI_Cache\huggingface`. Inspect the resolved path before opting
into the real-model test. No reranker is downloaded automatically by the standard test suite, and
existing caches are not moved or deleted automatically.

## Sample API usage

Create a logical document:

```powershell
$document = Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/documents `
  -ContentType application/json `
  -Body '{"stable_key":"hr-policy","title":"HR Policy","domain":"HR"}'
```

Upload and activate a version with `curl.exe`:

```powershell
curl.exe -X POST "http://localhost:8000/api/documents/$($document.logical_document_id)/versions" `
  -F "file=@data/samples/employee_handbook_en.md;type=text/markdown" `
  -F "version_label=2026" `
  -F "version_ordinal=1" `
  -F "revision_number=1" `
  -F "language=en" `
  -F "activate=true" `
  -F "chunk_size=400" `
  -F "chunk_overlap=50"
```

Index that version and run an exact-version dense search:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/api/document-versions/$versionId/index"

Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/retrieval/dense `
  -ContentType application/json `
  -Body (@{
    query = "How many remote-work days are allowed?"
    document_version_id = $versionId
    top_k = 5
  } | ConvertTo-Json)
```

Ask the explicitly active current version:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/query `
  -ContentType application/json `
  -Body (@{
    question = "How many remote-work days are allowed?"
    logical_document_id = $document.logical_document_id
    version_selector = @{ kind = "current" }
    top_k = 5
    debug = $false
  } | ConvertTo-Json -Depth 4)
```

For exact scope, use
`{"kind":"exact","document_version_id":"<uuid>"}`. For historical scope, use
`{"kind":"as_of","as_of":"2026-01-15T00:00:00Z"}`. The `all` selector uses
`{"kind":"all"}`.

## Migrations

```powershell
# Apply
.venv\Scripts\python -m alembic upgrade head

# Revert Phase 2 schema
.venv\Scripts\python -m alembic downgrade base

# Reapply
.venv\Scripts\python -m alembic upgrade head
```

The PostgreSQL migration also installs database triggers that prevent mutation of ready source
snapshots and their chunks. Application-level checks provide the same source-version protection
for ORM writes.

## Verification

```powershell
.venv\Scripts\python -m ruff format --check app tests
.venv\Scripts\python -m ruff check app tests
.venv\Scripts\python -m mypy app
.venv\Scripts\python -m pytest
docker compose config
```

No API keys belong in the repository. Copy `.env.example` to `.env` and provide secrets locally.

## Known limitations

- Ingestion executes synchronously in the API process; background workers are future work.
- PDF parsing handles embedded text only. Scanned/image-only PDFs return an explicit unsupported
  content error because OCR is outside Phase 2.
- DOCX page numbers are unavailable because the document format does not retain reliable rendered
  pagination.
- TXT and Markdown must be UTF-8 encoded.
- Chunking uses transparent whitespace-token windows and does not claim linguistic optimality.
- Deletion endpoints and retention policies are deferred; stored failed-upload sources remain
  available for diagnosis.
- PostgreSQL `simple` search is only a transparent Vietnamese lexical baseline; it has no
  Vietnamese stemmer and may miss differently segmented or inflected terms.
- Hybrid candidate limits and `RRF_K` have not been tuned by the later evaluation phase.
- Reranker candidate/final limits, latency, and English/Vietnamese quality have not been
  benchmark-tuned; no measured improvement over hybrid retrieval is claimed yet.
- Qdrant and PostgreSQL do not share a transaction. Explicit failed state and idempotent point IDs
  make retries safe, while reconciliation tooling remains future work.
- Dense retrieval quality has only been smoke-tested with the included English and Vietnamese
  samples; no benchmark or evaluation score is claimed.
- Citation-key validation proves that a key maps to supplied evidence, not that every natural-
  language claim is semantically entailed by that evidence. Automatic claim/evidence verification
  is deferred.
- Version comparison, agents, and final evaluation benchmarking are not implemented.
