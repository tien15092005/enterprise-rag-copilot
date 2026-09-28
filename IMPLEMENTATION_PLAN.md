# Enterprise Knowledge Copilot Implementation Plan

## Working principles

- Implement and verify one phase at a time; do not claim later-phase capability early.
- Keep retrieval, fusion, reranking, citation, and evaluation logic explicit and unit-testable.
- Use provider interfaces and deterministic fakes so paid APIs and Ollama are never test-suite requirements.
- Preserve logical-document, immutable-version, and source provenance from parsing through the API response.
- Never infer version grouping or ordering from filenames; require explicit catalog operations.
- Resolve `current` only from an explicitly active and ready version; never fall back to timestamps, labels, ordinals, or ingestion order.
- Treat successfully ingested document versions as immutable source snapshots; changed source content creates a new version/revision.
- Test English and Vietnamese behavior independently and report per-language results.
- Record experiment configuration and real outputs; never invent benchmark results.
- Add dependencies only in the phase that uses them and pin compatible ranges in `pyproject.toml`.

Commands below are the intended verification commands after each phase is implemented. They have not been executed during Phase 0 because the corresponding code does not exist yet.

## Phase 1 - Repository/backend skeleton

### Objective

Create the installable Python 3.12 project, configuration and logging foundations, FastAPI application, health endpoint, database session scaffolding, local development services, and test harness. Establish clean module boundaries—including placeholders for version-aware domain schemas—without implementing ingestion, comparison, or RAG.

### Files/modules affected

- `pyproject.toml`
- `.env.example`, `.gitignore`
- `app/main.py`
- `app/api/router.py`, `app/api/routes/health.py`, `app/api/error_handlers.py`
- `app/core/config.py`, `app/core/logging.py`, `app/core/exceptions.py`
- `app/db/session.py`, initial Alembic configuration
- `app/models/` base API/domain schemas
- `tests/conftest.py`, `tests/api/test_health.py`, `tests/test_config.py`
- `Dockerfile`, `docker-compose.yml`
- Initial `README.md`

### Test criteria

- Package installs under Python 3.12.
- Settings validate required/non-required variables without exposing secrets.
- `GET /health` returns a typed response and distinguishes liveness from dependency readiness.
- Structured logs contain level, timestamp, event, and correlation ID.
- FastAPI OpenAPI schema is generated.
- PostgreSQL and Qdrant containers become healthy when Docker Desktop is running.

### Verification commands

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/api/test_health.py tests/test_config.py
python -m ruff check app tests
python -m mypy app
docker compose config
docker compose up -d postgres qdrant
docker compose ps
```

## Phase 2 - Version-aware document ingestion

### Objective

Implement the logical-document/version catalog, safe upload storage, parsers for English and Vietnamese PDF/DOCX/TXT/Markdown, structured parsed blocks, configurable token-aware chunking, metadata persistence, separate business/indexing lifecycle states, listing, and version-scoped deletion. Embedding/indexing remains a stubbed boundary until Phase 3. Do not implement section comparison yet.

### Files/modules affected

- `app/ingestion/parsers/base.py`, `pdf.py`, `docx.py`, `text.py`, `markdown.py`
- `app/ingestion/chunker.py`, `storage.py`, `service.py`
- `app/db/models.py`, `app/db/repositories/documents.py`, `versions.py`
- Alembic logical-document, document-version, and chunk migration
- `app/models/documents.py`, `app/models/versions.py`
- `app/api/routes/documents.py`, `versions.py`
- `tests/ingestion/`, `tests/api/test_documents.py`
- `data/samples/` with small redistributable fixtures

### Test criteria

- Each supported format produces ordered text blocks.
- PDF pages and Markdown/DOCX sections are retained when available.
- Chunk size and overlap are configurable and validated (`overlap < size`).
- Creating a first upload creates one logical document and one immutable version; later uploads can explicitly join the logical document.
- Similar filenames never cause automatic grouping, and human version labels are never used as the sole ordering mechanism.
- Logical-document key, version label, and version ordinal uniqueness constraints are enforced.
- At most one version is active per logical document, and activation/supersession is distinct from indexing state.
- Only a ready version can be activated, and no heuristic fallback selects a current version when none is active and ready.
- Once ingestion succeeds, attempts to replace source bytes or source-semantic metadata are rejected; a changed source creates a new version/revision.
- Every chunk preserves `logical_document_id`, `document_version_id`, version label/ordinal, language, filename, page/section, ordinal, and `chunk_id`.
- English and Vietnamese Unicode text/metadata survive parsing and storage without lossy normalization.
- Upload filename cannot escape the storage root; unsupported/oversized files fail cleanly.
- Repeated ingestion and version-scoped deletion follow documented idempotency and active-version rules.
- API schemas list logical documents and their versions without conflating the two identities.

### Verification commands

```powershell
python -m pytest tests/ingestion tests/api/test_documents.py
python -m pytest tests/ingestion/test_chunker.py -k metadata
python -m ruff check app tests
python -m mypy app
python -m alembic upgrade head
```

## Phase 3 - Embeddings + Qdrant

### Objective

Add the provider-independent embedding interface, `intfloat/multilingual-e5-small` implementation, deterministic fake, pinned/recorded model revision, E5 query/passage prefixing, versioned Qdrant collections, batched vector upserts, version-filtered deletion, and dense similarity retrieval for English and Vietnamese.

### Files/modules affected

- `app/embeddings/base.py`, `factory.py`, `sentence_transformers.py`
- `app/retrieval/dense.py`
- `app/db/repositories/chunks.py`
- Qdrant client/configuration modules
- Ingestion service indexing integration
- `tests/embeddings/`, `tests/retrieval/test_dense.py`
- `tests/integration/test_qdrant_index.py`

### Test criteria

- Retrieval code works against the embedding protocol and deterministic fake.
- Vector dimension/model identity is validated against collection metadata.
- Batch embedding preserves input order.
- Qdrant payload retains logical-document/version provenance, language, and index identity and supports exact version filters.
- E5 prefixes are applied consistently to queries and passages, and model revision/config is recorded.
- Re-indexing is idempotent; version deletion removes only that version's points.
- Container-backed integration tests index a tiny bilingual, multi-version corpus and retrieve expected English and Vietnamese chunks from an exact version.

### Verification commands

```powershell
python -m pytest tests/embeddings tests/retrieval/test_dense.py
docker compose up -d postgres qdrant
python -m pytest -m integration tests/integration/test_qdrant_index.py
python -m ruff check app tests
python -m mypy app
```

## Phase 4 - Baseline RAG

### Objective

Implement the `/query` pipeline using dense retrieval, explicit version-scope resolution, context budgeting, stable evidence keys, validated structured citations, evidence-only prompting, cloud and optional Ollama LLM adapters, timing instrumentation, and deterministic no-evidence refusal. Support current, exact, all-version, and as-of selectors. The prompt must explicitly require an insufficient-evidence response when needed and answer in the requested English or Vietnamese language.

### Files/modules affected

- `app/generation/providers/base.py`, `openai_compatible.py`, `ollama.py`
- `app/generation/context.py`, `prompts.py`, `service.py`
- `app/retrieval/service.py`
- `app/retrieval/version_scope.py`
- `app/models/query.py`, `app/models/retrieval.py`
- `app/api/routes/query.py`, dependency wiring
- `app/core/telemetry.py`
- `tests/generation/`, `tests/api/test_query.py`

### Test criteria

- `/query` returns a schema-valid answer and evidence using a fake provider.
- Default queries resolve only the active/current ready version; exact selectors never fall back to another version.
- `all` and `as_of` selectors resolve deterministically and reject ambiguous/missing scope.
- Dense retrieval applies the resolved version IDs before ranking.
- Tests never require a cloud key or Ollama service.
- No-provider mode supports retrieval-only requests and reports generation configuration errors clearly.
- Context length is bounded and construction is deterministic.
- Citation keys resolve only to evidence supplied to the model; unknown keys are rejected.
- Prompt contains evidence boundaries and insufficient-evidence instruction.
- Retrieval, generation, and total latency are measured.
- Token usage is captured when the provider supplies it.

### Verification commands

```powershell
python -m pytest tests/generation tests/api/test_query.py
python -m pytest -m "not external"
python -m ruff check app tests
python -m mypy app
```

## Phase 5 - Citations

### Objective

Add stable inline citation keys, strict citation resolution/validation, source-rich version-aware citation objects, page/section reporting, and debug-only retrieval score and resolved-version details.

### Files/modules affected

- `app/generation/citations.py`, `context.py`, `prompts.py`, `service.py`
- `app/models/query.py`, `app/models/retrieval.py`
- `tests/generation/test_citations.py`
- `tests/api/test_query_citations.py`

### Test criteria

- Known citation keys map to the correct logical document, immutable version, and chunk metadata.
- Duplicate citations are handled deterministically; unknown keys are rejected or flagged.
- Logical-document ID, document-version ID, version label/effective date, filename, chunk ID, page, and section appear when available.
- Citations outside the resolved version scope are rejected.
- Retrieval and later reranker scores are absent by default and present only in debug mode.
- An answer without sufficient supporting context follows the insufficient-evidence behavior.

### Verification commands

```powershell
python -m pytest tests/generation/test_citations.py tests/api/test_query_citations.py
python -m pytest -k "citation or metadata"
python -m ruff check app tests
python -m mypy app
```

## Phase 6 - Hybrid retrieval

### Objective

Add language-aware PostgreSQL lexical/full-text retrieval over chunk text and explicit Reciprocal Rank Fusion with dense results. Use `english` text-search configuration for English and `simple` for Vietnamese initially. Support dense and hybrid modes without hiding score or version provenance.

### Files/modules affected

- PostgreSQL chunk search-vector migration and indexes
- `app/retrieval/lexical.py`, `fusion.py`, `hybrid.py`, `service.py`
- `app/db/repositories/chunks.py`
- Retrieval request/response schemas and settings
- `tests/retrieval/test_lexical.py`, `test_fusion.py`, `test_hybrid.py`
- PostgreSQL integration tests

### Test criteria

- Lexical retrieval finds exact enterprise terms that deterministic dense search misses.
- Vietnamese text with diacritics is indexed and queried predictably; limitations of tokenization/stemming are documented.
- Fusion deduplicates by `chunk_id`, is deterministic, and retains component ranks/scores.
- Logical-document and resolved-version filters apply consistently to both retrieval branches.
- Dense-only behavior remains backward compatible.
- `top_k` and branch candidate limits are configurable and bounded.

### Verification commands

```powershell
docker compose up -d postgres qdrant
python -m alembic upgrade head
python -m pytest tests/retrieval/test_lexical.py tests/retrieval/test_fusion.py tests/retrieval/test_hybrid.py
python -m pytest -m integration -k hybrid
python -m ruff check app tests
python -m mypy app
```

## Phase 7 - Reranker

Status: implemented. The pinned multilingual cross-encoder is lazy and injectable; the
`hybrid_rerank` mode bounds the post-RRF pool, preserves all retrieval metadata and immutable
version provenance, exposes separate latency, and supports retrieval-only execution.

### Objective

Introduce a replaceable reranker interface, optional multilingual local cross-encoder selected using English/Vietnamese evaluation, deterministic fake/no-op implementations, bounded candidate reranking, and separate reranker latency/score reporting.

### Files/modules affected

- `app/reranking/base.py`, `cross_encoder.py`, factory/configuration
- `app/retrieval/service.py`
- Query schemas and debug metadata
- `tests/reranking/`, `tests/retrieval/test_reranked_pipeline.py`

### Test criteria

- Pipeline can enable or disable reranking without code changes.
- The chosen baseline reranker declares English and Vietnamese support and is evaluated per language before becoming the default.
- `rerank_top_n` is validated against candidate counts and configured limits.
- Fake reranker produces a predictable ordering in unit tests.
- Original dense, lexical, and fused scores are retained beside reranker scores.
- Reranker failure follows an explicit policy (fail request or configured fallback) and is logged.
- Reranking latency is measured separately.

### Verification commands

```powershell
python -m pytest tests/reranking tests/retrieval/test_reranked_pipeline.py
python -m pytest -k rerank
python -m ruff check app tests
python -m mypy app
```

## Phase 8 - Evaluation

Status: implemented. A repository-owned bilingual, multi-version dataset, deterministic retrieval
metrics, version-leakage accounting, bounded configuration comparison, JSON/CSV persistence,
reproducible CLI, and generated human-readable report compare dense, hybrid, and hybrid-reranked
outcomes. Generation judge scoring remains deliberately unexecuted without a configured judge.

### Objective

Create a small licensed/repository-owned bilingual, multi-version evaluation corpus; version-scoped annotations; deterministic retrieval metrics; citation checks; optional judge-based generation metrics; experiment persistence; JSON/CSV export; and evaluation API endpoints. Add comparison fixtures and metrics only for the later comparison subphase; do not fabricate comparison scores before then.

### Files/modules affected

- `data/evaluation/corpus/`, `data/evaluation/questions.jsonl`
- `app/evaluation/datasets.py`, `retrieval_metrics.py`, `generation_metrics.py`, `runner.py`
- `app/models/evaluation.py`
- Evaluation database migration/repository
- `app/api/routes/evaluation.py`
- `scripts/run_evaluation.py`
- `tests/evaluation/`, evaluation integration tests
- `docs/evaluation.md`

### Test criteria

- Recall@K, Hit Rate@K, and MRR match hand-calculated fixture values.
- Metrics are reported overall and separately for English, Vietnamese, current-version, and exact-version subsets when sample sizes permit.
- Fixtures detect leakage from superseded versions when current-only retrieval is requested.
- Citation accuracy calculation is deterministic and tested against correct/incorrect references.
- Judge-based groundedness/faithfulness and answer relevance are marked `not_executed` with a reason when unavailable.
- Results contain dataset hash/version, full run configuration, per-example records, aggregates, and errors.
- JSON and CSV outputs are machine-readable and round-trip successfully.
- README labels only executed results as measured.

### Verification commands

```powershell
python -m pytest tests/evaluation
python scripts/run_evaluation.py --dataset data/evaluation/questions.jsonl --retrieval-only --output data/evaluation/results/baseline.json
python -m json.tool data/evaluation/results/baseline.json > $null
python -m ruff check app tests scripts
python -m mypy app
```

## Phase 9A - Version comparison application service

Status: implemented. The typed comparison API validates two explicit immutable versions, loads
only their authoritative sections, aligns them deterministically with an optional multilingual
semantic fallback, classifies four change types, emits sentence-level diffs and side-aware
citations, and optionally generates a citation-validated summary.

### Objective

Implement an explicit two-version comparison application service and `/compare` API. Load sections
only from each selected version, align them using interpretable metadata and semantic signals,
classify changes, optionally explain the deterministic result, and return citations from left and
right versions. Do not add agent orchestration in this subphase.

### Files/modules affected

- `app/comparison/models.py`, `alignment.py`, `service.py`
- `app/api/routes/comparisons.py`
- Comparison request, change-item, and response schemas
- Bilingual multi-version comparison fixtures
- `scripts/verify_comparison.py`
- `tests/comparison/`, `tests/api/test_compare.py`

### Test criteria

- Base and target version IDs are explicit, distinct, ready, and belong to the same logical document.
- Alignment preserves added/removed sections and does not force low-confidence pairs.
- Change items classify added, removed, modified, or unchanged evidence with alignment confidence.
- Modified-change claims cite both versions; additions/removals cite the applicable side and retain the missing-side semantics.
- Comparison retrieval cannot leak chunks from a third version.
- English and Vietnamese fixtures cover section moves, edits, additions, removals, and formatting-only changes.
- A fake generator makes tests deterministic and free of paid APIs.

### Verification commands

```powershell
python -m pytest tests/comparison tests/api/test_compare.py
python -m pytest -k "comparison or version_alignment"
python scripts/verify_comparison.py
python -m ruff check app tests scripts
python -m mypy app
```

## Phase 9B - Controlled agent tools

Status: implemented. A bounded provider-neutral orchestration loop validates calls against four
typed tools, delegates to existing application services, preserves version/citation provenance,
enforces a configurable step ceiling, and exposes only sanitized execution traces.

### Objective

Add a minimal typed tool registry and single-agent orchestration layer over existing retrieval, metadata, summarization, and comparison application services. Limit tool access, calls, and execution time. Keep direct RAG and comparison APIs independent.

### Files/modules affected

- `app/agent/models.py`, `provider.py`, `tools.py`, `service.py`, `prompts.py`, `errors.py`
- Agent/tool request and response schemas
- `app/api/routes/agent.py`
- `tests/agent/`, `tests/api/test_agent.py`
- `scripts/verify_agent.py`

### Test criteria

- Only allow-listed tools can be called and all arguments pass Pydantic validation.
- Tool-call count and execution timeout are enforced.
- `search_knowledge_base` accepts an explicit version selector and calls the existing retrieval service.
- `get_document_metadata` exposes logical-document and version metadata without source bytes.
- `compare_documents` requires two explicit versions and calls the Phase 9A comparison service rather than implementing comparison inside the tool.
- Fake tool-calling LLM makes tests deterministic and free of paid APIs.
- Direct `/query` and `/compare` work when the agent layer is disabled.

### Verification commands

```powershell
python -m pytest tests/agent tests/api/test_agent.py
python -m pytest -k "agent or tool"
python scripts/verify_agent.py
python -m ruff check app tests scripts
python -m mypy app
```

## Phase 10 - Docker, tests, and documentation hardening

Status: implemented. The runtime uses a CPU-oriented multi-stage image, configurable bind-mounted
state, migration-on-startup, non-root/read-only container controls, fail-fast production settings,
an agent deadline, disabled-by-default evidence score floors, deterministic full-workflow E2E
coverage, and portfolio-ready operational documentation.

### Objective

Finish reproducible containers, migrations/startup behavior, full test coverage, security and failure-path checks, end-to-end smoke tests, and interview-quality documentation. Report only measured experiment results produced by Phase 8.

### Files/modules affected

- `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.env.example`
- Startup/health scripts if required
- All test suites and CI workflow
- `README.md`
- `docs/` architecture decision records, operations, API examples, and evaluation guide
- Sample scripts and data licensing/attribution

### Test criteria

- Fresh clone can be configured from `.env.example`, built, migrated, and started with documented commands.
- Containers run as non-root where practical and have health checks and persistent named volumes.
- Full unit suite works without network access, paid APIs, or Ollama.
- Opt-in integration/end-to-end suite exercises PostgreSQL, Qdrant, logical-document creation, multi-version upload, version-scoped query, bilingual citations, comparison, and version deletion.
- API examples match the generated OpenAPI schema.
- README covers all required sections and separates measured results, expected behavior, and future work.
- Secrets, local models, uploaded documents, and generated result artifacts follow explicit commit/ignore rules.

### Verification commands

```powershell
python -m pytest
python -m ruff check app tests scripts
python -m ruff format --check app tests scripts
python -m mypy app
docker compose config
docker compose build
docker compose up -d --wait
python -m pytest -m e2e
docker compose ps
```

## Phase gates and dependency order

```text
1 Skeleton
   -> 2 Version-aware ingestion
      -> 3 Embeddings + dense retrieval
         -> 4 Baseline RAG
            -> 5 Citations
               -> 6 Hybrid retrieval
                  -> 7 Reranker
                     -> 8 Evaluation
                        -> 9A Version comparison service
                           -> 9B Controlled agent tools
                              -> 10 Hardening and documentation
```

Each phase should end with a small review of code, test evidence, configuration changes, and deviations from `ARCHITECTURE.md`. Decisions that change data ownership, provider contracts, or public API schemas should be recorded in `docs/decisions/` once that directory exists.

## Phase 0 completion record

Environment inspected on 2026-09-27:

- Workspace: `D:\Projects\enterprise-rag-copilot`
- Existing files: none found, including hidden entries.
- Git repository: not initialized.
- Python: 3.12.2 (also 3.11 and 3.6 installed; 3.12 is the active interpreter).
- Git CLI: 2.53.0.windows.3.
- Docker CLI: 29.1.3.
- Docker Compose: v2.40.3-desktop.1.
- Docker engine: unavailable during inspection because the Docker Desktop Linux engine named pipe was not running.

Phase 0 creates only `ARCHITECTURE.md` and `IMPLEMENTATION_PLAN.md`. Phase 1 has not been implemented.

Phase 0 was amended before approval to make version-aware knowledge retrieval the product differentiator, replace the English-only embedding proposal with a multilingual baseline, require English/Vietnamese evaluation, and split version comparison services from later agent exposure. Phase 1 remains unstarted.
