from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Chunk
from app.db.session import Database
from app.embeddings import create_embedding_provider
from app.evaluation.datasets import dataset_sha256, load_dataset
from app.evaluation.models import EvaluationDataset
from app.evaluation.reporting import build_report
from app.evaluation.runner import EvaluationCatalog, RetrievalEvaluationRunner
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, TextParser
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.ingestion.storage import LocalFilesystemStorage
from app.models.documents import DocumentVersionCreate, LogicalDocumentCreate
from app.models.retrieval import RetrievalMode
from app.reranking import create_reranker
from app.retrieval.hybrid import HybridRetrievalService, ModeAwareRetrievalService
from app.retrieval.lexical import LexicalRetrievalService
from app.retrieval.reranked import RerankedRetrievalService
from app.retrieval.service import DenseRetrievalService, VectorIndexingService
from app.retrieval.version_scope import VersionScopeResolver
from app.vector_store import QdrantVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bilingual version-aware retrieval evaluation"
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/evaluation/results"))
    parser.add_argument("--report", type=Path, default=Path("EVALUATION_REPORT.md"))
    parser.add_argument("--candidate-limits", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout.strip() or None) if result.returncode == 0 else None


def ensure_d_drive(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise RuntimeError(f"{label} must resolve to D:, got {resolved}")
    return resolved


async def prepare_catalog(
    dataset: EvaluationDataset,
    dataset_path: Path,
    catalog_service: DocumentCatalogService,
    ingestion: IngestionService,
    indexer: VectorIndexingService,
    session: AsyncSession,
) -> EvaluationCatalog:
    run_suffix = uuid4().hex[:10]
    document_ids: dict[str, UUID] = {}
    version_ids: dict[tuple[str, str], UUID] = {}
    for document in dataset.documents:
        created = await catalog_service.create(
            LogicalDocumentCreate(
                stable_key=f"evaluation-{dataset.revision}-{document.stable_id}-{run_suffix}",
                title=document.title,
                domain="phase8-evaluation",
            )
        )
        document_ids[document.stable_id] = created.logical_document_id
        for version in sorted(document.versions, key=lambda item: item.ordinal):
            source_path = (dataset_path.parent / version.path).resolve()
            with source_path.open("rb") as source:
                ingested = await ingestion.ingest(
                    created.logical_document_id,
                    DocumentVersionCreate(
                        version_label=version.label,
                        version_ordinal=version.ordinal,
                        language=document.language,
                        effective_from=version.effective_from,
                        effective_to=version.effective_to,
                        activate=version.active,
                        chunk_size=dataset.chunking.chunk_size,
                        chunk_overlap=dataset.chunking.chunk_overlap,
                    ),
                    source=source,
                    filename=source_path.name,
                )
            await indexer.index_version(ingested.document_version_id)
            version_ids[(document.stable_id, version.label)] = ingested.document_version_id
    evidence_chunk_ids: dict[str, UUID] = {}
    for evidence in dataset.evidence:
        version_id = version_ids[(evidence.document, evidence.version)]
        chunks = list(
            await session.scalars(select(Chunk).where(Chunk.document_version_id == version_id))
        )
        matches = [
            chunk.chunk_id
            for chunk in chunks
            if evidence.text_contains.casefold() in chunk.text.casefold()
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"evidence '{evidence.id}' resolved to {len(matches)} chunks; expected exactly one"
            )
        evidence_chunk_ids[evidence.id] = matches[0]
    return EvaluationCatalog(document_ids, version_ids, evidence_chunk_ids)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = ensure_d_drive(args.dataset, "dataset")
    output_dir = ensure_d_drive(args.output_dir, "evaluation output directory")
    ensure_d_drive(args.report, "evaluation report")
    dataset = load_dataset(dataset_path)
    if args.top_k < max(1, 3, 5):
        raise ValueError("top-k must be at least 5 for the required metrics")
    if any(limit < args.top_k for limit in args.candidate_limits):
        raise ValueError("candidate limits must be at least top-k")

    settings = Settings()
    ensure_d_drive(Path(settings.embedding_cache_directory), "embedding cache")
    ensure_d_drive(Path(settings.reranker_cache_directory), "reranker cache")
    database = Database(settings)
    embeddings = create_embedding_provider(settings)
    reranker = create_reranker(settings)
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        timeout_seconds=settings.qdrant_timeout_seconds,
    )
    storage = LocalFilesystemStorage(
        Path(settings.upload_directory), max_bytes=settings.max_upload_bytes
    )
    parsers = ParserRegistry((PdfParser(), DocxParser(), TextParser(), MarkdownParser()))
    experiments = []
    cold_start: dict[str, float] = {}
    try:
        async with database.session_factory() as session:
            ingestion = IngestionService(session, storage, parsers, TokenWindowChunker())
            catalog = await prepare_catalog(
                dataset,
                dataset_path,
                DocumentCatalogService(session),
                ingestion,
                VectorIndexingService(session, embeddings, vector_store),
                session,
            )
            dense = DenseRetrievalService(session, embeddings, vector_store)
            lexical = LexicalRetrievalService(session)
            first_case = dataset.cases[0]
            for profile_index, candidate_limit in enumerate(args.candidate_limits):
                profile = f"candidates_{candidate_limit}"
                hybrid = HybridRetrievalService(
                    dense,
                    lexical,
                    dense_candidate_limit=candidate_limit,
                    lexical_candidate_limit=candidate_limit,
                    rrf_k=settings.rrf_k,
                )
                reranked = RerankedRetrievalService(
                    hybrid,
                    reranker,
                    candidate_limit=candidate_limit,
                    final_top_k=args.top_k,
                )
                mode_service = ModeAwareRetrievalService(dense, hybrid, reranked)
                runner = RetrievalEvaluationRunner(
                    dataset,
                    catalog,
                    VersionScopeResolver(session),
                    mode_service,
                )
                for mode in RetrievalMode:
                    if profile_index == 0:
                        cold_start[mode.value] = await runner.cold_probe(
                            first_case, mode, args.top_k
                        )
                    experiments.append(
                        await runner.run(
                            mode,
                            profile=profile,
                            configuration={
                                "top_k": args.top_k,
                                "dense_candidate_limit": candidate_limit,
                                "lexical_candidate_limit": candidate_limit,
                                "rerank_candidate_limit": candidate_limit,
                                "rrf_k": settings.rrf_k,
                                "chunk_size": dataset.chunking.chunk_size,
                                "chunk_overlap": dataset.chunking.chunk_overlap,
                            },
                            top_k=args.top_k,
                        )
                    )
    finally:
        await vector_store.close()
        await database.close()

    timestamp = datetime.now(UTC).isoformat()
    dataset_hash = dataset_sha256(dataset_path)
    artifact = {
        "timestamp": timestamp,
        "git_commit": git_commit(),
        "dataset": {
            "id": dataset.dataset_id,
            "revision": dataset.revision,
            "sha256": dataset_hash,
            "path": str(dataset_path),
            "case_count": len(dataset.cases),
        },
        "models": {
            "embedding_model": settings.embedding_model_name,
            "embedding_revision": settings.embedding_model_revision,
            "reranker_model": settings.reranker_model_name,
            "reranker_revision": settings.reranker_model_revision,
        },
        "cold_start_ms": cold_start,
        "experiments": [item.model_dump(mode="json") for item in experiments],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(UTC).strftime("evaluation-%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, experiments)
    report = build_report(
        dataset,
        experiments,
        dataset_hash=dataset_hash,
        timestamp=timestamp,
        cold_start=cold_start,
    )
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "report": str(args.report)}))
    return artifact


def write_csv(path: Path, experiments: list[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "profile",
                "retrieval_mode",
                "language",
                "query_count",
                "hit_rate_1",
                "hit_rate_3",
                "hit_rate_5",
                "recall_1",
                "recall_3",
                "recall_5",
                "mrr",
                "leakage_count",
                "leakage_rate",
            ],
        )
        writer.writeheader()
        for experiment in experiments:
            for language, metrics in experiment.metrics.items():
                writer.writerow(
                    {
                        "profile": experiment.profile,
                        "retrieval_mode": experiment.retrieval_mode.value,
                        "language": language,
                        "query_count": metrics.evaluated_query_count,
                        "hit_rate_1": metrics.hit_rate[1],
                        "hit_rate_3": metrics.hit_rate[3],
                        "hit_rate_5": metrics.hit_rate[5],
                        "recall_1": metrics.recall[1],
                        "recall_3": metrics.recall[3],
                        "recall_5": metrics.recall[5],
                        "mrr": metrics.mrr,
                        "leakage_count": metrics.leakage_count,
                        "leakage_rate": metrics.leakage_rate,
                    }
                )


if __name__ == "__main__":
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8")
    asyncio.run(run(parse_args()))
