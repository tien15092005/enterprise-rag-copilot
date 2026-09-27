from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID, uuid4

from app.comparison.models import ComparisonRequest, ComparisonResponse
from app.comparison.service import DocumentComparisonService
from app.core.config import Settings
from app.db.session import Database
from app.embeddings import create_embedding_provider
from app.generation.providers.base import LLMRequest, LLMResponse, LLMTokenUsage
from app.ingestion.chunker import TokenWindowChunker
from app.ingestion.parsers import DocxParser, MarkdownParser, ParserRegistry, PdfParser, TextParser
from app.ingestion.service import DocumentCatalogService, IngestionService
from app.ingestion.storage import LocalFilesystemStorage
from app.models.documents import DocumentVersionCreate, LogicalDocumentCreate
from app.models.versioning import DocumentLanguage


class ControlledComparisonLLM:
    provider_name = "controlled_fake"
    model_name = "controlled-comparison-verifier-v1"

    async def generate(self, _request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            "Remote work increased from one day to two days per week [S1] [S2].",
            LLMTokenUsage(180, 16, 196),
        )

    async def close(self) -> None:
        return None


async def ingest_pair(
    catalog: DocumentCatalogService,
    ingestion: IngestionService,
    *,
    stable_key: str,
    title: str,
    language: DocumentLanguage,
    left_path: Path,
    right_path: Path,
) -> tuple[UUID, UUID, UUID]:
    document = await catalog.create(LogicalDocumentCreate(stable_key=stable_key, title=title))
    version_ids: list[UUID] = []
    for ordinal, (label, path) in enumerate((("2025", left_path), ("2026", right_path)), 1):
        with path.open("rb") as source:
            version = await ingestion.ingest(
                document.logical_document_id,
                DocumentVersionCreate(
                    version_label=label,
                    version_ordinal=ordinal,
                    language=language,
                    activate=label == "2026",
                    chunk_size=100,
                    chunk_overlap=10,
                ),
                source=source,
                filename=path.name,
            )
        version_ids.append(version.document_version_id)
    return document.logical_document_id, version_ids[0], version_ids[1]


def compact(response: ComparisonResponse) -> dict[str, object]:
    payload = response.model_dump(mode="json")
    return {
        "left_version": payload["left_version"],
        "right_version": payload["right_version"],
        "counts": payload["counts"],
        "summary": payload["summary"],
        "items": [
            {
                "order": item["order"],
                "change_type": item["change_type"],
                "left_section_path": item["left_section_path"],
                "right_section_path": item["right_section_path"],
                "left_text": item["left_text"],
                "right_text": item["right_text"],
                "alignment_method": item["alignment_method"],
                "alignment_score": item["alignment_score"],
                "citation_keys": item["citation_keys"],
            }
            for item in payload["items"]
        ],
        "citations": payload["citations"],
        "timing": payload["timing"],
        "debug": payload["debug"],
    }


def runtime_paths(settings: Settings) -> tuple[Path, Path]:
    return (
        Path(settings.embedding_cache_directory).resolve(),
        Path(settings.upload_directory).resolve(),
    )


async def run() -> None:
    settings = Settings()
    cache, uploads = runtime_paths(settings)
    if cache.drive.upper() != "D:" or uploads.drive.upper() != "D:":
        raise RuntimeError(f"comparison runtime paths must resolve to D:, got {cache}, {uploads}")

    database = Database(settings)
    embeddings = create_embedding_provider(settings)
    storage = LocalFilesystemStorage(uploads, max_bytes=settings.max_upload_bytes)
    parsers = ParserRegistry((PdfParser(), DocxParser(), TextParser(), MarkdownParser()))
    suffix = uuid4().hex[:8]
    output: dict[str, object] = {}
    try:
        async with database.session_factory() as session:
            catalog = DocumentCatalogService(session)
            ingestion = IngestionService(session, storage, parsers, TokenWindowChunker())
            en_doc, en_left, en_right = await ingest_pair(
                catalog,
                ingestion,
                stable_key=f"phase9a-en-{suffix}",
                title="Employee Policy",
                language=DocumentLanguage.ENGLISH,
                left_path=Path("data/comparison/employee_policy_en_2025.md"),
                right_path=Path("data/comparison/employee_policy_en_2026.md"),
            )
            vi_doc, vi_left, vi_right = await ingest_pair(
                catalog,
                ingestion,
                stable_key=f"phase9a-vi-{suffix}",
                title="Chính sách nhân sự",
                language=DocumentLanguage.VIETNAMESE,
                left_path=Path("data/comparison/leave_policy_vi_2025.md"),
                right_path=Path("data/comparison/leave_policy_vi_2026.md"),
            )
            service = DocumentComparisonService(
                session, embeddings, ControlledComparisonLLM(), settings
            )
            english = await service.compare(
                ComparisonRequest(
                    logical_document_id=en_doc,
                    left_document_version_id=en_left,
                    right_document_version_id=en_right,
                    include_unchanged=True,
                    generate_summary=False,
                    debug=True,
                )
            )
            vietnamese = await service.compare(
                ComparisonRequest(
                    logical_document_id=vi_doc,
                    left_document_version_id=vi_left,
                    right_document_version_id=vi_right,
                    include_unchanged=True,
                    generate_summary=False,
                    debug=True,
                )
            )
            generated = await service.compare(
                ComparisonRequest(
                    logical_document_id=en_doc,
                    left_document_version_id=en_left,
                    right_document_version_id=en_right,
                    include_unchanged=False,
                    generate_summary=True,
                    debug=True,
                )
            )
            expected_versions = {str(en_left), str(en_right)}
            actual_versions = {
                str(citation.citation.document_version_id) for citation in english.citations
            }
            assert actual_versions == expected_versions
            assert english.counts.modified >= 1
            assert english.counts.added >= 1
            assert english.counts.removed >= 1
            assert english.counts.unchanged >= 1
            assert generated.summary is not None
            output = {
                "runtime_paths": {"embedding_cache": str(cache), "uploads": str(uploads)},
                "embedding_model": {
                    "name": embeddings.model_name,
                    "revision": embeddings.revision,
                },
                "english_no_generation": compact(english),
                "vietnamese_no_generation": compact(vietnamese),
                "controlled_generated_summary": compact(generated),
                "citation_version_provenance": {
                    "expected": sorted(expected_versions),
                    "actual": sorted(actual_versions),
                    "passed": True,
                },
            }
    finally:
        await database.close()
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding="utf-8")
    asyncio.run(run())
