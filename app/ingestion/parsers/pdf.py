from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.errors import UnsupportedDocumentError
from app.ingestion.parsers.base import ParsedBlock, ParsedDocument


class PdfParser:
    supported_extensions = frozenset({".pdf"})

    def parse(self, content: bytes) -> ParsedDocument:
        try:
            reader = PdfReader(BytesIO(content))
            blocks = tuple(
                ParsedBlock(text=text, page=page_number)
                for page_number, page in enumerate(reader.pages, start=1)
                if (text := (page.extract_text() or "").strip())
            )
        except (PdfReadError, ValueError) as exc:
            raise UnsupportedDocumentError("invalid PDF document") from exc

        if not blocks:
            raise UnsupportedDocumentError(
                "PDF contains no extractable text; OCR is not supported in Phase 2"
            )
        return ParsedDocument(blocks=blocks)
