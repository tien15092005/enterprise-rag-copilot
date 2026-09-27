from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter
from reportlab.pdfgen import canvas

from app.ingestion.errors import UnsupportedDocumentError
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.markdown import MarkdownParser
from app.ingestion.parsers.pdf import PdfParser
from app.ingestion.parsers.text import TextParser


def test_text_parser_preserves_vietnamese_text() -> None:
    parsed = TextParser().parse("Chính sách nghỉ phép\n\nMười hai ngày mỗi năm.".encode())

    assert [block.text for block in parsed.blocks] == [
        "Chính sách nghỉ phép",
        "Mười hai ngày mỗi năm.",
    ]


def test_markdown_parser_preserves_section_path() -> None:
    parsed = MarkdownParser().parse(b"# Policy\n\nIntro\n\n## Leave\n\nTwelve days.")

    assert parsed.blocks[0].section_path == ("Policy",)
    assert parsed.blocks[1].section_path == ("Policy", "Leave")


def test_docx_parser_preserves_heading_structure() -> None:
    buffer = BytesIO()
    document = Document()
    document.add_heading("Handbook", level=1)
    document.add_heading("Security", level=2)
    document.add_paragraph("Use approved devices.")
    document.save(buffer)

    parsed = DocxParser().parse(buffer.getvalue())

    assert parsed.blocks[0].text == "Use approved devices."
    assert parsed.blocks[0].section_path == ("Handbook", "Security")


def test_pdf_parser_preserves_one_based_page_number() -> None:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "Remote work policy")
    pdf.showPage()
    pdf.drawString(72, 720, "Security policy")
    pdf.save()

    parsed = PdfParser().parse(buffer.getvalue())

    assert [block.page for block in parsed.blocks] == [1, 2]
    assert "Remote work policy" in parsed.blocks[0].text


def test_scanned_or_empty_pdf_returns_clear_error() -> None:
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)

    with pytest.raises(UnsupportedDocumentError, match="OCR is not supported"):
        PdfParser().parse(buffer.getvalue())
