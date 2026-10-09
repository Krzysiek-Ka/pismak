from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF

from .models import ExtractedPage

_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*(?:\|\s*S\s*t\s*r\s*o\s*n\s*a)?\s*$", re.I)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Konserwatywne czyszczenie:
    - normalizacja Unicode,
    - usuwanie soft-hyphen,
    - porządkowanie białych znaków.
    Nie próbujemy automatycznie poprawiać starego kodowania polskich znaków,
    bo mogłoby to zmienić sens źródła.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for raw in text.splitlines():
        line = _MULTI_SPACE_RE.sub(" ", raw).strip()
        if not line:
            lines.append("")
            continue
        if _PAGE_NUMBER_RE.match(line):
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str | Path) -> list[ExtractedPage]:
    """Extract normalized text and page quality metadata from a PDF file."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    pages: list[ExtractedPage] = []

    for idx, page in enumerate(doc):
        text = normalize_text(page.get_text("text"))
        char_count = len(text)

        pages.append(
            ExtractedPage(
                page_number=idx + 1,
                text=text,
                char_count=char_count,
                low_text_quality=char_count < 80,
            )
        )

    return pages
