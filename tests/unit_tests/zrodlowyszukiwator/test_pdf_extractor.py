from __future__ import annotations

from pathlib import Path

import pytest

from src.Zrodlowyszukiwator.rag_ingestion.pdf_extractor import (
    extract_pdf_pages,
    normalize_text,
)


class TestNormalizeText:
    def test_normalize_text__removes_page_markers_and_soft_hyphens(self) -> None:
        """Verify PDF artifacts are removed without changing meaningful line content."""
        result = normalize_text("  1 | S t r o n a  \r\nA\u00adla  \r\n\r\n\r\nB  ")

        assert result == "Ala\n\nB"

    def test_normalize_text__returns_empty_for_whitespace_only_text(self) -> None:
        """Verify pages containing only whitespace do not create fake content."""
        assert normalize_text(" \r\n\n\t") == ""


class TestExtractPdfPages:
    def test_extract_pdf_pages__raises_for_missing_file(self, tmp_path: Path) -> None:
        """Verify a missing PDF fails clearly before invoking the PDF library."""
        with pytest.raises(FileNotFoundError):
            extract_pdf_pages(tmp_path / "missing.pdf")
