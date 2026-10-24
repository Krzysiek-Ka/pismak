from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.Zrodlowyszukiwator.rag_ingestion import pdf_extractor
from src.Zrodlowyszukiwator.rag_ingestion.pdf_extractor import (
    extract_pdf_pages,
    find_running_lines,
    normalize_text,
    remove_running_lines,
    repair_legacy_polish_glyphs,
)


def make_pages(count: int, body: str = "Treść strony {n}.") -> list[str]:
    """Build page texts with a running header and footer around a numbered body."""
    return [
        f"Strategia miasta\n{body.format(n=n)}\nDruga linia {n}.\nStopka urzędu"
        for n in range(count)
    ]


class _FakePdf:
    """Stand-in for an opened PyMuPDF document with the given text layer on each page."""

    def __init__(self, page_texts: list[str]) -> None:
        self.page_texts = page_texts
        self.closed = False

    def __enter__(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(get_text=lambda kind, text=text: text) for text in self.page_texts]

    def __exit__(self, *exc_info: object) -> bool:
        self.closed = True
        return False


def fake_pdf(monkeypatch, tmp_path: Path, *page_texts: str) -> tuple[Path, _FakePdf]:
    """Put an empty PDF file in tmp_path and make PyMuPDF read the given pages from it."""
    path = tmp_path / "dokument.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    document = _FakePdf(list(page_texts))
    opened: list[Path] = []

    def fake_open(pdf_path: Path) -> _FakePdf:
        opened.append(pdf_path)
        return document

    monkeypatch.setattr(pdf_extractor.pymupdf, "open", fake_open)
    document.opened = opened
    return path, document


class TestNormalizeText:
    def test_normalize_text__removes_page_markers_and_soft_hyphens(self) -> None:
        """Verify PDF artifacts are removed without changing meaningful line content."""
        result = normalize_text("  1 | S t r o n a  \r\nA\u00adla  \r\n\r\n\r\nB  ")

        assert result == "Ala\n\nB"

    def test_normalize_text__keeps_a_number_line_inside_the_page(self) -> None:
        """Verify a value broken onto its own line is content, not a page number."""
        page = (
            "Wytyczne\nSkrzydło bramki powinno otwierać się do szerokości\n110\nstopni.\n"
            "Siła nie może przekraczać\n25\nN.\nStopka\n12"
        )

        result = normalize_text(page)

        assert "szerokości\n110\nstopni." in result
        assert "przekraczać\n25\nN." in result
        assert not result.endswith("12")

    def test_normalize_text__removes_a_page_number_at_the_top_or_bottom(self) -> None:
        """Verify the page number is dropped whether it is printed above or below the text."""
        assert normalize_text("7\nTreść strony\nDalsza treść") == "Treść strony\nDalsza treść"
        assert normalize_text("Treść strony\nDalsza treść\n7") == "Treść strony\nDalsza treść"

    def test_normalize_text__removes_control_characters_used_as_bullets(self) -> None:
        """Verify symbol-font bullets extracted as control characters do not reach the text."""
        result = normalize_text("\x01 zmniejszenie problemów;\n\x01 udrożnienie ulic\x0c")

        assert result == "zmniejszenie problemów;\nudrożnienie ulic"

    def test_normalize_text__returns_empty_for_whitespace_only_text(self) -> None:
        """Verify pages containing only whitespace do not create fake content."""
        assert normalize_text(" \r\n\n\t") == ""


class TestRepairLegacyPolishGlyphs:
    def test_repair_legacy_polish_glyphs__restores_z_dot_and_keeps_regular_s_acute(self) -> None:
        """Verify the broken font letters become 'ż'/'Ż' while ordinary 'ś' is untouched."""
        result = repair_legacy_polish_glyphs("CELE BIEśĄCE, podróŜy, środowisko")

        assert result == "CELE BIEŻĄCE, podróży, środowisko"


class TestExtractPdfPages:
    def test_extract_pdf_pages__raises_for_missing_file(self, tmp_path: Path) -> None:
        """Verify a missing PDF fails clearly before invoking the PDF library."""
        with pytest.raises(FileNotFoundError):
            extract_pdf_pages(tmp_path / "missing.pdf")

    def test_extract_pdf_pages__broken_font_document_repairs_every_page(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Verify one 'Ŝ' marks the whole document, so capitals on other pages are fixed too."""
        path, _ = fake_pdf(
            monkeypatch,
            tmp_path,
            "Zakres moŜliwości miasta",
            "CELE BIEśĄCE\nśrodowisko miejskie",
        )

        pages = pdf_extractor.extract_pdf_pages(path)

        assert [page.text for page in pages] == [
            "Zakres możliwości miasta",
            "CELE BIEŻĄCE\nśrodowisko miejskie",
        ]
        assert [page.legacy_glyphs_repaired for page in pages] == [True, True]

    def test_extract_pdf_pages__document_without_the_broken_glyph_is_left_unchanged(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Verify an 'ś' among capitals is kept when nothing shows the broken font pattern."""
        path, _ = fake_pdf(monkeypatch, tmp_path, "CELE BIEśĄCE\nśrodowisko miejskie")

        [page] = pdf_extractor.extract_pdf_pages(path)

        assert page.text == "CELE BIEśĄCE\nśrodowisko miejskie"
        assert page.legacy_glyphs_repaired is False

    def test_extract_pdf_pages__pages_are_numbered_from_one_and_short_ones_flagged(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Verify page numbers match the PDF and a near-empty page is marked as low quality."""
        path, _ = fake_pdf(monkeypatch, tmp_path, "a" * 100, "krótko", "")

        pages = pdf_extractor.extract_pdf_pages(path)

        assert [(p.page_number, p.char_count, p.low_text_quality) for p in pages] == [
            (1, 100, False),
            (2, 6, True),
            (3, 0, True),
        ]

    def test_extract_pdf_pages__running_header_is_removed_after_its_first_page(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Verify a header printed on every page stays once and is reported where it was cut."""
        texts = [
            f"Strategia Wrocław 2050\nTreść strony {n} jest unikalna.\nDruga linia {n}."
            for n in range(4)
        ]
        path, _ = fake_pdf(monkeypatch, tmp_path, *texts)

        pages = pdf_extractor.extract_pdf_pages(path)

        assert pages[0].text.startswith("Strategia Wrocław 2050")
        assert all("Strategia Wrocław 2050" not in page.text for page in pages[1:])
        assert [page.removed_running_lines for page in pages] == [
            [],
            ["Strategia Wrocław 2050"],
            ["Strategia Wrocław 2050"],
            ["Strategia Wrocław 2050"],
        ]

    def test_extract_pdf_pages__control_characters_and_page_number_are_cleaned(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Verify symbol-font bullets and the printed page number never reach the chunks."""
        path, _ = fake_pdf(monkeypatch, tmp_path, "\x01 punkt listy\nTreść strony.\n12")

        [page] = pdf_extractor.extract_pdf_pages(path)

        assert page.text == "punkt listy\nTreść strony."

    def test_extract_pdf_pages__document_is_closed_after_reading(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Verify the file is released at once, because Windows locks an open PDF against writes."""
        path, document = fake_pdf(monkeypatch, tmp_path, "Treść")

        pdf_extractor.extract_pdf_pages(path)

        assert document.opened == [path]
        assert document.closed is True


class TestRemoveRunningLines:
    def test_remove_running_lines__drops_repeated_headers_and_footers_but_keeps_the_first(
        self,
    ) -> None:
        """Verify page headers and footers appear once in the text instead of on every page."""
        result = remove_running_lines(make_pages(8))

        texts = [text for text, _ in result]
        assert texts[0] == "Strategia miasta\nTreść strony 0.\nDruga linia 0.\nStopka urzędu"
        assert texts[1] == "Treść strony 1.\nDruga linia 1."
        assert result[1][1] == ["Strategia miasta", "Stopka urzędu"]


class TestFindRunningLines:
    def test_find_running_lines__keeps_a_line_that_also_appears_inside_a_page(self) -> None:
        """Verify a repeated line is content, not a header, when it also occurs mid-page."""
        pages = make_pages(8)
        pages[3] = "Inny początek\nA\nB\nStrategia miasta\nC\nD\nE\nKoniec"

        assert find_running_lines(pages) == {"Stopka urzędu"}

    def test_find_running_lines__ignores_short_documents_and_rare_repeats(self) -> None:
        """Verify a few repeated edge lines in a short or long document are left alone."""
        assert find_running_lines(make_pages(3)) == set()

        pages = make_pages(20)
        for n in range(3, 20):
            pages[n] = f"Rozdział {n}\nTreść {n}.\nKoniec {n}."
        assert find_running_lines(pages) == set()
