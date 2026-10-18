from __future__ import annotations

from pathlib import Path

import pytest

from src.Zrodlowyszukiwator.rag_ingestion.pdf_extractor import (
    extract_pdf_pages,
    find_running_lines,
    normalize_text,
    remove_running_lines,
    repair_legacy_polish_glyphs,
)

RAW_DIR = Path(__file__).resolve().parents[3] / "src" / "Zrodlowyszukiwator" / "RAG" / "RAW"


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

    def test_extract_pdf_pages__repairs_the_broken_font_of_the_bicycle_policy(self) -> None:
        """Verify the committed policy is extracted with the letters its pages actually show."""
        pages = extract_pdf_pages(RAW_DIR / "01_Polityka-rowerowa-wroclawia.pdf")

        assert not any("Ŝ" in page.text for page in pages)
        assert "CELE BIEŻĄCE" in pages[4].text
        assert pages[4].legacy_glyphs_repaired

    def test_extract_pdf_pages__leaves_a_correctly_encoded_document_unchanged(self) -> None:
        """Verify the repair is not applied to a document without the broken font signature."""
        pages = extract_pdf_pages(RAW_DIR / "01_Wroclawska_polityka_mobilnosci.pdf")

        assert not any(page.legacy_glyphs_repaired for page in pages)

    def test_extract_pdf_pages__leaves_no_control_characters_in_the_committed_policies(
        self,
    ) -> None:
        """Verify the bullets of both committed policies are not kept as control characters."""
        for name in ("01_Polityka-rowerowa-wroclawia.pdf", "01_Wroclawska_polityka_mobilnosci.pdf"):
            text = "".join(page.text for page in extract_pdf_pages(RAW_DIR / name))

            assert not any(ord(c) < 32 and c not in "\n\t" for c in text), name

    def test_extract_pdf_pages__finds_no_running_headers_in_the_committed_policies(self) -> None:
        """Verify header detection removes nothing from documents that have no page headers."""
        for name in ("01_Polityka-rowerowa-wroclawia.pdf", "01_Wroclawska_polityka_mobilnosci.pdf"):
            pages = extract_pdf_pages(RAW_DIR / name)

            assert [line for page in pages for line in page.removed_running_lines] == [], name


def make_pages(count: int, body: str = "Treść strony {n}.") -> list[str]:
    """Build page texts with a running header and footer around a numbered body."""
    return [
        f"Strategia miasta\n{body.format(n=n)}\nDruga linia {n}.\nStopka urzędu"
        for n in range(count)
    ]


class TestRunningLines:
    def test_remove_running_lines__drops_repeated_headers_and_footers_but_keeps_the_first(
        self,
    ) -> None:
        """Verify page headers and footers appear once in the text instead of on every page."""
        result = remove_running_lines(make_pages(8))

        texts = [text for text, _ in result]
        assert texts[0] == "Strategia miasta\nTreść strony 0.\nDruga linia 0.\nStopka urzędu"
        assert texts[1] == "Treść strony 1.\nDruga linia 1."
        assert result[1][1] == ["Strategia miasta", "Stopka urzędu"]

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
