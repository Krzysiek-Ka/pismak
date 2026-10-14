from __future__ import annotations

from pathlib import Path

from src.Zrodlowyszukiwator.rag_ingestion.chunker import (
    _looks_like_heading,
    _split_page_to_units,
    build_text_units,
    chunk_units,
)
from src.Zrodlowyszukiwator.rag_ingestion.models import (
    DocumentKind,
    DocumentMetadata,
    ExtractedPage,
    TextUnit,
)
from src.Zrodlowyszukiwator.rag_ingestion.pdf_extractor import extract_pdf_pages

BICYCLE_POLICY_PDF = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "Zrodlowyszukiwator"
    / "RAG"
    / "RAW"
    / "01_Polityka-rowerowa-wroclawia.pdf"
)


def make_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc-1",
        title="Dokument testowy",
        document_kind=DocumentKind.POLICY,
    )


class TestLooksLikeHeading:
    def test_looks_like_heading__rejects_wrapped_lines_starting_with_a_year_or_a_date(self) -> None:
        """Verify sentence fragments that begin with a number are not taken for chapter headings."""
        assert _looks_like_heading(
            "2015 co najmniej 10% udziału ruchu rowerowego w liczbie podróży niepieszych"
        ) == (False, 0)
        assert _looks_like_heading("8 marca 1990 r. o samorządzie gminnym (Dz. U. z 2001 r.") == (
            False,
            0,
        )

    def test_looks_like_heading__keeps_numbered_section_levels(self) -> None:
        """Verify real numbered headings keep their nesting level."""
        assert _looks_like_heading("1. CELE STRATEGICZNE") == (True, 1)
        assert _looks_like_heading("3.2.1. Wysoka jakość życia w dobrej przestrzeni") == (True, 3)

    def test_looks_like_heading__rejects_list_introductions_and_hyphenated_lines(self) -> None:
        """Verify a line ending with a colon or a line-break hyphen is treated as running text."""
        assert _looks_like_heading("Czynniki będące przedmiotem analiz:") == (False, 0)
        assert _looks_like_heading("Udział podróży transportem niesa-") == (False, 0)


class TestSplitPageToUnits:
    def test_split_page_to_units__ignores_empty_page(self) -> None:
        """Verify an empty page produces no text units instead of an empty paragraph."""
        page = ExtractedPage(page_number=1, text="\n  \n", char_count=0)

        assert _split_page_to_units(page) == []

    def test_split_page_to_units__separates_heading_and_paragraph(self) -> None:
        """Verify headings become structural units and nearby lines form one paragraph."""
        page = ExtractedPage(
            page_number=1,
            text="1. Bezpieczeństwo\nPierwsza linia.\nDruga linia.",
            char_count=50,
        )

        assert _split_page_to_units(page) == [
            ("1. Bezpieczeństwo", 1),
            ("Pierwsza linia. Druga linia.", None),
        ]

    def test_split_page_to_units__skips_table_of_contents_lines(self) -> None:
        """Verify dotted table-of-contents entries do not duplicate headings in the text."""
        page = ExtractedPage(
            page_number=1,
            text="WPROWADZENIE.................. 3\nCELE POLITYKI.............\nTreść.",
            char_count=60,
        )

        assert _split_page_to_units(page) == [("Treść.", None)]

    def test_split_page_to_units__glues_words_hyphenated_at_a_line_break(self) -> None:
        """Verify a word split by the PDF is rejoined while a suspended hyphen stays."""
        page = ExtractedPage(
            page_number=1,
            text="Udział transportu niesamocho-\ndowego rośnie, a ścieżki jedno-\ni dwukierunkowe też.",
            char_count=80,
        )

        assert _split_page_to_units(page) == [
            (
                "Udział transportu niesamochodowego rośnie, a ścieżki jedno- i dwukierunkowe też.",
                None,
            )
        ]


class TestBuildTextUnits:
    def test_build_text_units__keeps_section_path_for_following_paragraphs(self) -> None:
        """Verify a heading stays in the text and is retained as section context."""
        pages = [
            ExtractedPage(
                page_number=2,
                text="I. Wstęp\nTreść sekcji.",
                char_count=30,
            )
        ]

        units = build_text_units(pages)

        assert [(unit.text, unit.section_path) for unit in units] == [
            ("I. Wstęp", ["I. Wstęp"]),
            ("Treść sekcji.", ["I. Wstęp"]),
        ]


class TestChunkUnits:
    def test_chunk_units__returns_empty_for_empty_units(self) -> None:
        """Verify an empty extraction result does not create synthetic chunks."""
        assert chunk_units([], make_metadata()) == []

    def test_chunk_units__splits_overlong_unit_at_character_boundary(self) -> None:
        """Verify one paragraph longer than the limit is split into bounded chunks."""
        units = [TextUnit(page_number=1, text="abcdefghij", section_path=[])]

        chunks = chunk_units(
            units,
            make_metadata(),
            max_chars=4,
            min_chars=1,
            overlap_units=0,
        )

        assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
        assert all(len(chunk.text) <= 4 for chunk in chunks)

    def test_chunk_units__keeps_lines_of_the_committed_policy_that_look_like_headings(
        self,
    ) -> None:
        """Verify the 2015 target and the cycling officer line from the real PDF reach the chunks."""
        chunks = chunk_units(
            build_text_units(extract_pdf_pages(BICYCLE_POLICY_PDF)), make_metadata()
        )
        texts = [" ".join(chunk.text.split()) for chunk in chunks]

        assert any(
            "w roku 2015 co najmniej 10% udziału ruchu rowerowego w liczbie podróży niepieszych"
            in text
            for text in texts
        )
        assert any("Koordynator (oficer rowerowy) oraz" in text for text in texts)
        assert any("Sekcja ds. rozwoju ruchu rowerowego" in text for text in texts)
