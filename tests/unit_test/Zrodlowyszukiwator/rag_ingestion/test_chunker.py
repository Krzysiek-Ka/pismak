from __future__ import annotations

import pytest

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

    def test_build_text_units__table_of_contents_heading_does_not_frame_later_pages(self) -> None:
        """Verify the contents page does not become the parent section of the document body."""
        pages = [
            ExtractedPage(
                page_number=1,
                text="SPIS TREŚCI\nWstęp ........ 3\nObszary działań ........ 5",
                char_count=50,
            ),
            ExtractedPage(
                page_number=3,
                text="Wstęp\nPlan zbiera kierunki rozwoju infrastruktury rowerowej.",
                char_count=60,
            ),
        ]

        units = build_text_units(pages)

        assert [unit.section_path for unit in units if unit.page_number == 3] == [
            ["Wstęp"],
            ["Wstęp"],
        ]
        assert units[0].text == "SPIS TREŚCI"

    @pytest.mark.parametrize("heading", ["Spis treści", "SPIS TRESCI", "Spis  treści"])
    def test_build_text_units__contents_heading_in_any_spelling_is_not_a_section(
        self, heading: str
    ) -> None:
        """Verify the heading is recognised regardless of case, spacing and diacritics."""
        pages = [
            ExtractedPage(page_number=2, text=f"{heading}\nTreść po spisie.", char_count=40),
        ]

        units = build_text_units(pages)

        assert [unit.section_path for unit in units] == [[], []]


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

    def test_chunk_units__starts_a_new_chunk_at_a_new_section_once_the_buffer_is_large_enough(
        self,
    ) -> None:
        """Verify chunks follow section boundaries so one chunk cites one section."""
        units = [
            TextUnit(page_number=1, section_path=["1. Cele"], text="a" * 30),
            TextUnit(page_number=2, section_path=["2. Działania"], text="b" * 30),
        ]

        chunks = chunk_units(units, make_metadata(), max_chars=200, min_chars=20, overlap_units=0)

        assert [(c.section_path, c.page_start, c.page_end) for c in chunks] == [
            (["1. Cele"], 1, 1),
            (["2. Działania"], 2, 2),
        ]

    def test_chunk_units__merges_a_short_section_with_the_next_one(self) -> None:
        """Verify a lone heading-sized section is not stored as a near-empty chunk."""
        units = [
            TextUnit(page_number=1, section_path=["1. Cele"], text="krótko"),
            TextUnit(page_number=2, section_path=["2. Działania"], text="b" * 30),
        ]

        [chunk] = chunk_units(units, make_metadata(), max_chars=200, min_chars=20, overlap_units=0)

        assert (chunk.page_start, chunk.page_end) == (1, 2)
        assert chunk.section_path == ["2. Działania"]
        assert chunk.text == "krótko\n\n" + "b" * 30

    def test_chunk_units__repeats_the_last_paragraph_of_a_chunk_as_overlap(self) -> None:
        """Verify context spanning two chunks is searchable from both of them."""
        units = [TextUnit(page_number=1, text=f"{letter * 40}") for letter in "abc"]

        chunks = chunk_units(units, make_metadata(), max_chars=90, min_chars=10, overlap_units=1)

        assert chunks[0].text.endswith("b" * 40)
        assert chunks[1].text.startswith("b" * 40)

    def test_chunk_units__gives_each_chunk_a_stable_unique_id_and_embedding_header(self) -> None:
        """Verify re-ingesting the same text yields the same IDs and the header names the place."""
        units = [TextUnit(page_number=3, text=f"{letter * 40}") for letter in "abc"]

        first = chunk_units(units, make_metadata(), max_chars=50, min_chars=10, overlap_units=0)
        second = chunk_units(units, make_metadata(), max_chars=50, min_chars=10, overlap_units=0)

        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert len({c.chunk_id for c in first}) == len(first) == 3
        assert first[0].chunk_id.startswith("doc-1:00000:")
        assert "Sekcja: (bez nagłówka)\nStrony PDF: 3-3" in first[0].embedding_text
        assert first[0].metadata["title"] == "Dokument testowy"

    def test_chunk_units__drops_units_that_contain_only_whitespace(self) -> None:
        """Verify blank extraction leftovers do not become empty chunks."""
        units = [TextUnit(page_number=1, text="   "), TextUnit(page_number=1, text="\n")]

        assert chunk_units(units, make_metadata(), min_chars=1) == []

    def test_chunk_units__never_exceeds_max_chars_when_overlap_is_added(self) -> None:
        """Verify the size limit holds for every chunk, including overlap."""
        units = [TextUnit(page_number=1, text=letter * 80) for letter in "abc"]

        chunks = chunk_units(units, make_metadata(), max_chars=100, min_chars=10, overlap_units=1)

        assert all(len(chunk.text) <= 100 for chunk in chunks), [len(c.text) for c in chunks]
        assert [chunk.text for chunk in chunks] == ["a" * 80, "b" * 80, "c" * 80]

    def test_chunk_units__keeps_the_overlap_when_it_fits_with_the_next_paragraph(self) -> None:
        """Verify the size fix drops the overlap only when it would break the limit."""
        units = [TextUnit(page_number=1, text=letter * 40) for letter in "abc"]

        chunks = chunk_units(units, make_metadata(), max_chars=90, min_chars=10, overlap_units=1)

        assert [chunk.text for chunk in chunks] == [
            "a" * 40 + "\n\n" + "b" * 40,
            "b" * 40 + "\n\n" + "c" * 40,
        ]

    def test_chunk_units__does_not_emit_a_chunk_made_only_of_overlap(self) -> None:
        """Verify the end of a document is not stored twice as two identical chunks."""
        text = "".join(chr(ord("a") + index % 26) for index in range(250))
        units = [TextUnit(page_number=1, text=text)]

        chunks = chunk_units(units, make_metadata(), max_chars=100, min_chars=10, overlap_units=1)

        texts = [chunk.text for chunk in chunks]
        assert len(texts) == len(set(texts)), texts
