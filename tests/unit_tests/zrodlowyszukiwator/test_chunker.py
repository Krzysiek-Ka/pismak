from __future__ import annotations

from src.Zrodlowyszukiwator.rag_ingestion.chunker import (
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


class TestBuildTextUnits:
    def test_build_text_units__keeps_section_path_for_following_paragraphs(self) -> None:
        """Verify a heading is retained as section context for subsequent text."""
        pages = [
            ExtractedPage(
                page_number=2,
                text="I. Wstęp\nTreść sekcji.",
                char_count=30,
            )
        ]

        units = build_text_units(pages)

        assert units[0].section_path == ["I. Wstęp"]
        assert units[0].text == "Treść sekcji."


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
