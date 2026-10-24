"""
Złoty zbiór wczytywania dokumentów (rag_ingestion): prawdziwe PDF-y korpusu i żywa baza RAG.

Każdy przypadek to jeden plik z `src/Zrodlowyszukiwator/RAG/RAW/` i jego wpis w katalogu.
Test wykonuje ten sam kod co `ingest_pdf` aż do embeddingów (ekstrakcja, jednostki tekstu,
fragmenty) i niczego nie zapisuje. Modelu językowego nie ma, więc dostawcą w identyfikatorze
testu jest ekstraktor PDF: `pymupdf-<case_id>`. Plik, którego nie ma na dysku (repozytorium
zawiera dwa z siedmiu PDF-ów), albo niedostępna baza kończą się pominięciem z powodem.

Zawsze sprawdzane: brak znaków sterujących i zepsutego „Ŝ”, fragmenty nie dłuższe niż
max_chars, brak fragmentu powtarzającego poprzedni (sama zakładka) oraz zgodność indeksu z bieżącym kodem (ten sam plik,
INGESTION_VERSION i identyczne fragmenty w bazie). Obsługiwane pola `expected`:
- phrases_present: każda fraza występuje w treści któregoś fragmentu (białe znaki ujednolicone),
- legacy_glyphs_repaired: czy naprawa zepsutego fontu zadziałała na jakiejś stronie,
- no_running_lines_removed: żadna linia nie została usunięta jako nagłówek lub stopka strony,
- toc_pages: strony spisu treści; poza nimi „Spis treści” nie może być sekcją fragmentu.
"""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf
import pytest

from src.Zrodlowyszukiwator.rag_ingestion.catalog import load_catalog
from src.Zrodlowyszukiwator.rag_ingestion.chunker import build_text_units, chunk_units
from src.Zrodlowyszukiwator.rag_ingestion.cli import DEFAULT_CATALOG_PATH
from src.Zrodlowyszukiwator.rag_ingestion.pdf_extractor import extract_pdf_pages
from src.Zrodlowyszukiwator.rag_ingestion.pipeline import INGESTION_VERSION, file_sha256
from src.Zrodlowyszukiwator.rag_retrieval.repository import RetrievalRepository
from tests.golden_set.golden_support import (
    PROJECT_ROOT,
    GoldenCase,
    Provider,
    assert_passed,
    database_skip_reason,
    golden_run,
    load_cases,
    normalize,
    skip_run,
)

pytestmark = pytest.mark.golden

MODULE = "rag_ingestion"
MODULE_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "src" / "Zrodlowyszukiwator" / "RAG" / "RAW"
SUPPORTED_EXPECTED = {
    "phrases_present",
    "legacy_glyphs_repaired",
    "no_running_lines_removed",
    "toc_pages",
}
CASES = load_cases(MODULE_DIR, supported_expected=SUPPORTED_EXPECTED)
CATALOG = load_catalog(DEFAULT_CATALOG_PATH)
# Domyślne parametry ingest_pdf, takie same jak w poleceniu z README.
MAX_CHARS = 3500
EXTRACTOR = Provider("pymupdf", "local", f"pymupdf {pymupdf.VersionBind}")


class TestGoldenCases:
    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__name_a_file_described_in_the_catalog(self, case: GoldenCase) -> None:
        """Verify each case points at a document the ingestion command would accept."""
        assert case.input["filename"] in CATALOG

    def test_cases__cover_every_catalog_document(self) -> None:
        """Verify no document of the corpus is left without a golden case."""
        assert {case.input["filename"] for case in CASES} == set(CATALOG)


def indexed_document(document_id: str) -> tuple[tuple | None, set[str]]:
    """Return the stored (sha256, ingestion_version) and the chunk ids of one document."""
    repository = RetrievalRepository(os.environ["DATABASE_URL"])
    with repository.connect() as conn:
        row = conn.execute(
            "SELECT content_sha256, ingestion_version FROM rag_documents WHERE document_id = %s",
            (document_id,),
        ).fetchone()
        chunk_ids = conn.execute(
            "SELECT chunk_id FROM rag_chunks WHERE document_id = %s", (document_id,)
        ).fetchall()
    return row, {r[0] for r in chunk_ids}


@pytest.mark.parametrize("case", [pytest.param(c, id=f"{EXTRACTOR.label}-{c.id}") for c in CASES])
def test_golden_rag_ingestion(case: GoldenCase) -> None:
    """Verify a real PDF is extracted and chunked cleanly and the index matches this code.

    The case file describes what is special about the document.
    """
    path = RAW_DIR / case.input["filename"]
    reason = None if path.exists() else f"brak pliku {path.name} w RAG/RAW"
    reason = reason or database_skip_reason()
    if reason:
        skip_run(MODULE, MODULE_DIR, EXTRACTOR, case, reason)
    metadata = CATALOG[case.input["filename"]]

    with golden_run(MODULE, MODULE_DIR, EXTRACTOR, case) as run:
        run.model = EXTRACTOR.model
        pages = extract_pdf_pages(path)
        chunks = chunk_units(build_text_units(pages), metadata, max_chars=MAX_CHARS)
        texts = [chunk.text for chunk in chunks]
        everything = "\n".join(page.text for page in pages)

        control = sorted({hex(ord(c)) for c in everything if ord(c) < 32 and c not in "\n\t"})
        run.check("no_control_characters", not control, str(control))
        run.check("no_broken_z_dot", "Ŝ" not in everything)
        longest = max((len(t) for t in texts), default=0)
        run.check("chunks_within_max_chars", 0 < longest <= MAX_CHARS, f"najdłuższy {longest}")
        # Dokument może sam powtarzać tekst (nagłówek tabeli na kilku stronach); błędem chunkera
        # jest fragment złożony z samej zakładki, czyli powtórzenie poprzedniego fragmentu.
        consecutive = [i for i in range(1, len(texts)) if texts[i] == texts[i - 1]]
        run.check("no_consecutive_repeated_chunks", not consecutive, f"fragmenty {consecutive}")

        stored, stored_ids = indexed_document(metadata.document_id)
        current_ids = {chunk.chunk_id for chunk in chunks}
        run.check("index_has_the_document", stored is not None, metadata.document_id)
        if stored is not None:
            run.check("index_from_this_file", stored[0] == file_sha256(path), "inny plik PDF")
            run.check(
                "index_from_this_version",
                stored[1] == INGESTION_VERSION,
                f"{stored[1]} ≠ {INGESTION_VERSION}",
            )
            run.check(
                "index_matches_current_chunks",
                stored_ids == current_ids,
                f"w bazie {len(stored_ids)}, z kodu {len(current_ids)}, "
                f"różnych {len(stored_ids ^ current_ids)}",
            )

        expected = case.expected
        normalized = [normalize(t) for t in texts]
        for phrase in expected.get("phrases_present", []):
            wanted = normalize(phrase)
            run.check(f"phrase:{phrase[:40]}", any(wanted in t for t in normalized), "brak")
        if "legacy_glyphs_repaired" in expected:
            repaired = any(page.legacy_glyphs_repaired for page in pages)
            run.check(
                "legacy_glyphs_repaired",
                repaired is expected["legacy_glyphs_repaired"],
                f"naprawa: {repaired}",
            )
        removed = sorted({line for page in pages for line in page.removed_running_lines})
        if expected.get("no_running_lines_removed"):
            run.check("no_running_lines_removed", not removed, str(removed[:5]))
        if "toc_pages" in expected:
            toc_pages = set(expected["toc_pages"])
            outside = [
                f"{c.chunk_index} (s. {c.page_start})"
                for c in chunks
                if c.section_path
                and normalize(c.section_path[0]).startswith("spis tresci")
                and c.page_start not in toc_pages
            ]
            run.check(
                "toc_heading_not_a_section_outside_toc",
                not outside,
                f"{len(outside)} fragmentów: {outside[:6]}",
            )
        run.details.update(
            pages=len(pages),
            repeated_chunk_texts=len(texts) - len(set(texts)),
            chunks=len(chunks),
            low_text_pages=[p.page_number for p in pages if p.low_text_quality],
            removed_running_lines=removed[:20],
            section_paths=sorted({" > ".join(c.section_path) for c in chunks})[:40],
        )

    assert_passed(run)
