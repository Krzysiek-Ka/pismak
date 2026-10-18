from __future__ import annotations

import hashlib
from pathlib import Path

from .chunker import build_text_units, chunk_document_metadata, chunk_units
from .embedder import OllamaEmbedder
from .models import DocumentMetadata
from .pdf_extractor import extract_pdf_pages
from .repository import PgVectorRepository

# Zmiana sposobu ekstrakcji lub dzielenia tekstu wymaga podniesienia wersji: dokumenty
# zapisane starszą wersją są wtedy przetwarzane od nowa mimo identycznego pliku.
# 0.2: nagłówki zostają w treści, sklejone przeniesienia, naprawa „Ŝ” → „ż”.
# 0.3: bez znaków sterujących i bez powtórzeń nagłówków/stopek stron.
# 0.4: zakładka nie przekracza max_chars i nie tworzy osobnego fragmentu.
# 0.5: linia z samą liczbą jest usuwana jako numer strony tylko na brzegu strony.
INGESTION_VERSION = "0.5"


def file_sha256(path: str | Path) -> str:
    """Calculate the SHA-256 digest of a file in streaming blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ingest_pdf(
    pdf_path: str | Path,
    metadata: DocumentMetadata,
    *,
    repository: PgVectorRepository,
    embedder: OllamaEmbedder,
    max_chars: int = 3500,
    min_chars: int = 250,
    overlap_units: int = 1,
    force: bool = False,
) -> dict:
    """Ingest a PDF after extracting, chunking, embedding, and storing its contents."""
    pdf_path = Path(pdf_path)
    digest = file_sha256(pdf_path)

    existing_hash = repository.document_hash(metadata.document_id)
    # Tytuł, typ i miasto są częścią tekstu, z którego liczony jest embedding każdego chunka,
    # więc ich zmiana wymaga pełnego przetworzenia tak samo jak zmiana treści pliku.
    embedding_metadata = (metadata.title, metadata.document_kind.value, metadata.city)
    if (
        existing_hash == digest
        and repository.embedding_metadata(metadata.document_id) == embedding_metadata
        and repository.ingestion_version(metadata.document_id) == INGESTION_VERSION
        and not force
    ):
        # Treść bez zmian: bez ponownego dzielenia i embeddingów. Metadane z katalogu są
        # jednak zapisywane, żeby poprawka statusu, numeru aktu czy adresu trafiła do bazy.
        repository.upsert_document(
            metadata,
            source_filename=pdf_path.name,
            content_sha256=digest,
            ingestion_version=INGESTION_VERSION,
        )
        repository.refresh_chunk_metadata(
            document_id=metadata.document_id,
            hierarchy_rank=metadata.hierarchy_rank,
            metadata=chunk_document_metadata(metadata),
        )
        return {
            "status": "skipped",
            "reason": "identical_content_metadata_refreshed",
            "document_id": metadata.document_id,
            "sha256": digest,
        }

    pages = extract_pdf_pages(pdf_path)
    units = build_text_units(pages)
    chunks = chunk_units(
        units,
        metadata,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_units=overlap_units,
    )

    if not chunks:
        raise RuntimeError("PDF nie wygenerował żadnych chunków.")

    embeddings = embedder.embed([c.embedding_text for c in chunks])

    # Jedna transakcja: po błędzie nie zostaje nowy hash dokumentu przy starych fragmentach.
    repository.store_document(
        metadata,
        source_filename=pdf_path.name,
        content_sha256=digest,
        ingestion_version=INGESTION_VERSION,
        chunks=chunks,
        embeddings=embeddings,
    )

    low_text_pages = [p.page_number for p in pages if p.low_text_quality]
    repaired_glyph_pages = [p.page_number for p in pages if p.legacy_glyphs_repaired]
    removed_running_lines = sorted({line for p in pages for line in p.removed_running_lines})

    return {
        "status": "ingested",
        "document_id": metadata.document_id,
        "title": metadata.title,
        "pages": len(pages),
        "chunks": len(chunks),
        "low_text_pages": low_text_pages,
        "repaired_glyph_pages": repaired_glyph_pages,
        "removed_running_lines": removed_running_lines,
        "sha256": digest,
        "embedding_dim": len(embeddings[0]),
    }
