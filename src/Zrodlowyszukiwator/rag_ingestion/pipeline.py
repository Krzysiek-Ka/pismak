from __future__ import annotations

import hashlib
from pathlib import Path

from .chunker import build_text_units, chunk_units
from .embedder import OllamaEmbedder
from .models import DocumentMetadata
from .pdf_extractor import extract_pdf_pages
from .repository import PgVectorRepository

INGESTION_VERSION = "0.1"


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
    if existing_hash == digest and not force:
        return {
            "status": "skipped",
            "reason": "identical_content_already_ingested",
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

    repository.upsert_document(
        metadata,
        source_filename=pdf_path.name,
        content_sha256=digest,
        ingestion_version=INGESTION_VERSION,
    )
    repository.replace_chunks(
        document_id=metadata.document_id,
        hierarchy_rank=metadata.hierarchy_rank,
        chunks=chunks,
        embeddings=embeddings,
    )

    low_text_pages = [p.page_number for p in pages if p.low_text_quality]

    return {
        "status": "ingested",
        "document_id": metadata.document_id,
        "title": metadata.title,
        "pages": len(pages),
        "chunks": len(chunks),
        "low_text_pages": low_text_pages,
        "sha256": digest,
        "embedding_dim": len(embeddings[0]),
    }
