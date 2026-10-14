from __future__ import annotations

from pathlib import Path

import pytest

from src.Zrodlowyszukiwator.rag_ingestion import pipeline
from src.Zrodlowyszukiwator.rag_ingestion.models import DocumentMetadata, ExtractedPage

PAGE_TEXT = "Miasto rozwija spójną sieć dróg rowerowych i poprawia bezpieczeństwo ruchu. " * 5


def make_metadata(**overrides: object) -> DocumentMetadata:
    values: dict[str, object] = {
        "document_id": "doc-1",
        "title": "Polityka",
        "document_kind": "policy",
        "adoption_act_number": "LV/1688/10",
        "is_active": True,
    }
    values.update(overrides)
    return DocumentMetadata.model_validate(values)


class FakeRepository:
    """Stand in for the database: report what is stored and record what gets written."""

    def __init__(self, *, stored_hash: str | None, stored_title: str | None) -> None:
        self.stored_hash = stored_hash
        self.stored_title = stored_title
        self.calls: list[str] = []
        self.chunk_metadata: dict[str, object] = {}

    def document_hash(self, document_id: str) -> str | None:
        return self.stored_hash

    def embedding_metadata(self, document_id: str) -> tuple[str, str, str] | None:
        return (self.stored_title, "policy", "Wrocław") if self.stored_title else None

    def upsert_document(self, metadata: DocumentMetadata, **kwargs: object) -> None:
        self.calls.append("upsert_document")

    def refresh_chunk_metadata(self, **kwargs: object) -> None:
        self.calls.append("refresh_chunk_metadata")
        self.chunk_metadata = kwargs["metadata"]

    def replace_chunks(self, **kwargs: object) -> None:
        self.calls.append("replace_chunks")


class FakeEmbedder:
    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.0, 1.0] for _ in texts]


@pytest.fixture
def pdf(tmp_path: Path, monkeypatch) -> Path:
    """Provide a file to hash and replace PDF parsing with one page of text."""
    path = tmp_path / "polityka.pdf"
    path.write_bytes(b"zawartosc pliku")
    monkeypatch.setattr(
        pipeline,
        "extract_pdf_pages",
        lambda pdf_path: [ExtractedPage(page_number=1, text=PAGE_TEXT, char_count=len(PAGE_TEXT))],
    )
    return path


class TestIngestPdf:
    def test_ingest_pdf__refreshes_metadata_without_embedding_when_only_the_catalog_changed(
        self, pdf: Path
    ) -> None:
        """Verify a corrected catalog entry reaches the database without re-embedding the file."""
        repository = FakeRepository(stored_hash=pipeline.file_sha256(pdf), stored_title="Polityka")
        embedder = FakeEmbedder()

        result = pipeline.ingest_pdf(
            pdf,
            make_metadata(adoption_act_number="XLVIII/1169/13"),
            repository=repository,
            embedder=embedder,
        )

        assert result["reason"] == "identical_content_metadata_refreshed"
        assert repository.calls == ["upsert_document", "refresh_chunk_metadata"]
        assert repository.chunk_metadata["adoption_act_number"] == "XLVIII/1169/13"
        assert embedder.embedded == []

    def test_ingest_pdf__reprocesses_the_file_when_the_title_in_the_catalog_changed(
        self, pdf: Path
    ) -> None:
        """Verify a new title, which is part of every embedding, is not left out of the vectors."""
        repository = FakeRepository(stored_hash=pipeline.file_sha256(pdf), stored_title="Polityka")
        embedder = FakeEmbedder()

        result = pipeline.ingest_pdf(
            pdf,
            make_metadata(title="Polityka Rowerowa"),
            repository=repository,
            embedder=embedder,
        )

        assert result["status"] == "ingested"
        assert repository.calls == ["upsert_document", "replace_chunks"]
        assert all(text.startswith("Dokument: Polityka Rowerowa\n") for text in embedder.embedded)

    def test_ingest_pdf__processes_a_document_that_is_not_in_the_database_yet(
        self, pdf: Path
    ) -> None:
        """Verify the metadata-only path cannot swallow the first ingestion of a document."""
        repository = FakeRepository(stored_hash=None, stored_title=None)
        embedder = FakeEmbedder()

        result = pipeline.ingest_pdf(pdf, make_metadata(), repository=repository, embedder=embedder)

        assert result["status"] == "ingested"
        assert "replace_chunks" in repository.calls
        assert len(embedder.embedded) == result["chunks"] > 0

    def test_ingest_pdf__reprocesses_an_unchanged_file_when_forced(self, pdf: Path) -> None:
        """Verify --force still rebuilds chunks and embeddings for identical content."""
        repository = FakeRepository(stored_hash=pipeline.file_sha256(pdf), stored_title="Polityka")

        result = pipeline.ingest_pdf(
            pdf, make_metadata(), repository=repository, embedder=FakeEmbedder(), force=True
        )

        assert result["status"] == "ingested"
        assert repository.calls == ["upsert_document", "replace_chunks"]
