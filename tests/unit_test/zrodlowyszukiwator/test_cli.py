from __future__ import annotations

import sys

import pytest

from src.Zrodlowyszukiwator.rag_ingestion import cli


class FakeRepository:
    def __init__(self, database_url: str, *, embedding_dim: int) -> None:
        self.database_url = database_url

    def ensure_schema(self) -> None:
        pass


@pytest.fixture
def ingested(monkeypatch) -> dict[str, object]:
    """Replace the database, the embedder and the pipeline; return what ingestion received."""
    captured: dict[str, object] = {}

    def fake_ingest_pdf(path, metadata, **kwargs):
        captured.update(path=path, metadata=metadata, force=kwargs["force"])
        return {"status": "ingested"}

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(cli, "PgVectorRepository", FakeRepository)
    monkeypatch.setattr(cli, "OllamaEmbedder", lambda **kwargs: object())
    monkeypatch.setattr(cli, "ingest_pdf", fake_ingest_pdf)
    return captured


class TestMain:
    def test_main__finds_metadata_in_the_repository_catalog_without_a_catalog_argument(
        self, monkeypatch, ingested, tmp_path
    ) -> None:
        """Verify the one-argument command from the README works from any working directory."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["cli", "--file", "anywhere/01_Polityka-rowerowa-wroclawia.pdf"]
        )

        cli.main()

        assert ingested["metadata"].document_id == "wroclaw-bicycle-policy-2010"
        assert ingested["force"] is False

    def test_main__rejects_a_file_that_has_no_catalog_metadata(self, monkeypatch, ingested) -> None:
        """Verify an unknown document stops before touching the database, naming the file."""
        monkeypatch.setattr(sys, "argv", ["cli", "--file", "nieznany.pdf"])

        with pytest.raises(SystemExit, match="nieznany.pdf"):
            cli.main()

        assert ingested == {}
