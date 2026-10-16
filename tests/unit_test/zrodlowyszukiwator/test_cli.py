from __future__ import annotations

import json
import sys

import pytest

from src.Zrodlowyszukiwator.rag_ingestion import cli


class FakeRepository:
    created: list[FakeRepository] = []

    def __init__(self, database_url: str, *, embedding_dim: int) -> None:
        self.database_url = database_url
        self.embedding_dim = embedding_dim
        self.schema_ready = False
        FakeRepository.created.append(self)

    def ensure_schema(self) -> None:
        self.schema_ready = True


@pytest.fixture
def ingested(monkeypatch) -> dict[str, object]:
    """Replace the database, the embedder and the pipeline; return what ingestion received."""
    captured: dict[str, object] = {}

    def fake_ingest_pdf(path, metadata, **kwargs):
        captured.update(
            path=path,
            metadata=metadata,
            force=kwargs["force"],
            repository=kwargs["repository"],
            embedder=kwargs["embedder"],
        )
        return {"status": "ingested", "document_id": metadata.document_id}

    FakeRepository.created = []
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://test")
    for name in ("RAG_EMBEDDING_MODEL", "OLLAMA_URL", "RAG_EMBEDDING_DIM"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "PgVectorRepository", FakeRepository)
    monkeypatch.setattr(cli, "OllamaEmbedder", lambda **kwargs: kwargs)
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

    def test_main__passes_force_and_prepares_the_schema_before_ingesting(
        self, monkeypatch, ingested, capsys
    ) -> None:
        """Verify --force reaches the pipeline and a fresh database gets its tables first."""
        monkeypatch.setattr(
            sys, "argv", ["cli", "--file", "01_Polityka-rowerowa-wroclawia.pdf", "--force"]
        )

        cli.main()

        assert ingested["force"] is True
        assert ingested["repository"].schema_ready is True
        assert "'status': 'ingested'" in capsys.readouterr().out

    def test_main__uses_the_default_embedding_settings(self, monkeypatch, ingested) -> None:
        """Verify a minimal .env ingests with bge-m3 on the local Ollama and 1024 dimensions."""
        monkeypatch.setattr(sys, "argv", ["cli", "--file", "01_Polityka-rowerowa-wroclawia.pdf"])

        cli.main()

        assert ingested["embedder"] == {"model": "bge-m3", "base_url": "http://127.0.0.1:11434"}
        assert ingested["repository"].database_url == "postgresql://test"
        assert ingested["repository"].embedding_dim == 1024

    def test_main__applies_embedding_settings_from_the_environment(
        self, monkeypatch, ingested
    ) -> None:
        """Verify another embedding model gets a matching vector size and Ollama address."""
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", "nomic-embed-text")
        monkeypatch.setenv("OLLAMA_URL", "http://ollama.test:11434")
        monkeypatch.setenv("RAG_EMBEDDING_DIM", "768")
        monkeypatch.setattr(sys, "argv", ["cli", "--file", "01_Polityka-rowerowa-wroclawia.pdf"])

        cli.main()

        assert ingested["embedder"] == {
            "model": "nomic-embed-text",
            "base_url": "http://ollama.test:11434",
        }
        assert ingested["repository"].embedding_dim == 768

    def test_main__reads_metadata_from_the_catalog_given_on_the_command_line(
        self, monkeypatch, ingested, tmp_path
    ) -> None:
        """Verify a user's own catalog can describe documents that are not in the repository."""
        catalog = tmp_path / "moj_katalog.json"
        catalog.write_text(
            json.dumps(
                {
                    "documents": [
                        {
                            "filename": "wlasny.pdf",
                            "document_id": "wlasny",
                            "title": "Własny dokument",
                            "document_kind": "plan",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            sys, "argv", ["cli", "--file", "wlasny.pdf", "--catalog", str(catalog)]
        )

        cli.main()

        assert ingested["metadata"].document_id == "wlasny"

    def test_main__stops_before_the_database_when_database_url_is_missing(
        self, monkeypatch, ingested
    ) -> None:
        """Verify a missing connection string fails before any table or vector is touched."""
        monkeypatch.delenv("DATABASE_URL")
        monkeypatch.setattr(sys, "argv", ["cli", "--file", "01_Polityka-rowerowa-wroclawia.pdf"])

        with pytest.raises(KeyError, match="DATABASE_URL"):
            cli.main()

        assert FakeRepository.created == []
        assert ingested == {}

    def test_main__requires_the_file_argument(self, monkeypatch, ingested, capsys) -> None:
        """Verify running without --file prints usage instead of ingesting something by default."""
        monkeypatch.setattr(sys, "argv", ["cli"])

        with pytest.raises(SystemExit) as exit_info:
            cli.main()

        assert exit_info.value.code == 2
        assert "--file" in capsys.readouterr().err
        assert ingested == {}
