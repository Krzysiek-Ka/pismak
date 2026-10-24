"""Tests for the pgvector repository that stores ingested documents and chunks."""

from __future__ import annotations

import builtins
from datetime import date

import numpy as np
import pytest

from src.Zrodlowyszukiwator.rag_ingestion.models import (
    ChunkRecord,
    DocumentMetadata,
    DocumentRelation,
)
from src.Zrodlowyszukiwator.rag_ingestion.repository import PgVectorRepository


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self.connection.statements.append((" ".join(sql.split()), params))

    def executemany(self, sql: str, rows: list[tuple[object, ...]]) -> None:
        self.connection.statements.append((" ".join(sql.split()), rows))
        self.connection.inserted_rows.extend(rows)


class FakeConnection:
    """Record statements, return canned rows and remember whether the work was committed."""

    def __init__(self, rows: list[tuple[object, ...] | None] | None = None) -> None:
        self.rows = list(rows or [])
        self.statements: list[tuple[str, object]] = []
        self.inserted_rows: list[tuple[object, ...]] = []
        self.commits = 0
        self.exited_with: type[BaseException] | None = None

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # psycopg wycofuje transakcję, gdy blok `with` kończy się wyjątkiem.
        self.exited_with = exc_type
        return False

    def execute(self, sql: str, params: object = None) -> FakeConnection:
        self.statements.append((" ".join(sql.split()), params))
        return self

    def fetchone(self) -> tuple[object, ...] | None:
        return self.rows.pop(0)

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


def make_repository(
    monkeypatch, rows: list[tuple[object, ...] | None] | None = None, *, embedding_dim: int = 3
) -> tuple[PgVectorRepository, FakeConnection]:
    connection = FakeConnection(rows)
    repository = PgVectorRepository("postgresql://unused", embedding_dim=embedding_dim)
    monkeypatch.setattr(repository, "connect", lambda: connection)
    return repository, connection


def make_chunk(index: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"doc-1:{index}",
        document_id="doc-1",
        chunk_index=index,
        page_start=index + 1,
        page_end=index + 1,
        section_path=["Rozdział 1"],
        text=f"Tekst {index}",
        embedding_text=f"Dokument | Tekst {index}",
        text_sha256=f"sha-{index}",
        metadata={"title": "Dokument"},
    )


class TestPgVectorRepositoryConnect:
    def test_connect__reports_actionable_error_when_driver_import_is_blocked(
        self, monkeypatch
    ) -> None:
        """Verify a missing native driver becomes a setup hint that names libpq, not [binary]."""
        original_import = builtins.__import__

        def block_psycopg(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("DLL load failed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_psycopg)

        with pytest.raises(RuntimeError, match="libpq.dll") as error:
            PgVectorRepository("postgresql://unused").connect()

        assert "[binary]" not in str(error.value)
        assert isinstance(error.value.__cause__, ImportError)

    def test_connect__limits_how_long_an_unreachable_database_can_block_ingestion(
        self, monkeypatch
    ) -> None:
        """Verify a wrong address fails after the connect timeout instead of hanging for minutes."""
        import pgvector.psycopg
        import psycopg

        calls: list[dict[str, object]] = []

        def fake_connect(database_url: str, **kwargs: object) -> FakeConnection:
            calls.append({"database_url": database_url, **kwargs})
            return FakeConnection()

        monkeypatch.setattr(psycopg, "connect", fake_connect)
        monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

        PgVectorRepository("postgresql://ingest@127.0.0.1/pismak").connect()

        assert calls == [
            {"database_url": "postgresql://ingest@127.0.0.1/pismak", "connect_timeout": 10}
        ]


class TestPgVectorRepositoryEnsureSchema:
    def test_ensure_schema__sizes_the_vector_column_for_the_configured_embedding_model(
        self, monkeypatch
    ) -> None:
        """Verify a different embedding model gets a matching column instead of the bge-m3 size."""
        repository, connection = make_repository(monkeypatch, embedding_dim=768)

        repository.ensure_schema()

        chunk_table = next(sql for sql, _ in connection.statements if "rag_chunks (" in sql)
        assert "embedding vector(768) NOT NULL" in chunk_table
        assert "metadata JSONB NOT NULL DEFAULT '{}'::jsonb" in chunk_table
        assert connection.commits == 1

    def test_ensure_schema__adds_columns_introduced_later_to_an_existing_database(
        self, monkeypatch
    ) -> None:
        """Verify a database created by an older version is upgraded without losing documents."""
        repository, connection = make_repository(monkeypatch)

        repository.ensure_schema()

        alter = next(sql for sql, _ in connection.statements if sql.startswith("ALTER TABLE"))
        for column in ("lifecycle_status", "retrieval_enabled", "metadata_url", "is_annex"):
            assert f"ADD COLUMN IF NOT EXISTS {column}" in alter
        assert "retrieval_enabled BOOLEAN NOT NULL DEFAULT TRUE" in alter

    def test_ensure_schema__can_run_again_on_an_existing_schema(self, monkeypatch) -> None:
        """Verify every statement is idempotent, because ingestion runs it on each start."""
        repository, connection = make_repository(monkeypatch)

        repository.ensure_schema()

        for sql, _ in connection.statements:
            assert "IF NOT EXISTS" in sql, sql


class TestPgVectorRepositoryDocumentHash:
    @pytest.mark.parametrize(
        ("row", "expected"), [(("abc",), "abc"), (None, None)], ids=["stored", "new_document"]
    )
    def test_document_hash__stored_or_missing_row_returns_the_hash_or_none(
        self, monkeypatch, row: tuple[object, ...] | None, expected: object
    ) -> None:
        """Verify an unknown document reads as absent, which triggers its first ingestion."""
        repository, connection = make_repository(monkeypatch, [row])

        assert repository.document_hash("doc-1") == expected
        assert connection.statements[0][1] == ("doc-1",)


class TestPgVectorRepositoryIngestionVersion:
    @pytest.mark.parametrize(
        ("row", "expected"), [(("v3",), "v3"), (None, None)], ids=["stored", "new_document"]
    )
    def test_ingestion_version__stored_or_missing_row_returns_the_version_or_none(
        self, monkeypatch, row: tuple[object, ...] | None, expected: object
    ) -> None:
        """Verify a document stored by an older version is recognised, and a new one is absent."""
        repository, connection = make_repository(monkeypatch, [row])

        assert repository.ingestion_version("doc-1") == expected
        assert connection.statements[0][1] == ("doc-1",)


class TestPgVectorRepositoryEmbeddingMetadata:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [(("Tytuł", "policy", "Wrocław"), ("Tytuł", "policy", "Wrocław")), (None, None)],
        ids=["stored", "new_document"],
    )
    def test_embedding_metadata__stored_or_missing_row_returns_the_triple_or_none(
        self, monkeypatch, row: tuple[object, ...] | None, expected: object
    ) -> None:
        """Verify title, kind and city come back in the order ingest_pdf compares them."""
        repository, connection = make_repository(monkeypatch, [row])

        assert repository.embedding_metadata("doc-1") == expected
        assert connection.statements[0][1] == ("doc-1",)


class TestPgVectorRepositoryCounts:
    def test_counts__two_queries_return_documents_then_chunks(self, monkeypatch) -> None:
        """Verify the UI shows document and chunk totals in the right places."""
        repository, _ = make_repository(monkeypatch, [(6,), (1450,)])

        assert repository.counts() == (6, 1450)


class TestPgVectorRepositoryUpsertDocument:
    def test_upsert_document__sends_one_parameter_per_placeholder_in_column_order(
        self, monkeypatch
    ) -> None:
        """Verify enums, dates and JSON fields reach the columns they belong to."""
        repository, connection = make_repository(monkeypatch)
        metadata = DocumentMetadata(
            document_id="doc-1",
            title="Polityka",
            document_kind="policy",
            legal_form="resolution",
            adoption_date=date(2023, 5, 18),
            is_active=True,
            lifecycle_status="current",
            tags=["mobilność"],
            relations=[DocumentRelation(target_document_id="doc-0", relation="implements")],
        )

        repository.upsert_document(
            metadata,
            source_filename="polityka.pdf",
            content_sha256="sha",
            ingestion_version="v3",
        )

        sql, params = connection.statements[0]
        assert sql.count("%s") == len(params) == 25
        assert params[:4] == ("doc-1", "Polityka", "policy", 80)
        assert params[7] == "resolution"
        assert params[9] == date(2023, 5, 18)
        assert params[14:16] == ("polityka.pdf", "sha")
        assert params[17].obj == ["mobilność"]
        assert params[18].obj == [
            {"target_document_id": "doc-0", "relation": "implements", "note": None}
        ]
        assert params[20:] == ("v3", "current", True, None, False)
        assert connection.commits == 1

    def test_upsert_document__stores_no_lifecycle_status_when_the_catalog_has_none(
        self, monkeypatch
    ) -> None:
        """Verify an older catalog entry without the field does not fail on a missing enum."""
        repository, connection = make_repository(monkeypatch)

        repository.upsert_document(
            DocumentMetadata(document_id="doc-1", title="Dokument", document_kind="other"),
            source_filename="dokument.pdf",
            content_sha256="sha",
            ingestion_version="v3",
        )

        params = connection.statements[0][1]
        assert params[7] == "unknown"
        assert params[21] is None

    def test_upsert_document__updates_every_column_of_an_already_stored_document(
        self, monkeypatch
    ) -> None:
        """Verify re-ingesting a corrected catalog entry overwrites all of its stored metadata."""
        repository, connection = make_repository(monkeypatch)

        repository.upsert_document(
            DocumentMetadata(document_id="doc-1", title="Dokument", document_kind="other"),
            source_filename="dokument.pdf",
            content_sha256="sha",
            ingestion_version="v3",
        )

        sql = connection.statements[0][0]
        inserted = sql.split("INSERT INTO rag_documents (")[1].split(")")[0]
        columns = [c.strip() for c in inserted.split(",")]
        for column in columns:
            if column not in ("document_id", "ingested_at"):
                assert f"{column} = EXCLUDED.{column}" in sql, column


class TestPgVectorRepositoryRefreshChunkMetadata:
    def test_refresh_chunk_metadata__rewrites_only_the_chunks_of_one_document(
        self, monkeypatch
    ) -> None:
        """Verify a catalog fix updates the copied metadata without touching other documents."""
        repository, connection = make_repository(monkeypatch)

        repository.refresh_chunk_metadata(
            document_id="doc-1", hierarchy_rank=80, metadata={"is_active": False}
        )

        sql, params = connection.statements[0]
        assert sql.endswith("WHERE document_id = %s")
        assert params[0].obj == {"is_active": False}
        assert params[1:] == (80, "doc-1")
        assert connection.commits == 1


class TestPgVectorRepositoryReplaceChunks:
    def test_replace_chunks__rejects_mismatched_counts_before_touching_the_database(
        self, monkeypatch
    ) -> None:
        """Verify a lost embedding cannot shift vectors onto the wrong chunks."""
        repository, connection = make_repository(monkeypatch)

        with pytest.raises(ValueError, match="Liczba chunków i embeddingów"):
            repository.replace_chunks(
                document_id="doc-1",
                hierarchy_rank=80,
                chunks=[make_chunk(0), make_chunk(1)],
                embeddings=[[0.1, 0.2, 0.3]],
            )

        assert connection.statements == []

    def test_replace_chunks__keeps_the_old_chunks_when_an_embedding_has_the_wrong_size(
        self, monkeypatch
    ) -> None:
        """Verify a bad vector is rejected before any chunk of the document is deleted."""
        repository, connection = make_repository(monkeypatch)

        with pytest.raises(ValueError, match=r"oczekiwano \(3,\)"):
            repository.replace_chunks(
                document_id="doc-1",
                hierarchy_rank=80,
                chunks=[make_chunk(0), make_chunk(1)],
                embeddings=[[0.1, 0.2, 0.3], [0.1, 0.2]],
            )

        assert connection.statements == []
        assert connection.commits == 0

    def test_replace_chunks__deletes_then_inserts_all_chunks_in_order_in_one_commit(
        self, monkeypatch
    ) -> None:
        """Verify re-ingestion leaves exactly the new chunks, each with its own vector."""
        repository, connection = make_repository(monkeypatch)

        repository.replace_chunks(
            document_id="doc-1",
            hierarchy_rank=80,
            chunks=[make_chunk(0), make_chunk(1)],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

        assert connection.statements[0] == (
            "DELETE FROM rag_chunks WHERE document_id = %s",
            ("doc-1",),
        )
        assert [row[0] for row in connection.inserted_rows] == ["doc-1:0", "doc-1:1"]
        first = connection.inserted_rows[0]
        assert first[5].obj == ["Rozdział 1"]
        assert first[9] == 80
        assert first[10].obj == {"title": "Dokument"}
        assert first[11].dtype == np.float32
        np.testing.assert_allclose(first[11], [0.1, 0.2, 0.3], rtol=1e-6)
        assert connection.commits == 1

    def test_replace_chunks__empties_a_document_that_no_longer_has_text(self, monkeypatch) -> None:
        """Verify a document re-ingested without chunks does not keep stale ones."""
        repository, connection = make_repository(monkeypatch)

        repository.replace_chunks(document_id="doc-1", hierarchy_rank=80, chunks=[], embeddings=[])

        assert connection.statements[0][0].startswith("DELETE FROM rag_chunks")
        assert connection.inserted_rows == []
        assert connection.commits == 1


class TestPgVectorRepositoryStoreDocument:
    def test_store_document__writes_the_document_and_its_chunks_in_one_commit(
        self, monkeypatch
    ) -> None:
        """Verify a new hash is never committed without the chunks that belong to it."""
        repository, connection = make_repository(monkeypatch)

        repository.store_document(
            DocumentMetadata(document_id="doc-1", title="Dokument", document_kind="other"),
            source_filename="polityka.pdf",
            content_sha256="sha-nowy",
            ingestion_version="0.5",
            chunks=[make_chunk(0), make_chunk(1)],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

        statements = [sql for sql, _ in connection.statements]
        assert statements[0].startswith("INSERT INTO rag_documents")
        assert statements[1].startswith("DELETE FROM rag_chunks")
        assert statements[2].startswith("INSERT INTO rag_chunks")
        assert [row[0] for row in connection.inserted_rows] == ["doc-1:0", "doc-1:1"]
        assert connection.commits == 1

    def test_store_document__a_bad_vector_leaves_the_stored_document_untouched(
        self, monkeypatch
    ) -> None:
        """Verify a failed ingestion cannot mark the document as current with stale chunks."""
        repository, connection = make_repository(monkeypatch)

        with pytest.raises(ValueError, match=r"oczekiwano \(3,\)"):
            repository.store_document(
                DocumentMetadata(document_id="doc-1", title="Dokument", document_kind="other"),
                source_filename="polityka.pdf",
                content_sha256="sha-nowy",
                ingestion_version="0.5",
                chunks=[make_chunk(0)],
                embeddings=[[0.1, 0.2]],
            )

        assert connection.statements == []
        assert connection.commits == 0

    def test_store_document__a_failed_chunk_insert_rolls_back_the_document_row(
        self, monkeypatch
    ) -> None:
        """Verify an error after the document upsert ends the transaction without a commit."""
        repository, connection = make_repository(monkeypatch)

        def fail_insert(sql: str, rows: list[tuple[object, ...]]) -> None:
            raise RuntimeError("połączenie zerwane")

        monkeypatch.setattr(
            FakeCursor, "executemany", lambda self, sql, rows: fail_insert(sql, rows)
        )

        with pytest.raises(RuntimeError, match="połączenie zerwane"):
            repository.store_document(
                DocumentMetadata(document_id="doc-1", title="Dokument", document_kind="other"),
                source_filename="polityka.pdf",
                content_sha256="sha-nowy",
                ingestion_version="0.5",
                chunks=[make_chunk(0)],
                embeddings=[[0.1, 0.2, 0.3]],
            )

        assert connection.statements[0][0].startswith("INSERT INTO rag_documents")
        assert connection.commits == 0
        assert connection.exited_with is RuntimeError
