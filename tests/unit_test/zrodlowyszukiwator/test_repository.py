from __future__ import annotations

import psycopg
import pytest

from src.Zrodlowyszukiwator.rag_retrieval import DatabaseConnectionError
from src.Zrodlowyszukiwator.rag_retrieval.repository import RetrievalRepository


class TestRetrievalRepository:
    def test_connect__gives_docker_guidance_for_unavailable_local_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify a local connection failure suggests starting Docker without exposing credentials."""

        def fail_to_connect(database_url: str) -> None:
            raise psycopg.OperationalError("connection timeout expired")

        monkeypatch.setattr(psycopg, "connect", fail_to_connect)

        with pytest.raises(DatabaseConnectionError) as error:
            RetrievalRepository(
                "postgresql://user:secret@localhost:5432/pismak"
            ).connect()

        assert "Docker Desktop" in str(error.value)
        assert "docker compose up -d postgres" in str(error.value)
        assert "secret" not in str(error.value)

    def test_connect__does_not_suggest_docker_for_unavailable_remote_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify remote database failures direct the user to connection settings instead."""

        def fail_to_connect(database_url: str) -> None:
            raise psycopg.OperationalError("connection timeout expired")

        monkeypatch.setattr(psycopg, "connect", fail_to_connect)

        with pytest.raises(DatabaseConnectionError) as error:
            RetrievalRepository("postgresql://user:secret@db.example.com/pismak").connect()

        assert "DATABASE_URL" in str(error.value)
        assert "Docker Desktop" not in str(error.value)

    @pytest.mark.parametrize(
        ("database_url", "shown"),
        [
            ("postgresql://user:secret@127.0.0.1:5433/pismak", "127.0.0.1:5433"),
            ("postgresql://user:secret@[::1]/pismak", "::1:5432"),
        ],
    )
    def test_connect__names_the_local_address_and_port_without_the_password(
        self, monkeypatch: pytest.MonkeyPatch, database_url: str, shown: str
    ) -> None:
        """Verify the operator sees which local port failed, with 5432 as the default."""

        def fail_to_connect(database_url: str) -> None:
            raise psycopg.OperationalError("connection refused")

        monkeypatch.setattr(psycopg, "connect", fail_to_connect)

        with pytest.raises(DatabaseConnectionError) as error:
            RetrievalRepository(database_url).connect()

        assert f"({shown})" in str(error.value)
        assert "secret" not in str(error.value)
        assert isinstance(error.value.__cause__, psycopg.OperationalError)

    def test_connect__registers_pgvector_on_the_opened_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify vectors can be sent as query parameters on every connection."""
        import pgvector.psycopg

        connection = object()
        registered: list[object] = []
        monkeypatch.setattr(psycopg, "connect", lambda database_url: connection)
        monkeypatch.setattr(pgvector.psycopg, "register_vector", registered.append)

        assert RetrievalRepository("postgresql://localhost/pismak").connect() is connection
        assert registered == [connection]


class TestCounts:
    def test_counts__returns_documents_and_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the UI caption shows document and chunk totals in the right order."""
        answers = iter([(2,), (63,)])

        class Connection:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def execute(self, sql):
                self.sql = sql
                return self

            def fetchone(self):
                return next(answers)

        repository = RetrievalRepository("postgresql://unused")
        monkeypatch.setattr(repository, "connect", Connection)

        assert repository.counts() == (2, 63)
