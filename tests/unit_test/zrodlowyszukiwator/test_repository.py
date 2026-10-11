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
