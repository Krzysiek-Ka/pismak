"""Tests for the read-only RAG repository used by retrieval."""

from __future__ import annotations

import builtins

import psycopg
import pytest

from src.Zrodlowyszukiwator.rag_retrieval import DatabaseConnectionError
from src.Zrodlowyszukiwator.rag_retrieval import repository as retrieval_repository
from src.Zrodlowyszukiwator.rag_retrieval.models import (
    RetrievalFilters,
)
from src.Zrodlowyszukiwator.rag_retrieval.repository import RetrievalRepository, lexical_tsquery

# Kolejność kolumn jak w zapytaniach repozytorium; na końcu wynik i status dokumentu.
ROW = (
    "chunk-1",
    "doc-1",
    "Strategia",
    "strategy",
    100,
    "resolution",
    "LI/1193/18",
    "2018-02-15",
    False,
    "https://example.org/strategia.pdf",
    "strategia.pdf",
    3,
    4,
    ["Rozdział 1"],
    "Treść fragmentu",
    0.5,
    "superseded",
)


class FakeConnection:
    """Record the statement sent to the database and return canned rows."""

    def __init__(self) -> None:
        self.sql = ""
        self.params: list[object] = []

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: list[object]) -> FakeConnection:
        self.sql = " ".join(sql.split())
        self.params = params
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return [ROW]


def make_repository(monkeypatch) -> tuple[RetrievalRepository, FakeConnection]:
    connection = FakeConnection()
    repository = RetrievalRepository("postgresql://unused")
    monkeypatch.setattr(repository, "connect", lambda: connection)
    return repository, connection


class TestLexicalTsquery:
    def test_lexical_tsquery__joins_words_with_or_so_one_missing_word_does_not_empty_the_result(
        self,
    ) -> None:
        """Verify a multi-word query matches chunks that contain only some of its words."""
        assert lexical_tsquery("bezpieczeństwo pieszych przejścia szkoła") == (
            "bezpieczeńs:* | piesz:* | przejś:* | szkoł:*"
        )

    def test_lexical_tsquery__matches_other_inflected_forms_of_long_words_by_prefix(
        self,
    ) -> None:
        """Verify the query built for one form also covers forms with a different ending."""
        query = lexical_tsquery("rowerowe")

        assert query == "rower:*"
        assert "rowerowych".startswith(query.removesuffix(":*"))

    def test_lexical_tsquery__keeps_short_words_and_numbers_exact(self) -> None:
        """Verify short names and act or year numbers are not widened into prefixes."""
        assert lexical_tsquery("drogi Wrocław 2050 uchwała XXIV/463/25") == (
            "drogi | wrocł:* | 2050 | uchwa:* | xxiv | 463 | 25"
        )

    def test_lexical_tsquery__drops_function_words_that_would_match_almost_every_chunk(
        self,
    ) -> None:
        """Verify Polish connectives and one- or two-letter words never become search terms."""
        assert lexical_tsquery(
            "brak przejścia dla pieszych przy szkole i w pobliżu oraz parku"
        ) == ("brak | przejś:* | piesz:* | szkol:* | pobli:* | parku")

    def test_lexical_tsquery__neutralizes_full_text_operators_in_the_input(self) -> None:
        """Verify text containing tsquery syntax cannot inject operators or break the query."""
        query = lexical_tsquery("rower & !piesi | (tramwaj) <-> 'autobus':*")

        assert query == "rower | piesi | tramw:* | autob:*"

    def test_lexical_tsquery__merges_forms_of_one_word_into_a_single_term(self) -> None:
        """Verify repeated words and different forms sharing a prefix appear only once."""
        assert lexical_tsquery("rowerowe Rowerowe rowerowa trasy trasy") == "rower:* | trasy"

    def test_lexical_tsquery__caps_the_number_of_terms(self) -> None:
        """Verify a long description such as the fallback plan's yields a bounded query."""
        query = lexical_tsquery(" ".join(str(number) for number in range(100, 140)))

        terms = query.split(" | ")
        assert terms[0] == "100"
        assert len(terms) == retrieval_repository.MAX_LEXICAL_TERMS

    def test_lexical_tsquery__returns_nothing_for_text_without_significant_words(self) -> None:
        """Verify a query made only of function words is recognized as empty."""
        assert lexical_tsquery("i w na oraz przez — ...") == ""


class TestRetrievalRepositoryLexicalSearch:
    def test_lexical_search__does_not_query_the_database_without_search_terms(
        self, monkeypatch
    ) -> None:
        """Verify an empty query returns nothing instead of sending an empty tsquery."""
        repository = RetrievalRepository("postgresql://unused")

        def fail_to_connect():
            raise AssertionError("the database must not be queried")

        monkeypatch.setattr(repository, "connect", fail_to_connect)

        assert repository.lexical_search(query="i w na oraz") == []

    def test_lexical_search__skips_disabled_and_replaced_documents_and_returns_the_status(
        self, monkeypatch
    ) -> None:
        """Verify the full-text query applies the same two exclusions as the vector query."""
        repository, connection = make_repository(monkeypatch)

        (chunk,) = repository.lexical_search(query="transport", limit=7)

        assert (
            "WHERE c.search_vector @@ q.query AND d.retrieval_enabled"
            " AND d.is_active IS NOT FALSE" in connection.sql
        )
        assert connection.params == ["transp:*", 7]
        assert (chunk.lifecycle_status, chunk.lexical_score) == ("superseded", 0.5)


class TestRetrievalRepositoryConnect:
    def test_connect__gives_docker_guidance_for_unavailable_local_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify a local connection failure suggests starting Docker without exposing credentials."""

        def fail_to_connect(database_url: str, **kwargs) -> None:
            raise psycopg.OperationalError("connection timeout expired")

        monkeypatch.setattr(psycopg, "connect", fail_to_connect)

        with pytest.raises(DatabaseConnectionError) as error:
            RetrievalRepository("postgresql://user:secret@localhost:5432/pismak").connect()

        assert "Docker Desktop" in str(error.value)
        assert "docker compose up -d postgres" in str(error.value)
        assert "secret" not in str(error.value)

    def test_connect__does_not_suggest_docker_for_unavailable_remote_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify remote database failures direct the user to connection settings instead."""

        def fail_to_connect(database_url: str, **kwargs) -> None:
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

        def fail_to_connect(database_url: str, **kwargs) -> None:
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
        monkeypatch.setattr(psycopg, "connect", lambda database_url, **kwargs: connection)
        monkeypatch.setattr(pgvector.psycopg, "register_vector", registered.append)

        assert RetrievalRepository("postgresql://localhost/pismak").connect() is connection
        assert registered == [connection]

    def test_connect__limits_how_long_an_unreachable_database_can_block_a_case(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify a wrong address fails after the connect timeout instead of hanging for minutes."""
        import pgvector.psycopg

        calls: list[dict[str, object]] = []

        def fake_connect(database_url: str, **kwargs: object) -> object:
            calls.append({"database_url": database_url, **kwargs})
            return object()

        monkeypatch.setattr(psycopg, "connect", fake_connect)
        monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

        RetrievalRepository("postgresql://app@127.0.0.1/pismak").connect()

        assert calls == [
            {"database_url": "postgresql://app@127.0.0.1/pismak", "connect_timeout": 10}
        ]

    def test_connect__reports_actionable_error_when_driver_import_is_blocked(
        self, monkeypatch
    ) -> None:
        """Verify a blocked native driver becomes a clear database setup error."""
        original_import = builtins.__import__

        def block_psycopg(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("blocked by application control")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_psycopg)

        with pytest.raises(RuntimeError, match="Nie można załadować sterownika PostgreSQL"):
            RetrievalRepository("postgresql://unused").connect()


class TestRetrievalRepositoryCounts:
    def test_counts__two_queries_return_documents_then_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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


class TestRetrievalRepositoryFilterSql:
    def test_filter_sql__drops_only_documents_explicitly_out_of_force_by_default(self) -> None:
        """Verify the default keeps documents with unknown status and removes replaced ones."""
        assert RetrievalRepository._filter_sql(RetrievalFilters()) == (
            " AND d.is_active IS NOT FALSE",
            [],
        )

    def test_filter_sql__keeps_documents_out_of_force_for_a_historical_search(self) -> None:
        """Verify a caller asking about policy history can get replaced documents back."""
        assert RetrievalRepository._filter_sql(RetrievalFilters(exclude_inactive=False)) == ("", [])

    def test_filter_sql__combines_every_filter_with_parameters_in_clause_order(self) -> None:
        """Verify all filters are applied together and their values stay out of the SQL text."""
        filters = RetrievalFilters(city="Wrocław", document_kinds=["policy"], active_only=True)

        sql, params = RetrievalRepository._filter_sql(filters)

        assert sql == (
            " AND d.city = %s AND d.document_kind = ANY(%s)"
            " AND d.is_active IS TRUE AND d.is_active IS NOT FALSE"
        )
        assert params == ["Wrocław", ["policy"]]


class TestRetrievalRepositoryVectorSearch:
    def test_vector_search__skips_disabled_and_replaced_documents_and_returns_the_status(
        self, monkeypatch
    ) -> None:
        """Verify the catalog switch and the validity filter both reach the vector query."""
        repository, connection = make_repository(monkeypatch)

        (chunk,) = repository.vector_search(query_embedding=[0.1, 0.2], limit=5)

        assert "WHERE d.retrieval_enabled AND d.is_active IS NOT FALSE" in connection.sql
        assert connection.params[-1] == 5
        assert (chunk.lifecycle_status, chunk.is_active, chunk.semantic_score) == (
            "superseded",
            False,
            0.5,
        )

    def test_vector_search__applies_the_catalog_switch_even_to_a_historical_search(
        self, monkeypatch
    ) -> None:
        """Verify a document disabled in the catalog stays hidden when validity is not filtered."""
        repository, connection = make_repository(monkeypatch)

        repository.vector_search(
            query_embedding=[0.1, 0.2], filters=RetrievalFilters(exclude_inactive=False)
        )

        assert "WHERE d.retrieval_enabled ORDER BY" in connection.sql
