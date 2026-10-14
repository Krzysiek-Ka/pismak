from __future__ import annotations

from src.Zrodlowyszukiwator.rag_retrieval.models import RetrievalFilters
from src.Zrodlowyszukiwator.rag_retrieval.repository import RetrievalRepository

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


class TestFilterSql:
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


class TestVectorSearch:
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


class TestLexicalSearch:
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
        assert connection.params == ["transport", 7]
        assert (chunk.lifecycle_status, chunk.lexical_score) == ("superseded", 0.5)
