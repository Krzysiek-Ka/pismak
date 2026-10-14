from __future__ import annotations

import re
from typing import Any

import numpy as np

from .models import RetrievalFilters, RetrievedChunk

# Najczęstsze polskie słowa funkcyjne. Konfiguracja 'simple' nie ma listy stop-słów,
# a przy łączeniu terminów przez OR takie słowo dopasowałoby niemal każdy fragment.
STOPWORDS = frozenset(
    """
    aby ale albo bez być czy dla gdy gdzie ich jak jako jego jej jest już jeśli jeżeli
    kiedy który która które którzy którego której którym lub mają może nad nie niż oraz
    pod przed przez przy się sobie swoje swój tak tam także tego tej ten też tylko tym
    więc został została zostało zostały jednak ponieważ dlatego wraz nas będzie było były
    mogą
    """.split()
)
MAX_LEXICAL_TERMS = 16


def lexical_tsquery(text: str) -> str:
    """Turn free text into an OR query of its significant words, tolerant to Polish inflection."""
    terms: list[str] = []
    for word in re.findall(r"[^\W_]+", text.lower()):
        if word.isdigit():
            terms.append(word)
        elif len(word) >= 3 and word not in STOPWORDS:
            # 'simple' nie odmienia słów, więc dłuższe słowo traci typową końcówkę
            # i jest dopasowywane prefiksem: „rowerowe” znajdzie też „rowerowych”.
            terms.append(word if len(word) <= 5 else f"{word[: max(5, len(word) - 3)]}:*")
    # Różne formy tego samego słowa dają ten sam prefiks, więc duplikaty usuwamy na końcu.
    return " | ".join(list(dict.fromkeys(terms))[:MAX_LEXICAL_TERMS])


class DatabaseConnectionError(RuntimeError):
    """Raised when the configured PostgreSQL server cannot be reached."""


class RetrievalRepository:
    """Czyta z tabel rag_documents i rag_chunks utworzonych przez rag_ingestion."""

    def __init__(self, database_url: str) -> None:
        """Initialize the retrieval repository with a database connection string."""
        self.database_url = database_url

    def connect(self) -> Any:
        """Open a PostgreSQL connection with pgvector adapters registered."""
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            from psycopg.conninfo import conninfo_to_dict
        except ImportError as exc:
            raise RuntimeError(
                "Nie można załadować sterownika PostgreSQL. "
                "Sprawdź, czy psycopg jest zainstalowany i czy biblioteka libpq.dll "
                "jest dostępna w PATH. Po zmianie PATH uruchom ponownie kernel."
            ) from exc

        try:
            conn = psycopg.connect(self.database_url)
        except psycopg.OperationalError as exc:
            conninfo = conninfo_to_dict(self.database_url)
            host = conninfo.get("host", "")
            if host.lower() in ("localhost", "127.0.0.1", "::1"):
                port = conninfo.get("port", "5432")
                message = (
                    f"Nie można połączyć się z lokalną bazą PostgreSQL ({host}:{port}). "
                    "Jeśli baza działa w Dockerze, uruchom Docker Desktop i kontener "
                    "poleceniem `docker compose up -d postgres`. Jeśli Docker już działa, "
                    "sprawdź port i DATABASE_URL."
                )
            else:
                message = (
                    "Nie można połączyć się z bazą PostgreSQL. "
                    "Sprawdź DATABASE_URL oraz dostępność serwera."
                )
            raise DatabaseConnectionError(message) from exc

        register_vector(conn)
        return conn

    @staticmethod
    def _filter_sql(filters: RetrievalFilters) -> tuple[str, list[object]]:
        """Build the SQL predicate and parameters represented by retrieval filters."""
        clauses: list[str] = []
        params: list[object] = []

        if filters.city:
            clauses.append("d.city = %s")
            params.append(filters.city)

        if filters.document_kinds:
            clauses.append("d.document_kind = ANY(%s)")
            params.append(filters.document_kinds)

        if filters.active_only:
            # NULL oznacza nieustalony status, więc przy active_only=True zachowujemy
            # tylko dokumenty jawnie aktywne.
            clauses.append("d.is_active IS TRUE")

        if filters.exclude_inactive:
            # Odpada tylko dokument jawnie nieobowiązujący; NULL (status nieustalony) zostaje.
            clauses.append("d.is_active IS NOT FALSE")

        if not clauses:
            return "", params

        return " AND " + " AND ".join(clauses), params

    def vector_search(
        self,
        *,
        query_embedding: list[float],
        limit: int = 20,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Search chunks by vector similarity with optional metadata filters."""
        filters = filters or RetrievalFilters()
        filter_sql, filter_params = self._filter_sql(filters)
        emb = np.asarray(query_embedding, dtype=np.float32)

        sql = f"""
            SELECT
                c.chunk_id,
                c.document_id,
                d.title,
                d.document_kind,
                d.hierarchy_rank,
                d.legal_form,
                d.adoption_act_number,
                d.adoption_date::text,
                d.is_active,
                d.source_url,
                d.source_filename,
                c.page_start,
                c.page_end,
                c.section_path,
                c.text,
                1 - (c.embedding <=> %s) AS semantic_score,
                d.lifecycle_status
            FROM rag_chunks c
            JOIN rag_documents d USING (document_id)
            WHERE d.retrieval_enabled {filter_sql}
            ORDER BY c.embedding <=> %s
            LIMIT %s
        """

        params = [emb, *filter_params, emb, limit]

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                title=row[2],
                document_kind=row[3],
                hierarchy_rank=row[4],
                legal_form=row[5],
                adoption_act_number=row[6],
                adoption_date=row[7],
                is_active=row[8],
                source_url=row[9],
                source_filename=row[10],
                page_start=row[11],
                page_end=row[12],
                section_path=row[13] or [],
                text=row[14],
                semantic_score=float(row[15]),
                lifecycle_status=row[16],
            )
            for row in rows
        ]

    def lexical_search(
        self,
        *,
        query: str,
        limit: int = 20,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Search chunks by PostgreSQL full-text ranking with optional metadata filters."""
        filters = filters or RetrievalFilters()
        filter_sql, filter_params = self._filter_sql(filters)
        # Wcześniej słowa były łączone przez AND, więc wielowyrazowe zapytanie prawie
        # nigdy nic nie znajdowało. OR plus ranking ts_rank_cd premiuje fragmenty
        # zawierające więcej terminów.
        tsquery = lexical_tsquery(query)
        if not tsquery:
            return []

        # Konfiguracja 'simple' odpowiada search_vector z rag_ingestion i dobrze
        # zachowuje nazwy własne, numery uchwał i terminy techniczne.
        sql = f"""
            WITH q AS (
                SELECT to_tsquery('simple', %s) AS query
            )
            SELECT
                c.chunk_id,
                c.document_id,
                d.title,
                d.document_kind,
                d.hierarchy_rank,
                d.legal_form,
                d.adoption_act_number,
                d.adoption_date::text,
                d.is_active,
                d.source_url,
                d.source_filename,
                c.page_start,
                c.page_end,
                c.section_path,
                c.text,
                ts_rank_cd(c.search_vector, q.query) AS lexical_score,
                d.lifecycle_status
            FROM rag_chunks c
            JOIN rag_documents d USING (document_id)
            CROSS JOIN q
            WHERE c.search_vector @@ q.query AND d.retrieval_enabled {filter_sql}
            ORDER BY lexical_score DESC
            LIMIT %s
        """

        params = [tsquery, *filter_params, limit]

        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                title=row[2],
                document_kind=row[3],
                hierarchy_rank=row[4],
                legal_form=row[5],
                adoption_act_number=row[6],
                adoption_date=row[7],
                is_active=row[8],
                source_url=row[9],
                source_filename=row[10],
                page_start=row[11],
                page_end=row[12],
                section_path=row[13] or [],
                text=row[14],
                lexical_score=float(row[15]),
                lifecycle_status=row[16],
            )
            for row in rows
        ]

    def counts(self) -> tuple[int, int]:
        """Return the number of documents and chunks available for retrieval."""
        with self.connect() as conn:
            docs = conn.execute("SELECT count(*) FROM rag_documents").fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM rag_chunks").fetchone()[0]
        return docs, chunks
