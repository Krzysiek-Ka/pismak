from __future__ import annotations

from typing import Any

import numpy as np

from .models import RetrievalFilters, RetrievedChunk


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
        except ImportError as exc:
            raise RuntimeError(
                "Nie można załadować sterownika PostgreSQL. "
                "Sprawdź, czy psycopg jest zainstalowany i czy biblioteka libpq.dll "
                "jest dostępna w PATH. Po zmianie PATH uruchom ponownie kernel."
            ) from exc

        conn = psycopg.connect(self.database_url)
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
                1 - (c.embedding <=> %s) AS semantic_score
            FROM rag_chunks c
            JOIN rag_documents d USING (document_id)
            WHERE 1=1 {filter_sql}
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

        # Konfiguracja 'simple' odpowiada search_vector z rag_ingestion i dobrze
        # zachowuje nazwy własne, numery uchwał i terminy techniczne.
        sql = f"""
            WITH q AS (
                SELECT websearch_to_tsquery('simple', %s) AS query
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
                ts_rank_cd(c.search_vector, q.query) AS lexical_score
            FROM rag_chunks c
            JOIN rag_documents d USING (document_id)
            CROSS JOIN q
            WHERE c.search_vector @@ q.query {filter_sql}
            ORDER BY lexical_score DESC
            LIMIT %s
        """

        params = [query, *filter_params, limit]

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
            )
            for row in rows
        ]

    def counts(self) -> tuple[int, int]:
        """Return the number of documents and chunks available for retrieval."""
        with self.connect() as conn:
            docs = conn.execute("SELECT count(*) FROM rag_documents").fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM rag_chunks").fetchone()[0]
        return docs, chunks
