from __future__ import annotations

from typing import Any

import numpy as np

from .models import ChunkRecord, DocumentMetadata

# Błędny adres bazy kończy się błędem po tym czasie zamiast wielominutowego zawieszenia.
CONNECT_TIMEOUT_SECONDS = 10


class PgVectorRepository:
    def __init__(self, database_url: str, *, embedding_dim: int = 1024) -> None:
        """Initialize the PostgreSQL vector repository configuration."""
        self.database_url = database_url
        self.embedding_dim = embedding_dim

    def connect(self) -> Any:
        """Open a PostgreSQL connection and register pgvector support."""
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError(
                "Nie można załadować sterownika PostgreSQL. "
                "Sprawdź, czy psycopg jest zainstalowany i czy biblioteka libpq.dll "
                "jest dostępna w PATH."
            ) from exc

        conn = psycopg.connect(self.database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        return conn

    def ensure_schema(self) -> None:
        """Create the document, chunk, and search indexes required by the RAG store."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_documents (
                        document_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        document_kind TEXT NOT NULL,
                        hierarchy_rank INTEGER NOT NULL,
                        city TEXT NOT NULL,
                        issuer TEXT,
                        adopting_body TEXT,
                        legal_form TEXT NOT NULL,
                        adoption_act_number TEXT,
                        adoption_date DATE,
                        valid_from DATE,
                        valid_to DATE,
                        is_active BOOLEAN,
                        source_url TEXT,
                        source_filename TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        language TEXT NOT NULL DEFAULT 'pl',
                        tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                        relations JSONB NOT NULL DEFAULT '[]'::jsonb,
                        notes TEXT,
                        ingestion_version TEXT NOT NULL,
                        ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                # Kolumny dodane po pierwszej wersji schematu. ALTER uzupełnia też bazy
                # utworzone wcześniej, a istniejące dokumenty pozostają włączone do wyszukiwania.
                cur.execute(
                    """
                    ALTER TABLE rag_documents
                        ADD COLUMN IF NOT EXISTS lifecycle_status TEXT,
                        ADD COLUMN IF NOT EXISTS retrieval_enabled BOOLEAN NOT NULL DEFAULT TRUE,
                        ADD COLUMN IF NOT EXISTS metadata_url TEXT,
                        ADD COLUMN IF NOT EXISTS is_annex BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        document_id TEXT NOT NULL
                            REFERENCES rag_documents(document_id)
                            ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        page_start INTEGER NOT NULL,
                        page_end INTEGER NOT NULL,
                        section_path JSONB NOT NULL DEFAULT '[]'::jsonb,
                        text TEXT NOT NULL,
                        embedding_text TEXT NOT NULL,
                        text_sha256 TEXT NOT NULL,
                        hierarchy_rank INTEGER NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.embedding_dim}) NOT NULL,
                        search_vector tsvector
                            GENERATED ALWAYS AS (
                                to_tsvector('simple', coalesce(text, ''))
                            ) STORED,
                        UNIQUE(document_id, chunk_index)
                    )
                    """
                )

                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rag_chunks_document
                    ON rag_chunks(document_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rag_chunks_fts
                    ON rag_chunks USING GIN(search_vector)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rag_chunks_metadata
                    ON rag_chunks USING GIN(metadata)
                    """
                )

                # HNSW wymaga pgvector wspierającego ten typ indeksu.
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding_hnsw
                    ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                    """
                )
            conn.commit()

    def document_hash(self, document_id: str) -> str | None:
        """Return the stored content hash for a document or None when it is absent."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT content_sha256 FROM rag_documents WHERE document_id = %s",
                (document_id,),
            ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self,
        metadata: DocumentMetadata,
        *,
        source_filename: str,
        content_sha256: str,
        ingestion_version: str,
    ) -> None:
        """Insert or update document metadata in the RAG document table."""
        with self.connect() as conn:
            self._upsert_document(
                conn,
                metadata,
                source_filename=source_filename,
                content_sha256=content_sha256,
                ingestion_version=ingestion_version,
            )
            conn.commit()

    def store_document(
        self,
        metadata: DocumentMetadata,
        *,
        source_filename: str,
        content_sha256: str,
        ingestion_version: str,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        """Write a processed document and all its chunks in one transaction."""
        # Osobne transakcje zostawiały po błędzie zapisu fragmentów nowy hash i wersję przy
        # starych fragmentach, a kolejne uruchomienie uznawało dokument za aktualny.
        rows = self._chunk_rows(metadata.hierarchy_rank, chunks, embeddings)
        with self.connect() as conn:
            self._upsert_document(
                conn,
                metadata,
                source_filename=source_filename,
                content_sha256=content_sha256,
                ingestion_version=ingestion_version,
            )
            self._replace_chunk_rows(conn, metadata.document_id, rows)
            conn.commit()

    @staticmethod
    def _upsert_document(
        conn: Any,
        metadata: DocumentMetadata,
        *,
        source_filename: str,
        content_sha256: str,
        ingestion_version: str,
    ) -> None:
        """Execute the document upsert on an open connection without committing."""
        from psycopg.types.json import Jsonb

        conn.execute(
            """
            INSERT INTO rag_documents (
                document_id, title, document_kind, hierarchy_rank,
                city, issuer, adopting_body, legal_form,
                adoption_act_number, adoption_date,
                valid_from, valid_to, is_active,
                source_url, source_filename, content_sha256,
                language, tags, relations, notes, ingestion_version,
                lifecycle_status, retrieval_enabled, metadata_url, is_annex,
                ingested_at
            )
            VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                %s,%s,%s,%s,now()
            )
            ON CONFLICT (document_id) DO UPDATE SET
                title = EXCLUDED.title,
                document_kind = EXCLUDED.document_kind,
                hierarchy_rank = EXCLUDED.hierarchy_rank,
                city = EXCLUDED.city,
                issuer = EXCLUDED.issuer,
                adopting_body = EXCLUDED.adopting_body,
                legal_form = EXCLUDED.legal_form,
                adoption_act_number = EXCLUDED.adoption_act_number,
                adoption_date = EXCLUDED.adoption_date,
                valid_from = EXCLUDED.valid_from,
                valid_to = EXCLUDED.valid_to,
                is_active = EXCLUDED.is_active,
                source_url = EXCLUDED.source_url,
                source_filename = EXCLUDED.source_filename,
                content_sha256 = EXCLUDED.content_sha256,
                language = EXCLUDED.language,
                tags = EXCLUDED.tags,
                relations = EXCLUDED.relations,
                notes = EXCLUDED.notes,
                ingestion_version = EXCLUDED.ingestion_version,
                lifecycle_status = EXCLUDED.lifecycle_status,
                retrieval_enabled = EXCLUDED.retrieval_enabled,
                metadata_url = EXCLUDED.metadata_url,
                is_annex = EXCLUDED.is_annex,
                ingested_at = now()
            """,
            (
                metadata.document_id,
                metadata.title,
                metadata.document_kind.value,
                metadata.hierarchy_rank,
                metadata.city,
                metadata.issuer,
                metadata.adopting_body,
                metadata.legal_form.value,
                metadata.adoption_act_number,
                metadata.adoption_date,
                metadata.valid_from,
                metadata.valid_to,
                metadata.is_active,
                metadata.source_url,
                source_filename,
                content_sha256,
                metadata.language,
                Jsonb(metadata.tags),
                Jsonb([x.model_dump(mode="json") for x in metadata.relations]),
                metadata.notes,
                ingestion_version,
                metadata.lifecycle_status.value if metadata.lifecycle_status else None,
                metadata.retrieval_enabled,
                metadata.metadata_url,
                metadata.is_annex,
            ),
        )

    def ingestion_version(self, document_id: str) -> str | None:
        """Return the processing version a document was stored with, or None when it is absent."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT ingestion_version FROM rag_documents WHERE document_id = %s",
                (document_id,),
            ).fetchone()
        return row[0] if row else None

    def embedding_metadata(self, document_id: str) -> tuple[str, str, str] | None:
        """Return the stored title, kind and city, which are part of every chunk's embedding."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT title, document_kind, city FROM rag_documents WHERE document_id = %s",
                (document_id,),
            ).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def refresh_chunk_metadata(
        self,
        *,
        document_id: str,
        hierarchy_rank: int,
        metadata: dict[str, Any],
    ) -> None:
        """Rewrite the document-level metadata copied into every chunk of one document."""
        from psycopg.types.json import Jsonb

        with self.connect() as conn:
            conn.execute(
                "UPDATE rag_chunks SET metadata = %s, hierarchy_rank = %s WHERE document_id = %s",
                (Jsonb(metadata), hierarchy_rank, document_id),
            )
            conn.commit()

    def replace_chunks(
        self,
        *,
        document_id: str,
        hierarchy_rank: int,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        """Replace all stored chunks for a document after validating embedding dimensions."""
        rows = self._chunk_rows(hierarchy_rank, chunks, embeddings)
        with self.connect() as conn:
            self._replace_chunk_rows(conn, document_id, rows)
            conn.commit()

    def _chunk_rows(
        self,
        hierarchy_rank: int,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> list[tuple[Any, ...]]:
        """Validate the embeddings and build the insert rows before anything is deleted."""
        from psycopg.types.json import Jsonb

        if len(chunks) != len(embeddings):
            raise ValueError("Liczba chunków i embeddingów różni się.")

        rows = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            arr = np.asarray(emb, dtype=np.float32)
            if arr.shape != (self.embedding_dim,):
                raise ValueError(
                    f"Embedding ma wymiar {arr.shape}, oczekiwano ({self.embedding_dim},)"
                )

            rows.append(
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.page_start,
                    chunk.page_end,
                    Jsonb(chunk.section_path),
                    chunk.text,
                    chunk.embedding_text,
                    chunk.text_sha256,
                    hierarchy_rank,
                    Jsonb(chunk.metadata),
                    arr,
                )
            )
        return rows

    @staticmethod
    def _replace_chunk_rows(conn: Any, document_id: str, rows: list[tuple[Any, ...]]) -> None:
        """Delete a document's chunks and insert the new rows on an open connection."""
        conn.execute(
            "DELETE FROM rag_chunks WHERE document_id = %s",
            (document_id,),
        )
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_chunks (
                    chunk_id, document_id, chunk_index,
                    page_start, page_end, section_path,
                    text, embedding_text, text_sha256,
                    hierarchy_rank, metadata, embedding
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                rows,
            )

    def counts(self) -> tuple[int, int]:
        """Return the number of stored documents and chunks."""
        with self.connect() as conn:
            docs = conn.execute("SELECT count(*) FROM rag_documents").fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM rag_chunks").fetchone()[0]
        return docs, chunks
