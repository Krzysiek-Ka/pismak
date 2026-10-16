from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .catalog import load_catalog
from .embedder import OllamaEmbedder
from .pipeline import ingest_pdf
from .repository import PgVectorRepository

# Katalog metadanych dołączony do repozytorium; działa niezależnie od katalogu roboczego.
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "RAG" / "document_catalog.json"


def main() -> None:
    """Parse ingestion arguments and load one PDF into the configured RAG database."""
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument(
        "--catalog",
        default=str(DEFAULT_CATALOG_PATH),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    path = Path(args.file)
    catalog = load_catalog(args.catalog)

    if path.name not in catalog:
        raise SystemExit(f"Brak metadanych dla {path.name!r} w katalogu {args.catalog}")

    # Ingestion zapisuje do bazy, więc używa własnej roli; DATABASE_URL aplikacji czyta tylko.
    database_url = os.getenv("RAG_INGESTION_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit(
            "Brak RAG_INGESTION_DATABASE_URL w .env. Ingestion łączy się jako rola "
            "pismak_ingest; rola z DATABASE_URL ma tylko prawo odczytu (README, punkt 5.2)."
        )
    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "bge-m3")
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    embedding_dim = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))

    repo = PgVectorRepository(
        database_url,
        embedding_dim=embedding_dim,
    )
    repo.ensure_schema()

    embedder = OllamaEmbedder(
        model=embedding_model,
        base_url=ollama_url,
    )

    result = ingest_pdf(
        path,
        catalog[path.name],
        repository=repo,
        embedder=embedder,
        force=args.force,
    )

    print(result)


if __name__ == "__main__":
    main()
