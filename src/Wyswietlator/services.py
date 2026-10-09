from __future__ import annotations

import json
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator_helper_functions
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.brief_to_evidence import (
    BriefToEvidencePipeline,
    EvidencePack,
    LLMQueryPlanner,
)
from src.Zrodlowyszukiwator.rag_ingestion.embedder import OllamaEmbedder
from src.Zrodlowyszukiwator.rag_retrieval import (
    HybridRetriever,
    LLMReranker,
    RetrievalRepository,
)

MAX_USER_MESSAGE_CHARS = 5000
MAX_INTERVIEW_QUESTIONS = 12
LOGGER = logging.getLogger("pismak")


def configure_logging(project_root: Path) -> logging.Logger:
    """Configure rotating local logs without recording user-provided content."""
    log_path = (project_root / "logs" / "pismak.log").resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in LOGGER.handlers
    ):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(handler)

    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    return LOGGER


def validate_user_message(message: str) -> None:
    """Reject oversized user input before adding it to the conversation state."""
    if len(message) > MAX_USER_MESSAGE_CHARS:
        raise ValueError(f"Wiadomość jest zbyt długa (limit: {MAX_USER_MESSAGE_CHARS} znaków).")


def get_openai_client(ollama_url: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client configured for the local Ollama endpoint."""
    base = (ollama_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
    return OpenAI(
        base_url=base + "/v1/",
        api_key="ollama",
        timeout=300.0,
    )


def list_generation_models(client: OpenAI) -> list[str]:
    """Return available model names excluding embedding-only models."""
    models = sorted(m.id for m in client.models.list().data)

    # Modele embeddingowe nie powinny być wybierane jako LLM.
    blocked = ("bge-m3", "nomic-embed", "embed", "embedding")
    return [m for m in models if not any(token in m.lower() for token in blocked)]


def build_zrodlowyszukiwator(
    *,
    client: OpenAI,
    model: str,
    on_wait: Any = None,
    session_id: str | None = None,
) -> tuple[BriefToEvidencePipeline, int, int]:
    """Assemble the existing RAG planner, retriever, reranker, and evidence pipeline."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Brak DATABASE_URL w .env.")

    repo = RetrievalRepository(database_url)
    docs, chunks = repo.counts()
    if chunks == 0:
        raise RuntimeError("Baza RAG jest pusta. Najpierw wykonaj rag_ingestion.")

    embedder = OllamaEmbedder(
        model=os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"),
        base_url=os.getenv(
            "OLLAMA_URL",
            "http://127.0.0.1:11434",
        ),
    )

    planner = LLMQueryPlanner(
        client=client,
        model=model,
        on_wait=on_wait,
        session_id=session_id,
    )
    retriever = HybridRetriever(
        repository=repo,
        embedder=embedder,
        vector_limit=20,
        lexical_limit=20,
        session_id=session_id,
    )
    reranker = LLMReranker(
        client=client,
        model=model,
        max_selected=6,
        on_wait=on_wait,
        session_id=session_id,
    )

    pipeline = BriefToEvidencePipeline(
        planner=planner,
        retriever=retriever,
        reranker=reranker,
        candidate_k=20,
        evidence_k=6,
        max_per_document=3,
    )
    return pipeline, docs, chunks


def render_letter(
    draft: DraftLetter,
    evidence_pack: EvidencePack,
    user_evidence: list[str] | None = None,
) -> str:
    """Render a draft using the evidence and user materials that produced it."""
    normalized_user_evidence = [
        evidence.strip() for evidence in user_evidence or [] if evidence.strip()
    ]
    return pismogenerator_helper_functions.render_letter(
        draft=draft,
        evidence_pack=evidence_pack,
        user_evidence=normalized_user_evidence,
    )


def save_run(
    *,
    project_root: Path,
    messages: list[dict[str, str]],
    case_brief: CaseBrief | None = None,
    evidence_pack: Any | None = None,
    draft: DraftLetter | None = None,
    letter_markdown: str | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Persist current UI artifacts to a uniquely named local output directory."""
    from datetime import datetime

    if run_dir is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = project_root / "outputs" / f"{run_id}_ui"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "conversation.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if case_brief is not None:
        (run_dir / "brief.json").write_text(
            case_brief.model_dump_json(indent=2),
            encoding="utf-8",
        )

    if evidence_pack is not None:
        (run_dir / "evidence.json").write_text(
            evidence_pack.model_dump_json(indent=2),
            encoding="utf-8",
        )

    if draft is not None:
        (run_dir / "draft.json").write_text(
            draft.model_dump_json(indent=2),
            encoding="utf-8",
        )

    if letter_markdown is not None:
        (run_dir / "letter.md").write_text(
            letter_markdown,
            encoding="utf-8",
        )

    return run_dir
