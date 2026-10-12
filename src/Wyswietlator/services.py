from __future__ import annotations

import json
import logging
import os
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from src._shared.context_budget import LLMInputLimitError, input_char_budget
from src._shared.model_context import detect_context_tokens
from src._shared.serwer import (
    OLLAMA_REQUEST_TIMEOUT_SECONDS,
    LLMConfigurationError,
    LLMOutputLimitError,
    LLMResponseError,
    set_up_client,
    structured_output_error_detail,
)
from src.Dopytywator import prompts as dopytywator_prompts
from src.Dopytywator.constants import DOPYTYWATOR_RESERVED_OUTPUT_TOKENS
from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator_helper_functions
from src.Pismogenerator.pismogenerator import DraftValidationError
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
from src.Zrodlowyszukiwator.rag_retrieval.repository import DatabaseConnectionError

MAX_USER_MESSAGE_CHARS = 5000
MAX_INTERVIEW_QUESTIONS = 12
LOGGER = logging.getLogger("pismak")
_CASE_LOG_HANDLERS: dict[str, logging.FileHandler] = {}
_CASE_LOG_HANDLERS_LOCK = threading.Lock()


class _SessionLogFilter(logging.Filter):
    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "session_id", None) == self.session_id


def configure_logging(project_root: Path, session_id: str) -> logging.Logger:
    """Create one private log file per case and route only that case's records to it."""
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    with _CASE_LOG_HANDLERS_LOCK:
        if session_id not in _CASE_LOG_HANDLERS:
            log_dir = (project_root / "logs").resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            handler = logging.FileHandler(
                log_dir / f"pismak_{timestamp}.log",
                encoding="utf-8",
            )
            handler.addFilter(_SessionLogFilter(session_id))
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            LOGGER.addHandler(handler)
            _CASE_LOG_HANDLERS[session_id] = handler

    return LOGGER


def close_case_logging(session_id: str) -> None:
    """Close and remove a case's file handler after that case has ended."""
    with _CASE_LOG_HANDLERS_LOCK:
        handler = _CASE_LOG_HANDLERS.pop(session_id, None)
        if handler is not None:
            LOGGER.removeHandler(handler)
            handler.close()


def conversation_char_limit(provider: str, context_tokens: int | None = None) -> int:
    """Return how many conversation characters fit in the interview and brief prompts."""
    interview_chars = len(dopytywator_prompts.DOPYTYWATOR_INTERVIEW_SYSTEM_PROMPT) + len(
        dopytywator_prompts.interviewer_user_prompt(
            [],
            question_number=MAX_INTERVIEW_QUESTIONS,
            max_interview_questions=MAX_INTERVIEW_QUESTIONS,
        )
    )
    brief_chars = len(dopytywator_prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT) + len(
        dopytywator_prompts.brief_user_prompt([])
    )
    budget = input_char_budget(
        provider,
        reserved_output_tokens=DOPYTYWATOR_RESERVED_OUTPUT_TOKENS,
        context_tokens=context_tokens,
    )
    return budget - max(interview_chars, brief_chars)


def validate_user_message(
    message: str,
    *,
    conversation_chars: int = 0,
    provider: str | None = None,
    context_tokens: int | None = None,
) -> None:
    """Reject oversized user input before adding it to the conversation state."""
    if len(message) > MAX_USER_MESSAGE_CHARS:
        raise ValueError(f"Wiadomość jest zbyt długa (limit: {MAX_USER_MESSAGE_CHARS} znaków).")
    if provider is None:
        return
    # Cała rozmowa trafia do promptu, więc limit zależy od okna kontekstu modelu.
    limit = conversation_char_limit(provider, context_tokens)
    remaining = max(0, limit - conversation_chars)
    if len(message) > remaining:
        raise ValueError(
            "Rozmowa jest zbyt długa dla wybranego modelu "
            f"(pozostało miejsca na {remaining} znaków). "
            "Skróć wiadomość albo rozpocznij nową sprawę."
        )


def get_openai_client(ollama_url: str | None = None) -> OpenAI:
    """Create an OpenAI-compatible client configured for the local Ollama endpoint."""
    base = (ollama_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
    return OpenAI(
        base_url=base + "/v1/",
        api_key="ollama",
        timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


def get_llm_client(provider: str) -> OpenAI:
    """Create a client for the explicitly selected provider."""
    if provider == "ollama":
        return get_openai_client()
    if provider == "openrouter":
        return set_up_client("openrouter")
    raise ValueError(f"Nieznany dostawca modelu: {provider}")


def llm_error_message(error: Exception) -> str:
    """Return a safe, actionable message without exposing request details."""
    if isinstance(error, DraftValidationError):
        return str(error)
    if isinstance(error, DatabaseConnectionError):
        return str(error)
    if isinstance(error, LLMConfigurationError):
        return str(error)
    if isinstance(error, (AuthenticationError, PermissionDeniedError)):
        return "OpenRouter odrzucił klucz API. Sprawdź OPENROUTER_KEY w pliku .env."
    if isinstance(error, RateLimitError):
        return "OpenRouter odrzucił zapytanie z powodu limitu. Spróbuj później."
    if isinstance(error, APITimeoutError):
        return "Przekroczono czas oczekiwania na odpowiedź modelu. Spróbuj ponownie."
    if isinstance(error, APIConnectionError):
        return "Nie udało się połączyć z usługą modelu. Sprawdź połączenie internetowe."
    if isinstance(error, BadRequestError):
        return "Wybrany model nie obsłużył wymaganej odpowiedzi strukturalnej."
    if isinstance(error, APIStatusError) and error.status_code >= 500:
        return "Usługa modelu jest chwilowo niedostępna. Spróbuj później."
    if isinstance(error, LLMInputLimitError):
        window = f" ({error.context_tokens} tokenów)" if error.context_tokens else ""
        return (
            f"Dane sprawy nie mieszczą się w oknie kontekstu wybranego modelu{window}. "
            "Opisz sprawę krócej albo wybierz model z większym oknem."
        )
    if isinstance(error, LLMOutputLimitError):
        return (
            "Odpowiedź modelu została ucięta. Aplikacja ponowiła generowanie z prośbą o "
            "krótszy wynik, ale nie udało się odzyskać kompletnej struktury."
        )
    if isinstance(error, LLMResponseError):
        return f"Model nie zwrócił poprawnych danych. {structured_output_error_detail(error)}"
    return "Wystąpił błąd podczas generowania odpowiedzi. Sprawdź log aplikacji."


def error_location(error: Exception) -> str | None:
    """Return the innermost project source location recorded on an exception."""
    project_root = Path(__file__).resolve().parents[2]
    for frame in reversed(traceback.extract_tb(error.__traceback__)):
        try:
            relative_path = Path(frame.filename).resolve().relative_to(project_root)
        except ValueError:
            continue
        # Środowisko wirtualne leży w katalogu projektu, ale to nie jest kod projektu.
        if "site-packages" in relative_path.parts:
            continue
        return f"{relative_path.as_posix()}:{frame.lineno} in {frame.name}"
    return None


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
    provider: str = "ollama",
) -> tuple[BriefToEvidencePipeline, int, int]:
    """Assemble the existing RAG planner, retriever, reranker, and evidence pipeline."""
    if provider not in ("ollama", "openrouter"):
        raise ValueError(f"Nieznany dostawca modelu: {provider}")
    allow_fallback = provider != "openrouter"

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

    # Po sprawdzeniu bazy, bo ustalenie okna może wymagać załadowania modelu.
    context_tokens = detect_context_tokens(
        provider,
        model,
        on_wait=on_wait,
        session_id=session_id,
    )
    planner = LLMQueryPlanner(
        client=client,
        model=model,
        on_wait=on_wait,
        session_id=session_id,
        allow_fallback=allow_fallback,
        provider=provider,
        context_tokens=context_tokens,
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
        allow_fallback=allow_fallback,
        provider=provider,
        context_tokens=context_tokens,
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
    draft_with_error: DraftLetter | None = None,
    draft_validation_errors: list[str] | None = None,
    letter_markdown: str | None = None,
    model_usage: dict[str, list[dict[str, str]]] | None = None,
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

    if draft_with_error is not None:
        (run_dir / "draft_with_error.json").write_text(
            json.dumps(
                {
                    "draft": draft_with_error.model_dump(mode="json"),
                    "validation_errors": draft_validation_errors or [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if letter_markdown is not None:
        (run_dir / "letter.md").write_text(
            letter_markdown,
            encoding="utf-8",
        )

    if model_usage is not None:
        (run_dir / "models.json").write_text(
            json.dumps(model_usage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return run_dir
