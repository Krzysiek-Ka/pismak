"""
Testy E2E Pismaka: sterownik przebiegu, kontrole, analiza logu i raport.

Sterownik przechodzi tę samą ścieżkę co aplikacja (src/Wyswietlator/app.py), tylko bez UI:
wywiad przez dopytywator.interview_step, źródła przez services.build_zrodlowyszukiwator,
pismo przez pismogenerator.build_draft. Nic nie jest zastępowane atrapą: modele, baza,
embedder i konfiguracja z .env są prawdziwe.

Kontrole „hard” (H1–H7) są niezmiennikami i decydują o wyniku testu. Kontrole „soft”
(S1–S9) to heurystyczna ocena jakości: trafiają do raportu jako WARN do ręcznego przeglądu.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import platform
import re
import subprocess
import unicodedata
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import pytest
from e2e_cases import Case, Provider, Setup

from src._shared.model_context import detect_context_tokens
from src._shared.serwer import OPENROUTER_KEY_PLACEHOLDER, LLMClient, error_log_detail
from src.Dopytywator import dopytywator
from src.Dopytywator.dopytywator_helper_functions import validate_brief_readiness
from src.Dopytywator.structures import CaseBrief, ClaimStatus, ConversationMessage
from src.Pismogenerator import pismogenerator
from src.Pismogenerator.pismogenerator import DraftValidationError
from src.Pismogenerator.structures import DraftLetter
from src.Wyswietlator import services
from src.Zrodlowyszukiwator.brief_to_evidence import EvidencePack
from src.Zrodlowyszukiwator.brief_to_evidence.planner import LLMQueryPlanner
from src.Zrodlowyszukiwator.rag_retrieval import RetrievalRepository

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "e2e"
MAX_QUESTIONS = 12
# Ollama przerywa żądanie po 900 s (OLLAMA_REQUEST_TIMEOUT_SECONDS); wolniejsze wywołania
# są blisko limitu i warto je widzieć w raporcie.
SLOW_CALL_SECONDS = 600.0
# Konfiguracja produkcyjna EvidencePack (services.build_zrodlowyszukiwator).
EVIDENCE_K = 6
MAX_PER_DOCUMENT = 3

HARD, SOFT = "hard", "soft"
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
PASSED, FAILED, NOT_RUN = "ZALICZONY", "NIEUDANY", "NIEURUCHOMIONY"
REDACTED = "***REDACTED***"
EVIDENCE_STATUSES = ("supported", "unsupported", "unverified")


# ---------------------------------------------------------------------------
# Tekst


def normalize(text: str) -> str:
    """Lowercase the text and drop Polish diacritics so keyword checks ignore them."""
    text = text.replace("ł", "l").replace("Ł", "L")
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(plain.lower().split())


def contains_any(text: str, variants: Iterable[str]) -> bool:
    """Return whether the text contains any of the variants, ignoring case and diacritics."""
    normalized = normalize(text)
    return any(normalize(variant) in normalized for variant in variants)


def missing_key_info(text: str, key_info: dict[str, tuple[str, ...]]) -> list[str]:
    """Return the names of key facts none of whose variants occur in the text."""
    return [name for name, variants in key_info.items() if not contains_any(text, variants)]


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Przypisy [1] i identyfikatory dowodów E1/U1 nie są faktami podanymi w treści.
_REFERENCES = re.compile(r"\[\d+\]|\b[EU]\d+\b")
# Numeracja listy („1. ”) też nie jest faktem; modele piszą czasem dosłowne „\n” zamiast
# nowego wiersza, więc oba warianty początku wiersza są rozpoznawane.
_LIST_MARKERS = re.compile(r"(?m)(?:^|\\n)[ \t*]*\d+[.)](?=\s)")


def _without_references(text: str) -> str:
    return _REFERENCES.sub(" ", _LIST_MARKERS.sub(" ", text))


def unknown_numbers(text: str, known_text: str) -> list[str]:
    """Return numbers in the text that occur nowhere in the known text."""
    known = set(_NUMBER.findall(_without_references(known_text)))
    found = _NUMBER.findall(_without_references(text))
    return sorted({number for number in found if number not in known})


def words_outside(name: str, source: str) -> list[str]:
    """Return words of a name (4+ letters) whose 5-letter stem does not occur in the source."""
    normalized = normalize(source)
    return [word for word in re.findall(r"\w{4,}", normalize(name)) if word[:5] not in normalized]


def repeated_questions(questions: list[str], threshold: float = 0.9) -> list[tuple[int, int]]:
    """Return 1-based pairs of questions that are near-identical."""
    pairs = []
    normalized = [normalize(question) for question in questions]
    for later, text in enumerate(normalized):
        for earlier in range(later):
            if difflib.SequenceMatcher(None, normalized[earlier], text).ratio() >= threshold:
                pairs.append((earlier + 1, later + 1))
    return pairs


def text_values(value: Any) -> list[str]:
    """Collect every string inside a JSON-like value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in text_values(item)]
    if isinstance(value, list | tuple):
        return [text for item in value for text in text_values(item)]
    return []


# Wzorce podstaw prawnych szersze niż zwroty blokowane przez walidację Pismogeneratora.
LEGAL_PATTERNS = {
    "§": r"§",
    "Dz. U.": r"\bdz\.?\s*u\b",
    "ustawa z dnia": r"\bustaw\w*\s+z\s+dnia",
    "rozporządzenie": r"rozporzadzen",
    "uchwała nr": r"\buchwal\w*\s+nr",
    "art.": r"\bart\.\s*\d",
}


def legal_basis_outside(text: str, allowed_text: str) -> list[str]:
    """Return legal-basis patterns found in the text but not in the allowed sources."""
    normalized, allowed = normalize(text), normalize(allowed_text)
    return [
        label
        for label, pattern in LEGAL_PATTERNS.items()
        if re.search(pattern, normalized) and not re.search(pattern, allowed)
    ]


# ---------------------------------------------------------------------------
# Wyniki


@dataclass
class Check:
    id: str
    title: str
    severity: str
    status: str
    detail: str = ""


@dataclass
class Turn:
    number: int
    question: str
    reason_for_question: str | None
    missing_topics: list[str]
    short_summary: str
    answer: str
    model: str | None
    elapsed_seconds: float


@dataclass
class PhaseError:
    phase: str
    error_type: str
    user_message: str
    location: str | None
    detail: str


@dataclass
class RunResult:
    run_id: str
    setup_id: str
    setup_label: str
    # Etap → skonfigurowany dostawca i model, jak w panelu konfiguracji aplikacji.
    stages: dict[str, dict[str, str]]
    case_id: str
    case_title: str
    status: str = FAILED
    not_run_reason: str | None = None
    session_id: str | None = None
    started_at: str = ""
    context_tokens: dict[str, int | None] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    questions_asked: int = 0
    answers_used: int = 0
    interview_calls: int = 0
    end_reason: str | None = None
    final_turn: dict[str, Any] | None = None
    brief: dict[str, Any] | None = None
    brief_readiness_errors: list[str] = field(default_factory=list)
    rag: dict[str, Any] | None = None
    draft: dict[str, Any] | None = None
    draft_validation_errors: list[str] = field(default_factory=list)
    letter_chars: int | None = None
    model_usage: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    log_stats: dict[str, Any] = field(default_factory=dict)
    errors: list[PhaseError] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    reproduce: str = ""

    def hard_failures(self) -> list[Check]:
        """Return the failed invariants that make the run fail."""
        return [c for c in self.checks if c.severity == HARD and c.status == FAIL]

    def warnings(self) -> list[Check]:
        """Return the quality findings left for manual review."""
        return [c for c in self.checks if c.status == WARN]

    def actual_models(self) -> list[str]:
        """Return the distinct models that answered, in order of first use."""
        models = [call["model"] for calls in self.model_usage.values() for call in calls]
        return list(dict.fromkeys(models))


def run_id(setup: Setup, case: Case) -> str:
    """Return the identifier of one setup/case run, also used as the pytest id."""
    return f"{setup.id}-{case.id}"


def new_result(setup: Setup, case: Case) -> RunResult:
    """Create an empty result describing one setup/case combination."""
    rid = run_id(setup, case)
    return RunResult(
        run_id=rid,
        setup_id=setup.id,
        setup_label=setup.label,
        stages={
            stage: {"name": provider.id, "provider": provider.provider, "model": provider.model}
            for stage, provider in setup.stages
        },
        case_id=case.id,
        case_title=case.title,
        started_at=datetime.now().isoformat(timespec="seconds"),
        # Pełny identyfikator testu, bo `-k bielik-buspas` wybrałby też konfiguracje mieszane.
        reproduce=(
            '$env:PISMAK_E2E="1"; .venv\\Scripts\\python.exe -m pytest '
            f'"tests/e2e/test_pismak_e2e.py::test_pismak_end_to_end[{rid}]" -v -rA -s'
        ),
    )


# ---------------------------------------------------------------------------
# Przebieg


@dataclass
class _RunState:
    database_url: str
    messages: list[ConversationMessage] = field(default_factory=list)
    brief: CaseBrief | None = None
    evidence_pack: EvidencePack | None = None
    draft: DraftLetter | None = None
    rejected_draft: DraftLetter | None = None
    letter: str | None = None


def detach_log_handler(project_root: Path) -> None:
    """Remove the handler that configure_logging attached for a run directory."""
    handler = services._LOG_HANDLERS.pop((project_root / "logs").resolve(), None)
    if handler is not None:
        services.LOGGER.removeHandler(handler)
        logging.getLogger(services.UNCAUGHT_ERROR_LOGGER_NAME).removeHandler(handler)


def _record_model(result: RunResult, stage: str, provider: Provider, model: str | None) -> None:
    """Keep the model that answered a stage, in the format of the app's models.json."""
    result.model_usage.setdefault(stage, []).append(
        {"provider": provider.provider, "model": model or provider.model}
    )


def _stage_client(provider: Provider) -> Any:
    """Create the client of one stage the way the app does for the provider selected there."""
    return services.get_llm_client(provider.provider)


def _context(result: RunResult, stage: str, provider: Provider) -> int | None:
    """Detect and remember the context window the stage's model is served with."""
    tokens = detect_context_tokens(provider.provider, provider.model, session_id=result.session_id)
    result.context_tokens[stage] = tokens
    return tokens


def _interview(setup: Setup, case: Case, result: RunResult, state: _RunState) -> None:
    """Run the interview with the case's answers in order until the brief is built."""
    provider = setup.dopytywator
    context_tokens = _context(result, "Dopytywator", provider)
    llm = LLMClient(
        client=_stage_client(provider),
        model_name=provider.model,
        stage="dopytywator",
        session_id=result.session_id,
        provider=provider.provider,
        context_tokens=context_tokens,
    )

    def add_user_message(message: str) -> None:
        # Ta sama kontrola co w aplikacji: cała rozmowa musi zmieścić się w oknie modelu.
        services.validate_user_message(
            message,
            conversation_chars=sum(len(m.content) for m in state.messages),
            provider=provider.provider,
            context_tokens=context_tokens,
        )
        state.messages.append(ConversationMessage(role="user", content=message))

    add_user_message(case.description)
    while True:
        step_started = perf_counter()
        step = dopytywator.interview_step(
            state.messages,
            llm,
            questions_asked=result.questions_asked,
            max_interview_questions=MAX_QUESTIONS,
            after_llm_call=lambda: _record_model(result, "Dopytywator", provider, llm.last_model),
        )
        elapsed = round(perf_counter() - step_started, 3)
        if step.turn is not None:
            result.interview_calls += 1

        if step.case_brief is not None:
            state.brief = step.case_brief
            result.end_reason = step.end_reason
            result.final_turn = step.turn.model_dump(mode="json") if step.turn else None
            result.brief = step.case_brief.model_dump(mode="json")
            result.brief_readiness_errors = validate_brief_readiness(step.case_brief)
            result.timings["wywiad_krok_z_briefem"] = elapsed
            return

        result.questions_asked += 1
        answer = case.answers[result.questions_asked - 1]
        result.turns.append(
            Turn(
                number=result.questions_asked,
                question=step.question,
                reason_for_question=step.turn.reason_for_question,
                missing_topics=list(step.turn.missing_topics),
                short_summary=step.turn.short_summary,
                answer=answer,
                model=llm.last_model,
                elapsed_seconds=elapsed,
            )
        )
        state.messages.append(ConversationMessage(role="assistant", content=step.question))
        add_user_message(answer)
        result.answers_used += 1


def verify_evidence_in_db(database_url: str, pack: EvidencePack) -> list[dict[str, Any]]:
    """Compare every evidence item with its chunk in rag_chunks (read-only query)."""
    chunk_ids = [evidence.chunk_id for evidence in pack.evidence]
    if not chunk_ids:
        return []
    with RetrievalRepository(database_url).connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id, document_id, page_start, page_end, text FROM rag_chunks "
            "WHERE chunk_id = ANY(%s)",
            (chunk_ids,),
        ).fetchall()
    by_id = {row[0]: row for row in rows}
    checked = []
    for evidence in pack.evidence:
        row = by_id.get(evidence.chunk_id)
        problems = []
        if row is None:
            problems.append("chunk_id nie istnieje w rag_chunks")
        else:
            _, document_id, page_start, page_end, text = row
            if document_id != evidence.document_id:
                problems.append(f"document_id w bazie: {document_id}")
            if (page_start, page_end) != (evidence.page_start, evidence.page_end):
                problems.append(f"strony w bazie: {page_start}-{page_end}")
            if text != evidence.excerpt:
                problems.append("fragment różni się od tekstu chunka w bazie")
        checked.append(
            {
                "evidence_id": evidence.evidence_id,
                "chunk_id": evidence.chunk_id,
                "in_db": row is not None,
                "problems": problems,
            }
        )
    return checked


def _retrieve(setup: Setup, case: Case, result: RunResult, state: _RunState) -> None:
    """Search the RAG database for the brief with the production pipeline configuration."""
    provider = setup.zrodla
    pipeline, documents, chunks = services.build_zrodlowyszukiwator(
        client=_stage_client(provider),
        model=provider.model,
        session_id=result.session_id,
        provider=provider.provider,
    )
    # Pipeline ustalił okno sam; ponowne sprawdzenie tylko odczytuje je do raportu.
    _context(result, "Źródłowyszukiwacz", provider)
    result.rag = {"documents": documents, "chunks": chunks}
    try:
        state.evidence_pack = pipeline.run(state.brief)
    finally:
        for stage, component in (("Planner", pipeline.planner), ("Reranker", pipeline.reranker)):
            if component is not None and component.last_model:
                _record_model(result, stage, provider, component.last_model)

    pack = state.evidence_pack
    fallback_plan = LLMQueryPlanner._fallback_plan(state.brief.model_dump(mode="json"))
    result.rag.update(
        {
            "evidence_status": pack.evidence_status,
            "query_summary": pack.query_summary,
            "intents": [intent.model_dump(mode="json") for intent in pack.research_plan.intents],
            "planner_fallback": pack.research_plan.model_dump() == fallback_plan.model_dump(),
            "unsupported_topics": list(pack.unsupported_topics),
            "search_notes": list(pack.search_notes),
            "evidence": [
                {
                    "evidence_id": e.evidence_id,
                    "chunk_id": e.chunk_id,
                    "document_id": e.document_id,
                    "document_title": e.document_title,
                    "pages": f"{e.page_start}-{e.page_end}",
                    "section_path": list(e.section_path),
                    "retrieval_score": e.retrieval_score,
                    "rerank_score": e.rerank_score,
                    "relevance_reason": e.relevance_reason,
                    "excerpt_start": " ".join(e.excerpt.split())[:300],
                }
                for e in pack.evidence
            ],
            "chunk_verification": verify_evidence_in_db(state.database_url, pack),
        }
    )


def _write_letter(setup: Setup, case: Case, result: RunResult, state: _RunState) -> None:
    """Generate and render the letter from the brief and the evidence found."""
    provider = setup.pismo
    llm = LLMClient(
        client=_stage_client(provider),
        model_name=provider.model,
        stage="pismogenerator",
        session_id=result.session_id,
        provider=provider.provider,
        context_tokens=_context(result, "Pismogenerator", provider),
    )
    try:
        state.draft = pismogenerator.build_draft(
            case_brief=state.brief,
            evidence_pack=state.evidence_pack,
            llm=llm,
            llm_temperature=0.2,
        )
    except DraftValidationError as exc:
        state.rejected_draft = exc.draft
        result.draft_validation_errors = list(exc.validation_errors)
        if exc.draft is not None:
            result.draft = exc.draft.model_dump(mode="json")
        raise
    finally:
        if llm.last_model:
            _record_model(result, "Pismogenerator", provider, llm.last_model)
    result.draft = state.draft.model_dump(mode="json")
    state.letter = services.render_letter(
        state.draft, state.evidence_pack, state.brief.available_evidence
    )
    result.letter_chars = len(state.letter)


def _record_error(result: RunResult, phase: str, exc: Exception) -> None:
    """Keep a phase failure with the message the UI would show and log its details."""
    message = services.llm_error_message(exc)
    location = services.error_location(exc)
    detail = error_log_detail(exc)
    result.errors.append(PhaseError(phase, type(exc).__name__, message, location, detail))
    services.LOGGER.error(
        "e2e_phase_failed session_id=%s phase=%s error_type=%s error_location=%s "
        "error_detail=%s error_message=%s",
        result.session_id,
        phase,
        type(exc).__name__,
        location or "-",
        detail,
        message,
        exc_info=exc,
        extra={"session_id": result.session_id},
    )


def _save(result: RunResult, state: _RunState, run_dir: Path) -> None:
    """Store the run artifacts the way the app stores a case."""
    artifacts = run_dir / "artifacts"
    services.save_run(
        project_root=run_dir,
        messages=[{"role": m.role, "content": m.content} for m in state.messages],
        case_brief=state.brief,
        evidence_pack=state.evidence_pack,
        draft=state.draft,
        draft_with_error=state.rejected_draft,
        draft_validation_errors=result.draft_validation_errors or None,
        letter_markdown=state.letter,
        model_usage=result.model_usage,
        run_dir=artifacts,
    )
    result.paths["artifacts"] = artifacts.relative_to(PROJECT_ROOT).as_posix()


PHASES: tuple[tuple[str, Callable[[Setup, Case, RunResult, _RunState], None]], ...] = (
    ("wywiad", _interview),
    ("rag", _retrieve),
    ("pismo", _write_letter),
)


def run_case(setup: Setup, case: Case, run_dir: Path, *, database_url: str) -> RunResult:
    """Run the whole Pismak flow for one setup and case and evaluate it."""
    run_dir.mkdir(parents=True, exist_ok=True)
    result = new_result(setup, case)
    result.session_id = uuid.uuid4().hex
    state = _RunState(database_url=database_url)
    extra = {"session_id": result.session_id}

    services.configure_logging(run_dir, result.session_id)
    services.LOGGER.info(
        "e2e_run_started session_id=%s run=%s dopytywator=%s/%s zrodla=%s/%s pismo=%s/%s",
        result.session_id,
        result.run_id,
        setup.dopytywator.provider,
        setup.dopytywator.model,
        setup.zrodla.provider,
        setup.zrodla.model,
        setup.pismo.provider,
        setup.pismo.model,
        extra=extra,
    )
    started = perf_counter()
    try:
        for phase, action in PHASES:
            phase_started = perf_counter()
            try:
                action(setup, case, result, state)
            except Exception as exc:
                _record_error(result, phase, exc)
                break
            finally:
                result.timings[phase] = round(perf_counter() - phase_started, 3)
                _save(result, state, run_dir)
    finally:
        result.timings["razem"] = round(perf_counter() - started, 3)
        services.LOGGER.info(
            "e2e_run_finished session_id=%s run=%s elapsed_seconds=%.3f errors=%d",
            result.session_id,
            result.run_id,
            result.timings["razem"],
            len(result.errors),
            extra=extra,
        )
        services.close_case_logging(result.session_id)
        detach_log_handler(run_dir)

    result.paths["logs"] = (run_dir / "logs").relative_to(PROJECT_ROOT).as_posix()
    result.llm_calls, result.log_stats = analyze_log(run_dir / "logs")
    evaluate(result, case, state)
    result.status = FAILED if result.hard_failures() else PASSED
    return result


# ---------------------------------------------------------------------------
# Log

_COMPLETED = re.compile(
    r"llm_request_completed stage=(?P<stage>\S+) session_id=\S+ provider=(?P<provider>\S+) "
    r"model=(?P<model>\S+) max_output_tokens=(?P<max_output>\d+) attempt=(?P<attempt>\d+) "
    r"elapsed_seconds=(?P<elapsed>[\d.]+)"
    r"(?: prompt_tokens=(?P<prompt_tokens>\S+) completion_tokens=(?P<completion_tokens>\S+))?"
)
_FAILED_CALL = re.compile(
    r"llm_request_failed stage=(?P<stage>\S+) .*?attempt=(?P<attempt>\d+)"
    r".*?error_type=(?P<error_type>\S+)"
)
_ELAPSED = re.compile(r"elapsed_seconds=([\d.]+)\s*$")
_RETRIEVAL = re.compile(
    r"stage_completed stage=retrieval session_id=\S+ intents=(?P<intents>\d+) "
    r"candidates=(?P<candidates>\d+)"
)
_COUNTED_EVENTS = (
    "llm_output_truncated",
    "llm_partial_response_recovered",
    "llm_output_limit",
    "llm_failed",
    "context_window_detection_failed",
    "e2e_phase_failed",
)


def _token(value: str | None) -> int | None:
    """Return a logged token count, or None when the provider did not report it."""
    return int(value) if value and value.isdigit() else None


def analyze_log(log_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract model calls, retries, truncations and retrieval runs from the run's log."""
    calls: list[dict[str, Any]] = []
    stats: dict[str, Any] = {name: 0 for name in _COUNTED_EVENTS}
    stats["failed_calls"] = []
    stats["retrieval"] = []
    for path in sorted(log_dir.glob("*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if match := _COMPLETED.search(line):
                calls.append(
                    {
                        "stage": match["stage"],
                        "model": match["model"],
                        "attempt": int(match["attempt"]),
                        "max_output_tokens": int(match["max_output"]),
                        "elapsed_seconds": float(match["elapsed"]),
                        "prompt_tokens": _token(match["prompt_tokens"]),
                        "completion_tokens": _token(match["completion_tokens"]),
                    }
                )
            elif match := _FAILED_CALL.search(line):
                elapsed = _ELAPSED.search(line)
                stats["failed_calls"].append(
                    {
                        "stage": match["stage"],
                        "attempt": int(match["attempt"]),
                        "error_type": match["error_type"],
                        "elapsed_seconds": float(elapsed[1]) if elapsed else None,
                    }
                )
            if match := _RETRIEVAL.search(line):
                stats["retrieval"].append(
                    {"intents": int(match["intents"]), "candidates": int(match["candidates"])}
                )
            for name in _COUNTED_EVENTS:
                if f"{name} " in line:
                    stats[name] += 1
    return calls, stats


# ---------------------------------------------------------------------------
# Kontrole


def _add(result: RunResult, id: str, title: str, severity: str, status: str, detail: str = ""):
    result.checks.append(Check(id, title, severity, status, detail))


def _hard(result: RunResult, id: str, title: str, problems: list[str]) -> None:
    _add(result, id, title, HARD, FAIL if problems else PASS, "; ".join(problems))


def _soft(result: RunResult, id: str, title: str, findings: list[str], ok: str = "") -> None:
    _add(result, id, title, SOFT, WARN if findings else PASS, "; ".join(findings) or ok)


def _phase_errors(result: RunResult, phase: str) -> list[str]:
    return [
        f"{e.error_type}: {e.user_message} ({e.detail})" for e in result.errors if e.phase == phase
    ]


def interview_invariants(result: RunResult, case: Case, messages: list[ConversationMessage]):
    """Add H1-H3: the description opens the talk, every turn works, the limit holds."""
    first = messages[0] if messages else None
    if first is None:
        _hard(result, "H1", "Rozmowa zaczyna się od opisu sprawy", ["opis nie trafił do rozmowy"])
    else:
        ok = first.role == "user" and first.content == case.description
        _hard(
            result,
            "H1",
            "Rozmowa zaczyna się od opisu sprawy",
            [] if ok else [f"pierwsza wiadomość ({first.role}): {first.content[:120]}"],
        )

    problems = _phase_errors(result, "wywiad")
    problems += [f"puste pytanie nr {t.number}" for t in result.turns if not t.question.strip()]
    _hard(result, "H2", "Pytania i odpowiedzi przetworzone bez błędów", problems)

    problems = []
    for label, value in (
        ("pytań", result.questions_asked),
        ("odpowiedzi", result.answers_used),
        ("wywołań wywiadu", result.interview_calls),
    ):
        if value > MAX_QUESTIONS:
            problems.append(f"{label}: {value} > {MAX_QUESTIONS}")
    final = result.final_turn
    if result.end_reason == "question_limit":
        if result.questions_asked != MAX_QUESTIONS:
            problems.append(f"limit zgłoszony po {result.questions_asked} pytaniach")
        if result.interview_calls != MAX_QUESTIONS:
            problems.append(f"{result.interview_calls} wywołań wywiadu przy limicie")
    elif result.end_reason == "model_ready":
        if result.questions_asked >= MAX_QUESTIONS:
            problems.append("gotowość zgłoszona dopiero po wyczerpaniu limitu")
        if not final or not final.get("ready_to_build_brief") or final.get("next_question"):
            problems.append("ostatnia tura nie sygnalizuje gotowości (ready/next_question)")
        if result.interview_calls != result.questions_asked + 1:
            problems.append(
                f"{result.interview_calls} wywołań wywiadu na {result.questions_asked} pytań"
            )
    detail_ok = "" if result.end_reason else "wywiad nie dobiegł końca (zob. H2), liczniki w normie"
    _add(
        result,
        "H3",
        f"Limit {MAX_QUESTIONS} pytań i sygnał gotowości do CaseBrief",
        HARD,
        FAIL if problems else PASS,
        "; ".join(problems) or detail_ok,
    )


def brief_invariant(result: RunResult) -> None:
    """Add H4: the brief exists and has the minimal Phase 0 fields."""
    if result.brief is None:
        _hard(result, "H4", "CaseBrief o poprawnej strukturze", ["CaseBrief nie powstał"])
    else:
        _hard(result, "H4", "CaseBrief o poprawnej strukturze", result.brief_readiness_errors)


def rag_invariant(result: RunResult) -> None:
    """Add H5: retrieval queried the database and returned consistent evidence."""
    title = "RAG odpytał bazę i zwrócił spójne dowody albo sygnał braku źródeł"
    if result.brief is None:
        _hard(result, "H5", title, ["nie uruchomiono: brak CaseBrief"])
        return
    problems = _phase_errors(result, "rag")
    rag = result.rag or {}
    if not problems:
        if not rag.get("chunks"):
            problems.append("baza RAG bez chunków")
        if not result.log_stats.get("retrieval"):
            problems.append("brak wpisu stage_completed stage=retrieval w logu")
        status = rag.get("evidence_status")
        evidence = rag.get("evidence", [])
        if status not in EVIDENCE_STATUSES:
            problems.append(f"nieznany evidence_status: {status}")
        if not evidence and status != "unsupported":
            problems.append(f"brak dowodów przy evidence_status={status}")
        if len(evidence) > EVIDENCE_K:
            problems.append(f"{len(evidence)} dowodów > {EVIDENCE_K}")
        per_document = Counter(item["document_id"] for item in evidence)
        problems += [
            f"{doc}: {count} dowodów > {MAX_PER_DOCUMENT}"
            for doc, count in per_document.items()
            if count > MAX_PER_DOCUMENT
        ]
        ids = [item["evidence_id"] for item in evidence]
        if len(ids) != len(set(ids)):
            problems.append("powtórzone evidence_id")
        problems += [
            f"{item['evidence_id']} ({item['chunk_id']}): {', '.join(item['problems'])}"
            for item in rag.get("chunk_verification", [])
            if item["problems"]
        ]
    _hard(result, "H5", title, problems)


def draft_invariant(result: RunResult) -> None:
    """Add H6: the letter was generated, passed the app's validation and was rendered."""
    title = "Pismogenerator utworzył projekt pisma, który przeszedł walidację aplikacji"
    if result.rag is None or result.rag.get("evidence_status") is None:
        _hard(result, "H6", title, ["nie uruchomiono: brak EvidencePack"])
        return
    problems = _phase_errors(result, "pismo")
    problems += [f"walidacja: {error}" for error in result.draft_validation_errors]
    if not problems:
        evidence_ids = {item["evidence_id"] for item in result.rag.get("evidence", [])}
        user_ids = {f"U{i}" for i, _ in enumerate(result.brief["available_evidence"], start=1)}
        illegal = [s for s in result.draft["sources"] if s not in evidence_ids | user_ids]
        if illegal:
            problems.append("niedozwolone źródła: " + ", ".join(illegal))
        if not result.letter_chars:
            problems.append("letter.md nie powstał")
    _hard(result, "H6", title, problems)


def interview_quality(result: RunResult, case: Case) -> None:
    """Add S1-S3: early end, repeated questions and whether ending was justified."""
    if result.end_reason is None:
        return
    if case.expects_early_end:
        findings = (
            [
                f"wywiad doszedł do limitu {MAX_QUESTIONS} pytań, model nie uznał danych za wystarczające"
            ]
            if result.end_reason == "question_limit"
            else []
        )
        _soft(
            result,
            "S1",
            "Zakończenie przed 12. pytaniem",
            findings,
            ok=f"model zakończył po {result.questions_asked} pytaniach",
        )
    else:
        ending = (
            f"model zakończył po {result.questions_asked} pytaniach"
            if result.end_reason == "model_ready"
            else f"limit {MAX_QUESTIONS} pytań"
        )
        _soft(result, "S1", "Zakończenie wywiadu (informacyjnie)", [], ok=ending)

    pairs = repeated_questions([turn.question for turn in result.turns])
    _soft(
        result,
        "S2",
        "Brak powtórzonych pytań",
        [f"pytanie {b} powtarza pytanie {a}" for a, b in pairs],
    )

    if result.brief is not None:
        missing = missing_key_info("\n".join(text_values(result.brief)), case.key_info)
        findings = [f"brief gubi: {', '.join(missing)}"] if missing else []
        if missing and result.end_reason == "model_ready":
            findings.append("zakończenie mogło być przedwczesne")
        _soft(result, "S3", "Brief zachowuje kluczowe informacje sprawy", findings)


def case_quality(result: RunResult, case: Case, letter: str) -> None:
    """Add S4-S5: the case-specific expectations (humour, observations vs data)."""
    brief = result.brief
    if brief is None:
        return
    if case.humor_markers:
        findings = []
        if not contains_any("\n".join(text_values(brief)), case.humor_markers):
            findings.append("brief nie oddaje satyrycznego/happeningowego charakteru sprawy")
        if letter and not contains_any(letter, case.humor_markers):
            findings.append("pismo nie oddaje satyrycznego/happeningowego charakteru sprawy")
        literal = [
            action
            for action in brief["requested_actions"]
            if contains_any(action, case.literal_demand_markers)
            and not contains_any(action, case.humor_markers)
        ]
        if literal:
            findings.append("dosłowne żądanie w postulatach: " + " | ".join(literal))
        _soft(result, "S4", "Rozpoznanie humorystycznego charakteru sprawy", findings)

    if case.user_has_no_measurements:
        findings = [
            f"twierdzenie oznaczone jako poparte dowodem: {claim['statement']}"
            for claim in brief["claims"]
            if claim["status"] == ClaimStatus.SUPPORTED_BY_USER_EVIDENCE.value
        ]
        if brief["available_evidence"]:
            findings.append(
                "available_evidence niepuste, choć użytkownik nie ma pomiarów: "
                + " | ".join(brief["available_evidence"])
            )
        _soft(result, "S5", "Obserwacje użytkownika nie są traktowane jako dane", findings)


def invention_quality(result: RunResult, conversation: str, evidence_text: str) -> None:
    """Add S6-S7: numbers, institutions and legal bases absent from the inputs."""
    brief = result.brief
    if brief is not None:
        findings = []
        numbers = unknown_numbers("\n".join(text_values(brief)), conversation)
        if numbers:
            findings.append("liczby spoza rozmowy: " + ", ".join(numbers))
        for name in ("institution", "addressee"):
            value = brief.get(name)
            if value and words_outside(value, conversation):
                findings.append(f"{name} spoza rozmowy: {value}")
        _soft(result, "S6", "Brief bez faktów spoza rozmowy", findings)

    draft = result.draft
    if draft is not None and result.letter_chars:
        main_text = draft["request_summary"] + "\n" + draft["argumentation_markdown"]
        known = conversation + "\n" + "\n".join(text_values(brief)) + "\n" + evidence_text
        findings = []
        numbers = unknown_numbers(main_text, known)
        if numbers:
            findings.append("liczby spoza rozmowy, briefu i dowodów: " + ", ".join(numbers))
        legal = legal_basis_outside(main_text, evidence_text)
        if legal:
            findings.append("podstawy prawne spoza dowodów: " + ", ".join(legal))
        recipient = draft["recipient"]
        if "[" not in recipient and words_outside(recipient, known):
            findings.append(f"adresat spoza rozmowy i dowodów: {recipient}")
        _soft(result, "S7", "Pismo bez wymyślonych faktów i podstaw prawnych", findings)


def rag_quality(result: RunResult, case: Case) -> None:
    """Add S8: planner/reranker fallbacks and the evidence status expected for the case."""
    rag = result.rag
    if not rag or rag.get("evidence_status") is None:
        return
    findings = []
    status = rag["evidence_status"]
    planner_failures = [
        c for c in result.log_stats["failed_calls"] if c["stage"] == "query_planner"
    ]
    if rag["planner_fallback"]:
        findings.append("planner użył planu awaryjnego (zapytania = pola briefu)")
    elif planner_failures:
        findings.append(f"planner: {len(planner_failures)} nieudanych prób przed sukcesem")
    if status == "unverified":
        findings.append("reranker nie ocenił trafności (status unverified)")
    if case.expected_evidence_status and status not in case.expected_evidence_status:
        findings.append(
            f"evidence_status={status}, oczekiwano {'/'.join(case.expected_evidence_status)}"
        )
    if not case.expected_evidence_status and status == "supported":
        titles = sorted({item["document_title"] for item in rag["evidence"]})
        findings.append(
            "dowody uznane za wspierające — oceń trafność ręcznie: " + "; ".join(titles)
        )
    _soft(
        result,
        "S8",
        "Jakość wyszukiwania źródeł",
        findings,
        ok=f"evidence_status={status}, dowodów: {len(rag['evidence'])}",
    )


def runtime_quality(result: RunResult) -> None:
    """Add S9: truncated outputs, repairs, slow calls and context-window problems."""
    stats = result.log_stats
    findings = []
    for name, label in (
        ("llm_output_truncated", "ucięte odpowiedzi (limit tokenów)"),
        ("llm_partial_response_recovered", "odzyskane częściowe odpowiedzi"),
        ("llm_output_limit", "przekroczony limit wyjścia"),
        ("context_window_detection_failed", "nieudane ustalenie okna kontekstu"),
    ):
        if stats.get(name):
            findings.append(f"{label}: {stats[name]}")
    repairs = [c for c in result.llm_calls if c["attempt"] > 1]
    if repairs:
        stages = Counter(c["stage"] for c in repairs)
        findings.append(
            "udane dopiero przy 2. próbie: " + ", ".join(f"{s}×{n}" for s, n in stages.items())
        )
    if stats.get("failed_calls"):
        stages = Counter(f"{c['stage']}/{c['error_type']}" for c in stats["failed_calls"])
        findings.append("nieudane próby: " + ", ".join(f"{s}×{n}" for s, n in stages.items()))
    # Nieudane wywołania też się liczą: zbyt długa odpowiedź bywa odrzucona dopiero po fakcie.
    slow = [
        f"{c['stage']} {c['elapsed_seconds']:.0f} s"
        + ("" if "error_type" not in c else f" (nieudane: {c['error_type']})")
        for c in [*result.llm_calls, *stats.get("failed_calls", [])]
        if (c.get("elapsed_seconds") or 0) > SLOW_CALL_SECONDS
    ]
    if slow:
        findings.append(f"wywołania > {SLOW_CALL_SECONDS:.0f} s: " + ", ".join(slow))
    longest = max((c["elapsed_seconds"] for c in result.llm_calls), default=0.0)
    _soft(
        result,
        "S9",
        "Limity tokenów, ponowienia i czas wykonania",
        findings,
        ok=f"{len(result.llm_calls)} wywołań, najdłuższe {longest:.0f} s",
    )


def evaluate(result: RunResult, case: Case, state: _RunState) -> None:
    """Run every check for a finished run."""
    conversation = "\n".join(m.content for m in state.messages)
    evidence_text = ""
    if state.evidence_pack is not None:
        evidence_text = "\n".join(
            text_values(state.evidence_pack.model_dump(mode="json", exclude={"research_plan"}))
        )
    interview_invariants(result, case, state.messages)
    brief_invariant(result)
    rag_invariant(result)
    draft_invariant(result)
    interview_quality(result, case)
    case_quality(result, case, state.letter or "")
    invention_quality(result, conversation, evidence_text)
    rag_quality(result, case)
    runtime_quality(result)


# ---------------------------------------------------------------------------
# Sekrety


_KEY_PATTERN = re.compile(rb"sk-or-v1-[A-Za-z0-9]{16,}")


def secret_values() -> list[str]:
    """Return the configured credentials that must never be written to the report."""
    values = []
    key = os.getenv("OPENROUTER_KEY", "").strip()
    if key and key != OPENROUTER_KEY_PLACEHOLDER:
        values.append(key)
    password = urlsplit(os.getenv("DATABASE_URL", "")).password
    if password and len(password) >= 6:
        values.append(password)
    return values


def scrub_secrets(root: Path, secrets: list[str]) -> list[Path]:
    """Redact credentials in every file under root and return the files that contained one."""
    leaked = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        cleaned = _KEY_PATTERN.sub(REDACTED.encode(), data)
        for secret in secrets:
            cleaned = cleaned.replace(secret.encode("utf-8"), REDACTED.encode())
        if cleaned != data:
            path.write_bytes(cleaned)
            leaked.append(path)
    return leaked


# ---------------------------------------------------------------------------
# Raport


def git_commit() -> str:
    """Return the short commit hash with a marker for uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "nieznany"
    return f"{commit}{' + niezacommitowane zmiany' if dirty else ''}"


def environment_header(documents: int | None, chunks: int | None) -> dict[str, Any]:
    """Describe the environment of the run without any credentials."""
    versions = {}
    for package in ("openai", "pydantic", "psycopg", "streamlit"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "brak"
    return {
        "start": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "python": platform.python_version(),
        "pakiety": ", ".join(f"{name} {version}" for name, version in versions.items()),
        "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        "model embeddingów": os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"),
        "PISMAK_CONTEXT_TOKENS": os.getenv("PISMAK_CONTEXT_TOKENS") or "nie ustawiono",
        "PISMAK_OPENROUTER_CONTEXT_TOKENS": (
            os.getenv("PISMAK_OPENROUTER_CONTEXT_TOKENS") or "nie ustawiono"
        ),
        "baza RAG": f"{documents} dokumentów, {chunks} chunków"
        if chunks is not None
        else "niedostępna",
        "limit pytań": MAX_QUESTIONS,
    }


LIMITATIONS = (
    "Odpowiedzi są podawane w ustalonej kolejności i celowo nie pasują do pytań modelu.",
    "Każda kombinacja jest uruchamiana raz; modele są niedeterministyczne, więc powtórny "
    "przebieg może dać inny wynik.",
    "`openrouter/free` dobiera darmowy model przy każdym wywołaniu; kolumna „modele "
    "faktyczne” pokazuje, które odpowiedziały.",
    "Kontrole S1–S9 są heurystyczne (słowa-klucze, liczby, wzorce) — WARN wymaga ręcznej "
    "oceny na podstawie pełnego briefu, dowodów i pisma.",
    "Embeddingi zawsze liczy lokalny model z Ollamy, także w przebiegach OpenRouter.",
    "Klient Ollamy ma limit 900 s, klient OpenRouter 60 s; przekroczenie kończy przebieg "
    "jako NIEUDANY z podanym powodem. Limit nie zawsze przerywa wywołanie po tym czasie — "
    "rzeczywiste czasy są w tabelach wywołań i w S9.",
    "W konfiguracjach mieszanych błąd należy do dostawcy etapu, w którym wystąpił: faza "
    "„wywiad” to Dopytywator, „rag” — Źródłowyszukiwacz, „pismo” — Pismogenerator.",
)


def _cell(value: Any, limit: int = 160) -> str:
    """Return a value as a one-line Markdown table cell."""
    text = " ".join(str(value if value is not None else "—").split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "/") or "—"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes} min {rest:02d} s" if minutes else f"{rest} s"


def _end_reason(result: RunResult) -> str:
    if result.end_reason == "model_ready":
        return f"model gotowy po {result.questions_asked} pyt."
    if result.end_reason == "question_limit":
        return f"limit {MAX_QUESTIONS} pyt."
    if result.errors:
        return f"błąd ({result.errors[0].phase})"
    return "—"


def _short_end(result: RunResult) -> str:
    if result.end_reason == "model_ready":
        return f"model po {result.questions_asked}"
    if result.end_reason == "question_limit":
        return f"limit {MAX_QUESTIONS}"
    return f"błąd ({result.errors[0].phase})" if result.errors else "—"


def _brief_cell(result: RunResult) -> str:
    if result.brief is None:
        return "brak"
    return "braki" if result.brief_readiness_errors else "OK"


def _rag_cell(result: RunResult) -> str:
    if result.rag is None or result.rag.get("evidence_status") is None:
        return "błąd" if any(e.phase == "rag" for e in result.errors) else "—"
    return f"{result.rag['evidence_status']} ({len(result.rag['evidence'])})"


def _letter_cell(result: RunResult) -> str:
    if result.letter_chars:
        return "OK"
    if result.draft_validation_errors:
        return "odrzucone"
    return "błąd" if any(e.phase == "pismo" for e in result.errors) else "—"


def failure_reason(result: RunResult) -> str:
    """Return why a run did not pass: the not-run reason, the first error or failed invariants."""
    if result.status == NOT_RUN:
        return result.not_run_reason or ""
    if not result.hard_failures():
        return ""
    if result.errors:
        error = result.errors[0]
        return f"{error.phase}: {error.error_type} — {error.detail}"
    return "; ".join(f"{check.id}: {check.detail}" for check in result.hard_failures())


def summary_rows(results: list[RunResult]) -> list[dict[str, str]]:
    """Return one summary row per run, shared by the Markdown report and the terminal table."""
    rows = []
    for result in results:
        ran = result.status != NOT_RUN
        rows.append(
            {
                "uruchomienie": result.setup_id,
                "sprawa": result.case_id,
                # Model tak, jak wpisano go w URUCHOMIENIA (skrót albo nazwa modelu).
                **{
                    stage: config.get("name") or config["model"]
                    for stage, config in result.stages.items()
                },
                "pytań": str(result.questions_asked) if ran else "—",
                "zakończenie": _short_end(result) if ran else "—",
                "brief": _brief_cell(result) if ran else "—",
                "RAG": _rag_cell(result) if ran else "—",
                "pismo": _letter_cell(result) if ran else "—",
                "czas": _duration(result.timings.get("razem")) if ran else "—",
                "wynik": result.status,
                "WARN": str(len(result.warnings())) if ran else "—",
            }
        )
    return rows


def markdown_summary(results: list[RunResult]) -> list[str]:
    """Render the summary table and the reasons of failed or skipped runs as Markdown."""
    rows = summary_rows(results)
    if not rows:
        return ["Brak uruchomień."]
    columns = list(rows[0])
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = [_cell(row[column], 60) for column in columns]
        cells[0] = f"`{cells[0]}`"
        cells[columns.index("wynik")] = f"**{row['wynik']}**"
        out.append("| " + " | ".join(cells) + " |")
    reasons = [(r.run_id, failure_reason(r)) for r in results if failure_reason(r)]
    if reasons:
        out.append("\nPrzyczyny niepowodzeń:\n")
        out += [f"- `{rid}`: {_cell(reason, 400)}" for rid, reason in reasons]
    return out


def text_summary(results: list[RunResult], *, reason_width: int = 150) -> list[str]:
    """Render the summary as a plain-text table for the terminal, reasons listed below it."""
    rows = summary_rows(results)
    if not rows:
        return []
    columns = list(rows[0])
    widths = [max(len(column), *(len(row[column]) for row in rows)) for column in columns]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths, strict=True)) + " |"

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    out = [border, line(columns), border]
    out += [line([row[column] for column in columns]) for row in rows]
    out.append(border)
    for result in results:
        reason = " ".join(failure_reason(result).split())
        if reason:
            if len(reason) > reason_width:
                reason = reason[: reason_width - 1] + "…"
            out.append(f"  {result.run_id}: {reason}")
    return out


def _run_section(result: RunResult, case: Case | None) -> list[str]:
    out = [f"## `{result.run_id}` — {result.setup_label}, {result.case_title}\n"]
    if result.status == NOT_RUN:
        out.append(f"**{NOT_RUN}:** {result.not_run_reason}\n")
        return out
    out.append(f"- Wynik: **{result.status}**, ostrzeżeń: {len(result.warnings())}")
    for stage, config in result.stages.items():
        tokens = result.context_tokens.get(stage)
        window = f"{tokens} tokenów" if tokens else "nieustalone (etap nie ruszył albo brak danych)"
        out.append(
            f"- {stage}: `{config['model']}` ({config['provider']}); okno kontekstu: {window}"
        )
    for stage, calls in result.model_usage.items():
        models = ", ".join(dict.fromkeys(f"`{c['model']}`" for c in calls))
        out.append(f"- Modele faktyczne — {stage}: {models}")
    out.append(
        "- Czasy: "
        + ", ".join(f"{name} {_duration(seconds)}" for name, seconds in result.timings.items())
    )
    out.append(
        f"- Sesja: `{result.session_id}`; log: `{result.paths.get('logs')}`; "
        f"artefakty: `{result.paths.get('artifacts')}`"
    )
    out.append(f"- Odtworzenie: `{result.reproduce}`\n")

    out.append("### Kontrole\n")
    out.append("| id | kontrola | waga | status | szczegóły |")
    out.append("|---|---|---|---|---|")
    for check in result.checks:
        out.append(
            f"| {check.id} | {_cell(check.title)} | {check.severity} | **{check.status}** "
            f"| {_cell(check.detail, 400)} |"
        )
    out.append("")

    out.append("### Rozmowa\n")
    if case is not None:
        out.append(f"Opis sprawy: {case.description}\n")
    if result.turns:
        out.append(
            "| # | pytanie modelu | powód pytania | brakujące tematy | podana odpowiedź | czas |"
        )
        out.append("|---|---|---|---|---|---|")
        for turn in result.turns:
            out.append(
                f"| {turn.number} | {_cell(turn.question, 300)} | "
                f"{_cell(turn.reason_for_question, 200)} | {_cell(', '.join(turn.missing_topics))} "
                f"| {_cell(turn.answer, 200)} | {_duration(turn.elapsed_seconds)} |"
            )
    out.append(f"\nZakończenie: **{_end_reason(result)}**.")
    if result.final_turn:
        out.append(
            f" Ostatnie podsumowanie modelu: {_cell(result.final_turn['short_summary'], 500)}"
        )
    out.append("")

    if result.brief is not None:
        brief = result.brief
        out.append("### CaseBrief\n")
        out.append(f"- Tytuł: {_cell(brief['title'], 300)}")
        out.append(f"- Problem: {_cell(brief['problem_description'], 600)}")
        out.append(f"- Oczekiwany rezultat: {_cell(brief['desired_outcome'], 400)}")
        for action in brief["requested_actions"]:
            out.append(f"- Postulat: {_cell(action, 300)}")
        out.append(
            f"- Lokalizacja: {_cell(brief['location'])}; czas: {_cell(brief['time_context'])}"
        )
        out.append(
            f"- Instytucja: {_cell(brief['institution'])}; adresat: {_cell(brief['addressee'])}"
        )
        out.append(f"- Ton: {brief['preferred_tone']}")
        for claim in brief["claims"]:
            out.append(f"- Twierdzenie [{claim['status']}]: {_cell(claim['statement'], 300)}")
        for label, key in (
            ("Brakujące informacje", "missing_information"),
            ("Fakty do weryfikacji", "facts_to_verify"),
            ("Dostępne dowody", "available_evidence"),
        ):
            if brief[key]:
                out.append(f"- {label}: " + " | ".join(_cell(item, 200) for item in brief[key]))
        out.append("")

    if result.rag is not None:
        rag = result.rag
        out.append("### Źródłowyszukiwacz\n")
        out.append(f"- Baza: {rag.get('documents')} dokumentów, {rag.get('chunks')} chunków")
        for run in result.log_stats.get("retrieval", []):
            out.append(f"- Wyszukiwanie: {run['intents']} zapytań, {run['candidates']} kandydatów")
        if "evidence_status" in rag:
            out.append(
                f"- evidence_status: **{rag['evidence_status']}**; plan awaryjny plannera: "
                f"{'tak' if rag['planner_fallback'] else 'nie'}"
            )
            for intent in rag["intents"]:
                out.append(f"- Zapytanie: {_cell(intent['semantic_query'], 200)}")
            if rag["unsupported_topics"]:
                out.append("- Tematy bez źródeł: " + " | ".join(rag["unsupported_topics"]))
            if rag["evidence"]:
                verified = {v["evidence_id"]: v for v in rag["chunk_verification"]}
                out.append("\n| id | dokument | s. | ocena | w bazie | uzasadnienie | fragment |")
                out.append("|---|---|---|---|---|---|---|")
                for item in rag["evidence"]:
                    check = verified.get(item["evidence_id"], {})
                    in_db = "tak" if check.get("in_db") and not check.get("problems") else "NIE"
                    out.append(
                        f"| {item['evidence_id']} | {_cell(item['document_title'], 80)} "
                        f"| {item['pages']} | {_cell(item['rerank_score'])} | {in_db} "
                        f"| {_cell(item['relevance_reason'], 160)} "
                        f"| {_cell(item['excerpt_start'], 160)} |"
                    )
        out.append("")

    if result.draft is not None:
        draft = result.draft
        out.append("### Pismo\n")
        out.append(f"- Adresat: {_cell(draft['recipient'])}")
        out.append(f"- Wniosek: {_cell(draft['request_summary'], 400)}")
        out.append(f"- Źródła: {', '.join(draft['sources']) or 'brak'}")
        if result.draft_validation_errors:
            out.append("- Odrzucone przez walidację: " + "; ".join(result.draft_validation_errors))
        if result.letter_chars:
            out.append(f"- Pełna treść: `{result.paths.get('artifacts')}/letter.md`")
        out.append("")

    out.append("### Wywołania modeli\n")
    if result.llm_calls:
        out.append(
            "| etap | model | próba | czas | tokeny wejścia | tokeny wyjścia | limit wyjścia |"
        )
        out.append("|---|---|---|---|---|---|---|")
        for call in result.llm_calls:
            out.append(
                f"| {call['stage']} | `{call['model']}` | {call['attempt']} "
                f"| {call['elapsed_seconds']:.1f} s | {_cell(call['prompt_tokens'])} "
                f"| {_cell(call['completion_tokens'])} | {call['max_output_tokens']} |"
            )
    events = {k: v for k, v in result.log_stats.items() if isinstance(v, int) and v}
    out.append(f"\nZdarzenia w logu: {events or 'brak ostrzeżeń'}\n")

    if result.errors:
        out.append("### Błędy\n")
        for error in result.errors:
            out.append(
                f"- {error.phase}: `{error.error_type}` — komunikat UI: „{error.user_message}”; "
                f"miejsce: `{error.location or '-'}`; szczegóły: {_cell(error.detail, 600)}"
            )
        out.append("")
    return out


def render_report(
    label: str, header: dict[str, Any], results: list[RunResult], cases: dict[str, Case]
) -> str:
    """Render the session report as Markdown."""
    out = [f"# Testy E2E Pismaka — przebieg `{label}`\n"]
    out += [f"- {key}: {value}" for key, value in header.items()]
    out.append(
        "- Zasada oceny: wynik testu zależy tylko od niezmienników H1–H7 (FAIL). "
        "S1–S9 to ocena jakości (WARN) do ręcznego przeglądu.\n"
    )

    out.append("## Podsumowanie\n")
    out.append(
        "Kolumny etapów podają model wpisany w `URUCHOMIENIA`; modele, które faktycznie "
        "odpowiedziały (ważne przy `openrouter/free`), są w sekcji każdego uruchomienia. "
        "„RAG” to evidence_status i liczba dowodów, „zakończenie” — kto zakończył wywiad.\n"
    )
    out += markdown_summary(results)
    out.append("")

    executed = [r for r in results if r.status != NOT_RUN]
    not_run = [r for r in results if r.status == NOT_RUN]
    out.append("## Wykonane\n")
    for result in executed:
        failures = "; ".join(f"{c.id}: {c.detail}" for c in result.hard_failures())
        out.append(
            f"- `{result.run_id}`: **{result.status}**" + (f" — {failures}" if failures else "")
        )
    if not executed:
        out.append("- brak")
    out.append("\n## Nieuruchomione\n")
    for result in not_run:
        out.append(f"- `{result.run_id}`: {result.not_run_reason}")
    if not not_run:
        out.append("- brak")
    out.append("\n## Ograniczenia testu\n")
    out += [f"- {item}" for item in LIMITATIONS]
    out.append("")

    for result in results:
        out += _run_section(result, cases.get(result.case_id))
    return "\n".join(out)


class E2ESession:
    """Collect the runs of one pytest session and keep the report up to date after each."""

    # Pod tym kluczem sesja trafia do konfiguracji pytest, skąd conftest.py drukuje tabelę.
    KEY: pytest.StashKey[E2ESession] = pytest.StashKey()

    def __init__(self, label: str, header: dict[str, Any], cases: Iterable[Case]) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.label = label
        self.header = header
        self.cases = {case.id: case for case in cases}
        self.out_dir = REPORTS_DIR / f"{stamp}_{label}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[RunResult] = []
        self.secrets = secret_values()

    @property
    def report_path(self) -> Path:
        return self.out_dir / "report.md"

    def run_dir(self, setup: Setup, case: Case) -> Path:
        return self.out_dir / f"{setup.id}__{case.id}"

    def add_not_run(self, setup: Setup, case: Case, reason: str) -> RunResult:
        """Record a run that could not start, with the reason."""
        result = new_result(setup, case)
        result.status = NOT_RUN
        result.not_run_reason = reason
        self.add(result)
        return result

    def add(self, result: RunResult) -> None:
        """Store a run, rewrite the report and make sure no credential was written."""
        self.results.append(result)
        self._write(result)
        leaked = scrub_secrets(self.out_dir, self.secrets)
        if result.status != NOT_RUN:
            names = [path.relative_to(PROJECT_ROOT).as_posix() for path in leaked]
            _hard(
                result,
                "H7",
                "Brak kluczy i haseł w logach i raporcie",
                [f"usunięto dane uwierzytelniające z: {', '.join(names)}"] if names else [],
            )
            if leaked:
                result.status = FAILED
            self._write(result)
            scrub_secrets(self.out_dir, self.secrets)

    def _write(self, result: RunResult) -> None:
        if result.status != NOT_RUN:
            run_dir = self.out_dir / f"{result.setup_id}__{result.case_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str)
            (run_dir / "result.json").write_text(payload, encoding="utf-8")
            turns = json.dumps(
                [asdict(turn) for turn in result.turns], ensure_ascii=False, indent=2
            )
            (run_dir / "turns.json").write_text(turns, encoding="utf-8")
        summary = {
            "label": self.label,
            "header": self.header,
            "results": [asdict(r) for r in self.results],
        }
        (self.out_dir / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        self.report_path.write_text(
            render_report(self.label, self.header, self.results, self.cases), encoding="utf-8"
        )
