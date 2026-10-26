"""
Wspólna infrastruktura złotych zbiorów: dostawcy modeli, przypadki, limity i raport przebiegu.

Dostawcy pochodzą z konfiguracji projektu, a nie z listy w teście: każdy katalog
`modele/<nazwa>/` z plikiem `Modelfile` to model Ollamy `pismak-<nazwa>` (README, rozdział 3).
OpenRouter (`OPENROUTER_MODEL` z `src/_shared/serwer.py`) dokładają tylko testy E2E
(`include_openrouter=True`): darmowe konto ma 50 żądań dziennie, a same złote zbiory zużyłyby
je w jednym przebiegu. Zmienna PISMAK_GOLDEN_PROVIDERS (np. "bielik") zawęża listę. Klienta
LLM, okno kontekstu i limit czasu ustawia ten sam kod, którego używa aplikacja
(`src/Wyswietlator/services.py`).

Każdy przebieg zapisuje `results/<czas>_<dostawca>_<przypadek>.json` w katalogu modułu
(status, model, wynik każdej kontroli, czas, ponowienia, błędy, manual_review) oraz log
przebiegu obok. Wartości sekretów z `.env` są w zapisie zastępowane znacznikiem.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values, load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from src._shared.llm_runtime import LLMBusyError  # noqa: E402
from src._shared.model_context import detect_context_tokens  # noqa: E402
from src._shared.serwer import (  # noqa: E402
    OPENROUTER_MODEL,
    LLMClient,
    LLMConfigurationError,
    LLMServiceError,
    error_log_detail,
)
from src.Wyswietlator import services  # noqa: E402
from src.Zrodlowyszukiwator.rag_retrieval.repository import (  # noqa: E402
    DatabaseConnectionError,
    RetrievalRepository,
)

PASSED, FAILED_CHECK, INFRA_ERROR, SKIPPED = "passed", "failed_check", "infra_error", "skipped"
CASE_KEYS = {"id", "description", "input", "user_answers", "limits", "expected", "manual_review"}
REQUIRED_CASE_KEYS = {"id", "description", "input", "expected", "manual_review"}
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
SECRET_SETTINGS = (
    "OPENROUTER_KEY",
    "POSTGRES_PASSWORD",
    "PISMAK_APP_DB_PASSWORD",
    "PISMAK_INGEST_DB_PASSWORD",
)
# Błędy otoczenia (sieć, limit czasu, kolejka, baza), a nie modułu: status infra_error.
INFRA_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    LLMBusyError,
    DatabaseConnectionError,
    LLMConfigurationError,
    LLMServiceError,
)


# ---------------------------------------------------------------------------
# Dostawcy


@dataclass(frozen=True)
class Provider:
    label: str  # w identyfikatorze testu i nazwie raportu: "bielik", "pllum"
    provider: str  # "ollama", jak w services.get_llm_client
    model: str


def configured_providers(*, include_openrouter: bool = False) -> list[Provider]:
    """Return the project's models, one per modele/<name>/Modelfile, and OpenRouter for E2E."""
    providers = [
        Provider(path.parent.name, "ollama", f"pismak-{path.parent.name}:latest")
        for path in sorted((PROJECT_ROOT / "modele").glob("*/Modelfile"))
    ]
    if include_openrouter:
        providers.append(Provider("openrouter", "openrouter", OPENROUTER_MODEL))
    wanted = [x.strip() for x in os.getenv("PISMAK_GOLDEN_PROVIDERS", "").split(",") if x.strip()]
    if wanted:
        unknown = sorted(set(wanted) - {p.label for p in providers})
        if unknown:
            raise ValueError(
                f"PISMAK_GOLDEN_PROVIDERS: nieznani dostawcy {unknown}; "
                f"dostępni: {', '.join(p.label for p in providers)}"
            )
        providers = [p for p in providers if p.label in wanted]
    return providers


_OLLAMA_MODELS: list[str] | None = None
SKIP_DETAIL_CHARS = 200


def failure(exc: Exception) -> str:
    """Name an exception with its shortened message for a skip reason."""
    # Sam typ („RuntimeError”) nie mówił, dlaczego 21 przypadków pominięto (2026-10-24).
    detail = error_log_detail(exc)[:SKIP_DETAIL_CHARS]
    name = type(exc).__name__
    return name if detail == name else f"{name}: {detail}"


def provider_skip_reason(provider: Provider) -> str | None:
    """Return why the local model cannot be used on this machine, or None when it can."""
    global _OLLAMA_MODELS
    if _OLLAMA_MODELS is None:
        try:
            _OLLAMA_MODELS = services.list_generation_models(services.get_openai_client())
        except Exception as exc:
            return f"Ollama nie odpowiada ({failure(exc)})"
    if provider.model not in _OLLAMA_MODELS:
        return f"model {provider.model} nie jest zarejestrowany w Ollamie"
    return None


def database_skip_reason() -> str | None:
    """Return why the RAG database cannot be read, or None when it holds chunks."""
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        return "brak DATABASE_URL w .env"
    try:
        _, chunks = RetrievalRepository(database_url).counts()
    except Exception as exc:
        return f"baza RAG niedostępna ({failure(exc)})"
    return None if chunks else "baza RAG jest pusta — wykonaj rag_ingestion"


def embeddings_skip_reason() -> str | None:
    """Return why the embedding model cannot be called, or None when it answers."""
    from src.Zrodlowyszukiwator.rag_ingestion.embedder import OllamaEmbedder

    try:
        OllamaEmbedder(
            model=os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"),
            base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
            timeout=60.0,
        ).embed(["test"])
    except Exception as exc:
        return f"model embeddingów niedostępny ({failure(exc)})"
    return None


def llm_client(provider: Provider, *, timeout_seconds: float | None = None) -> Any:
    """Return the OpenAI-compatible client the app uses, optionally with the case's timeout."""
    client = services.get_llm_client(provider.provider)
    if timeout_seconds is not None:
        client = client.with_options(timeout=timeout_seconds, max_retries=0)
    return client


def make_llm(
    provider: Provider, *, stage: str, session_id: str, timeout_seconds: float | None = None
) -> LLMClient:
    """Build the stage's LLMClient exactly as src/Wyswietlator/app.py does."""
    return LLMClient(
        client=llm_client(provider, timeout_seconds=timeout_seconds),
        model_name=provider.model,
        stage=stage,
        session_id=session_id,
        provider=provider.provider,
        context_tokens=detect_context_tokens(
            provider.provider, provider.model, session_id=session_id
        ),
    )


# ---------------------------------------------------------------------------
# Przypadki


@dataclass(frozen=True)
class GoldenCase:
    id: str
    description: str
    input: Any
    expected: dict[str, Any]
    manual_review: list[str]
    user_answers: list[str] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)


def load_cases(
    module_dir: Path,
    *,
    supported_expected: set[str],
    required_limits: set[str] = frozenset(),
    supported_limits: set[str] = frozenset({"timeout_seconds"}),
) -> list[GoldenCase]:
    """Load cases/<case_id>.json, rejecting unknown fields so a typo cannot pass silently."""
    cases: list[GoldenCase] = []
    for path in sorted((module_dir / "cases").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        where = path.relative_to(PROJECT_ROOT).as_posix()
        missing = REQUIRED_CASE_KEYS - set(raw)
        if missing:
            raise ValueError(f"{where}: brak pól {sorted(missing)}")
        unknown = set(raw) - CASE_KEYS
        if unknown:
            raise ValueError(f"{where}: nieznane pola {sorted(unknown)}")
        if raw["id"] != path.stem:
            raise ValueError(f"{where}: id {raw['id']!r} różni się od nazwy pliku")
        unknown_expected = set(raw["expected"]) - supported_expected
        if unknown_expected:
            raise ValueError(
                f"{where}: pola expected {sorted(unknown_expected)} nie są obsługiwane; "
                f"obsługiwane: {sorted(supported_expected)}"
            )
        limits = raw.get("limits", {})
        if required_limits - set(limits):
            raise ValueError(f"{where}: brak limitów {sorted(required_limits - set(limits))}")
        if set(limits) - set(supported_limits) - set(required_limits):
            raise ValueError(f"{where}: nieznane limity {sorted(set(limits) - supported_limits)}")
        if not str(raw["description"]).strip() or not isinstance(raw["manual_review"], list):
            raise ValueError(f"{where}: description i manual_review (lista) są wymagane")
        cases.append(
            GoldenCase(
                id=raw["id"],
                description=raw["description"],
                input=raw["input"],
                expected=raw["expected"],
                manual_review=list(raw["manual_review"]),
                user_answers=list(raw.get("user_answers", [])),
                limits=dict(limits),
            )
        )
    if not cases:
        raise ValueError(f"{module_dir}: brak przypadków w cases/")
    return cases


def runs(cases: list[GoldenCase]) -> list[Any]:
    """Parametrize over providers × cases with the id <provider>-<case_id>."""
    return [
        pytest.param(provider, case, id=f"{provider.label}-{case.id}")
        for provider in configured_providers()
        for case in cases
    ]


# ---------------------------------------------------------------------------
# Porównywanie tekstu


def normalize(text: str) -> str:
    """Lowercase, drop Polish diacritics and collapse whitespace for keyword checks."""
    decomposed = unicodedata.normalize("NFKD", str(text).replace("ł", "l").replace("Ł", "L"))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(plain.casefold().split())


def mentions(text: str, term: str) -> bool:
    """Return whether the term occurs in the text, ignoring case and Polish characters."""
    return normalize(term) in normalize(text)


def as_text(value: Any) -> str:
    """Flatten a model, dict or list into one searchable string."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers_in(text: str) -> set[str]:
    """Return the numbers of a text, with a decimal comma read as a point."""
    return {match.replace(",", ".") for match in _NUMBER.findall(text)}


# ---------------------------------------------------------------------------
# Przebieg i raport


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def secret_values() -> list[str]:
    """Return the secret values configured in .env and the process, longest first."""
    values = dotenv_values(PROJECT_ROOT / ".env")
    secrets = {str(values.get(name) or "") for name in SECRET_SETTINGS}
    secrets |= {os.getenv(name, "") for name in SECRET_SETTINGS}
    return sorted((s for s in secrets if len(s) >= 6), key=len, reverse=True)


def scrub(text: str) -> str:
    """Replace every configured secret value in a text."""
    for secret in secret_values():
        text = text.replace(secret, "[SEKRET]")
    return text


@dataclass
class GoldenRun:
    module: str
    module_dir: Path
    provider: Provider
    case: GoldenCase
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = PASSED
    model: str | None = None
    checks: list[Check] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    retries: int = 0
    log_lines: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        """Record one check; the run fails at the end if any check did not pass."""
        self.checks.append(Check(name, bool(passed), detail))
        if not passed and self.status == PASSED:
            self.status = FAILED_CHECK
        return bool(passed)

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def result_path(self) -> Path:
        name = f"{self.started_at}_{self.provider.label}_{self.case.id}"
        return self.module_dir / "results" / f"{name}.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "case_id": self.case.id,
            "description": self.case.description,
            "provider": self.provider.provider,
            "provider_label": self.provider.label,
            "requested_model": self.provider.model,
            "model": self.model,
            "status": self.status,
            "checks": [c.__dict__ for c in self.checks],
            "duration_seconds": round(self.duration_seconds, 1),
            "retries": self.retries,
            "errors": self.errors,
            "manual_review": self.case.manual_review,
            "limits": self.case.limits,
            "details": self.details,
            "session_id": self.session_id,
        }

    def write(self) -> Path:
        """Write the run report and its log; secrets are replaced before anything is saved."""
        path = self.result_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            scrub(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)),
            encoding="utf-8",
        )
        if self.log_lines:
            path.with_suffix(".log").write_text(
                scrub("\n".join(self.log_lines) + "\n"), encoding="utf-8"
            )
        return path

    def summary(self) -> str:
        failed = "; ".join(f"{c.name}: {c.detail}" for c in self.failed_checks)
        return (
            f"{self.provider.label}-{self.case.id}: {self.status} ({self.model or '-'}, "
            f"{self.duration_seconds:.0f} s, ponowienia {self.retries})"
            + (f" — {failed}" if failed else "")
            + (f" — błędy: {'; '.join(self.errors)}" if self.errors else "")
        )


SESSION_RUNS: list[GoldenRun] = []


@contextmanager
def golden_run(
    module: str, module_dir: Path, provider: Provider, case: GoldenCase
) -> Iterator[GoldenRun]:
    """Time the run, capture the pismak log, classify errors and always write the report."""
    run = GoldenRun(module, module_dir, provider, case)
    logger = logging.getLogger("pismak")
    capture = _Capture()
    previous_level = logger.level
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)
    started = time.perf_counter()
    try:
        yield run
    except Exception as exc:
        run.status = (
            INFRA_ERROR if isinstance(exc, INFRA_ERRORS) or _is_server_error(exc) else FAILED_CHECK
        )
        run.errors.append(f"{type(exc).__name__}: {' '.join(str(exc).split())[:500]}")
        raise
    finally:
        run.duration_seconds = time.perf_counter() - started
        logger.removeHandler(capture)
        logger.setLevel(previous_level)
        run.log_lines = capture.lines
        run.retries = sum("retry_scheduled=true" in line for line in capture.lines)
        if run.model is None:
            models = re.findall(r"llm_request_completed .*?model=(\S+)", "\n".join(capture.lines))
            run.model = ", ".join(dict.fromkeys(models)) or None
        run.write()
        SESSION_RUNS.append(run)
        print(f"\n{run.summary()}\n  raport: {run.result_path.relative_to(PROJECT_ROOT)}")


def _is_server_error(exc: Exception) -> bool:
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


def skip_run(
    module: str, module_dir: Path, provider: Provider, case: GoldenCase, reason: str
) -> None:
    """Record a run that could not start, then skip the test with the reason."""
    run = GoldenRun(module, module_dir, provider, case, status=SKIPPED, errors=[reason])
    run.write()
    SESSION_RUNS.append(run)
    pytest.skip(reason)


def assert_passed(run: GoldenRun) -> None:
    """Fail the test with every failed check listed and the report path."""
    assert run.status == PASSED, (
        f"{run.summary()}\nraport: {run.result_path.relative_to(PROJECT_ROOT)}"
    )
