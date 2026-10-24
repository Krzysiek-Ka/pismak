"""
Testy end-to-end Pismaka na prawdziwych modelach, bazie RAG i konfiguracji z .env.

Sprawy leżą w tests/e2e/cases/<id>.json. Każdy dostawca z konfiguracji projektu obsługuje
wszystkie etapy (Dopytywator, Źródłowyszukiwacz, Pismogenerator, Pismopoprawiator); uruchomienia
mieszane są opcjonalne (PISMAK_E2E_MIXED, zob. e2e_cases.py). Identyfikator testu:
<dostawca>-<sprawa>. Uruchamiane tylko z `-m e2e`. Opcjonalnie PISMAK_E2E_LABEL nazywa przebieg.
Każde uruchomienie zapisuje results/<czas>_<dostawca>_<sprawa>.json; raport zbiorczy
results/<czas>_<etykieta>/report.md jest przepisywany po każdym uruchomieniu, zanim zostaną
sprawdzone asercje, a tabela podsumowania pojawia się też na końcu wyjścia pytest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest
from dotenv import load_dotenv

from src._shared.serwer import LLMConfigurationError
from src.Wyswietlator import services
from src.Zrodlowyszukiwator.rag_ingestion.embedder import OllamaEmbedder
from src.Zrodlowyszukiwator.rag_retrieval import DatabaseConnectionError, RetrievalRepository
from tests.e2e.e2e_cases import CASES, Case, Provider, Setup, run_plan
from tests.e2e.e2e_support import (
    MAX_QUESTIONS,
    PROJECT_ROOT,
    E2ESession,
    environment_header,
    run_case,
    run_id,
)

pytestmark = pytest.mark.e2e


@dataclass
class Environment:
    """What the machine offers; a missing service marks the affected runs as not run."""

    database_url: str | None = None
    documents: int | None = None
    chunks: int | None = None
    common_problem: str | None = None
    ollama_models: list[str] = field(default_factory=list)

    def not_run_reason(self, setup: Setup) -> str | None:
        """Return why a setup cannot start (first unavailable stage), or None when it can."""
        if self.common_problem:
            return self.common_problem
        for stage, provider in setup.stages:
            reason = self._provider_problem(provider)
            if reason:
                return f"{stage}: {reason}"
        return None

    def _provider_problem(self, provider: Provider) -> str | None:
        if provider.provider == "ollama" and provider.model not in self.ollama_models:
            return (
                f"model {provider.model} nie jest zarejestrowany w Ollamie "
                f"(dostępne: {', '.join(self.ollama_models) or 'brak'}); zob. README, punkt 3.1"
            )
        if provider.provider == "openrouter":
            try:
                services.get_llm_client("openrouter")
            except LLMConfigurationError as exc:
                return str(exc)
        return None


def check_environment() -> Environment:
    """Check the database, Ollama and the embedding model shared by every run."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    env = Environment(database_url=os.getenv("DATABASE_URL"))
    if not env.database_url:
        env.common_problem = "brak DATABASE_URL w .env"
        return env
    try:
        env.documents, env.chunks = RetrievalRepository(env.database_url).counts()
    except (DatabaseConnectionError, RuntimeError) as exc:
        env.common_problem = f"baza RAG niedostępna: {exc}"
        return env
    if not env.chunks:
        env.common_problem = "baza RAG jest pusta — najpierw wykonaj rag_ingestion"
        return env

    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    try:
        env.ollama_models = services.list_generation_models(services.get_openai_client())
    except Exception as exc:
        env.common_problem = f"Ollama niedostępna pod {ollama_url}: {type(exc).__name__}"
        return env
    embedding_model = os.getenv("RAG_EMBEDDING_MODEL", "bge-m3")
    try:
        OllamaEmbedder(model=embedding_model, base_url=ollama_url).embed(["test połączenia"])
    except RuntimeError as exc:
        env.common_problem = f"model embeddingów {embedding_model} niedostępny: {exc}"
    return env


@pytest.fixture(scope="session")
def environment() -> Environment:
    return check_environment()


@pytest.fixture(scope="session")
def e2e_session(environment: Environment, request: pytest.FixtureRequest) -> E2ESession:
    session = E2ESession(
        label=os.getenv("PISMAK_E2E_LABEL", "run"),
        header=environment_header(environment.documents, environment.chunks),
        cases=CASES,
    )
    # conftest.py drukuje z niej tabelę podsumowania na końcu wyjścia pytest.
    request.config.stash[E2ESession.KEY] = session
    return session


def test_interview_limit_matches_the_app() -> None:
    """Verify the tested limit is the app's and every case has an answer for each question."""
    assert services.MAX_INTERVIEW_QUESTIONS == MAX_QUESTIONS
    for case in CASES:
        assert case.max_questions == MAX_QUESTIONS, case.id
        assert len(case.answers) == MAX_QUESTIONS, case.id


RUNS = [pytest.param(setup, case, id=run_id(setup, case)) for setup, case in run_plan()]


@pytest.mark.parametrize(("setup", "case"), RUNS)
def test_e2e_pipeline(
    setup: Setup, case: Case, environment: Environment, e2e_session: E2ESession
) -> None:
    """Run description → interview → CaseBrief → RAG → letter and check the invariants."""
    reason = environment.not_run_reason(setup)
    if reason:
        e2e_session.add_not_run(setup, case, reason)
        pytest.skip(f"NIEURUCHOMIONY: {reason}")

    result = run_case(
        setup, case, e2e_session.run_dir(setup, case), database_url=environment.database_url
    )
    e2e_session.add(result)

    print(
        f"\n{result.run_id}: {result.status}, pytań {result.questions_asked}, "
        f"zakończenie {result.end_reason or '-'}, ostrzeżeń {len(result.warnings())}, "
        f"czas {result.timings.get('razem', 0):.0f} s"
    )
    failures = result.hard_failures()
    assert not failures, (
        f"{result.run_id}: "
        + " | ".join(f"{c.id} {c.title}: {c.detail}" for c in failures)
        + f" — raport: {e2e_session.report_path}"
    )
