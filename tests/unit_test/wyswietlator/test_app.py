from __future__ import annotations

import importlib
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit import config
from streamlit.testing.v1 import AppTest

from src._shared import llm_runtime, model_context
from src.Dopytywator.structures import CaseBrief
from src.Zrodlowyszukiwator.brief_to_evidence.models import (
    Evidence,
    EvidencePack,
    ResearchPlan,
)
from src.Zrodlowyszukiwator.rag_retrieval import DatabaseConnectionError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WYSWIETLATOR_DIR = PROJECT_ROOT / "src" / "Wyswietlator"
if str(WYSWIETLATOR_DIR) not in sys.path:
    sys.path.insert(0, str(WYSWIETLATOR_DIR))

services = importlib.import_module("services")

MODEL_ERROR = "szczegół wewnętrzny: host=10.1.2.3 plik=C:\\tajne\\modul.py"
DATABASE_ERROR = (
    "Nie można połączyć się z lokalną bazą PostgreSQL (127.0.0.1:5432). "
    "Uruchom Docker Desktop i `docker compose up -d postgres`."
)


class FailingCompletions:
    def create(self, **kwargs):
        raise RuntimeError(MODEL_ERROR)


@pytest.fixture
def app(monkeypatch, tmp_path):
    """Run the real UI script against fakes, writing logs and outputs to a temporary directory."""
    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    monkeypatch.setenv("PISMAK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(services, "get_openai_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(services, "list_generation_models", lambda client: ["model-testowy"])
    monkeypatch.setattr(model_context, "detect_context_tokens", lambda *args, **kwargs: 8192)
    st.cache_data.clear()
    st.cache_resource.clear()

    app_test = AppTest.from_file(str(WYSWIETLATOR_DIR / "app.py"), default_timeout=30)
    app_test.session_state["cfg_server_storage_consent"] = True
    yield app_test

    detach_log_handler(tmp_path)
    st.cache_data.clear()
    st.cache_resource.clear()


def detach_log_handler(project_root: Path) -> None:
    """Remove the handler that configure_logging attached for a temporary project directory."""
    handler = services._LOG_HANDLERS.pop((project_root / "logs").resolve(), None)
    if handler is not None:
        services.LOGGER.removeHandler(handler)
        logging.getLogger(services.UNCAUGHT_ERROR_LOGGER_NAME).removeHandler(handler)


def shown_text(app_test: AppTest) -> str:
    """Return everything the page tells the user through messages and exception boxes."""
    elements = [*app_test.error, *app_test.warning, *app_test.info, *app_test.exception]
    return "\n".join(str(element.value) for element in elements)


def case_log(project_root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in (project_root / "logs").glob("pismak_*.log")
    )


class TestUnhandledErrors:
    def test_app__unhandled_error_is_hidden_on_the_page_and_written_to_the_shared_log(
        self, tmp_path
    ) -> None:
        """Verify Streamlit's own error handling shows no details and loses none of them."""
        script = (
            "from pathlib import Path\n"
            "import services\n"
            f"services.configure_logging(Path(r'{tmp_path}'), 'case-x')\n"
            "raise RuntimeError('tajny szczegół w C:/sekret/plik.py')\n"
        )
        # Te same wartości ustawia .streamlit/config.toml; tu jawnie, by test nie zależał
        # od katalogu, z którego uruchomiono pytest.
        settings = {"client.showErrorDetails": "none", "logger.enableRich": False}
        originals = {name: config.get_option(name) for name in settings}
        for name, value in settings.items():
            config.set_option(name, value)

        try:
            app_test = AppTest.from_string(script, default_timeout=30).run()

            shown = shown_text(app_test)
            assert len(app_test.exception) == 1
            for hidden in ("tajny szczegół", "sekret", "RuntimeError", "Traceback"):
                assert hidden not in shown

            log = (tmp_path / "logs" / services.UNCAUGHT_ERROR_LOG_NAME).read_text(encoding="utf-8")
            assert "RuntimeError: tajny szczegół w C:/sekret/plik.py" in log
            assert "Traceback (most recent call last)" in log
        finally:
            for name, value in originals.items():
                config.set_option(name, value)
            detach_log_handler(tmp_path)


class TestErrorReporting:
    def test_app__model_failure_shows_a_general_message_and_logs_the_details(
        self, app, tmp_path
    ) -> None:
        """Verify a failed interview turn tells the user nothing internal and the log everything."""
        app.run()
        app.chat_input[0].set_value("Opis sprawy").run()

        shown = shown_text(app)
        assert "Wystąpił błąd podczas przetwarzania sprawy" in shown
        for hidden in ("szczegół wewnętrzny", "10.1.2.3", "modul.py", "Miejsce błędu", "Traceback"):
            assert hidden not in shown

        log = case_log(tmp_path)
        assert "interview_failed" in log
        assert "error_type=RuntimeError" in log
        assert re.search(r"error_location=\S+\.py:\d+ in create ", log)
        assert MODEL_ERROR in log
        assert "Traceback (most recent call last)" in log

    def test_app__database_failure_shows_a_general_message_and_logs_host_and_cause(
        self, app, tmp_path, monkeypatch
    ) -> None:
        """Verify an unreachable database is named to the operator only, with its root cause."""

        def unavailable(**kwargs):
            try:
                raise ConnectionRefusedError("connection to server at 127.0.0.1 refused")
            except ConnectionRefusedError as cause:
                raise DatabaseConnectionError(DATABASE_ERROR) from cause

        monkeypatch.setattr(services, "build_zrodlowyszukiwator", unavailable)
        app.session_state["case_brief"] = CaseBrief(
            title="Sprawa",
            problem_description="Problem",
            desired_outcome="Rozwiązanie",
            requested_actions=["Sprawdzenie"],
        )
        app.run()
        next(b for b in app.button if b.label == "Znajdź źródła i wygeneruj pismo").click().run()

        shown = shown_text(app)
        assert "Źródłowyszukiwacz jest obecnie niedostępny" in shown
        for hidden in ("127.0.0.1", "5432", "Docker", "PostgreSQL", "Miejsce błędu"):
            assert hidden not in shown

        log = case_log(tmp_path)
        assert "pipeline_failed" in log
        assert "error_type=DatabaseConnectionError" in log
        assert DATABASE_ERROR in log
        assert "ConnectionRefusedError: connection to server at 127.0.0.1 refused" in log


INTERVIEW = "InterviewTurnResult"
BRIEF = "CaseBrief"
DRAFT = "DraftLetter"
CORRECTION = "CorrectionResponse"
GENERATE_LABEL = "Znajdź źródła i wygeneruj pismo"

QUESTION = {
    "short_summary": "Brak przejścia przy szkole.",
    "next_question": "Przy której szkole brakuje przejścia?",
    "reason_for_question": "Potrzebna lokalizacja.",
    "missing_topics": ["lokalizacja"],
    "ready_to_build_brief": False,
}
READY = {
    "short_summary": "Sprawa jest jasna.",
    "next_question": None,
    "reason_for_question": None,
    "missing_topics": [],
    "ready_to_build_brief": True,
}
CASE_BRIEF = {
    "title": "Przejście przy szkole",
    "problem_description": "Dzieci przechodzą przez ulicę bez przejścia.",
    "desired_outcome": "Wyznaczenie przejścia dla pieszych.",
    "requested_actions": ["Wyznaczenie przejścia"],
}
VALID_DRAFT = {
    "recipient": "Zarząd Dróg i Utrzymania Miasta",
    "request_summary": "wyznaczenie przejścia dla pieszych przy szkole",
    "argumentation_markdown": (
        "Dzieci codziennie przechodzą przez ruchliwą ulicę w miejscu bez oznakowania. "
        "Polityka mobilności miasta stawia bezpieczeństwo pieszych na pierwszym miejscu [1]."
    ),
    "sources": ["E1"],
}


class ScriptedCompletions:
    """Answer each structured request with the next scripted reply for its response model."""

    def __init__(self, replies: dict[str, list[object]]) -> None:
        self.replies = {name: list(items) for name, items in replies.items()}
        self.calls: list[str] = []

    def create(self, **kwargs):
        name = kwargs["response_format"]["json_schema"]["name"]
        self.calls.append(name)
        reply = self.replies[name].pop(0)
        finish_reason = "stop"
        if isinstance(reply, tuple):
            reply, finish_reason = reply
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return SimpleNamespace(
            model=f"fake/{name}",
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    message=SimpleNamespace(content=content),
                )
            ],
        )


def make_evidence_pack(*, evidence: int = 1, status: str = "supported") -> EvidencePack:
    return EvidencePack(
        query_summary="Przejście przy szkole",
        research_plan=ResearchPlan(case_summary="Przejście przy szkole", intents=[]),
        evidence=[
            Evidence(
                evidence_id=f"E{index}",
                chunk_id=f"chunk-{index}",
                document_id="mobilnosc",
                document_title="Wrocławska polityka mobilności",
                document_kind="policy",
                hierarchy_rank=100,
                legal_form="resolution",
                page_start=3,
                page_end=4,
                section_path=["Cele", "Bezpieczeństwo"],
                excerpt="Bezpieczeństwo pieszych jest priorytetem.",
                retrieval_score=0.9,
                relevance_reason="Dotyczy bezpieczeństwa pieszych.",
            )
            for index in range(1, evidence + 1)
        ],
        evidence_status=status,
    )


class FakePipeline:
    def __init__(self, result: EvidencePack | Exception) -> None:
        self.result = result
        self.planner = SimpleNamespace(last_model="fake/planner")
        self.reranker = SimpleNamespace(last_model=None)
        self.briefs: list[CaseBrief] = []

    def run(self, brief: CaseBrief) -> EvidencePack:
        self.briefs.append(brief)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def ui(monkeypatch, tmp_path):
    """Return a factory that starts the real UI script with a scripted model and fake services."""

    def start(
        replies: dict[str, list[object]] | None = None,
        *,
        models: list[str] | Exception = ("model-testowy",),
        storage_consent: bool = True,
        korekta: bool | None = False,
    ) -> tuple[AppTest, ScriptedCompletions]:
        completions = ScriptedCompletions(replies or {})
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        # Pismopoprawiator ustawia własny limit czasu przez with_options klienta OpenAI.
        client.with_options = lambda **kwargs: client

        def list_models(client):
            if isinstance(models, Exception):
                raise models
            return list(models)

        monkeypatch.setattr(services, "get_openai_client", lambda *args, **kwargs: client)
        monkeypatch.setattr(services, "list_generation_models", list_models)
        st.cache_data.clear()
        st.cache_resource.clear()
        app_test = AppTest.from_file(str(WYSWIETLATOR_DIR / "app.py"), default_timeout=30)
        app_test.session_state["cfg_server_storage_consent"] = storage_consent
        # Korekta językowa jest domyślnie włączona; testy innych etapów ją wyłączają, żeby
        # sprawdzać dokładną listę wywołań. None zostawia ustawienie aplikacji.
        if korekta is not None:
            app_test.session_state["cfg_korekta"] = korekta
        return app_test, completions

    monkeypatch.setenv("PISMAK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(model_context, "detect_context_tokens", lambda *args, **kwargs: 8192)
    yield start

    detach_log_handler(tmp_path)
    st.cache_data.clear()
    st.cache_resource.clear()


def run_dirs(project_root: Path) -> list[Path]:
    return sorted((project_root / "outputs").glob("*_ui"))


def chat_messages(app_test: AppTest) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in app_test.session_state["messages"]]


def with_brief(app_test: AppTest, **overrides: object) -> AppTest:
    app_test.session_state["case_brief"] = CaseBrief.model_validate({**CASE_BRIEF, **overrides})
    return app_test


def click(app_test: AppTest, label: str) -> AppTest:
    return next(b for b in app_test.button if b.label == label).click().run()


class TestFirstRender:
    def test_app__first_render_offers_a_new_case_without_errors(self, ui) -> None:
        """Verify an opened page is ready for a case description and calls no model yet."""
        app_test, completions = ui()

        app_test.run()

        assert not app_test.exception
        assert app_test.title[0].value == "Pismak - twój asystent w pisaniu wniosków"
        assert app_test.chat_input[0].placeholder == "Opisz sprawę własnymi słowami..."
        assert not app_test.chat_input[0].disabled
        assert app_test.selectbox(key="cfg_dopytywator_model").options == ["model-testowy"]
        assert app_test.checkbox(key="cfg_server_storage_consent") is not None
        assert GENERATE_LABEL not in [b.label for b in app_test.button]
        assert "OpenRouter wysyła treść" not in shown_text(app_test)
        assert re.fullmatch(r"[0-9a-f]{32}", app_test.session_state["session_id"])
        assert app_test.session_state["stage"] == "INTERVIEW"
        assert completions.calls == []

    def test_app__first_render_warns_that_refreshing_the_page_starts_a_new_case(self, ui) -> None:
        """Verify the user learns before writing that the case lives only in this browser tab."""
        app_test, _ = ui()

        app_test.run()

        captions = [c.value for c in app_test.caption]
        assert any("Odświeżenie strony rozpoczyna nową sprawę" in c for c in captions)

    def test_app__first_render_without_generation_models_explains_what_is_missing(self, ui) -> None:
        """Verify an Ollama with only embedding models blocks the interview with a clear hint."""
        app_test, completions = ui(models=[])

        app_test.run()
        app_test.chat_input[0].set_value("Opis sprawy").run()

        shown = shown_text(app_test)
        assert "Brak modeli generatywnych w Ollama" in shown
        assert "Wybierz dostawcę Ollama z dostępnym modelem albo OpenRouter." in shown
        assert app_test.session_state["messages"] == []
        assert completions.calls == []

    def test_app__first_render_with_ollama_down_shows_a_general_message_and_logs_it(
        self, ui, tmp_path
    ) -> None:
        """Verify a failing model list neither crashes the page nor shows connection details."""
        app_test, _ = ui(models=ConnectionError("connection refused 127.0.0.1:11434"))

        app_test.run()

        shown = shown_text(app_test)
        assert not app_test.exception
        assert "Nie udało się pobrać modeli Ollama. Sprawdź, czy Ollama działa." in shown
        assert "11434" not in shown
        assert "model_list_failed" in case_log(tmp_path)


class TestConsent:
    def test_app__message_is_not_processed_without_server_storage_consent(
        self, ui, tmp_path
    ) -> None:
        """Verify nothing is sent, kept in the session, or written to disk before consent."""
        app_test, completions = ui(storage_consent=False)

        app_test.run()
        app_test.chat_input[0].set_value("Opis sprawy").run()

        assert "Potwierdź zapoznanie się z informacją o zapisie danych" in shown_text(app_test)
        assert app_test.session_state["messages"] == []
        assert completions.calls == []
        assert run_dirs(tmp_path) == []

    def test_app__openrouter_requires_its_own_consent_before_the_interview(
        self, ui, monkeypatch
    ) -> None:
        """Verify selecting the external provider warns and blocks sending until consent."""

        def must_not_connect(provider):
            raise AssertionError("OpenRouter must not be contacted without consent")

        monkeypatch.setattr(services, "get_llm_client", must_not_connect)
        app_test, _ = ui()
        app_test.run()

        app_test.selectbox(key="cfg_dopytywator_provider").set_value("openrouter").run()
        app_test.chat_input[0].set_value("Opis sprawy").run()

        shown = shown_text(app_test)
        assert "OpenRouter wysyła treść zapytań do usługi zewnętrznej" in shown
        assert "Potwierdź zgodę na przetwarzanie zewnętrzne" in shown
        assert app_test.session_state["messages"] == []

    def test_app__openrouter_with_consent_runs_the_interview_on_the_selected_provider(
        self, ui, monkeypatch
    ) -> None:
        """Verify consent unlocks the external provider and the run records it as used."""
        app_test, completions = ui()
        remote = SimpleNamespace(
            chat=SimpleNamespace(completions=ScriptedCompletions({INTERVIEW: [QUESTION]}))
        )
        requested: list[str] = []

        def remote_client(provider):
            requested.append(provider)
            return remote

        monkeypatch.setattr(services, "get_llm_client", remote_client)
        app_test.run()
        app_test.selectbox(key="cfg_dopytywator_provider").set_value("openrouter").run()
        app_test.checkbox(key="cfg_external_processing_consent").check().run()

        app_test.chat_input[0].set_value("Opis sprawy").run()

        assert requested == ["openrouter"]
        assert completions.calls == []
        assert chat_messages(app_test)[-1] == ("assistant", QUESTION["next_question"])
        assert app_test.session_state["model_usage"]["Dopytywator"] == [
            {"provider": "openrouter", "model": "fake/InterviewTurnResult"}
        ]

    def test_app__pipeline_requires_consent_when_only_the_sources_use_openrouter(
        self, ui, monkeypatch
    ) -> None:
        """Verify the external-processing check covers later stages, not only the interview."""
        built: list[object] = []
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: built.append(kw))
        app_test, _ = ui()
        app_test.run()
        app_test.selectbox(key="cfg_zrodla_provider").set_value("openrouter").run()
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        assert "Potwierdź zgodę na przetwarzanie zewnętrzne" in shown_text(app_test)
        assert built == []
        assert app_test.session_state["letter_markdown"] is None


class TestMessageLimits:
    def test_app__message_over_the_limit_is_rejected_and_not_added_to_the_case(self, ui) -> None:
        """Verify an oversized message is refused before it enters the history or the model."""
        app_test, completions = ui()
        app_test.run()

        app_test.chat_input[0].set_value("x" * (services.MAX_USER_MESSAGE_CHARS + 1)).run()

        assert "Wiadomość jest zbyt długa (limit: 5000 znaków)." in shown_text(app_test)
        assert app_test.session_state["messages"] == []
        assert completions.calls == []

    def test_app__message_exactly_at_the_limit_is_accepted(self, ui) -> None:
        """Verify the advertised maximum is usable for a long first description."""
        app_test, completions = ui({INTERVIEW: [QUESTION]})
        app_test.run()

        app_test.chat_input[0].set_value("x" * services.MAX_USER_MESSAGE_CHARS).run()

        assert completions.calls == [INTERVIEW]
        assert chat_messages(app_test)[0] == ("user", "x" * services.MAX_USER_MESSAGE_CHARS)

    def test_app__answer_that_overflows_the_model_window_is_rejected(self, ui) -> None:
        """Verify the whole conversation, not just one message, must fit the model window."""
        app_test, completions = ui()
        history = [
            {"role": "user", "content": "a" * 4000},
            {"role": "assistant", "content": "Pytanie?"},
            {"role": "user", "content": "b" * 4000},
        ]
        app_test.session_state["messages"] = list(history)
        app_test.session_state["question_count"] = 1
        app_test.run()

        app_test.chat_input[0].set_value("c" * 4000).run()

        assert "Rozmowa jest zbyt długa dla wybranego modelu" in shown_text(app_test)
        assert app_test.session_state["messages"] == history
        assert completions.calls == []


class TestInterview:
    def test_app__model_question_is_added_and_locks_the_interview_settings(self, ui) -> None:
        """Verify a question is shown and provider, model and limit cannot change mid-interview."""
        app_test, completions = ui({INTERVIEW: [QUESTION]})
        app_test.run()
        app_test.number_input[0].set_value(3).run()

        app_test.chat_input[0].set_value("  Brak przejścia przy szkole  ").run()

        assert chat_messages(app_test) == [
            ("user", "Brak przejścia przy szkole"),
            ("assistant", QUESTION["next_question"]),
        ]
        assert app_test.session_state["question_count"] == 1
        assert app_test.session_state["case_brief"] is None
        assert app_test.selectbox(key="cfg_dopytywator_provider").disabled
        assert app_test.selectbox(key="cfg_dopytywator_model").disabled
        assert app_test.number_input[0].disabled
        assert app_test.chat_input[0].placeholder == "Odpowiedz Dopytywatorowi..."
        assert completions.calls == [INTERVIEW]

    def test_app__links_and_images_in_a_model_question_are_shown_as_text(self, ui) -> None:
        """Verify a model question cannot make the browser load an image or hide a link address."""
        asked = "Czy to ta szkoła? ![](http://tracker.test/p.png) [mapa](http://phish.test)"
        app_test, _ = ui({INTERVIEW: [{**QUESTION, "next_question": asked}]})
        app_test.run()

        app_test.chat_input[0].set_value("Brak przejścia").run()

        rendered = [m.value for m in app_test.markdown]
        assert (
            "Czy to ta szkoła? !\\[](http://tracker.test/p.png) \\[mapa](http://phish.test)"
            in rendered
        )
        assert chat_messages(app_test)[-1] == ("assistant", asked)

    def test_app__ready_answer_builds_the_brief_and_saves_the_run(self, ui, tmp_path) -> None:
        """Verify a complete interview produces a brief, closes the chat and stores the case."""
        app_test, completions = ui({INTERVIEW: [READY], BRIEF: [CASE_BRIEF]})
        app_test.run()

        app_test.chat_input[0].set_value("Brak przejścia przy SP 1").run()

        assert completions.calls == [INTERVIEW, BRIEF]
        assert app_test.session_state["stage"] == "BRIEF_READY"
        assert app_test.session_state["case_brief"].title == CASE_BRIEF["title"]
        assert chat_messages(app_test)[-1][1].startswith("Mam wystarczające informacje")
        assert app_test.chat_input[0].disabled
        assert GENERATE_LABEL in [b.label for b in app_test.button]
        [run_dir] = run_dirs(tmp_path)
        assert app_test.session_state["run_dir"] == run_dir
        assert (
            json.loads((run_dir / "brief.json").read_text(encoding="utf-8"))["title"]
            == (CASE_BRIEF["title"])
        )
        assert json.loads((run_dir / "models.json").read_text(encoding="utf-8")) == {
            "Dopytywator": [
                {"provider": "ollama", "model": "fake/InterviewTurnResult"},
                {"provider": "ollama", "model": "fake/CaseBrief"},
            ]
        }

    def test_app__reaching_the_question_limit_builds_the_brief_from_the_next_answer(
        self, ui
    ) -> None:
        """Verify the answer to the last allowed question goes straight to the brief."""
        app_test, completions = ui({INTERVIEW: [QUESTION], BRIEF: [CASE_BRIEF]})
        app_test.run()

        app_test.chat_input[0].set_value("Brak przejścia").run()
        app_test.chat_input[0].set_value("Przy SP 1").run()

        assert completions.calls == [INTERVIEW, BRIEF]
        assert app_test.session_state["question_count"] == 1
        assert app_test.session_state["stage"] == "BRIEF_READY"

    def test_app__question_limit_set_before_the_interview_still_applies_once_locked(
        self, ui
    ) -> None:
        """Verify the disabled 'Max pytań' field keeps the chosen limit (acceptance test A7)."""
        second_question = {**QUESTION, "next_question": "Od kiedy trwa problem?"}
        app_test, completions = ui({INTERVIEW: [QUESTION, second_question], BRIEF: [CASE_BRIEF]})
        app_test.run()
        app_test.number_input[0].set_value(2).run()

        app_test.chat_input[0].set_value("Brak przejścia").run()
        app_test.chat_input[0].set_value("Przy SP 1").run()

        assert app_test.number_input[0].value == 2
        assert completions.calls == [INTERVIEW, INTERVIEW]
        assert app_test.session_state["question_count"] == 2

    def test_app__brief_with_gaps_is_kept_and_the_gaps_are_reported(self, ui) -> None:
        """Verify an incomplete brief does not block the case but tells the user what is missing."""
        app_test, _ = ui({INTERVIEW: [READY], BRIEF: [{**CASE_BRIEF, "requested_actions": [" "]}]})
        app_test.run()

        app_test.chat_input[0].set_value("Brak przejścia").run()

        assert app_test.session_state["case_brief"] is not None
        assert chat_messages(app_test)[-1] == (
            "assistant",
            "Brief został utworzony, ale ma braki: Brak co najmniej jednego konkretnego postulatu",
        )

    def test_app__truncated_model_answer_shows_a_warning_instead_of_an_error(
        self, ui, tmp_path
    ) -> None:
        """Verify an output-limit stop is explained as truncation and logged as such."""
        app_test, completions = ui({INTERVIEW: [('{"short_summary": "Spra', "length")] * 2})
        app_test.run()

        app_test.chat_input[0].set_value("Opis sprawy").run()

        assert completions.calls == [INTERVIEW, INTERVIEW]
        assert any("Odpowiedź modelu została ucięta" in w.value for w in app_test.warning)
        assert not app_test.error
        assert "interview_output_truncated" in case_log(tmp_path)

    def test_app__invalid_model_answer_after_repair_shows_a_general_message(
        self, ui, tmp_path
    ) -> None:
        """Verify a model that twice breaks the schema gets one repair and then a clear message."""
        app_test, completions = ui({INTERVIEW: ["to nie jest JSON", '{"short_summary": "x"}']})
        app_test.run()

        app_test.chat_input[0].set_value("Opis sprawy").run()

        assert completions.calls == [INTERVIEW, INTERVIEW]
        assert "Model nie zwrócił poprawnych danych" in shown_text(app_test)
        assert app_test.session_state["question_count"] == 0
        assert "error_type=LLMResponseError" in case_log(tmp_path)

    def test_app__second_case_gets_a_busy_message_instead_of_waiting_forever(
        self, ui, tmp_path, monkeypatch
    ) -> None:
        """Verify a case queued behind another generation is told to retry after the wait limit."""
        monkeypatch.setattr(llm_runtime, "LLM_LOCK_WAIT_TIMEOUT_SECONDS", 0.1)
        app_test, completions = ui({INTERVIEW: [QUESTION]})
        app_test.run()

        with llm_runtime.llm_generation_lock():
            app_test.chat_input[0].set_value("Opis sprawy").run()

        shown = shown_text(app_test)
        assert "Lokalny model jest zajęty obsługą innych spraw" in shown
        assert completions.calls == []
        assert app_test.session_state["question_count"] == 0
        assert "llm_wait_timeout stage=dopytywator" in case_log(tmp_path)
        assert "error_type=LLMBusyError" in case_log(tmp_path)


class TestNewCase:
    def test_app__new_case_clears_the_case_and_starts_a_new_session(self, ui) -> None:
        """Verify 'Nowa sprawa' drops the case data but keeps the chosen provider settings."""
        app_test, _ = ui({INTERVIEW: [QUESTION]})
        app_test.run()
        app_test.chat_input[0].set_value("Opis sprawy").run()
        first_session = app_test.session_state["session_id"]

        click(app_test, "Nowa sprawa")

        assert app_test.session_state["messages"] == []
        assert app_test.session_state["question_count"] == 0
        assert app_test.session_state["session_id"] != first_session
        assert app_test.session_state["cfg_server_storage_consent"] is True
        assert not app_test.selectbox(key="cfg_dopytywator_provider").disabled
        assert app_test.chat_input[0].placeholder == "Opisz sprawę własnymi słowami..."


class TestSessionIsolation:
    def test_app__two_sessions_keep_their_conversations_logs_and_outputs_apart(
        self, ui, tmp_path
    ) -> None:
        """Verify two users of one server never see or overwrite each other's case."""
        first, _ = ui({INTERVIEW: [READY], BRIEF: [{**CASE_BRIEF, "title": "Sprawa A"}]})
        first.run()
        first.chat_input[0].set_value("Opis sprawy A").run()
        second, _ = ui({INTERVIEW: [READY], BRIEF: [{**CASE_BRIEF, "title": "Sprawa B"}]})
        second.run()

        assert second.session_state["messages"] == []
        assert second.session_state["case_brief"] is None

        second.chat_input[0].set_value("Opis sprawy B").run()

        assert first.session_state["session_id"] != second.session_state["session_id"]
        assert "Opis sprawy B" not in str(first.session_state["messages"])
        assert "Opis sprawy A" not in str(second.session_state["messages"])
        assert first.session_state["case_brief"].title == "Sprawa A"
        assert second.session_state["case_brief"].title == "Sprawa B"
        assert first.session_state["run_dir"] != second.session_state["run_dir"]
        assert len(run_dirs(tmp_path)) == 2
        logs = [
            path.read_text(encoding="utf-8") for path in (tmp_path / "logs").glob("pismak_*.log")
        ]
        assert len(logs) == 2
        for session in (first, second):
            session_id = session.session_state["session_id"]
            assert sum(session_id in log for log in logs) == 1


class TestPipeline:
    def test_app__pipeline_renders_the_letter_and_saves_every_artifact(
        self, ui, tmp_path, monkeypatch
    ) -> None:
        """Verify one click turns the brief into a letter with its sources and a stored run."""
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, completions = ui({DRAFT: [VALID_DRAFT]})
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        assert not app_test.exception
        assert completions.calls == [DRAFT]
        assert pipeline.briefs[0].title == CASE_BRIEF["title"]
        letter = app_test.session_state["letter_markdown"]
        assert "**Adresat:** Zarząd Dróg i Utrzymania Miasta" in letter
        assert "[1] E1: Wrocławska polityka mobilności, s. 3-4" in letter
        assert app_test.session_state["stage"] == "DONE"
        assert GENERATE_LABEL not in [b.label for b in app_test.button]
        assert "Projekt pisma" in [h.value for h in app_test.subheader]
        [run_dir] = run_dirs(tmp_path)
        for name in ("conversation.json", "brief.json", "evidence.json", "draft.json", "letter.md"):
            assert (run_dir / name).exists(), name
        assert (run_dir / "letter.md").read_text(encoding="utf-8") == letter
        models = json.loads((run_dir / "models.json").read_text(encoding="utf-8"))
        assert models["Planner"] == [{"provider": "ollama", "model": "fake/planner"}]
        assert "Reranker" not in models
        assert models["Pismogenerator"] == [{"provider": "ollama", "model": "fake/DraftLetter"}]

    def test_app__letter_shows_model_links_as_text_but_keeps_them_in_the_file(
        self, ui, tmp_path, monkeypatch
    ) -> None:
        """Verify the page loads no image from the letter while the saved letter is unchanged."""
        image = "![](http://tracker.test/p.png)"
        draft = {
            **VALID_DRAFT,
            "argumentation_markdown": VALID_DRAFT["argumentation_markdown"] + " " + image,
        }
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, _ = ui({DRAFT: [draft]})
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        letter = app_test.session_state["letter_markdown"]
        rendered = [m.value for m in app_test.markdown]
        assert image in letter
        assert services.escape_markdown_links(letter) in rendered
        assert not any(image in value for value in rendered)
        [run_dir] = run_dirs(tmp_path)
        assert image in (run_dir / "letter.md").read_text(encoding="utf-8")

    def test_app__rejected_draft_is_saved_for_diagnosis_and_can_be_retried(
        self, ui, tmp_path, monkeypatch
    ) -> None:
        """Verify a draft failing the app checks is kept on disk, explained, and not shown."""
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, _ = ui({DRAFT: [{**VALID_DRAFT, "argumentation_markdown": "Za krótko [1]."}]})
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        assert "nie przeszedł kontroli aplikacji" in shown_text(app_test)
        assert app_test.session_state["letter_markdown"] is None
        assert GENERATE_LABEL in [b.label for b in app_test.button]
        [run_dir] = run_dirs(tmp_path)
        saved = json.loads((run_dir / "draft_with_error.json").read_text(encoding="utf-8"))
        assert saved["draft"]["argumentation_markdown"] == "Za krótko [1]."
        assert saved["validation_errors"] == ["Argumentacja projektu jest podejrzanie krótka"]
        assert not (run_dir / "letter.md").exists()

    def test_app__unknown_source_id_is_repaired_instead_of_rejecting_the_letter(
        self, ui, monkeypatch
    ) -> None:
        """Verify a source outside the schema triggers one repair request and the letter is made."""
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, completions = ui({DRAFT: [{**VALID_DRAFT, "sources": ["1"]}, VALID_DRAFT]})
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        assert completions.calls == [DRAFT, DRAFT]
        assert "[1] E1: Wrocławska polityka mobilności" in app_test.session_state["letter_markdown"]

    def test_app__retrieval_failure_still_records_the_planner_model_and_saves_the_case(
        self, ui, tmp_path, monkeypatch
    ) -> None:
        """Verify diagnostics about the models used survive a failure inside the RAG stage."""
        pipeline = FakePipeline(RuntimeError("embedding service down"))
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, completions = ui()
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        assert "Wystąpił błąd podczas przetwarzania sprawy" in shown_text(app_test)
        assert "embedding service down" not in shown_text(app_test)
        assert completions.calls == []
        [run_dir] = run_dirs(tmp_path)
        models = json.loads((run_dir / "models.json").read_text(encoding="utf-8"))
        assert models == {"Planner": [{"provider": "ollama", "model": "fake/planner"}]}

    def test_app__after_a_failed_search_the_sources_provider_can_be_changed_and_retried(
        self, ui, monkeypatch
    ) -> None:
        """Verify the advice to pick another model can be followed without losing the case."""
        providers: list[str] = []
        outcomes = [RuntimeError("planner zwrócił zły JSON"), make_evidence_pack()]

        def build(**kwargs):
            providers.append(kwargs["provider"])
            return FakePipeline(outcomes.pop(0)), 2, 63

        monkeypatch.setattr(services, "build_zrodlowyszukiwator", build)
        monkeypatch.setenv("OPENROUTER_KEY", "sk-or-v1-klucz-testowy")
        app_test, _ = ui({DRAFT: [VALID_DRAFT]})
        app_test.session_state["cfg_external_processing_consent"] = True
        app_test.session_state["cfg_zrodla_provider"] = "openrouter"
        with_brief(app_test).run()
        click(app_test, GENERATE_LABEL)
        assert not app_test.selectbox(key="cfg_zrodla_provider").disabled

        app_test.selectbox(key="cfg_zrodla_provider").set_value("ollama").run()
        click(app_test, GENERATE_LABEL)

        assert providers == ["openrouter", "ollama"]
        assert app_test.session_state["letter_markdown"] is not None
        assert app_test.selectbox(key="cfg_zrodla_provider").disabled

    def test_app__no_matching_sources_is_explained_and_the_letter_has_none(
        self, ui, monkeypatch
    ) -> None:
        """Verify an unsupported case says why no source was used instead of inventing one."""
        pipeline = FakePipeline(make_evidence_pack(evidence=0, status="unsupported"))
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        draft = {**VALID_DRAFT, "sources": [], "argumentation_markdown": "Uzasadnienie. " * 10}
        app_test, _ = ui({DRAFT: [draft]})
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        assert "Nie znaleziono źródeł, które odpowiadają na sprawę." in shown_text(app_test)
        assert (
            "Brak źródeł wskazanych przez użytkownika."
            in (app_test.session_state["letter_markdown"])
        )

    def test_app__unassessed_sources_are_labelled_as_not_checked(self, ui, monkeypatch) -> None:
        """Verify sources without a reranker verdict are not presented as confirmed."""
        pipeline = FakePipeline(make_evidence_pack(status="unverified"))
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, _ = ui({DRAFT: [VALID_DRAFT]})
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        captions = [c.value for c in app_test.caption]
        assert any("Trafność fragmentów nie została oceniona" in c for c in captions)
        assert "Cele > Bezpieczeństwo" in captions


def correction_fix(oryginal: str, poprawka: str, kategoria: str = "ortografia") -> dict:
    return {
        "akapit": "A5",
        "oryginal": oryginal,
        "poprawka": poprawka,
        "kategoria": kategoria,
        "uzasadnienie": "błąd językowy",
    }


TYPO_DRAFT = {
    **VALID_DRAFT,
    "argumentation_markdown": VALID_DRAFT["argumentation_markdown"].replace(
        "przechodzą", "przechodza"
    ),
}


class TestPismopoprawiator:
    def test_app__correction_is_on_by_default_and_both_versions_are_kept(
        self, ui, tmp_path, monkeypatch
    ) -> None:
        """Verify a new case corrects the letter and keeps the uncorrected version available."""
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        answer = {
            "poprawki": [
                correction_fix("codziennie przechodza", "codziennie przechodzą"),
                correction_fix("ruchliwą ulicę", "ruchliwą ulicę.", "styl"),
            ]
        }
        app_test, completions = ui({DRAFT: [TYPO_DRAFT], CORRECTION: [answer]}, korekta=None)
        with_brief(app_test).run()

        assert app_test.toggle(key="cfg_korekta").value is True
        click(app_test, GENERATE_LABEL)

        assert not app_test.exception
        assert completions.calls == [DRAFT, CORRECTION]
        letter = app_test.session_state["letter_markdown"]
        original = app_test.session_state["letter_original"]
        assert "codziennie przechodzą" in letter
        assert "codziennie przechodza" in original
        assert app_test.session_state["correction"].status == "corrected"
        assert [b.label for b in app_test.get("download_button")] == [
            "Pobierz pismo.md (po korekcie)",
            "Pobierz pismo_bez_korekty.md",
        ]
        assert [t.label for t in app_test.tabs] == ["Wersja poprawiona", "Bez korekty", "Różnice"]
        assert any("Korekta językowa: wprowadzono poprawki" in c.value for c in app_test.caption)
        [run_dir] = run_dirs(tmp_path)
        assert (run_dir / "letter.md").read_text(encoding="utf-8") == letter
        assert (run_dir / "letter_bez_korekty.md").read_text(encoding="utf-8") == original
        saved = json.loads((run_dir / "korekta.json").read_text(encoding="utf-8"))
        assert saved["status"] == "corrected"
        assert saved["applied"][0]["poprawka"] == "codziennie przechodzą"
        assert saved["suggestions"][0]["kategoria"] == "styl"
        assert "original_text" not in saved
        models = json.loads((run_dir / "models.json").read_text(encoding="utf-8"))
        assert models["Pismopoprawiator"] == [
            {"provider": "ollama", "model": "fake/CorrectionResponse"}
        ]

    def test_app__diff_view_lists_applied_and_rejected_corrections(self, ui, monkeypatch) -> None:
        """Verify the user sees what changed, what was refused and why."""
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        answer = {
            "poprawki": [
                correction_fix("codziennie przechodza", "codziennie przechodzą"),
                correction_fix("miejscu [1]", "miejscu [2]"),
            ]
        }
        app_test, _ = ui({DRAFT: [TYPO_DRAFT], CORRECTION: [answer]}, korekta=True)
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        [diff] = [c.value for c in app_test.code if c.language == "diff"]
        assert "-Dzieci codziennie przechodza" in diff
        assert "+Dzieci codziennie przechodzą" in diff
        [applied, rejected] = [frame.value for frame in app_test.dataframe]
        assert list(applied["Jest"]) == ["codziennie przechodzą"]
        assert list(rejected["Powód odrzucenia"]) == [
            "zmienia liczby i daty: usuwa „1”; dodaje „2”"
        ]
        assert "Odrzucone poprawki (1)" in [e.label for e in app_test.expander]

    def test_app__failed_correction_keeps_the_letter_without_an_error(
        self, ui, tmp_path, monkeypatch
    ) -> None:
        """Verify a broken correction answer never stops the pipeline or hides the letter."""
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, completions = ui(
            {DRAFT: [VALID_DRAFT], CORRECTION: ["nie JSON", "nadal nie JSON"]}, korekta=True
        )
        with_brief(app_test).run()

        click(app_test, GENERATE_LABEL)

        assert not app_test.exception
        assert not app_test.error
        assert completions.calls == [DRAFT, CORRECTION, CORRECTION]
        correction = app_test.session_state["correction"]
        assert correction.status == "failed"
        assert app_test.session_state["stage"] == "DONE"
        assert app_test.session_state["letter_markdown"] == correction.original_text
        assert any("nie powiodła się, pismo bez korekty" in c.value for c in app_test.caption)
        assert [b.label for b in app_test.get("download_button")] == ["Pobierz pismo.md"]
        models = json.loads((run_dirs(tmp_path)[0] / "models.json").read_text(encoding="utf-8"))
        assert "Pismopoprawiator" not in models

    def test_app__switched_off_correction_calls_no_model_and_offers_one_file(
        self, ui, tmp_path, monkeypatch
    ) -> None:
        """Verify turning the correction off restores the two-stage pipeline exactly."""
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, completions = ui({DRAFT: [VALID_DRAFT]}, korekta=True)
        with_brief(app_test).run()

        app_test.toggle(key="cfg_korekta").set_value(False).run()
        click(app_test, GENERATE_LABEL)

        assert completions.calls == [DRAFT]
        assert app_test.session_state["correction"] is None
        assert [b.label for b in app_test.get("download_button")] == ["Pobierz pismo.md"]
        assert not (run_dirs(tmp_path)[0] / "korekta.json").exists()
        assert app_test.toggle(key="cfg_korekta").disabled

    def test_app__openrouter_correction_needs_consent_before_the_pipeline(
        self, ui, monkeypatch
    ) -> None:
        """Verify the letter is not sent out for correction without the external consent."""
        pipeline = FakePipeline(make_evidence_pack())
        monkeypatch.setattr(services, "build_zrodlowyszukiwator", lambda **kw: (pipeline, 2, 63))
        app_test, completions = ui({DRAFT: [VALID_DRAFT]}, korekta=True)
        with_brief(app_test).run()

        app_test.selectbox(key="cfg_poprawiator_provider").set_value("openrouter").run()
        click(app_test, GENERATE_LABEL)

        assert "Potwierdź zgodę na przetwarzanie zewnętrzne" in shown_text(app_test)
        assert completions.calls == []
        assert pipeline.briefs == []
