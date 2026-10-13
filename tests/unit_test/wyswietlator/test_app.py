from __future__ import annotations

import importlib
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit import config
from streamlit.testing.v1 import AppTest

from src._shared import model_context
from src.Dopytywator.structures import CaseBrief
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

    if "session_id" in app_test.session_state:
        services.close_case_logging(app_test.session_state["session_id"])
    close_uncaught_error_log(tmp_path)
    st.cache_data.clear()
    st.cache_resource.clear()


def close_uncaught_error_log(project_root: Path) -> None:
    """Release the shared error-log file that configure_logging opened for a test directory."""
    handler = services._UNCAUGHT_ERROR_LOG_HANDLERS.pop((project_root / "logs").resolve(), None)
    if handler is not None:
        logging.getLogger(services.UNCAUGHT_ERROR_LOGGER_NAME).removeHandler(handler)
        handler.close()


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
            services.close_case_logging("case-x")
            close_uncaught_error_log(tmp_path)


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
