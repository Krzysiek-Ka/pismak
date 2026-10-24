"""Tests for the Wyswietlator service adapters."""

from __future__ import annotations

import importlib
import json
import logging
import shutil
import sys
from pathlib import Path
from threading import Thread

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import ValidationError

from src._shared.context_budget import LLMInputLimitError
from src._shared.llm_runtime import LLMBusyError
from src._shared.serwer import LLMConfigurationError, LLMResponseError
from src.Pismogenerator.pismogenerator import DraftValidationError
from src.Pismogenerator.structures import DraftLetter
from src.Pismopoprawiator.structures import Correction, CorrectionResult
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack, ResearchPlan
from src.Zrodlowyszukiwator.rag_retrieval import DatabaseConnectionError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WYSWIETLATOR_DIR = PROJECT_ROOT / "src" / "Wyswietlator"


if str(WYSWIETLATOR_DIR) not in sys.path:
    sys.path.insert(0, str(WYSWIETLATOR_DIR))


services = importlib.import_module("services")


def detach_log_handler(project_root: Path) -> None:
    """Remove the handler that configure_logging attached for a temporary project directory."""
    handler = services._LOG_HANDLERS.pop((project_root / "logs").resolve(), None)
    if handler is not None:
        services.LOGGER.removeHandler(handler)
        logging.getLogger(services.UNCAUGHT_ERROR_LOGGER_NAME).removeHandler(handler)


@pytest.fixture
def logs(tmp_path: Path):
    """Yield the log directory of a temporary project and detach its handler afterwards."""
    yield tmp_path / "logs"
    detach_log_handler(tmp_path)


def log_as(session_id: str, message: str) -> None:
    services.LOGGER.info(message, extra={"session_id": session_id})


def make_empty_pack() -> EvidencePack:
    return EvidencePack(
        query_summary="Brak wyników",
        research_plan=ResearchPlan(case_summary="Brak wyników", intents=[]),
    )


OPENROUTER_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def status_error(error_class: type[Exception], status_code: int) -> Exception:
    return error_class(
        f"Error code: {status_code} - {{'error': 'sk-or-v1-tajny upstream detail'}}",
        response=httpx.Response(status_code, request=OPENROUTER_REQUEST),
        body=None,
    )


class TestGetLlmClient:
    def test_get_llm_client__uses_local_ollama_without_openrouter_key(self, monkeypatch) -> None:
        """Verify choosing Ollama never reads or requires the OpenRouter credential."""
        sentinel = object()
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        monkeypatch.setattr(services, "get_openai_client", lambda: sentinel)

        assert services.get_llm_client("ollama") is sentinel

    def test_get_llm_client__rejects_unknown_provider(self) -> None:
        """Verify invalid provider names produce a clear error rather than a local fallback."""
        with pytest.raises(ValueError, match="Nieznany dostawca"):
            services.get_llm_client("other")

    def test_get_llm_client__routes_openrouter_to_the_provider_factory(self, monkeypatch) -> None:
        """Verify OpenRouter selection is passed to the provider factory explicitly."""
        sentinel = object()
        calls: list[str] = []

        def make_client(provider: str):
            calls.append(provider)
            return sentinel

        monkeypatch.setattr(services, "set_up_client", make_client)

        assert services.get_llm_client("openrouter") is sentinel
        assert calls == ["openrouter"]


class TestRunPismopoprawiator:
    def test_run_pismopoprawiator__returns_the_uncorrected_letter_without_an_openrouter_key(
        self, monkeypatch, caplog
    ) -> None:
        """Verify a misconfigured provider ends the correction, not the pipeline."""
        monkeypatch.delenv("OPENROUTER_KEY", raising=False)
        logger = logging.getLogger("pismak")
        logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger="pismak"):
                result = services.run_pismopoprawiator(
                    "Pismo ktore", provider="openrouter", model="openrouter/free"
                )
        finally:
            logger.removeHandler(caplog.handler)

        assert result.status == "failed"
        assert result.corrected_text == "Pismo ktore"
        assert result.errors == ["LLMConfigurationError"]
        messages = [record.getMessage() for record in caplog.records]
        assert any("correction_setup_failed" in message for message in messages)
        assert any(
            "correction_finished" in message and "status=failed" in message for message in messages
        )

    def test_run_pismopoprawiator__uses_the_window_of_the_selected_model(self, monkeypatch) -> None:
        """Verify the correction plans its prompt for the model's real context window."""
        seen: dict = {}

        def correct(letter, llm, *, session_id=None):
            seen.update(llm=llm, session_id=session_id)
            return "wynik"

        client = type("Client", (), {"with_options": lambda self, **kwargs: self})()
        monkeypatch.setattr(
            services.model_context, "detect_context_tokens", lambda *args, **kwargs: 32768
        )
        monkeypatch.setattr(services.pismopoprawiator, "correct_letter", correct)

        result = services.run_pismopoprawiator(
            "Pismo", provider="ollama", model="pismak-pllum", session_id="s1", client=client
        )

        assert result == "wynik"
        assert seen["llm"].context_tokens == 32768
        assert seen["llm"].stage == "pismopoprawiator"
        assert seen["session_id"] == "s1"


class TestLlmErrorMessage:
    @pytest.mark.parametrize(
        ("error", "internal_details"),
        [
            (
                DatabaseConnectionError(
                    "Nie można połączyć się z lokalną bazą PostgreSQL (127.0.0.1:5432). "
                    "Uruchom Docker Desktop i `docker compose up -d postgres`."
                ),
                ("127.0.0.1", "5432", "Docker", "PostgreSQL"),
            ),
            (
                DraftValidationError(["Model dodał niedozwolone źródła: U1; U2"]),
                ("DraftLetter", "niedozwolone źródła", "U1"),
            ),
            (
                LLMConfigurationError("Brak OPENROUTER_KEY. Dodaj klucz do pliku .env."),
                ("OPENROUTER_KEY", ".env"),
            ),
            (
                LLMResponseError(
                    "Model dwukrotnie zwrócił niepoprawny structured output dla "
                    "InterviewTurnResult. Ostatni błąd: next_question: Field required (missing)"
                ),
                ("InterviewTurnResult", "next_question", "Field required"),
            ),
            (RuntimeError("Brak DATABASE_URL w .env."), ("DATABASE_URL", ".env")),
            (
                OSError("C:\\Krzysiek\\Pismak\\src\\modul.py: brak dostępu"),
                ("C:\\", "modul.py", "brak dostępu"),
            ),
        ],
    )
    def test_llm_error_message__keeps_the_content_of_the_error_out_of_the_ui(
        self, error: Exception, internal_details: tuple[str, ...]
    ) -> None:
        """Verify no host, path, setting name or schema detail of a failure reaches the user."""
        message = services.llm_error_message(error)

        assert message
        assert [detail for detail in internal_details if detail in message] == []

    def test_llm_error_message__still_tells_the_user_which_part_failed(self) -> None:
        """Verify hiding details does not collapse every failure into one indistinct message."""
        messages = {
            services.llm_error_message(DatabaseConnectionError("x")),
            services.llm_error_message(DraftValidationError(["x"])),
            services.llm_error_message(LLMConfigurationError("x")),
            services.llm_error_message(LLMResponseError("x")),
            services.llm_error_message(RuntimeError("x")),
        }

        assert len(messages) == 5
        assert "Źródłowyszukiwacz" in services.llm_error_message(DatabaseConnectionError("x"))
        assert "projekt pisma" in services.llm_error_message(DraftValidationError(["x"]))

    def test_llm_error_message__explains_input_beyond_the_context_window(self) -> None:
        """Verify an oversized case is explained to the user instead of the generic log hint."""
        message = services.llm_error_message(LLMInputLimitError("12000 znaków, 8192 tokenów"))

        assert "oknie kontekstu" in message
        assert "12000" not in message

    def test_llm_error_message__names_the_model_window_when_it_is_known(self) -> None:
        """Verify the user learns which window applied, since it now differs between models."""
        message = services.llm_error_message(LLMInputLimitError("details", context_tokens=4096))

        assert "4096 tokenów" in message
        assert "details" not in message

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (status_error(AuthenticationError, 401), "OpenRouter odrzucił dostęp aplikacji"),
            (status_error(PermissionDeniedError, 403), "OpenRouter odrzucił dostęp aplikacji"),
            (status_error(RateLimitError, 429), "z powodu limitu"),
            (status_error(BadRequestError, 400), "nie obsłużył wymaganej odpowiedzi strukturalnej"),
            (status_error(InternalServerError, 503), "chwilowo niedostępna"),
            (APITimeoutError(request=OPENROUTER_REQUEST), "Przekroczono czas oczekiwania"),
            (APIConnectionError(request=OPENROUTER_REQUEST), "Nie udało się połączyć z usługą"),
        ],
        ids=["401", "403", "429", "400", "503", "timeout", "connection"],
    )
    def test_llm_error_message__explains_each_provider_failure_without_its_details(
        self, error: Exception, expected: str
    ) -> None:
        """Verify each OpenRouter failure gets its own hint and never the upstream message."""
        message = services.llm_error_message(error)

        assert expected in message
        assert "sk-or-v1" not in message
        assert "upstream" not in message

    def test_llm_error_message__busy_local_model_suggests_retrying_or_openrouter(self) -> None:
        """Verify a case that waited too long learns the model is busy, not that it failed."""
        message = services.llm_error_message(
            LLMBusyError("Lokalny model był zajęty dłużej niż 900 s.")
        )

        assert "zajęty obsługą innych spraw" in message
        assert "OpenRouter" in message
        assert "900" not in message

    def test_llm_error_message__does_not_call_a_timeout_a_connection_failure(self) -> None:
        """Verify a slow model is not reported as unreachable, though it subclasses that error."""
        timeout = services.llm_error_message(APITimeoutError(request=OPENROUTER_REQUEST))
        connection = services.llm_error_message(APIConnectionError(request=OPENROUTER_REQUEST))

        assert timeout != connection

    @pytest.mark.parametrize("status_code", [404, 422])
    def test_llm_error_message__uses_the_general_message_for_other_client_errors(
        self, status_code: int
    ) -> None:
        """Verify an unexpected 4xx is not presented as a temporary server outage."""
        message = services.llm_error_message(status_error(APIStatusError, status_code))

        assert message.startswith("Wystąpił błąd podczas przetwarzania sprawy")


class TestGetOpenaiClient:
    def test_get_openai_client__gives_local_model_fifteen_minutes_without_sdk_retries(
        self, monkeypatch
    ) -> None:
        """Verify the UI client waits for a slow CPU generation once instead of re-sending it."""
        calls: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                calls.update(kwargs)

        monkeypatch.setenv("OLLAMA_URL", "http://ollama.test/")
        monkeypatch.setattr(services, "OpenAI", FakeOpenAI)

        services.get_openai_client()

        assert calls == {
            "base_url": "http://ollama.test/v1/",
            "api_key": "ollama",
            "timeout": 900.0,
            "max_retries": 0,
        }

    def test_get_openai_client__prefers_an_explicit_address_over_the_environment(
        self, monkeypatch
    ) -> None:
        """Verify a caller can point the client at another Ollama than the one in .env."""
        calls: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                calls.update(kwargs)

        monkeypatch.setenv("OLLAMA_URL", "http://ollama.test/")
        monkeypatch.setattr(services, "OpenAI", FakeOpenAI)

        services.get_openai_client("http://inna-ollama.test:11434/")

        assert calls["base_url"] == "http://inna-ollama.test:11434/v1/"


class TestEscapeMarkdownLinks:
    def test_escape_markdown_links__turns_images_and_links_into_literal_text(self) -> None:
        """Verify Markdown cannot embed an image or disguise a link address."""
        text = "Zob. ![](http://tracker.test/p.png) i [mapa](http://phish.test)."

        assert services.escape_markdown_links(text) == (
            "Zob. !\\[](http://tracker.test/p.png) i \\[mapa](http://phish.test)."
        )

    def test_escape_markdown_links__a_backslash_in_the_text_cannot_undo_the_escape(self) -> None:
        """Verify a model-written backslash before a bracket still leaves the link inert."""
        assert services.escape_markdown_links("\\[mapa](http://phish.test)") == (
            "\\\\\\[mapa](http://phish.test)"
        )

    def test_escape_markdown_links__keeps_text_without_brackets_unchanged(self) -> None:
        """Verify ordinary letter formatting such as bold and headings is left alone."""
        text = "**Adresat:** ZDiUM\n\n# WNIOSEK\n\n- punkt"

        assert services.escape_markdown_links(text) == text


class TestErrorLocation:
    def test_error_location__identifies_project_file_and_line(self) -> None:
        """Verify exception diagnostics point to the project code that raised the error."""

        def raise_validation_error() -> None:
            raise DraftValidationError(["Model dodał niedozwolone źródła: U1; U2"])

        try:
            raise_validation_error()
        except DraftValidationError as error:
            location = services.error_location(error)

        assert location is not None
        assert location.startswith("tests/unit_test/Wyswietlator/test_services.py:")
        assert "in raise_validation_error" in location

    def test_error_location__skips_packages_installed_inside_the_project(self) -> None:
        """Verify an error raised by a dependency points to the project code that called it."""

        def parse_invalid_json() -> None:
            DraftLetter.model_validate_json("not json")

        try:
            parse_invalid_json()
        except ValidationError as error:
            location = services.error_location(error)

        assert location is not None
        assert location.startswith("tests/unit_test/Wyswietlator/test_services.py:")
        assert "in parse_invalid_json" in location


class TestConversationCharLimit:
    def test_conversation_char_limit__admits_a_full_length_first_message_by_default(
        self, monkeypatch
    ) -> None:
        """Verify the default window never rejects an opening message the UI itself allows."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        assert services.conversation_char_limit("ollama") >= services.MAX_USER_MESSAGE_CHARS

    def test_conversation_char_limit__grows_with_the_configured_window(self, monkeypatch) -> None:
        """Verify raising the model window lets the interview run longer."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        default_limit = services.conversation_char_limit("ollama")
        monkeypatch.setenv("PISMAK_CONTEXT_TOKENS", "16384")

        assert services.conversation_char_limit("ollama") > default_limit

    def test_conversation_char_limit__follows_the_window_of_the_selected_model(
        self, monkeypatch
    ) -> None:
        """Verify a model with a large window allows a much longer interview."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        assert services.conversation_char_limit("ollama", 131072) > (
            10 * services.conversation_char_limit("ollama")
        )


class TestValidateUserMessage:
    def test_validate_user_message__rejects_message_beyond_the_remaining_conversation_room(
        self, monkeypatch
    ) -> None:
        """Verify a reply that would overflow the model window is refused with the room left."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        limit = services.conversation_char_limit("ollama")

        with pytest.raises(ValueError, match="pozostało miejsca na 100 znaków"):
            services.validate_user_message(
                "a" * 101, conversation_chars=limit - 100, provider="ollama"
            )

    def test_validate_user_message__accepts_message_that_exactly_fills_the_remaining_room(
        self, monkeypatch
    ) -> None:
        """Verify the conversation limit is inclusive, like the per-message limit."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        limit = services.conversation_char_limit("ollama")

        assert (
            services.validate_user_message(
                "a" * 100, conversation_chars=limit - 100, provider="ollama"
            )
            is None
        )

    def test_validate_user_message__reports_zero_room_when_conversation_already_overflows(
        self, monkeypatch
    ) -> None:
        """Verify an already overlong conversation never shows a negative character count."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        limit = services.conversation_char_limit("ollama")

        with pytest.raises(ValueError, match="pozostało miejsca na 0 znaków"):
            services.validate_user_message("a", conversation_chars=limit + 500, provider="ollama")

    def test_validate_user_message__accepts_long_conversation_for_a_model_with_a_large_window(
        self, monkeypatch
    ) -> None:
        """Verify a conversation too long for the fallback window passes for a larger model."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        with pytest.raises(ValueError, match="Rozmowa jest zbyt długa"):
            services.validate_user_message("a" * 100, conversation_chars=50_000, provider="ollama")

        services.validate_user_message(
            "a" * 100, conversation_chars=50_000, provider="ollama", context_tokens=131072
        )

    def test_validate_user_message__accepts_message_at_limit(self) -> None:
        """Verify the configured maximum remains usable for legitimate long descriptions."""
        message = "x" * services.MAX_USER_MESSAGE_CHARS

        assert services.validate_user_message(message) is None

    def test_validate_user_message__rejects_message_over_limit(self) -> None:
        """Verify oversized user text is rejected before it enters session history."""
        message = "x" * (services.MAX_USER_MESSAGE_CHARS + 1)

        with pytest.raises(ValueError, match="Wiadomość jest zbyt długa"):
            services.validate_user_message(message)


class TestListGenerationModels:
    def test_list_generation_models__filters_embedding_models_case_insensitively(self) -> None:
        """Verify embedding-only models cannot be selected for generation."""

        class Models:
            data = [
                type("Model", (), {"id": "Bielik"})(),
                type("Model", (), {"id": "BGE-M3"})(),
                type("Model", (), {"id": "nomic-embed-text"})(),
            ]

        class Client:
            class ModelsApi:
                def list(self):
                    return Models()

            models = ModelsApi()

        assert services.list_generation_models(Client()) == ["Bielik"]


class TestBuildZrodlowyszukiwator:
    def test_build_zrodlowyszukiwator__routes_openrouter_without_changing_embeddings(
        self, monkeypatch
    ) -> None:
        """Verify RAG generation uses the chosen provider while embeddings remain on Ollama."""
        captures: dict[str, object] = {}

        class FakeRepository:
            def __init__(self, database_url: str) -> None:
                captures["database_url"] = database_url

            def counts(self) -> tuple[int, int]:
                return 2, 4

        class FakeEmbedder:
            def __init__(self, **kwargs) -> None:
                captures["embedder"] = kwargs

        class FakePlanner:
            def __init__(self, **kwargs) -> None:
                captures["planner"] = kwargs

        class FakeReranker:
            def __init__(self, **kwargs) -> None:
                captures["reranker"] = kwargs

        class FakeRetriever:
            def __init__(self, **kwargs) -> None:
                captures["retriever"] = kwargs

        class FakePipeline:
            def __init__(self, **kwargs) -> None:
                captures["pipeline"] = kwargs

        monkeypatch.setenv("DATABASE_URL", "postgresql://test")
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", "bge-m3")
        monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:11434")
        monkeypatch.setattr(services, "RetrievalRepository", FakeRepository)
        monkeypatch.setattr(services, "OllamaEmbedder", FakeEmbedder)
        monkeypatch.setattr(services, "LLMQueryPlanner", FakePlanner)
        monkeypatch.setattr(services, "LLMReranker", FakeReranker)
        monkeypatch.setattr(services, "HybridRetriever", FakeRetriever)
        monkeypatch.setattr(services, "BriefToEvidencePipeline", FakePipeline)
        monkeypatch.setattr(
            services,
            "detect_context_tokens",
            lambda provider, model, **kwargs: {("openrouter", "openrouter/free"): 65536}[
                (provider, model)
            ],
        )

        pipeline, documents, chunks = services.build_zrodlowyszukiwator(
            client=object(),
            model="openrouter/free",
            provider="openrouter",
        )

        assert isinstance(pipeline, FakePipeline)
        assert (documents, chunks) == (2, 4)
        assert captures["planner"]["context_tokens"] == 65536
        assert captures["reranker"]["context_tokens"] == 65536
        assert captures["planner"]["model"] == "openrouter/free"
        assert captures["planner"]["allow_fallback"] is False
        assert captures["reranker"]["allow_fallback"] is False
        assert captures["embedder"] == {
            "model": "bge-m3",
            "base_url": "http://127.0.0.1:11434",
        }

    def test_build_zrodlowyszukiwator__fails_before_connecting_without_database_url(
        self, monkeypatch
    ) -> None:
        """Verify missing database configuration produces a clear setup error."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with pytest.raises(RuntimeError, match=r"^Brak DATABASE_URL w \.env\.$"):
            services.build_zrodlowyszukiwator(client=object(), model="test")

    def test_build_zrodlowyszukiwator__rejects_an_unknown_provider_before_the_database(
        self, monkeypatch
    ) -> None:
        """Verify a typo in the provider fails fast instead of opening a connection."""
        monkeypatch.setattr(
            services, "RetrievalRepository", lambda url: pytest.fail("must not connect")
        )

        with pytest.raises(ValueError, match="Nieznany dostawca modelu: azure"):
            services.build_zrodlowyszukiwator(client=object(), model="m", provider="azure")

    def test_build_zrodlowyszukiwator__reports_an_empty_rag_database(self, monkeypatch) -> None:
        """Verify an un-ingested database is named as such instead of returning no sources."""

        class EmptyRepository:
            def __init__(self, database_url: str) -> None:
                pass

            def counts(self) -> tuple[int, int]:
                return 0, 0

        monkeypatch.setenv("DATABASE_URL", "postgresql://test")
        monkeypatch.setattr(services, "RetrievalRepository", EmptyRepository)

        with pytest.raises(RuntimeError, match="Baza RAG jest pusta"):
            services.build_zrodlowyszukiwator(client=object(), model="m")

    def test_build_zrodlowyszukiwator__lets_the_local_model_fall_back(self, monkeypatch) -> None:
        """Verify planner and reranker may use deterministic fallbacks only on Ollama."""
        captures: dict[str, dict] = {}

        def capture(name):
            return lambda **kwargs: captures.setdefault(name, kwargs)

        class Repository:
            def __init__(self, database_url: str) -> None:
                pass

            def counts(self) -> tuple[int, int]:
                return 1, 1

        monkeypatch.setenv("DATABASE_URL", "postgresql://test")
        monkeypatch.setattr(services, "RetrievalRepository", Repository)
        for name in ("OllamaEmbedder", "LLMQueryPlanner", "LLMReranker", "HybridRetriever"):
            monkeypatch.setattr(services, name, capture(name))
        monkeypatch.setattr(services, "BriefToEvidencePipeline", capture("pipeline"))
        monkeypatch.setattr(services, "detect_context_tokens", lambda *args, **kwargs: 8192)

        services.build_zrodlowyszukiwator(client=object(), model="pismak-bielik", session_id="s1")

        assert captures["LLMQueryPlanner"]["allow_fallback"] is True
        assert captures["LLMReranker"]["allow_fallback"] is True
        assert captures["HybridRetriever"]["session_id"] == "s1"
        assert captures["pipeline"]["evidence_k"] == 6


class TestRenderLetter:
    def test_render_letter__normalizes_blank_user_evidence_before_resolving_ids(self) -> None:
        """Verify blank user materials do not shift U-source identifiers in the rendered letter."""
        draft = DraftLetter(
            recipient="Urząd",
            request_summary="sprawdzenie sprawy",
            argumentation_markdown="A" * 120,
            sources=["U1"],
        )

        result = services.render_letter(
            draft,
            make_empty_pack(),
            ["  ", "Zdjęcie"],
        )

        assert "U1: Zdjęcie" in result


class TestSaveRun:
    def test_save_run__persists_rejected_draft_and_validation_errors(self, tmp_path: Path) -> None:
        """Verify a rejected draft is preserved separately from any valid draft artifact."""
        rejected_draft = DraftLetter(
            recipient="Urząd",
            request_summary="sprawdzenie sprawy",
            argumentation_markdown="A" * 120,
            sources=["U1", "U2"],
        )

        run_dir = services.save_run(
            project_root=tmp_path,
            messages=[],
            draft_with_error=rejected_draft,
            draft_validation_errors=["Model dodał niedozwolone źródła: U1; U2"],
            run_dir=tmp_path / "run",
        )

        saved = json.loads((run_dir / "draft_with_error.json").read_text(encoding="utf-8"))
        assert saved["draft"]["sources"] == ["U1", "U2"]
        assert saved["validation_errors"] == ["Model dodał niedozwolone źródła: U1; U2"]
        assert not (run_dir / "draft.json").exists()

    def test_save_run__persists_actual_provider_and_model_metadata(self, tmp_path: Path) -> None:
        """Verify each run records provider/model diagnostics without storing credentials."""
        usage = {"Dopytywator": [{"provider": "openrouter", "model": "vendor/actual-model"}]}

        run_dir = services.save_run(
            project_root=tmp_path,
            messages=[],
            model_usage=usage,
            run_dir=tmp_path / "run",
        )

        assert (run_dir / "models.json").read_text(encoding="utf-8") == (
            '{\n  "Dopytywator": [\n    {\n      "provider": "openrouter",\n'
            '      "model": "vendor/actual-model"\n    }\n  ]\n}'
        )

    def test_save_run__writes_only_provided_optional_artifacts(self, tmp_path: Path) -> None:
        """Verify a partial UI run saves its conversation without creating absent artifacts."""
        run_dir = services.save_run(
            project_root=tmp_path,
            messages=[{"role": "user", "content": "Opis"}],
            run_dir=tmp_path / "run",
        )

        assert run_dir == tmp_path / "run"
        assert json.loads((run_dir / "conversation.json").read_text(encoding="utf-8")) == [
            {"role": "user", "content": "Opis"}
        ]
        assert not (run_dir / "brief.json").exists()
        assert not (run_dir / "letter.md").exists()

    def test_save_run__keeps_both_letter_versions_and_the_correction_report(
        self, tmp_path: Path
    ) -> None:
        """Verify a corrected run stores the uncorrected letter and every correction decision."""
        correction = CorrectionResult(
            original_text="Pismo ktore",
            corrected_text="Pismo, które",
            applied=[
                Correction(akapit="A5", oryginal="ktore", poprawka="które", kategoria="ortografia")
            ],
            status="corrected",
            provider="ollama",
            model="pismak-bielik:latest",
        )

        run_dir = services.save_run(
            project_root=tmp_path,
            messages=[],
            letter_markdown="Pismo, które",
            letter_original_markdown="Pismo ktore",
            correction=correction,
            run_dir=tmp_path / "run",
        )

        assert (run_dir / "letter.md").read_text(encoding="utf-8") == "Pismo, które"
        assert (run_dir / "letter_bez_korekty.md").read_text(encoding="utf-8") == "Pismo ktore"
        saved = json.loads((run_dir / "korekta.json").read_text(encoding="utf-8"))
        assert saved["status"] == "corrected"
        assert saved["applied"][0]["poprawka"] == "które"
        assert saved["model"] == "pismak-bielik:latest"
        # Teksty pisma są już w dwóch plikach .md obok.
        assert "original_text" not in saved and "corrected_text" not in saved

    def test_save_run__creates_unique_directories_for_runs_in_same_second(
        self, tmp_path: Path
    ) -> None:
        """Verify simultaneous sessions cannot write artifacts into one output directory."""
        first = services.save_run(project_root=tmp_path, messages=[])
        second = services.save_run(project_root=tmp_path, messages=[])

        assert first != second
        assert first.parent == second.parent == tmp_path / "outputs"


class TestConfigureLogging:
    def test_configure_logging__writes_each_case_to_its_own_timestamped_file(
        self, tmp_path: Path
    ) -> None:
        """Verify concurrent cases create distinct files and never share log records."""
        logger = services.configure_logging(tmp_path, "case-a")
        services.configure_logging(tmp_path, "case-b")

        try:
            logger.info("message for case A", extra={"session_id": "case-a"})
            logger.info("message for case B", extra={"session_id": "case-b"})

            log_files = sorted((tmp_path / "logs").glob("pismak_*.log"))
            assert len(log_files) == 2
            assert all(path.stem.startswith("pismak_") for path in log_files)

            contents = [path.read_text(encoding="utf-8") for path in log_files]
            assert any("message for case A" in content for content in contents)
            assert any("message for case B" in content for content in contents)
            assert all(
                not ("message for case A" in content and "message for case B" in content)
                for content in contents
            )
        finally:
            services.close_case_logging("case-a")
            services.close_case_logging("case-b")
            detach_log_handler(tmp_path)

    def test_configure_logging__records_unhandled_errors_with_their_traceback(
        self, tmp_path: Path, logs: Path
    ) -> None:
        """Verify an exception the UI hides is still written down, in full, for the operator."""
        services.configure_logging(tmp_path, "case-a")
        log_as("case-a", "message for case A")

        try:
            raise RuntimeError("szczegóły tylko dla operatora")
        except RuntimeError as error:
            logging.getLogger(services.UNCAUGHT_ERROR_LOGGER_NAME).error(
                "Uncaught app execution", exc_info=error
            )

        content = (logs / services.UNCAUGHT_ERROR_LOG_NAME).read_text(encoding="utf-8")
        assert "RuntimeError: szczegóły tylko dla operatora" in content
        assert "Traceback (most recent call last)" in content
        (case_log,) = logs.glob("pismak_*.log")
        assert "szczegóły tylko dla operatora" not in case_log.read_text(encoding="utf-8")

    def test_configure_logging__uses_one_handler_and_no_open_file_for_fifty_cases(
        self, tmp_path: Path, logs: Path
    ) -> None:
        """Verify opening many sessions adds neither handlers nor file handles to the process."""
        streamlit_logger = logging.getLogger(services.UNCAUGHT_ERROR_LOGGER_NAME)
        before = (len(services.LOGGER.handlers), len(streamlit_logger.handlers))

        for index in range(50):
            services.configure_logging(tmp_path, f"case-{index}")
            log_as(f"case-{index}", f"message for case {index};")

        assert len(services.LOGGER.handlers) == before[0] + 1
        assert len(streamlit_logger.handlers) == before[1] + 1
        case_logs = sorted(logs.glob("pismak_*.log"))
        assert len(case_logs) == 50
        assert all(
            path.read_text(encoding="utf-8").count("message for case") == 1 for path in case_logs
        )
        # Windows odmawia usunięcia pliku, który jakikolwiek proces trzyma otwarty.
        for path in case_logs:
            path.unlink()

    def test_configure_logging__keeps_one_file_for_a_case_across_reruns(
        self, tmp_path: Path, logs: Path
    ) -> None:
        """Verify Streamlit reruns, which configure logging again, do not split a case's log."""
        for turn in range(3):
            services.configure_logging(tmp_path, "case-a")
            log_as("case-a", f"turn {turn}")

        (case_log,) = logs.glob("pismak_*.log")
        assert [line.split()[-1] for line in case_log.read_text(encoding="utf-8").splitlines()] == [
            "0",
            "1",
            "2",
        ]

    def test_configure_logging__keeps_cases_apart_when_sessions_log_at_the_same_time(
        self, tmp_path: Path, logs: Path
    ) -> None:
        """Verify writes from parallel session threads never land in, or tear, another case's file."""
        sessions = [f"case-{index}" for index in range(8)]
        for session_id in sessions:
            services.configure_logging(tmp_path, session_id)

        def write_many(session_id: str) -> None:
            for number in range(100):
                log_as(session_id, f"{session_id} line {number} end")

        threads = [Thread(target=write_many, args=(session_id,)) for session_id in sessions]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        case_logs = list(logs.glob("pismak_*.log"))
        assert len(case_logs) == 8
        for path in case_logs:
            lines = path.read_text(encoding="utf-8").splitlines()
            owners = {line.split()[3] for line in lines}
            assert len(lines) == 100
            assert len(owners) == 1
            assert all(line.endswith(" end") for line in lines)

    def test_configure_logging__writes_records_without_a_known_case_to_the_shared_file(
        self, tmp_path: Path, logs: Path
    ) -> None:
        """Verify a record that carries no session, or an unknown one, is never silently lost."""
        services.configure_logging(tmp_path, "case-a")

        services.LOGGER.info("record without a session")
        log_as("never-configured", "record of an unknown session")

        shared = (logs / services.UNCAUGHT_ERROR_LOG_NAME).read_text(encoding="utf-8")
        assert "record without a session" in shared
        assert "record of an unknown session" in shared
        assert list(logs.glob("pismak_*.log")) == []

    def test_configure_logging__forgets_only_the_least_recently_used_cases_beyond_the_limit(
        self, tmp_path: Path, logs: Path, monkeypatch
    ) -> None:
        """Verify abandoned sessions cannot grow the registry, while active ones keep their file."""
        monkeypatch.setattr(services, "MAX_TRACKED_CASES", 3)
        for session_id in ("case-a", "case-b", "case-c"):
            services.configure_logging(tmp_path, session_id)
        services.configure_logging(tmp_path, "case-a")
        services.configure_logging(tmp_path, "case-d")

        for session_id in ("case-a", "case-b", "case-c", "case-d"):
            log_as(session_id, f"message of {session_id};")

        assert len(services._LOG_HANDLERS[logs.resolve()].case_paths) == 3
        case_text = "".join(path.read_text(encoding="utf-8") for path in logs.glob("pismak_*.log"))
        assert [name for name in "abcd" if f"message of case-{name};" in case_text] == [
            "a",
            "c",
            "d",
        ]
        shared = (logs / services.UNCAUGHT_ERROR_LOG_NAME).read_text(encoding="utf-8")
        assert "message of case-b;" in shared

    def test_configure_logging__recreates_the_log_directory_removed_while_running(
        self, tmp_path: Path, logs: Path
    ) -> None:
        """Verify an operator can delete the logs of a running app without breaking logging."""
        services.configure_logging(tmp_path, "case-a")
        log_as("case-a", "before cleanup")
        shutil.rmtree(logs)

        log_as("case-a", "after cleanup")

        (case_log,) = logs.glob("pismak_*.log")
        assert case_log.read_text(encoding="utf-8").count("after cleanup") == 1

    def test_configure_logging__a_failed_log_write_does_not_break_the_case(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Verify a full disk or locked log file cannot stop the user's case from running."""
        (tmp_path / "logs").write_text("to jest plik, nie katalog", encoding="utf-8")
        handled: list[logging.LogRecord] = []
        services.configure_logging(tmp_path, "case-1")
        try:
            handler = services._LOG_HANDLERS[(tmp_path / "logs").resolve()]
            monkeypatch.setattr(handler, "handleError", handled.append)

            log_as("case-1", "zdarzenie sprawy")
        finally:
            detach_log_handler(tmp_path)

        assert [record.getMessage() for record in handled] == ["zdarzenie sprawy"]


class TestCloseCaseLogging:
    def test_close_case_logging__sends_a_late_record_of_a_finished_case_to_the_shared_file(
        self, tmp_path: Path, logs: Path
    ) -> None:
        """Verify "Nowa sprawa" ends the case file without dropping what still arrives for it."""
        services.configure_logging(tmp_path, "case-a")
        log_as("case-a", "during the case")

        services.close_case_logging("case-a")
        log_as("case-a", "after the case was closed")

        (case_log,) = logs.glob("pismak_*.log")
        assert "after the case was closed" not in case_log.read_text(encoding="utf-8")
        shared = (logs / services.UNCAUGHT_ERROR_LOG_NAME).read_text(encoding="utf-8")
        assert "after the case was closed" in shared
