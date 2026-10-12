from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from src._shared.context_budget import LLMInputLimitError
from src._shared.serwer import LLMConfigurationError, LLMResponseError
from src.Pismogenerator.pismogenerator import DraftValidationError
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.rag_retrieval import DatabaseConnectionError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WYSWIETLATOR_DIR = PROJECT_ROOT / "src" / "Wyswietlator"
if str(WYSWIETLATOR_DIR) not in sys.path:
    sys.path.insert(0, str(WYSWIETLATOR_DIR))

services = importlib.import_module("services")


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

    def test_llm_error_message__explains_missing_openrouter_key(self) -> None:
        """Verify missing credentials are shown as setup guidance, not a traceback."""
        message = services.llm_error_message(LLMConfigurationError("Brak OPENROUTER_KEY."))

        assert "OPENROUTER_KEY" in message


class TestLlmErrorMessage:
    def test_llm_error_message__explains_local_postgres_docker_failure(self) -> None:
        """Verify the UI preserves actionable Docker guidance for local database failures."""
        expected = "Uruchom Docker Desktop i kontener PostgreSQL."

        assert services.llm_error_message(DatabaseConnectionError(expected)) == expected

    def test_llm_error_message__preserves_draft_validation_details(self) -> None:
        """Verify draft validation failures show their rejected-source explanation."""
        error = DraftValidationError(["Model dodał niedozwolone źródła: U1; U2"])

        message = services.llm_error_message(error)

        assert "DraftLetter nie przeszedł walidacji" in message
        assert "Model dodał niedozwolone źródła: U1; U2" in message

    def test_llm_error_message__preserves_structured_output_failure_reason(self) -> None:
        """Verify structured-output failures expose the specific schema rejection."""
        error = LLMResponseError(
            "Model dwukrotnie zwrócił niepoprawny structured output dla InterviewTurnResult. "
            "Ostatni błąd: next_question: Field required (missing)"
        )

        message = services.llm_error_message(error)

        assert "InterviewTurnResult" in message
        assert "next_question: Field required (missing)" in message

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

    def test_error_location__identifies_project_file_and_line(self) -> None:
        """Verify exception diagnostics point to the project code that raised the error."""

        def raise_validation_error() -> None:
            raise DraftValidationError(["Model dodał niedozwolone źródła: U1; U2"])

        try:
            raise_validation_error()
        except DraftValidationError as error:
            location = services.error_location(error)

        assert location is not None
        assert location.startswith("tests/unit_test/wyswietlator/test_services.py:")
        assert "in raise_validation_error" in location


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

        services.validate_user_message("a" * 100, conversation_chars=limit - 100, provider="ollama")

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
