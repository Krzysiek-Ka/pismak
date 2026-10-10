from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from src._shared.serwer import LLMConfigurationError

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

        pipeline, documents, chunks = services.build_zrodlowyszukiwator(
            client=object(),
            model="openrouter/free",
            provider="openrouter",
        )

        assert isinstance(pipeline, FakePipeline)
        assert (documents, chunks) == (2, 4)
        assert captures["planner"]["model"] == "openrouter/free"
        assert captures["planner"]["allow_fallback"] is False
        assert captures["reranker"]["allow_fallback"] is False
        assert captures["embedder"] == {
            "model": "bge-m3",
            "base_url": "http://127.0.0.1:11434",
        }


class TestSaveRun:
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
