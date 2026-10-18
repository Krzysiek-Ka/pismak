from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from src.Zrodlowyszukiwator.rag_ingestion.catalog import load_catalog
from src.Zrodlowyszukiwator.rag_ingestion.cli import DEFAULT_CATALOG_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_RAG_DOCUMENTS = (
    "01_Polityka-rowerowa-wroclawia.pdf",
    "01_Wroclawska_polityka_mobilnosci.pdf",
)
# Odwołania w README sprawdzane jako ścieżki plików repozytorium.
REPOSITORY_FILE_SUFFIXES = (
    ".py",
    ".md",
    ".json",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".bat",
    ".sql",
    ".ipynb",
    ".pdf",
)


class TestMainNotebookConfiguration:
    def test_main_notebook__discovers_root_and_imports_rag_dependencies(self) -> None:
        """Verify the notebook does not depend on a hard-coded working directory or hidden exports."""
        notebook = json.loads((PROJECT_ROOT / "src" / "main.ipynb").read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        assert '"docker-compose.yaml"' in source
        assert "from src.Zrodlowyszukiwator.rag_ingestion.embedder import OllamaEmbedder" in source
        assert "RetrievalFilters" in source
        assert "sys.path.insert(0, str(Path.cwd().parent))" not in source
        # Jupyter Lab, w przeciwieństwie do VS Code, nie wczytuje .env samodzielnie.
        assert 'load_dotenv(PROJECT_ROOT / ".env")' in source


class TestReleaseFiles:
    def test_requirements__includes_notebook_and_ui_dependencies(self) -> None:
        """Verify pandas remains available for notebooks and Streamlit for the UI."""
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

        assert "llama-cpp-python" not in requirements
        assert "pandas" in requirements
        assert "streamlit" in requirements

    def test_readme__uses_existing_ui_installation_and_module_names(self) -> None:
        """Verify publication instructions reference files and package names that exist in the repository."""
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        assert "requirements-ui.txt" not in readme
        assert "Zrodlowyszukiwacz" not in readme
        assert "src\\Wyswietlator\\app.py" in readme

    def test_rag_documents__published_files_exist_and_have_catalog_metadata(self) -> None:
        """Verify a clean clone can ingest the two documents that the README tells it to load."""
        catalog = load_catalog(DEFAULT_CATALOG_PATH)
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        for name in PUBLISHED_RAG_DOCUMENTS:
            assert (PROJECT_ROOT / "src" / "Zrodlowyszukiwator" / "RAG" / "RAW" / name).is_file()
            assert name in catalog
            assert f"--file src\\Zrodlowyszukiwator\\RAG\\RAW\\{name}" in readme

    def test_modelfiles__register_both_local_models_with_an_explicit_context_window(self) -> None:
        """Verify the model setup from the README does not depend on Ollama's small default window."""
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        for model in ("bielik", "pllum"):
            modelfile = (PROJECT_ROOT / "modele" / model / "Modelfile").read_text(encoding="utf-8")
            gguf = re.search(r"^FROM \./(\S+\.gguf)$", modelfile, flags=re.M)

            assert gguf is not None
            assert re.search(r"^PARAMETER num_ctx \d+$", modelfile, flags=re.M)
            assert f"modele\\{model}\\{gguf.group(1)}" in readme
            assert f"ollama create pismak-{model} -f modele\\{model}\\Modelfile" in readme

    def test_streamlit_config__hides_error_details_and_routes_unhandled_errors_to_the_logger(
        self,
    ) -> None:
        """Verify the published configuration cannot show a traceback to a demo user or lose it."""
        settings = tomllib.loads(
            (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
        )

        assert settings["client"]["showErrorDetails"] == "none"
        assert settings["logger"]["enableRich"] is False

    def test_readmes__reference_only_files_that_exist_in_the_repository(self) -> None:
        """Verify setup instructions never point a new user at a file the repository lacks."""
        readmes = [PROJECT_ROOT / "README.md", *sorted((PROJECT_ROOT / "src").rglob("README.md"))]
        missing = []

        for readme in readmes:
            text = readme.read_text(encoding="utf-8")
            for reference in re.findall(r"`([^`\n]+)`", text):
                path = reference.replace("\\", "/")
                if "/" not in path or not path.endswith(REPOSITORY_FILE_SUFFIXES):
                    continue
                if any(marker in path for marker in ("<", "*", " ", ":", "%", "...")):
                    continue
                if not ((PROJECT_ROOT / path).exists() or (readme.parent / path).exists()):
                    missing.append(f"{readme.relative_to(PROJECT_ROOT).as_posix()}: {reference}")

        assert missing == []
