from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
