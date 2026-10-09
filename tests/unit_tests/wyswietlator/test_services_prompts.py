from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WYSWIETLATOR_DIR = PROJECT_ROOT / "src" / "Wyswietlator"
if str(WYSWIETLATOR_DIR) not in sys.path:
    sys.path.insert(0, str(WYSWIETLATOR_DIR))


class TestPromptSources:
    def test_services__uses_canonical_domain_prompts(self) -> None:
        """Verify Wyswietlator reuses prompts from Dopytywator and Pismogenerator."""
        services = importlib.import_module("services")

        assert not hasattr(services, "generate_structured")
        assert not hasattr(services, "build_case_brief")
        assert not hasattr(services, "generate_draft")
