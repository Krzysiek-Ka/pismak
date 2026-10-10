from __future__ import annotations

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator
from src.Pismogenerator.constants import PISMOGENERATOR_MAX_OUTPUT_TOKENS
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack, ResearchPlan


class TestBuildDraft:
    def test_build_draft__passes_stage_output_token_limit(self, monkeypatch) -> None:
        """Verify letter generation uses its larger, dedicated output token budget."""
        captured: dict[str, object] = {}

        class FakeLLM:
            def generate_structured(self, **kwargs):
                captured.update(kwargs)
                return DraftLetter(
                    recipient="Urząd",
                    request_summary="Wniosek",
                    argumentation_markdown="A" * 200,
                    sources=[],
                )

        monkeypatch.setattr(
            pismogenerator.pismogenerator_helper_functions,
            "validate_draft",
            lambda **kwargs: [],
        )
        brief = CaseBrief(
            title="Sprawa",
            problem_description="Problem",
            desired_outcome="Rozwiązanie",
            requested_actions=["Sprawdzenie"],
        )
        evidence = EvidencePack(
            query_summary="Brak wyników",
            research_plan=ResearchPlan(case_summary="Brak wyników", intents=[]),
        )

        pismogenerator.build_draft(brief, evidence, FakeLLM())

        assert captured["max_output_tokens"] == PISMOGENERATOR_MAX_OUTPUT_TOKENS
