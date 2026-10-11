from __future__ import annotations

import pytest

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator
from src.Pismogenerator.constants import PISMOGENERATOR_MAX_OUTPUT_TOKENS
from src.Pismogenerator.pismogenerator import DraftValidationError
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

    def test_build_draft__includes_validation_details_in_exception(self, monkeypatch) -> None:
        """Verify rejected model sources are preserved for the UI and diagnostic log."""

        class FakeLLM:
            def generate_structured(self, **kwargs):
                return DraftLetter(
                    recipient="Urząd",
                    request_summary="Wniosek",
                    argumentation_markdown="A" * 200,
                    sources=["U1", "U2"],
                )

        monkeypatch.setattr(
            pismogenerator.pismogenerator_helper_functions,
            "validate_draft",
            lambda **kwargs: ["Model dodał niedozwolone źródła: U1; U2"],
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

        with pytest.raises(DraftValidationError) as error:
            pismogenerator.build_draft(brief, evidence, FakeLLM())

        assert "DraftLetter nie przeszedł walidacji" in str(error.value)
        assert "Model dodał niedozwolone źródła: U1; U2" in str(error.value)
        assert error.value.draft is not None
        assert error.value.draft.sources == ["U1", "U2"]
