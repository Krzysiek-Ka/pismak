"""Tests for the Pismogenerator prompt builders."""

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import prompts
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack, ResearchPlan


class TestDraftUserPrompt:
    def test_drafr_user_prompt__contains_case_brief_and_allowed_sources(self) -> None:
        """Verify the draft prompt packages the brief and explicit source list for the model."""
        brief = CaseBrief(
            title="Brak dostępu",
            problem_description="Brak dostępu do dokumentacji.",
            desired_outcome="Przywrócenie dostępu.",
            requested_actions=["Weryfikacja dokumentacji"],
        )

        pack = EvidencePack(
            query_summary="Sprawa",
            research_plan=ResearchPlan(case_summary="Sprawa", intents=[]),
        )
        prompt = prompts.drafr_user_prompt(brief, pack, ["Dokument 1"])

        assert "CASE BRIEF:" in prompt
        assert "DOZWOLONE IDENTYFIKATORY ŹRÓDEŁ:" in prompt
        assert "Dokument 1" in prompt
        assert "U1" in prompt
        assert "EvidencePack" in prompt
        assert "Utwórz DraftLetter" in prompt
