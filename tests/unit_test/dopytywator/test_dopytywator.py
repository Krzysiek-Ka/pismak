from __future__ import annotations

from src.Dopytywator import dopytywator
from src.Dopytywator.constants import DOPYTYWATOR_MAX_OUTPUT_TOKENS
from src.Dopytywator.structures import CaseBrief


class TestBuildBrief:
    def test_build_brief__passes_stage_output_token_limit(self, monkeypatch) -> None:
        """Verify brief generation explicitly uses the Dopytywator's short-output budget."""
        captured: dict[str, object] = {}

        class FakeLLM:
            def generate_structured(self, **kwargs):
                captured.update(kwargs)
                return CaseBrief(
                    title="Sprawa",
                    problem_description="Problem",
                    desired_outcome="Rozwiązanie",
                    requested_actions=["Sprawdzenie"],
                )

        monkeypatch.setattr(
            dopytywator.dopytywator_helper_functions,
            "print_case_brief",
            lambda brief: None,
        )

        dopytywator.build_brief([], FakeLLM())

        assert captured["max_output_tokens"] == DOPYTYWATOR_MAX_OUTPUT_TOKENS
