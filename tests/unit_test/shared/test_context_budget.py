from __future__ import annotations

from types import SimpleNamespace

import pytest

from src._shared import context_budget


class TestContextWindowTokens:
    def test_context_window_tokens__uses_provider_default_without_configuration(
        self, monkeypatch
    ) -> None:
        """Verify an unconfigured installation gets the window the local models can hold."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        monkeypatch.delenv("PISMAK_OPENROUTER_CONTEXT_TOKENS", raising=False)

        assert context_budget.context_window_tokens("ollama") == 8192
        assert context_budget.context_window_tokens("openrouter") == 32768

    def test_context_window_tokens__reads_override_for_the_selected_provider_only(
        self, monkeypatch
    ) -> None:
        """Verify a larger local window does not change the budget of the remote provider."""
        monkeypatch.setenv("PISMAK_CONTEXT_TOKENS", "16384")
        monkeypatch.delenv("PISMAK_OPENROUTER_CONTEXT_TOKENS", raising=False)

        assert context_budget.context_window_tokens("ollama") == 16384
        assert context_budget.context_window_tokens("openrouter") == 32768

    def test_context_window_tokens__treats_unknown_provider_as_local(self, monkeypatch) -> None:
        """Verify an unrecognized provider falls back to the smaller, safer local window."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        assert context_budget.context_window_tokens("local") == 8192

    def test_context_window_tokens__rejects_non_numeric_value(self, monkeypatch) -> None:
        """Verify a typo in the setting is named instead of surfacing as a bare int() error."""
        monkeypatch.setenv("PISMAK_CONTEXT_TOKENS", "8k")

        with pytest.raises(ValueError, match="PISMAK_CONTEXT_TOKENS"):
            context_budget.context_window_tokens("ollama")

    def test_context_window_tokens__rejects_window_too_small_for_any_answer(
        self, monkeypatch
    ) -> None:
        """Verify a window that could never hold an answer is reported as a configuration error."""
        monkeypatch.setenv("PISMAK_CONTEXT_TOKENS", "1024")

        with pytest.raises(ValueError, match="PISMAK_CONTEXT_TOKENS"):
            context_budget.context_window_tokens("ollama")

    def test_context_window_tokens__uses_full_model_window_when_no_limit_is_set(
        self, monkeypatch
    ) -> None:
        """Verify the selected model's own window is used whole, above and below the fallback."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        assert context_budget.context_window_tokens("ollama", 131072) == 131072
        assert context_budget.context_window_tokens("ollama", 4096) == 4096

    def test_context_window_tokens__caps_model_window_with_the_optional_limit(
        self, monkeypatch
    ) -> None:
        """Verify the setting only ever lowers the window and never raises a smaller model."""
        monkeypatch.setenv("PISMAK_CONTEXT_TOKENS", "32768")

        assert context_budget.context_window_tokens("ollama", 131072) == 32768
        assert context_budget.context_window_tokens("ollama", 8192) == 8192


class TestFitOutputTokens:
    def test_fit_output_tokens__keeps_stage_limit_for_short_prompt(self, monkeypatch) -> None:
        """Verify a short prompt leaves the configured output limit untouched."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        result = context_budget.fit_output_tokens(
            provider="ollama", prompt_chars=400, max_output_tokens=4096, stage="test"
        )

        assert result == 4096

    def test_fit_output_tokens__lowers_limit_so_prompt_and_answer_share_the_window(
        self, monkeypatch
    ) -> None:
        """Verify the output limit shrinks when the prompt takes most of the context window."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        prompt_chars = 12_000

        result = context_budget.fit_output_tokens(
            provider="ollama", prompt_chars=prompt_chars, max_output_tokens=8192, stage="test"
        )

        assert context_budget.MIN_OUTPUT_TOKENS <= result < 8192
        assert (
            context_budget.estimate_tokens(prompt_chars)
            + result
            + context_budget.SAFETY_MARGIN_TOKENS
            <= 8192
        )

    def test_fit_output_tokens__rejects_prompt_that_leaves_no_room_for_an_answer(
        self, monkeypatch
    ) -> None:
        """Verify an oversized prompt fails loudly instead of being truncated by the server."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        with pytest.raises(context_budget.LLMInputLimitError, match="oknie kontekstu"):
            context_budget.fit_output_tokens(
                provider="ollama", prompt_chars=40_000, max_output_tokens=4096, stage="test"
            )

    def test_fit_output_tokens__allows_small_stage_limit_below_the_general_minimum(
        self, monkeypatch
    ) -> None:
        """Verify a stage asking for a short answer is not rejected for being under the floor."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        result = context_budget.fit_output_tokens(
            provider="ollama", prompt_chars=400, max_output_tokens=256, stage="test"
        )

        assert result == 256

    def test_fit_output_tokens__keeps_stage_limit_for_long_prompt_in_a_large_model_window(
        self, monkeypatch
    ) -> None:
        """Verify a prompt too big for the fallback window is fine for a model that can hold it."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        result = context_budget.fit_output_tokens(
            provider="ollama",
            prompt_chars=40_000,
            max_output_tokens=4096,
            stage="test",
            context_tokens=131072,
        )

        assert result == 4096

    def test_fit_output_tokens__reports_the_model_window_on_rejection(self, monkeypatch) -> None:
        """Verify the error carries the window so the UI can tell the user which limit applied."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        with pytest.raises(context_budget.LLMInputLimitError) as error:
            context_budget.fit_output_tokens(
                provider="ollama",
                prompt_chars=12_000,
                max_output_tokens=4096,
                stage="test",
                context_tokens=4096,
            )

        assert error.value.context_tokens == 4096


class TestInputCharBudget:
    def test_input_char_budget__prompt_at_the_budget_keeps_the_reserved_output(
        self, monkeypatch
    ) -> None:
        """Verify a prompt trimmed to the budget still gets the output room it was promised."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        budget = context_budget.input_char_budget("ollama", reserved_output_tokens=2048)

        result = context_budget.fit_output_tokens(
            provider="ollama", prompt_chars=budget, max_output_tokens=8192, stage="test"
        )

        assert result >= 2048

    def test_input_char_budget__returns_zero_when_reservation_exceeds_the_window(
        self, monkeypatch
    ) -> None:
        """Verify an impossible reservation yields an empty budget rather than a negative one."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        assert context_budget.input_char_budget("ollama", reserved_output_tokens=20_000) == 0

    def test_input_char_budget__grows_with_the_model_window(self, monkeypatch) -> None:
        """Verify a model with a larger window is given proportionally more prompt room."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        fallback = context_budget.input_char_budget("ollama", reserved_output_tokens=2048)

        result = context_budget.input_char_budget(
            "ollama", reserved_output_tokens=2048, context_tokens=131072
        )

        assert result > 10 * fallback


class TestUsageSummary:
    def test_usage_summary__reports_provider_token_counts(self) -> None:
        """Verify real token counts reach the log so the estimate can be checked against them."""
        completion = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=340)
        )

        assert context_budget.usage_summary(completion) == (
            "prompt_tokens=1200 completion_tokens=340"
        )

    def test_usage_summary__uses_placeholders_when_provider_omits_usage(self) -> None:
        """Verify a response without usage data still produces a parseable log fragment."""
        assert context_budget.usage_summary(SimpleNamespace()) == (
            "prompt_tokens=- completion_tokens=-"
        )
