from __future__ import annotations

from types import SimpleNamespace

import pytest

from src._shared import serwer
from src._shared.context_budget import LLMInputLimitError, estimate_tokens


class TestSetUpClient:
    def test_set_up_client__configures_openrouter_without_sdk_retries(self, monkeypatch) -> None:
        """Verify the selected remote provider uses its exact endpoint, key, and model alias."""
        calls: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                calls.update(kwargs)

        monkeypatch.setenv("OPENROUTER_KEY", "test-key")
        monkeypatch.setattr(serwer, "OpenAI", FakeOpenAI)

        client = serwer.set_up_client("openrouter")

        assert isinstance(client, FakeOpenAI)
        assert calls == {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
            "timeout": 60.0,
            "max_retries": 0,
        }
        assert serwer.OPENROUTER_MODEL == "openrouter/free"

    @pytest.mark.parametrize(
        "api_key",
        ["", "   ", serwer.OPENROUTER_KEY_PLACEHOLDER],
    )
    def test_set_up_client__rejects_missing_or_blank_openrouter_key(
        self, monkeypatch, api_key: str
    ) -> None:
        """Verify empty credentials fail before constructing a network client."""
        monkeypatch.setenv("OPENROUTER_KEY", api_key)

        with pytest.raises(serwer.LLMConfigurationError, match="OPENROUTER_KEY"):
            serwer.set_up_client("openrouter")

    def test_set_up_client__rejects_unknown_provider(self) -> None:
        """Verify an unsupported provider is reported instead of silently selecting Ollama."""
        with pytest.raises(ValueError, match="Nieznany dostawca"):
            serwer.set_up_client("unsupported")


class _Answer(serwer.BaseModel):
    value: str


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model="vendor/model",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"value":"ok"}'),
                )
            ],
        )


class TestCallStructured:
    def test_call_structured__lowers_output_limit_for_a_long_prompt(self, monkeypatch) -> None:
        """Verify a long prompt reduces max_tokens so prompt and answer fit the local window."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        completions = _Completions()
        llm = serwer.LLMClient(SimpleNamespace(chat=SimpleNamespace(completions=completions)), "m")
        user_prompt = "a" * 12_000

        llm._call_structured(
            system_prompt="system",
            user_prompt=user_prompt,
            response_model=_Answer,
            temperature=0.0,
            max_output_tokens=8192,
            attempt=1,
        )

        sent = completions.calls[0]["max_tokens"]
        assert sent < 8192
        assert estimate_tokens(len("system") + len(user_prompt)) + sent <= 8192

    def test_call_structured__rejects_oversized_prompt_without_calling_the_model(
        self, monkeypatch
    ) -> None:
        """Verify a prompt beyond the window is refused before the server can truncate it."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        completions = _Completions()
        llm = serwer.LLMClient(SimpleNamespace(chat=SimpleNamespace(completions=completions)), "m")

        with pytest.raises(LLMInputLimitError):
            llm.generate_structured(
                system_prompt="system",
                user_prompt="a" * 40_000,
                response_model=_Answer,
                max_output_tokens=4096,
            )

        assert completions.calls == []

    def test_call_structured__uses_larger_window_for_openrouter(self, monkeypatch) -> None:
        """Verify the remote provider keeps its full output limit for a prompt too big locally."""
        monkeypatch.delenv("PISMAK_OPENROUTER_CONTEXT_TOKENS", raising=False)
        completions = _Completions()
        llm = serwer.LLMClient(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            "openrouter/free",
            provider="openrouter",
        )

        llm._call_structured(
            system_prompt="system",
            user_prompt="a" * 12_000,
            response_model=_Answer,
            temperature=0.0,
            max_output_tokens=8192,
            attempt=1,
        )

        assert completions.calls[0]["max_tokens"] == 8192
