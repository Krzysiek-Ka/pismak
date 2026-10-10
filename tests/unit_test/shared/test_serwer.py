from __future__ import annotations

import pytest

from src._shared import serwer


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
