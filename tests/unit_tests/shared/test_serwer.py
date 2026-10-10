"""Tests for the shared LLM server client utilities."""

from __future__ import annotations

import logging
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from src._shared import serwer
from src._shared.llm_runtime import llm_generation_lock


def _completion(
    content: str | None,
    *,
    finish_reason: str = "stop",
    model: str = "vendor/model",
) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ],
    )


def _client(completions) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class TestCheckAvailableModel:
    def test_check_available_model__prints_visible_models(self, monkeypatch, capsys) -> None:
        """Verify that available Ollama models are listed from the API response."""

        class FakeModel:
            def __init__(self, model_id: str) -> None:
                self.id = model_id

        class FakeListResponse:
            def __init__(self, items) -> None:
                self.data = items

        class FakeModels:
            def list(self):
                return FakeListResponse([FakeModel("model-a"), FakeModel("model-b")])

        class FakeClient:
            def __init__(self, **kwargs) -> None:
                self.models = FakeModels()

        monkeypatch.setattr(serwer, "OpenAI", FakeClient)

        serwer.check_available_model()

        captured = capsys.readouterr().out
        assert "Modele widoczne w Ollama:" in captured
        assert "model-a" in captured
        assert "model-b" in captured


class TestSetUpClient:
    def test_set_up_client__creates_local_ollama_client(self, monkeypatch) -> None:
        """Verify the local OpenAI-compatible client is configured for Ollama."""
        calls = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                calls["kwargs"] = kwargs
                self.models = object()

        monkeypatch.setattr(serwer, "OpenAI", FakeOpenAI)

        client = serwer.set_up_client("local")

        assert isinstance(client, FakeOpenAI)
        assert calls["kwargs"]["base_url"] == serwer.OLLAMA_BASE_URL
        assert calls["kwargs"]["api_key"] == "ollama"


class TestLLMClient:
    def test_llm_generation_lock__serializes_calls_and_notifies_waiters(self) -> None:
        """Verify concurrent local generations queue and notify a waiting session."""
        first_entered = Event()
        release_first = Event()
        second_entered = Event()
        notified = Event()

        def first_call() -> None:
            with llm_generation_lock():
                first_entered.set()
                release_first.wait(timeout=2)

        def second_call() -> None:
            with llm_generation_lock(on_wait=notified.set):
                second_entered.set()

        first = Thread(target=first_call)
        second = Thread(target=second_call)
        first.start()
        assert first_entered.wait(timeout=1)
        second.start()

        assert notified.wait(timeout=1)
        assert not second_entered.is_set()

        release_first.set()
        first.join(timeout=1)
        second.join(timeout=1)

        assert not first.is_alive()
        assert not second.is_alive()
        assert second_entered.is_set()

    def test_llm_client__call_structured_returns_parsed_model(self) -> None:
        """Verify raw structured JSON is validated and returned as the response model."""

        class FakeModel(serwer.BaseModel):
            value: str

        class FakeCompletions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return _completion('{"value":"ok"}', model="vendor/actual-model")

        completions = FakeCompletions()
        client = serwer.LLMClient(_client(completions), "model-name")

        result = client._call_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=FakeModel,
            temperature=0.0,
            max_output_tokens=512,
            attempt=1,
        )

        assert isinstance(result, FakeModel)
        assert result.value == "ok"
        assert client.last_model == "vendor/actual-model"
        assert completions.calls[0]["max_tokens"] == 512
        assert completions.calls[0]["response_format"]["json_schema"]["strict"] is True

    def test_llm_client__generate_structured_raises_error_when_retries_fail(self) -> None:
        """Verify a repeated structured-output failure raises the project-specific runtime error."""

        class ResponseModel(serwer.BaseModel):
            ok: bool

        class FakeCompletions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise serwer.LLMResponseError("bad payload")

        completions = FakeCompletions()
        client = serwer.LLMClient(_client(completions), "model-name")

        with pytest.raises(serwer.LLMResponseError, match="dwukrotnie"):
            client.generate_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=ResponseModel,
                max_output_tokens=768,
            )
        assert completions.calls == 2

    def test_llm_client__does_not_retry_transport_errors(self) -> None:
        """Verify network/provider failures are not repeated as structured-output repairs."""

        class ResponseModel(serwer.BaseModel):
            ok: bool

        class FakeCompletions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise RuntimeError("provider unavailable")

        completions = FakeCompletions()
        client = serwer.LLMClient(_client(completions), "model-name")

        with pytest.raises(RuntimeError, match="provider unavailable"):
            client.generate_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=ResponseModel,
                max_output_tokens=768,
            )

        assert completions.calls == 1

    def test_llm_client__caps_each_repair_attempt_with_the_same_output_limit(self) -> None:
        """Verify the one repair request cannot exceed the original output token cap."""

        class ResponseModel(serwer.BaseModel):
            ok: bool

        class FakeCompletions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise serwer.LLMResponseError("invalid response")
                return _completion('{"ok":true}', model="model-name")

        completions = FakeCompletions()
        client = _client(completions)
        llm = serwer.LLMClient(client, "model-name")

        result = llm.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ResponseModel,
            max_output_tokens=384,
        )

        assert result == ResponseModel(ok=True)
        assert [call["max_tokens"] for call in completions.calls] == [384, 384]

    def test_llm_client__retries_with_a_compact_prompt_when_output_limit_is_reached(
        self,
    ) -> None:
        """Verify a truncated response is retried once with an instruction to shorten it."""

        class ResponseModel(serwer.BaseModel):
            ok: bool

        class FakeCompletions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    return _completion(
                        '{"short_summary":"long',
                        finish_reason="length",
                        model="vendor/long-output",
                    )
                return _completion('{"ok":true}', model="vendor/long-output")

        completions = FakeCompletions()
        client = _client(completions)
        llm = serwer.LLMClient(client, "openrouter/free", provider="openrouter")

        capture = _CaptureHandler()
        serwer.LOGGER.addHandler(capture)
        try:
            result = llm.generate_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=ResponseModel,
                max_output_tokens=1024,
            )
        finally:
            serwer.LOGGER.removeHandler(capture)

        assert result == ResponseModel(ok=True)
        assert len(completions.calls) == 2
        assert "krótszą" in completions.calls[1]["messages"][1]["content"]
        assert all(call["max_tokens"] == 1024 for call in completions.calls)
        assert llm.last_model == "vendor/long-output"
        assert completions.calls[0]["response_format"] == completions.calls[1]["response_format"]
        messages = "\n".join(capture.messages)
        assert "llm_model_response outcome=truncated" in messages
        assert "short_summary" in messages

    def test_llm_client__logs_invalid_model_response_content(self) -> None:
        """Verify both rejected structured responses are recorded for diagnosing schema errors."""

        class ResponseModel(serwer.BaseModel):
            required: str

        class FakeCompletions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return _completion('{"unexpected":"bad model answer"}')

        completions = FakeCompletions()
        llm = serwer.LLMClient(_client(completions), "model-name")

        capture = _CaptureHandler()
        serwer.LOGGER.addHandler(capture)
        try:
            with pytest.raises(serwer.LLMResponseError, match="dwukrotnie"):
                llm.generate_structured(
                    system_prompt="system",
                    user_prompt="user",
                    response_model=ResponseModel,
                    max_output_tokens=512,
                )
        finally:
            serwer.LOGGER.removeHandler(capture)

        assert completions.calls == 2
        messages = "\n".join(capture.messages)
        assert messages.count("llm_model_response outcome=invalid") == 2
        assert "bad model answer" in messages

    def test_llm_client__returns_valid_partial_response_after_compact_retry_is_truncated(
        self,
    ) -> None:
        """Verify a truncated JSON tail is recovered only when the partial model validates."""

        class ResponseModel(serwer.BaseModel):
            summary: str
            completed: bool

        class FakeCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return _completion(
                    '{"summary":"partial but complete","completed":true',
                    finish_reason="length",
                    model="vendor/long-output",
                )

        completions = FakeCompletions()
        client = _client(completions)
        llm = serwer.LLMClient(client, "openrouter/free", provider="openrouter")

        result = llm.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=ResponseModel,
            max_output_tokens=1024,
        )

        assert result == ResponseModel(summary="partial but complete", completed=True)
        assert completions.calls == 2

    def test_llm_client__keeps_unrecoverable_partial_response_on_output_limit_error(self) -> None:
        """Verify unusable truncated JSON is retained for the UI warning instead of logged as an API error."""

        class ResponseModel(serwer.BaseModel):
            required: str

        class FakeCompletions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return _completion(
                    '{"incomplete":',
                    finish_reason="length",
                    model="vendor/long-output",
                )

        completions = FakeCompletions()
        client = _client(completions)
        llm = serwer.LLMClient(client, "openrouter/free", provider="openrouter")

        with pytest.raises(serwer.LLMOutputLimitError) as error:
            llm.generate_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=ResponseModel,
                max_output_tokens=1024,
            )

        assert completions.calls == 2
        assert error.value.partial_response == '{"incomplete":'
