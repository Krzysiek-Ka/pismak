"""Tests for the shared LLM server client utilities."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from openai import APITimeoutError

from src._shared import llm_runtime, serwer
from src._shared.context_budget import LLMInputLimitError, estimate_tokens
from src._shared.llm_runtime import LLMBusyError, llm_generation_lock
from tests.unit_test.conftest import OPENROUTER_REQUEST


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


class _Plan(serwer.BaseModel):
    summary: str
    items: list[_Answer] = []


class _Replies:
    """Return prepared completions one per call."""

    def __init__(self, *completions: SimpleNamespace) -> None:
        self.completions = list(completions)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.completions.pop(0)


class TestCheckAvailableModel:
    def test_check_available_model__listed_models_are_printed(self, monkeypatch, capsys) -> None:
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

    def test_check_available_model__explains_that_ollama_is_not_running(self, monkeypatch) -> None:
        """Verify a refused connection names the Ollama address to check."""

        class Unreachable:
            def __init__(self, **kwargs) -> None:
                self.models = SimpleNamespace(list=self.fail)

            def fail(self):
                raise ConnectionError("refused")

        monkeypatch.setenv("OLLAMA_URL", "http://ollama.test:11434")
        monkeypatch.setattr(serwer, "OpenAI", Unreachable)

        with pytest.raises(RuntimeError, match="Sprawdź, czy Ollama działa") as error:
            serwer.check_available_model()

        assert "http://ollama.test:11434/v1/" in str(error.value)
        assert isinstance(error.value.__cause__, ConnectionError)

    def test_check_available_model__reports_an_ollama_without_models(self, monkeypatch) -> None:
        """Verify a fresh Ollama install is told to pull a model instead of showing an empty list."""

        class Empty:
            def __init__(self, **kwargs) -> None:
                self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

        monkeypatch.setattr(serwer, "OpenAI", Empty)

        with pytest.raises(RuntimeError, match="ollama list"):
            serwer.check_available_model()


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
            "timeout": 120.0,
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

    def test_set_up_client__gives_local_model_fifteen_minutes_without_sdk_retries(
        self, monkeypatch
    ) -> None:
        """Verify a slow local generation gets one long attempt instead of three repeated ones."""
        calls: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                calls.update(kwargs)

        monkeypatch.setattr(serwer, "OpenAI", FakeOpenAI)

        serwer.set_up_client("local")

        assert calls["timeout"] == 900.0
        assert calls["max_retries"] == 0

    def test_set_up_client__creates_local_ollama_client(self, monkeypatch) -> None:
        """Verify the local OpenAI-compatible client is configured for Ollama."""
        calls = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                calls["kwargs"] = kwargs
                self.models = object()

        monkeypatch.delenv("OLLAMA_URL", raising=False)
        monkeypatch.setattr(serwer, "OpenAI", FakeOpenAI)

        client = serwer.set_up_client("local")

        assert isinstance(client, FakeOpenAI)
        assert calls["kwargs"]["base_url"] == "http://127.0.0.1:11434/v1/"
        assert calls["kwargs"]["api_key"] == "ollama"

    def test_set_up_client__uses_the_ollama_address_from_the_environment(self, monkeypatch) -> None:
        """Verify OLLAMA_URL from .env moves the client the same way it moves the UI client."""
        calls: dict[str, object] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                calls.update(kwargs)

        monkeypatch.setenv("OLLAMA_URL", "http://ollama.test:11434/")
        monkeypatch.setattr(serwer, "OpenAI", FakeOpenAI)

        serwer.set_up_client("local")

        assert calls["base_url"] == "http://ollama.test:11434/v1/"


class TestLLMClientGenerateStructured:
    def test_generate_structured__openrouter_request_requires_providers_that_honour_the_schema(
        self,
    ) -> None:
        """Verify OpenRouter gets require_parameters and the local Ollama request stays plain."""
        openrouter, ollama = _Completions(), _Completions()

        for completions, provider in ((openrouter, "openrouter"), (ollama, "ollama")):
            serwer.LLMClient(_client(completions), "m", provider=provider).generate_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=_Answer,
                max_output_tokens=512,
            )

        assert openrouter.calls[0]["extra_body"] == {"provider": {"require_parameters": True}}
        assert "extra_body" not in ollama.calls[0]

    def test_generate_structured__seed_is_sent_to_ollama_on_every_attempt(self) -> None:
        """Verify the first request and the schema repair both carry the caller's seed."""
        completions = _Replies(_completion("nie JSON"), _completion('{"value": "ok"}'))

        serwer.LLMClient(_client(completions), "m", provider="ollama").generate_structured(
            system_prompt="s",
            user_prompt="u",
            response_model=_Answer,
            max_output_tokens=512,
            seed=47,
        )

        assert [call["seed"] for call in completions.calls] == [47, 47]

    def test_generate_structured__seed_is_not_sent_to_openrouter_or_without_a_seed(self) -> None:
        """Verify OpenRouter, where require_parameters would narrow routing, gets no seed."""
        openrouter, unseeded = _Completions(), _Completions()

        serwer.LLMClient(_client(openrouter), "m", provider="openrouter").generate_structured(
            system_prompt="s",
            user_prompt="u",
            response_model=_Answer,
            max_output_tokens=512,
            seed=47,
        )
        serwer.LLMClient(_client(unseeded), "m", provider="ollama").generate_structured(
            system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=512
        )

        assert "seed" not in openrouter.calls[0]
        assert "seed" not in unseeded.calls[0]

    def test_generate_structured__json_cut_off_with_stop_logs_finish_reason_and_usage(self) -> None:
        """Verify a JSON cut off with finish_reason=stop can be told apart from a length limit."""

        class CutOff:
            def create(self, **kwargs):
                completion = _completion('{"value": "urwa', finish_reason="stop")
                completion.usage = SimpleNamespace(prompt_tokens=900, completion_tokens=12)
                return completion

        capture = _CaptureHandler()
        serwer.LOGGER.addHandler(capture)
        try:
            with pytest.raises(serwer.LLMResponseError, match="dwukrotnie"):
                serwer.LLMClient(_client(CutOff()), "m").generate_structured(
                    system_prompt="system",
                    user_prompt="user",
                    response_model=_Answer,
                    max_output_tokens=512,
                )
        finally:
            serwer.LOGGER.removeHandler(capture)

        failed = next(m for m in capture.messages if m.startswith("llm_request_failed"))
        assert "finish_reason=stop" in failed
        assert "prompt_tokens=900 completion_tokens=12" in failed

    def test_generate_structured__openrouter_while_local_model_is_busy_runs_without_waiting(
        self,
    ) -> None:
        """Verify an OpenRouter stage is not queued behind a long Ollama generation."""
        completions = _Completions()
        waits: list[str] = []
        llm = serwer.LLMClient(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            "openrouter/free",
            on_wait=lambda: waits.append("wait"),
            provider="openrouter",
        )

        with llm_generation_lock():
            result = llm.generate_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=_Answer,
                max_output_tokens=512,
            )

        assert result.value == "ok"
        assert waits == []

    def test_generate_structured__local_model_busy_past_the_wait_limit_raises_without_a_request(
        self, monkeypatch
    ) -> None:
        """Verify an Ollama stage that waited too long fails clearly and sends nothing."""
        monkeypatch.setattr(llm_runtime, "LLM_LOCK_WAIT_TIMEOUT_SECONDS", 0.05)
        completions = _Completions()
        llm = serwer.LLMClient(SimpleNamespace(chat=SimpleNamespace(completions=completions)), "m")

        with llm_generation_lock():
            with pytest.raises(LLMBusyError):
                llm.generate_structured(
                    system_prompt="system",
                    user_prompt="user",
                    response_model=_Answer,
                    max_output_tokens=512,
                )

        assert completions.calls == []

    def test_generate_structured__long_prompt_lowers_the_output_limit_to_fit_the_window(
        self, monkeypatch
    ) -> None:
        """Verify a long prompt reduces max_tokens so prompt and answer fit the local window."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        completions = _Completions()
        llm = serwer.LLMClient(SimpleNamespace(chat=SimpleNamespace(completions=completions)), "m")
        user_prompt = "a" * 12_000

        llm.generate_structured(
            system_prompt="system",
            user_prompt=user_prompt,
            response_model=_Answer,
            max_output_tokens=8192,
        )

        sent = completions.calls[0]["max_tokens"]
        assert sent < 8192
        assert estimate_tokens(len("system") + len(user_prompt)) + sent <= 8192

    def test_generate_structured__prompt_beyond_the_window_raises_without_a_request(
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

    def test_generate_structured__openrouter_long_prompt_keeps_the_full_output_limit(
        self, monkeypatch
    ) -> None:
        """Verify the remote provider keeps its full output limit for a prompt too big locally."""
        monkeypatch.delenv("PISMAK_OPENROUTER_CONTEXT_TOKENS", raising=False)
        completions = _Completions()
        llm = serwer.LLMClient(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            "openrouter/free",
            provider="openrouter",
        )

        llm.generate_structured(
            system_prompt="system",
            user_prompt="a" * 12_000,
            response_model=_Answer,
            max_output_tokens=8192,
        )

        assert completions.calls[0]["max_tokens"] == 8192

    def test_generate_structured__large_model_window_keeps_the_full_output_limit(
        self, monkeypatch
    ) -> None:
        """Verify the selected model's own window replaces the small fallback budget."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        completions = _Completions()
        llm = serwer.LLMClient(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            "pismak-pllum",
            context_tokens=131072,
        )

        llm.generate_structured(
            system_prompt="system",
            user_prompt="a" * 40_000,
            response_model=_Answer,
            max_output_tokens=8192,
        )

        assert completions.calls[0]["max_tokens"] == 8192

    def test_generate_structured__prompt_beyond_a_small_model_window_raises_with_the_window(
        self, monkeypatch
    ) -> None:
        """Verify a model served with a small window is not sent a prompt it would truncate."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        completions = _Completions()
        llm = serwer.LLMClient(
            SimpleNamespace(chat=SimpleNamespace(completions=completions)),
            "small-model",
            context_tokens=4096,
        )

        with pytest.raises(LLMInputLimitError) as error:
            llm.generate_structured(
                system_prompt="system",
                user_prompt="a" * 8_000,
                response_model=_Answer,
                max_output_tokens=4096,
            )

        assert error.value.context_tokens == 4096
        assert completions.calls == []

    def test_generate_structured__valid_json_returns_the_parsed_model(self) -> None:
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

        result = client.generate_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=FakeModel,
            max_output_tokens=512,
        )

        assert isinstance(result, FakeModel)
        assert result.value == "ok"
        assert client.last_model == "vendor/actual-model"
        assert completions.calls[0]["max_tokens"] == 512
        assert completions.calls[0]["response_format"]["json_schema"]["strict"] is True

    def test_generate_structured__two_invalid_answers_raise_a_readable_error(self) -> None:
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

        with pytest.raises(serwer.LLMResponseError, match="dwukrotnie") as error:
            client.generate_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=ResponseModel,
                max_output_tokens=768,
            )

        assert "ResponseModel" in str(error.value)
        assert "bad payload" in str(error.value)
        assert completions.calls == 2

    def test_generate_structured__provider_error_is_raised_without_a_retry(self) -> None:
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

    def test_generate_structured__repair_attempt_keeps_the_same_output_limit(self) -> None:
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

    def test_generate_structured__truncated_answer_is_retried_once_with_a_compact_prompt(
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

    def test_generate_structured__invalid_answers_are_logged_with_their_content(self) -> None:
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

    def test_generate_structured__second_truncation_with_a_valid_prefix_returns_the_prefix(
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

    def test_generate_structured__second_truncation_without_a_valid_prefix_raises_with_the_partial_answer(
        self,
    ) -> None:
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

    def test_generate_structured__non_positive_output_limit_raises_before_any_request(self) -> None:
        """Verify a misconfigured stage budget fails before any request is sent."""
        completions = _Replies()
        llm = serwer.LLMClient(_client(completions), "m")

        with pytest.raises(ValueError, match="max_output_tokens musi być dodatnie"):
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=0
            )

        assert completions.calls == []

    def test_generate_structured__empty_openrouter_answer_is_repaired_once(self) -> None:
        """Verify an empty remote message is treated as invalid output and repaired once."""
        completions = _Replies(_completion(None), _completion('{"value": "ok"}'))
        llm = serwer.LLMClient(_client(completions), "openrouter/free", provider="openrouter")

        result = llm.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
        )

        assert result == _Answer(value="ok")
        assert "Model nie zwrócił danych." in completions.calls[1]["messages"][1]["content"]

    def test_generate_structured__empty_ollama_answer_raises_service_error_without_a_retry(
        self,
    ) -> None:
        """Verify a crashed local runner is reported as unavailable instead of being repaired."""
        completions = _Replies(_completion(None), _completion('{"value": "ok"}'))
        llm = serwer.LLMClient(_client(completions), "pismak-pllum:latest", provider="ollama")

        with pytest.raises(serwer.LLMServiceError, match="bez odpowiedzi"):
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
            )

        assert len(completions.calls) == 1

    def test_generate_structured__repair_prompt_names_the_error_without_the_rejected_value(
        self,
    ) -> None:
        """Verify the repair request explains the schema error but does not echo the answer."""
        completions = _Replies(
            _completion('{"value": ["tajny-fragment-odpowiedzi"]}'),
            _completion('{"value": "ok"}'),
        )
        llm = serwer.LLMClient(_client(completions), "m")

        llm.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
        )

        repair = completions.calls[1]["messages"][1]["content"]
        assert "value: Input should be a valid string (string_type)" in repair
        assert "tajny-fragment-odpowiedzi" not in repair

    def test_generate_structured__compact_retry_breaking_the_schema_raises_output_limit_error(
        self,
    ) -> None:
        """Verify a short but invalid retry is still reported as an output-limit problem."""
        completions = _Replies(
            _completion('{"val', finish_reason="length"), _completion('{"other": 1}')
        )
        llm = serwer.LLMClient(_client(completions), "m")

        with pytest.raises(serwer.LLMOutputLimitError, match="Skrócona odpowiedź") as error:
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
            )

        assert error.value.partial_response == '{"val'

    def test_generate_structured__truncation_during_the_repair_raises_output_limit_error(
        self,
    ) -> None:
        """Verify a repair that runs out of tokens is reported as truncation, not invalid data."""
        completions = _Replies(
            _completion("nie JSON"), _completion('{"value": "o', finish_reason="length")
        )
        llm = serwer.LLMClient(_client(completions), "m")

        with pytest.raises(serwer.LLMOutputLimitError):
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
            )

        assert len(completions.calls) == 2

    def test_generate_structured__routed_answer_records_the_model_that_answered(self) -> None:
        """Verify the model chosen by OpenRouter's router is recorded for the run metadata."""
        completions = _Replies(_completion('{"value": "ok"}', model="vendor/routed"))
        llm = serwer.LLMClient(_client(completions), "openrouter/free")

        llm.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
        )

        assert llm.last_model == "vendor/routed"

    @pytest.mark.parametrize(
        "first_answer",
        [
            pytest.param(None, id="empty"),
            pytest.param("Oto odpowiedź: wartość ok.", id="prose"),
            pytest.param('```json\n{"value": "ok"}\n```', id="markdown_fence"),
            pytest.param('{"other": "ok"}', id="wrong_schema"),
        ],
    )
    def test_generate_structured__unusable_first_answer_is_repaired_once(
        self, first_answer: str | None
    ) -> None:
        """Verify an answer that is not the bare schema JSON gets exactly one repair request."""
        completions = _Replies(_completion(first_answer), _completion('{"value": "ok"}'))
        llm = serwer.LLMClient(_client(completions), "openrouter/free", provider="openrouter")

        result = llm.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
        )

        assert result == _Answer(value="ok")
        assert len(completions.calls) == 2
        assert "Błędy walidacji" in completions.calls[1]["messages"][1]["content"]

    def test_generate_structured__two_fenced_answers_raise_a_readable_error(self) -> None:
        """Verify JSON in a Markdown fence is never unwrapped, so two such answers end the stage."""
        fenced = '```json\n{"value": "ok"}\n```'
        completions = _Replies(_completion(fenced), _completion(fenced))
        llm = serwer.LLMClient(_client(completions), "m")

        with pytest.raises(serwer.LLMResponseError, match="dwukrotnie.*_Answer"):
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
            )

        assert len(completions.calls) == 2

    def test_generate_structured__transient_openrouter_error_is_retried_once_with_the_same_prompt(
        self, transient_provider_error: Exception
    ) -> None:
        """Verify a passing OpenRouter failure gets one more identical request, which can succeed."""

        class FailsOnce:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise transient_provider_error
                return _completion('{"value": "ok"}')

        completions = FailsOnce()
        llm = serwer.LLMClient(_client(completions), "openrouter/free", provider="openrouter")

        result = llm.generate_structured(
            system_prompt="s",
            user_prompt="u",
            response_model=_Answer,
            temperature=0.2,
            max_output_tokens=50,
        )

        assert result == _Answer(value="ok")
        first, second = completions.calls
        assert first["messages"] == second["messages"]
        assert first["temperature"] == second["temperature"] == 0.2

    def test_generate_structured__transient_error_twice_is_raised_after_one_retry(
        self, transient_provider_error: Exception
    ) -> None:
        """Verify the retry is bounded: a second failure reaches the caller unchanged."""

        class Failing:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise transient_provider_error

        completions = Failing()
        llm = serwer.LLMClient(_client(completions), "openrouter/free", provider="openrouter")

        with pytest.raises(type(transient_provider_error)):
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
            )

        assert completions.calls == 2

    def test_generate_structured__permanent_provider_error_is_raised_without_a_retry(
        self, permanent_provider_error: Exception
    ) -> None:
        """Verify a rejected request, a bad key or the daily limit is not sent a second time."""

        class Failing:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise permanent_provider_error

        completions = Failing()
        llm = serwer.LLMClient(_client(completions), "openrouter/free", provider="openrouter")

        with pytest.raises(type(permanent_provider_error)):
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
            )

        assert completions.calls == 1

    def test_generate_structured__ollama_timeout_is_raised_without_a_retry(self) -> None:
        """Verify a local request that already took up to 15 minutes is not repeated."""

        class Failing:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise APITimeoutError(request=OPENROUTER_REQUEST)

        completions = Failing()
        llm = serwer.LLMClient(_client(completions), "pismak-bielik:latest", provider="ollama")

        with pytest.raises(APITimeoutError):
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
            )

        assert completions.calls == 1

    def test_generate_structured__every_attempt_sends_the_callers_output_limit(self) -> None:
        """Verify the truncation retry and the schema repair keep the limit the stage asked for."""
        truncated = _Replies(
            _completion('{"val', finish_reason="length"), _completion('{"value": "a"}')
        )
        invalid = _Replies(_completion("nie JSON"), _completion('{"value": "b"}'))

        for completions in (truncated, invalid):
            serwer.LLMClient(_client(completions), "m").generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=321
            )

        assert [call["max_tokens"] for call in truncated.calls + invalid.calls] == [321] * 4

    def test_generate_structured__answer_without_choices_raises_a_readable_error(self) -> None:
        """Verify a provider error sent with status 200 ends in the module's readable error."""
        error = {"error": {"message": "Provider returned error", "code": 502}}
        completions = _Replies(
            SimpleNamespace(model="m", choices=None, model_extra=error),
            SimpleNamespace(model="m", choices=None, model_extra=error),
        )
        llm = serwer.LLMClient(_client(completions), "openrouter/free", provider="openrouter")

        with pytest.raises(serwer.LLMResponseError, match="dwukrotnie") as raised:
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
            )

        assert "Dostawca modelu zwrócił błąd zamiast odpowiedzi: Provider returned error" in str(
            raised.value
        )
        assert len(completions.calls) == 2

    def test_generate_structured__answer_without_choices_is_repaired_once(self) -> None:
        """Verify one provider error inside a 200 response costs one retry, not the stage."""
        completions = _Replies(
            SimpleNamespace(model="m", choices=None), _completion('{"value": "ok"}')
        )
        llm = serwer.LLMClient(_client(completions), "openrouter/free", provider="openrouter")

        result = llm.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
        )

        assert result == _Answer(value="ok")
        assert "Dostawca modelu zwrócił błąd" in completions.calls[1]["messages"][1]["content"]


class TestStructuredOutputErrorDetail:
    def test_structured_output_error_detail__describes_validation_without_input_values(
        self,
    ) -> None:
        """Verify schema diagnostics identify missing fields but never include rejected content."""

        class ResponseModel(serwer.BaseModel):
            required: str

        with pytest.raises(serwer.ValidationError) as error:
            ResponseModel.model_validate({"extra": "private user content"})

        detail = serwer.structured_output_error_detail(error.value)

        assert "required" in detail
        assert "missing" in detail
        assert "private user content" not in detail

    def test_structured_output_error_detail__names_only_the_type_of_other_errors(self) -> None:
        """Verify an unexpected error cannot leak its message through the schema summary."""
        detail = serwer.structured_output_error_detail(RuntimeError("tajna treść"))

        assert detail == "RuntimeError"

    def test_structured_output_error_detail__keeps_the_message_of_an_llm_response_error(
        self,
    ) -> None:
        """Verify the application's own response errors stay readable in the log."""
        detail = serwer.structured_output_error_detail(serwer.LLMResponseError("pusta odpowiedź"))

        assert detail == "pusta odpowiedź"


class TestErrorLogDetail:
    def test_error_log_detail__keeps_the_provider_message_on_one_line(self) -> None:
        """Verify non-schema failures are logged with their own text instead of only a type."""
        detail = serwer.error_log_detail(RuntimeError("Error code: 429\n  rate limit exceeded"))

        assert detail == "Error code: 429 rate limit exceeded"

    def test_error_log_detail__hides_openrouter_and_provider_account_ids(self) -> None:
        """Verify a 404 relayed by OpenRouter keeps its text but not the account identifiers."""
        error = RuntimeError(
            "Error code: 404 - {'error': {'metadata': {'raw': '{\"detail\":\"Function "
            "\\'f1\\': Not found for account \\'acct-TEST-0000\\'\"}', 'provider_name': "
            "'Nvidia'}}, 'user_id': 'user_2abcDEF'}"
        )

        detail = serwer.error_log_detail(error)

        assert "acct-TEST-0000" not in detail
        assert "user_2abcDEF" not in detail
        assert "Not found for account \\'<ukryte>" in detail
        assert "'user_id': '<ukryte>'" in detail
        assert "'provider_name': 'Nvidia'" in detail

    def test_error_log_detail__hides_the_user_id_in_a_json_error_body(self) -> None:
        """Verify the same id is hidden when the provider sends the error as JSON."""
        detail = serwer.error_log_detail(RuntimeError('{"user_id":"user_2abcDEF","code":429}'))

        assert detail == '{"user_id":"<ukryte>","code":429}'

    def test_error_log_detail__names_the_type_of_a_message_less_error(self) -> None:
        """Verify an exception raised without text still leaves something to search for."""
        assert serwer.error_log_detail(TimeoutError()) == "TimeoutError"

    def test_error_log_detail__describes_validation_without_input_values(self) -> None:
        """Verify schema failures keep the value-free summary used for structured output."""

        class ResponseModel(serwer.BaseModel):
            required: str

        with pytest.raises(serwer.ValidationError) as error:
            ResponseModel.model_validate({"extra": "private user content"})

        detail = serwer.error_log_detail(error.value)

        assert "required: Field required (missing)" in detail
        assert "private user content" not in detail


class TestRecoverPartialStructuredResponse:
    def test_recover_partial_structured_response__keeps_complete_items_cut_between_them(
        self,
    ) -> None:
        """Verify a list cut right after a finished item is still usable."""
        content = '{"summary": "ok", "items": [{"value": "a"}, '

        result = serwer.recover_partial_structured_response(content, _Plan)

        assert result == _Plan(summary="ok", items=[_Answer(value="a")])

    def test_recover_partial_structured_response__rejects_a_cut_inside_a_list_item(self) -> None:
        """Verify a half-written item invalidates the prefix instead of being guessed."""
        content = '{"summary": "ok", "items": [{"value": "a"}, {"val'

        assert serwer.recover_partial_structured_response(content, _Plan) is None

    def test_recover_partial_structured_response__rejects_a_prefix_missing_required_fields(
        self,
    ) -> None:
        """Verify a cut before a required field is not filled with a default."""
        assert serwer.recover_partial_structured_response('{"items": []', _Plan) is None

    @pytest.mark.parametrize("content", [None, ""])
    def test_recover_partial_structured_response__returns_none_for_no_content(
        self, content: str | None
    ) -> None:
        """Verify an empty truncated answer is reported as unrecoverable."""
        assert serwer.recover_partial_structured_response(content, _Plan) is None

    def test_recover_partial_structured_response__returns_none_for_text_that_is_not_json(
        self,
    ) -> None:
        """Verify prose returned instead of JSON does not raise from the recovery path."""
        assert serwer.recover_partial_structured_response("Oto plan:", _Plan) is None


class TestLogModelResponse:
    def test_log_model_response__answer_over_the_limit_is_cut_and_marked(self) -> None:
        """Verify a runaway answer cannot fill the case log, and the cut is visible in it."""
        capture = _CaptureHandler()
        serwer.LOGGER.addHandler(capture)
        try:
            serwer.log_model_response(
                outcome="invalid",
                stage="pismogenerator",
                session_id="case-1",
                response="x" * (serwer.MAX_LOGGED_MODEL_RESPONSE_CHARS + 10),
                provider="ollama",
                model="m",
                attempt=1,
            )
        finally:
            serwer.LOGGER.removeHandler(capture)

        [message] = capture.messages
        assert f"response_chars={serwer.MAX_LOGGED_MODEL_RESPONSE_CHARS + 10}" in message
        assert "response_log_truncated=true" in message
        assert message.count("x") == serwer.MAX_LOGGED_MODEL_RESPONSE_CHARS

    def test_log_model_response__polish_text_with_quotes_stays_on_one_line(self) -> None:
        """Verify the logged answer is JSON-encoded, so newlines cannot forge another log line."""
        capture = _CaptureHandler()
        serwer.LOGGER.addHandler(capture)
        try:
            serwer.log_model_response(
                outcome="truncated",
                stage="dopytywator",
                session_id=None,
                response='Zażółć "gęślą"\nllm_request_completed fałszywy wpis',
                provider="openrouter",
                model="vendor/model",
                attempt=2,
            )
        finally:
            serwer.LOGGER.removeHandler(capture)

        [message] = capture.messages
        assert "\n" not in message
        assert 'response_json="Zażółć \\"gęślą\\"\\nllm_request_completed fałszywy wpis"' in message
        assert "session_id=- " in message
        assert "response_log_truncated=false" in message

    def test_log_model_response__no_answer_writes_nothing(self) -> None:
        """Verify a request that failed before any answer does not log an empty response."""
        capture = _CaptureHandler()
        serwer.LOGGER.addHandler(capture)
        try:
            serwer.log_model_response(
                outcome="invalid",
                stage="s",
                session_id=None,
                response=None,
                provider="ollama",
                model="m",
                attempt=1,
            )
        finally:
            serwer.LOGGER.removeHandler(capture)

        assert capture.messages == []
