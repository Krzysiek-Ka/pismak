"""Tests for the shared LLM server client utilities."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src._shared import llm_runtime, serwer
from src._shared.context_budget import LLMInputLimitError, estimate_tokens
from src._shared.llm_runtime import LLMBusyError, llm_generation_lock


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


class TestCallStructuredQueue:
    def test_call_structured__openrouter_runs_while_the_local_model_is_busy(self) -> None:
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
            result = llm._call_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=_Answer,
                temperature=0.0,
                max_output_tokens=512,
                attempt=1,
            )

        assert result.value == "ok"
        assert waits == []

    def test_call_structured__busy_local_model_raises_without_calling_it(self, monkeypatch) -> None:
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

    def test_call_structured__sends_long_prompt_to_a_model_with_a_large_window(
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

        llm._call_structured(
            system_prompt="system",
            user_prompt="a" * 40_000,
            response_model=_Answer,
            temperature=0.0,
            max_output_tokens=8192,
            attempt=1,
        )

        assert completions.calls[0]["max_tokens"] == 8192

    def test_call_structured__rejects_prompt_beyond_a_small_model_window(self, monkeypatch) -> None:
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


class TestLLMClient:
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


class TestErrorLogDetail:
    def test_error_log_detail__keeps_the_provider_message_on_one_line(self) -> None:
        """Verify non-schema failures are logged with their own text instead of only a type."""
        detail = serwer.error_log_detail(RuntimeError("Error code: 429\n  rate limit exceeded"))

        assert detail == "Error code: 429 rate limit exceeded"

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


class _Plan(serwer.BaseModel):
    summary: str
    items: list[_Answer] = []


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


class _Replies:
    """Return prepared completions one per call."""

    def __init__(self, *completions: SimpleNamespace) -> None:
        self.completions = list(completions)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.completions.pop(0)


class TestCheckAvailableModelErrors:
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


class TestGenerateStructuredEdgeCases:
    def test_generate_structured__rejects_a_non_positive_output_limit(self) -> None:
        """Verify a misconfigured stage budget fails before any request is sent."""
        completions = _Replies()
        llm = serwer.LLMClient(_client(completions), "m")

        with pytest.raises(ValueError, match="max_output_tokens musi być dodatnie"):
            llm.generate_structured(
                system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=0
            )

        assert completions.calls == []

    def test_generate_structured__repairs_an_empty_answer_once(self) -> None:
        """Verify an empty message is treated as invalid output and repaired with the schema."""
        completions = _Replies(_completion(None), _completion('{"value": "ok"}'))
        llm = serwer.LLMClient(_client(completions), "m")

        result = llm.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
        )

        assert result == _Answer(value="ok")
        assert "Model nie zwrócił danych." in completions.calls[1]["messages"][1]["content"]

    def test_generate_structured__reports_truncation_when_the_compact_retry_breaks_the_schema(
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

    def test_generate_structured__keeps_truncation_raised_during_the_repair_attempt(self) -> None:
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

    def test_generate_structured__remembers_the_model_that_answered(self) -> None:
        """Verify the model chosen by OpenRouter's router is recorded for the run metadata."""
        completions = _Replies(_completion('{"value": "ok"}', model="vendor/routed"))
        llm = serwer.LLMClient(_client(completions), "openrouter/free")

        llm.generate_structured(
            system_prompt="s", user_prompt="u", response_model=_Answer, max_output_tokens=50
        )

        assert llm.last_model == "vendor/routed"


class TestStructuredOutputErrorDetailOtherErrors:
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
