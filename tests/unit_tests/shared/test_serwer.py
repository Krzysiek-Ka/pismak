"""Tests for the shared LLM server client utilities."""

from __future__ import annotations

from threading import Event, Thread

from src._shared import serwer
from src._shared.llm_runtime import llm_generation_lock


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
        """Verify structured responses are returned directly from parsed model output."""

        class FakeMessage:
            def __init__(self, parsed) -> None:
                self.parsed = parsed
                self.content = None

        class FakeChoice:
            def __init__(self, parsed) -> None:
                self.message = FakeMessage(parsed)

        class FakeCompletion:
            def __init__(self, parsed) -> None:
                self.choices = [FakeChoice(parsed)]

        class FakeCompletions:
            def __init__(self, parsed) -> None:
                self._parsed = parsed

            def parse(self, **kwargs):
                return FakeCompletion(self._parsed)

        class FakeChat:
            def __init__(self, parsed) -> None:
                self.completions = FakeCompletions(parsed)

        class FakeBeta:
            def __init__(self, parsed) -> None:
                self.chat = FakeChat(parsed)

        class FakeClient:
            def __init__(self, parsed) -> None:
                self.beta = FakeBeta(parsed)

        class FakeModel:
            def __init__(self, value: str) -> None:
                self.value = value

        client = serwer.LLMClient(FakeClient(FakeModel("ok")), "model-name")

        result = client._call_structured(
            system_prompt="system",
            user_prompt="user",
            response_model=FakeModel,
            temperature=0.0,
        )

        assert isinstance(result, FakeModel)
        assert result.value == "ok"

    def test_llm_client__generate_structured_raises_error_when_retries_fail(self) -> None:
        """Verify a repeated structured-output failure raises the project-specific runtime error."""

        class FakeCompletions:
            def parse(self, **kwargs):
                raise RuntimeError("bad payload")

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeBeta:
            def __init__(self) -> None:
                self.chat = FakeChat()

        class FakeClient:
            def __init__(self) -> None:
                self.beta = FakeBeta()

        client = serwer.LLMClient(FakeClient(), "model-name")

        try:
            client.generate_structured(
                system_prompt="system",
                user_prompt="user",
                response_model=dict,
            )
            raise AssertionError("Expected LLMResponseError")
        except serwer.LLMResponseError:
            pass
