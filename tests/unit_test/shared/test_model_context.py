from __future__ import annotations

import pytest

from src._shared import llm_runtime, model_context

PS_URL = "http://ollama.test/api/ps"
GENERATE_URL = "http://ollama.test/api/generate"


class FakeHttp:
    """Replay canned responses in order and record every request."""

    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def __call__(self, url: str, payload=None, *, timeout: float):
        self.calls.append((url, payload))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def isolated_lookups(monkeypatch):
    """Keep module-level caches from leaking between tests and pin the Ollama address."""
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.test")
    model_context._LAST_KNOWN.clear()
    model_context._OPENROUTER_CACHE.clear()
    yield
    model_context._LAST_KNOWN.clear()
    model_context._OPENROUTER_CACHE.clear()


def loaded(name: str, context_length: int) -> dict[str, object]:
    return {"models": [{"name": name, "context_length": context_length}]}


class TestDetectContextTokens:
    def test_detect_context_tokens__reads_window_of_a_loaded_ollama_model(
        self, monkeypatch
    ) -> None:
        """Verify the window Ollama actually serves is used, without loading anything again."""
        http = FakeHttp(loaded("pismak-bielik:latest", 8192))
        monkeypatch.setattr(model_context, "_request_json", http)

        assert model_context.detect_context_tokens("ollama", "pismak-bielik:latest") == 8192
        assert http.calls == [(PS_URL, None)]

    def test_detect_context_tokens__loads_ollama_model_before_reading_its_window(
        self, monkeypatch
    ) -> None:
        """Verify an unloaded model is loaded first, because only then is its window known."""
        http = FakeHttp({"models": []}, {"done": True}, loaded("pismak-pllum:latest", 131072))
        monkeypatch.setattr(model_context, "_request_json", http)

        assert model_context.detect_context_tokens("ollama", "pismak-pllum:latest") == 131072
        assert http.calls == [
            (PS_URL, None),
            (GENERATE_URL, {"model": "pismak-pllum:latest", "stream": False}),
            (PS_URL, None),
        ]

    def test_detect_context_tokens__loads_model_while_holding_the_generation_lock(
        self, monkeypatch
    ) -> None:
        """Verify loading a model cannot overlap with another session's running generation."""
        lock_states: list[bool] = []

        def http(url: str, payload=None, *, timeout: float):
            if payload is not None:
                lock_states.append(llm_runtime._LLM_LOCK.locked())
                return {"done": True}
            return loaded("model:latest", 4096) if lock_states else {"models": []}

        monkeypatch.setattr(model_context, "_request_json", http)

        assert model_context.detect_context_tokens("ollama", "model:latest") == 4096
        assert lock_states == [True]
        assert not llm_runtime._LLM_LOCK.locked()

    def test_detect_context_tokens__matches_ollama_name_without_an_explicit_tag(
        self, monkeypatch
    ) -> None:
        """Verify `model` and `model:latest` are the same model, as they are for Ollama."""
        monkeypatch.setattr(
            model_context, "_request_json", FakeHttp(loaded("pismak-bielik:latest", 8192))
        )

        assert model_context.detect_context_tokens("ollama", "pismak-bielik") == 8192

    def test_detect_context_tokens__returns_none_when_ollama_is_unreachable(
        self, monkeypatch
    ) -> None:
        """Verify a failed lookup yields no window instead of breaking the request path."""
        monkeypatch.setattr(model_context, "_request_json", FakeHttp(OSError("refused")))

        assert model_context.detect_context_tokens("ollama", "pismak-bielik:latest") is None

    def test_detect_context_tokens__reuses_last_known_window_when_a_later_lookup_fails(
        self, monkeypatch
    ) -> None:
        """Verify a transient lookup failure does not shrink the budget back to the fallback."""
        http = FakeHttp(loaded("pismak-pllum:latest", 131072), OSError("refused"))
        monkeypatch.setattr(model_context, "_request_json", http)

        model_context.detect_context_tokens("ollama", "pismak-pllum:latest")

        assert model_context.detect_context_tokens("ollama", "pismak-pllum:latest") == 131072

    def test_detect_context_tokens__uses_smallest_window_among_free_structured_models(
        self, monkeypatch
    ) -> None:
        """Verify the free router is budgeted for the smallest model it may pick for Pismak."""
        models = {
            "data": [
                {"id": "openrouter/free", "context_length": 200000},
                {
                    "id": "a:free",
                    "context_length": 65536,
                    "supported_parameters": ["response_format"],
                },
                {"id": "b:free", "context_length": 32000, "supported_parameters": ["max_tokens"]},
                {
                    "id": "c:free",
                    "context_length": 262144,
                    "supported_parameters": ["structured_outputs"],
                },
                {
                    "id": "paid/m",
                    "context_length": 8000,
                    "supported_parameters": ["response_format"],
                },
            ]
        }
        monkeypatch.setattr(model_context, "_request_json", FakeHttp(models))

        assert model_context.detect_context_tokens("openrouter", "openrouter/free") == 65536

    def test_detect_context_tokens__considers_every_free_model_when_none_declares_structure(
        self, monkeypatch
    ) -> None:
        """Verify missing capability data still yields a conservative window, not none at all."""
        models = {
            "data": [
                {"id": "a:free", "context_length": 65536},
                {"id": "b:free", "context_length": 32000},
            ]
        }
        monkeypatch.setattr(model_context, "_request_json", FakeHttp(models))

        assert model_context.detect_context_tokens("openrouter", "openrouter/free") == 32000

    def test_detect_context_tokens__reads_window_of_a_specific_openrouter_model(
        self, monkeypatch
    ) -> None:
        """Verify a concrete model id gets its own window rather than the router minimum."""
        models = {
            "data": [
                {
                    "id": "a:free",
                    "context_length": 65536,
                    "supported_parameters": ["response_format"],
                },
                {"id": "vendor/model", "context_length": 1000000},
            ]
        }
        monkeypatch.setattr(model_context, "_request_json", FakeHttp(models))

        assert model_context.detect_context_tokens("openrouter", "vendor/model") == 1000000

    def test_detect_context_tokens__fetches_openrouter_list_once_within_the_cache_period(
        self, monkeypatch
    ) -> None:
        """Verify repeated stages of one case do not each download the public model list."""
        http = FakeHttp({"data": [{"id": "a:free", "context_length": 65536}]})
        monkeypatch.setattr(model_context, "_request_json", http)

        model_context.detect_context_tokens("openrouter", "openrouter/free")
        model_context.detect_context_tokens("openrouter", "openrouter/free")

        assert http.calls == [(model_context.OPENROUTER_MODELS_URL, None)]

    def test_detect_context_tokens__makes_no_request_for_an_unknown_provider(
        self, monkeypatch
    ) -> None:
        """Verify an unsupported provider is left to the fallback without any network call."""
        http = FakeHttp()
        monkeypatch.setattr(model_context, "_request_json", http)

        assert model_context.detect_context_tokens("other", "model") is None
        assert http.calls == []


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self) -> bytes:
        return self.body


class TestRequestJson:
    def test_request_json__sends_a_get_without_a_body_when_there_is_no_payload(
        self, monkeypatch
    ) -> None:
        """Verify listing endpoints such as /api/ps are queried with GET."""
        sent: list[object] = []

        def urlopen(request, timeout):
            sent.append(
                (request.get_method(), request.data, timeout, request.get_header("User-agent"))
            )
            return _Response(b'{"models": []}')

        monkeypatch.setattr(model_context.urllib.request, "urlopen", urlopen)

        assert model_context._request_json("http://ollama.test/api/ps", timeout=3.0) == {
            "models": []
        }
        assert sent == [("GET", None, 3.0, "pismak")]

    def test_request_json__posts_the_payload_as_json(self, monkeypatch) -> None:
        """Verify /api/show receives the model name as a JSON body."""
        sent: list[object] = []

        def urlopen(request, timeout):
            sent.append((request.get_method(), request.data, request.get_header("Content-type")))
            return _Response(b'{"ok": true}')

        monkeypatch.setattr(model_context.urllib.request, "urlopen", urlopen)

        model_context._request_json("http://ollama.test/api/show", {"model": "bielik"}, timeout=1)

        assert sent == [("POST", b'{"model": "bielik"}', "application/json")]


class TestTagged:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("pismak-bielik", "pismak-bielik:latest"),
            ("pismak-bielik:latest", "pismak-bielik:latest"),
            ("hf.co/org/model:Q4_K_M", "hf.co/org/model:Q4_K_M"),
            ("localhost:5000/model", "localhost:5000/model:latest"),
        ],
    )
    def test_tagged__adds_latest_only_when_the_model_name_has_no_tag(
        self, model: str, expected: str
    ) -> None:
        """Verify a registry port is not mistaken for a tag when matching loaded models."""
        assert model_context._tagged(model) == expected
