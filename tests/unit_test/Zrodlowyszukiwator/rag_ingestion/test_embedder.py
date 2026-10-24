"""Tests for the Ollama embedding client used by ingestion and retrieval."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from src.Zrodlowyszukiwator.rag_ingestion import embedder as embedder_module
from src.Zrodlowyszukiwator.rag_ingestion.embedder import OllamaEmbedder


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


class FakeOllama:
    """Answer /api/embed requests with one vector per input text, encoding its position."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def __call__(self, request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        self.requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": payload,
                "method": request.method,
            }
        )
        vectors = [[float(len(text))] for text in payload["input"]]
        return FakeResponse(json.dumps({"embeddings": vectors}).encode("utf-8"))


def answer_with(body: bytes):
    return lambda request, timeout: FakeResponse(body)


class TestOllamaEmbedderEmbed:
    def test_embed__splits_input_into_batches_and_keeps_the_input_order(self, monkeypatch) -> None:
        """Verify each vector stays aligned with its text across batch boundaries."""
        ollama = FakeOllama()
        monkeypatch.setattr(embedder_module.urllib.request, "urlopen", ollama)
        texts = ["x" * length for length in range(1, 36)]

        vectors = OllamaEmbedder(batch_size=16).embed(texts)

        assert [len(r["payload"]["input"]) for r in ollama.requests] == [16, 16, 3]
        assert vectors == [[float(length)] for length in range(1, 36)]

    def test_embed__sends_no_request_for_no_texts(self, monkeypatch) -> None:
        """Verify an empty document does not call Ollama with an empty batch."""
        ollama = FakeOllama()
        monkeypatch.setattr(embedder_module.urllib.request, "urlopen", ollama)

        assert OllamaEmbedder().embed([]) == []
        assert ollama.requests == []

    def test_embed__posts_to_the_native_endpoint_with_truncation_and_timeout(
        self, monkeypatch
    ) -> None:
        """Verify a trailing slash in OLLAMA_URL does not break the endpoint address."""
        ollama = FakeOllama()
        monkeypatch.setattr(embedder_module.urllib.request, "urlopen", ollama)

        OllamaEmbedder(model="bge-m3", base_url="http://ollama.test:11434/", timeout=12.5).embed(
            ["tekst"]
        )

        [request] = ollama.requests
        assert request["url"] == "http://ollama.test:11434/api/embed"
        assert request["method"] == "POST"
        assert request["timeout"] == 12.5
        assert request["payload"] == {"model": "bge-m3", "input": ["tekst"], "truncate": True}

    def test_embed__http_error_raises_with_the_status_and_body(self, monkeypatch) -> None:
        """Verify a missing model is reported with Ollama's own explanation."""

        def not_found(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", {}, io.BytesIO(b'{"error":"model not found"}')
            )

        monkeypatch.setattr(embedder_module.urllib.request, "urlopen", not_found)

        with pytest.raises(RuntimeError, match=r'HTTP 404: \{"error":"model not found"\}'):
            OllamaEmbedder().embed(["tekst"])

    def test_embed__connection_failure_raises_a_clear_error(self, monkeypatch) -> None:
        """Verify a stopped Ollama produces an error that says embeddings could not be fetched."""

        def refused(request, timeout):
            raise urllib.error.URLError(ConnectionRefusedError("refused"))

        monkeypatch.setattr(embedder_module.urllib.request, "urlopen", refused)

        with pytest.raises(RuntimeError, match="Nie udało się pobrać embeddingów") as error:
            OllamaEmbedder().embed(["tekst"])

        assert isinstance(error.value.__cause__, urllib.error.URLError)

    @pytest.mark.parametrize(
        "body",
        [
            b'{"embeddings": [[0.1], [0.2]]}',
            b'{"embeddings": []}',
            b'{"embedding": [0.1]}',
        ],
        ids=["too-many-vectors", "no-vectors", "legacy-key"],
    )
    def test_embed__wrong_number_of_vectors_raises(self, monkeypatch, body: bytes) -> None:
        """Verify a wrong number of vectors is never silently paired with the wrong texts."""
        monkeypatch.setattr(embedder_module.urllib.request, "urlopen", answer_with(body))

        with pytest.raises(RuntimeError, match="nieoczekiwany format embeddings"):
            OllamaEmbedder().embed(["tekst"])

    def test_embed__non_json_response_raises_a_fetch_error(self, monkeypatch) -> None:
        """Verify a proxy error page is reported instead of crashing with a decoding error."""
        monkeypatch.setattr(
            embedder_module.urllib.request, "urlopen", answer_with(b"<html>Bad gateway</html>")
        )

        with pytest.raises(RuntimeError, match="Nie udało się pobrać embeddingów"):
            OllamaEmbedder().embed(["tekst"])
