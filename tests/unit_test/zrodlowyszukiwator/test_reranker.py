from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from src._shared.context_budget import SAFETY_MARGIN_TOKENS, estimate_tokens, input_char_budget
from src._shared.serwer import LLMResponseError
from src.Zrodlowyszukiwator.rag_retrieval.models import RetrievedChunk
from src.Zrodlowyszukiwator.rag_retrieval.reranker import (
    RERANKER_MAX_TEXT_CHARS,
    RERANKER_MIN_TEXT_CHARS,
    RERANKER_RESERVED_OUTPUT_TOKENS,
    ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS,
    LLMReranker,
    RerankResult,
)


def make_candidate(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        title="Dokument",
        document_kind="strategy",
        hierarchy_rank=1,
        legal_form="uchwała",
        page_start=1,
        page_end=1,
        text=text,
    )


class _ScriptedClient:
    """Return the given response contents one per call, each finished normally."""

    def __init__(self, *contents: str | None) -> None:
        class Completions:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):
                content = contents[len(self.calls)]
                self.calls.append(kwargs)
                message = SimpleNamespace(content=content)
                choice = SimpleNamespace(finish_reason="stop", message=message)
                return SimpleNamespace(model="vendor/model", choices=[choice])

        self.chat = SimpleNamespace(completions=Completions())


class TestFittedUserPrompt:
    def test_fitted_user_prompt__shortens_texts_then_drops_lowest_ranked_candidates(
        self, monkeypatch
    ) -> None:
        """Verify a full candidate list is cut until the prompt leaves room for the answer."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        reranker = LLMReranker(client=object(), model="local")
        candidates = [make_candidate(f"chunk-{i}", "ź" * 3000) for i in range(20)]

        prompt = reranker._fitted_user_prompt({"problem_description": "p" * 6000}, candidates)

        sent = json.loads(prompt)["candidates"]
        budget = input_char_budget("ollama", reserved_output_tokens=RERANKER_RESERVED_OUTPUT_TOKENS)
        assert len(reranker.SYSTEM_PROMPT) + len(prompt) <= budget
        assert 1 <= len(sent) < 20
        assert [c["chunk_id"] for c in sent] == [f"chunk-{i}" for i in range(len(sent))]
        assert all(len(c["text"]) == RERANKER_MIN_TEXT_CHARS for c in sent)

    def test_fitted_user_prompt__keeps_every_candidate_and_full_text_when_they_fit(
        self, monkeypatch
    ) -> None:
        """Verify a prompt already inside the window is sent without losing any evidence."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        reranker = LLMReranker(client=object(), model="local")
        candidates = [make_candidate("chunk-0", "a" * 3000), make_candidate("chunk-1", "b" * 100)]

        prompt = reranker._fitted_user_prompt({}, candidates)

        sent = json.loads(prompt)["candidates"]
        assert [len(c["text"]) for c in sent] == [RERANKER_MAX_TEXT_CHARS, 100]

    def test_fitted_user_prompt__sends_every_candidate_in_full_to_a_large_model_window(
        self, monkeypatch
    ) -> None:
        """Verify a model with a large window gets the complete candidate list, untrimmed."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        reranker = LLMReranker(client=object(), model="pismak-pllum", context_tokens=131072)
        candidates = [make_candidate(f"chunk-{i}", "ź" * 3000) for i in range(20)]

        prompt = reranker._fitted_user_prompt({"problem_description": "p" * 6000}, candidates)

        sent = json.loads(prompt)["candidates"]
        assert [len(c["text"]) for c in sent] == [RERANKER_MAX_TEXT_CHARS] * 20


class TestLlmReranker:
    def test_rerank__sends_prompt_and_output_limit_that_fit_the_context_window(
        self, monkeypatch
    ) -> None:
        """Verify the request for twenty long candidates stays inside the local model window."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)

        class Completions:
            kwargs: dict[str, object] = {}

            def create(self, **kwargs):
                self.kwargs = kwargs
                raise RuntimeError("stop after capturing the request")

        class Client:
            def __init__(self) -> None:
                self.chat = type("Chat", (), {"completions": Completions()})()

        client = Client()
        reranker = LLMReranker(client=client, model="local", allow_fallback=False)
        candidates = [make_candidate(f"chunk-{i}", "ź" * 3000) for i in range(20)]

        with pytest.raises(RuntimeError, match="stop after capturing"):
            reranker.rerank(case_brief={"problem_description": "p" * 6000}, candidates=candidates)

        kwargs = client.chat.completions.kwargs
        prompt_chars = sum(len(message["content"]) for message in kwargs["messages"])
        assert kwargs["max_tokens"] >= RERANKER_RESERVED_OUTPUT_TOKENS
        assert estimate_tokens(prompt_chars) + kwargs["max_tokens"] + SAFETY_MARGIN_TOKENS <= 8192

    def test_structured__retries_truncated_output_with_a_compact_prompt(self) -> None:
        """Verify a token-truncated reranker response is retried once in concise form."""

        class Completions:
            def __init__(self) -> None:
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                content = '{"selected":' if len(self.calls) == 1 else '{"selected":[]}'
                finish_reason = "length" if len(self.calls) == 1 else "stop"
                return type(
                    "Completion",
                    (),
                    {
                        "model": "vendor/model",
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "finish_reason": finish_reason,
                                    "message": type("Message", (), {"content": content})(),
                                },
                            )()
                        ],
                    },
                )()

        class Client:
            def __init__(self) -> None:
                self.chat = type("Chat", (), {"completions": Completions()})()

        client = Client()
        reranker = LLMReranker(client=client, model="openrouter/free", allow_fallback=False)

        result = reranker._structured("prompt", RerankResult)

        assert result == RerankResult(selected=[])
        assert len(client.chat.completions.calls) == 2
        assert "krótszą" in client.chat.completions.calls[1]["messages"][1]["content"]

    def test_structured__retries_invalid_output_with_the_validation_errors(self) -> None:
        """Verify a response that fails the schema is repaired by one more request."""
        client = _ScriptedClient('{"selected":"none"}', '{"selected":[]}')
        reranker = LLMReranker(client=client, model="openrouter/free", allow_fallback=False)

        result = reranker._structured("prompt", RerankResult)

        calls = client.chat.completions.calls
        assert result == RerankResult(selected=[])
        assert len(calls) == 2
        assert calls[0]["messages"][1]["content"] == "prompt"
        assert "Błędy walidacji" in calls[1]["messages"][1]["content"]
        assert "selected" in calls[1]["messages"][1]["content"]

    def test_structured__reports_the_schema_error_after_two_invalid_responses(self, caplog) -> None:
        """Verify the raised error and the log both say what the model got wrong."""
        client = _ScriptedClient(None, '{"selected":[{"chunk_id":"chunk-1","score":500}]}')
        reranker = LLMReranker(client=client, model="openrouter/free", allow_fallback=False)
        logger = logging.getLogger("pismak")
        logger.addHandler(caplog.handler)

        try:
            with pytest.raises(LLMResponseError, match="RerankResult") as error:
                reranker._structured("prompt", RerankResult)
        finally:
            logger.removeHandler(caplog.handler)

        assert len(client.chat.completions.calls) == 2
        assert "selected.0.score" in str(error.value)
        failures = [r.getMessage() for r in caplog.records if "llm_request_failed" in r.msg]
        assert "attempt=1 retry_scheduled=true" in failures[0]
        assert "pustą odpowiedź" in failures[0]
        assert "attempt=2 retry_scheduled=false" in failures[-1]
        assert "selected.0.score" in failures[-1]
        assert '\\"score\\":500' in caplog.text

    def test_rerank__does_not_use_hybrid_fallback_after_openrouter_error(self) -> None:
        """Verify remote reranking failures are surfaced instead of hiding the provider error."""

        class Completions:
            calls = 0
            kwargs: dict[str, object] = {}

            def create(self, **kwargs):
                self.calls += 1
                self.kwargs = kwargs
                raise RuntimeError("remote request failed")

        class Client:
            def __init__(self) -> None:
                self.chat = type("Chat", (), {"completions": Completions()})()

        client = Client()
        reranker = LLMReranker(client=client, model="openrouter/free", allow_fallback=False)
        candidate = RetrievedChunk(
            chunk_id="chunk-1",
            document_id="document-1",
            title="Dokument",
            document_kind="strategy",
            hierarchy_rank=1,
            legal_form="uchwała",
            page_start=1,
            page_end=1,
            text="Treść",
        )

        with pytest.raises(RuntimeError, match="remote request failed"):
            reranker.rerank(case_brief={}, candidates=[candidate])

        assert client.chat.completions.calls == 1
        assert client.chat.completions.kwargs["max_tokens"] == (
            ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS
        )


class TestRerankSupportAssessment:
    def test_rerank__treats_a_valid_empty_selection_as_no_support(self) -> None:
        """Verify an empty answer means 'nothing supports the case', not a reason for fallback."""
        client = _ScriptedClient('{"selected": []}')
        reranker = LLMReranker(client=client, model="local", allow_fallback=True)

        result = reranker.rerank(case_brief={}, candidates=[make_candidate("chunk-1", "Treść")])

        assert result.selected == []
        assert result.assessed is True

    def test_rerank__drops_candidates_scored_below_the_support_threshold(self) -> None:
        """Verify fragments the model rates as merely similar are not presented as evidence."""
        payload = {
            "selected": [
                {"chunk_id": "chunk-1", "score": 85, "reason": "popiera postulat"},
                {"chunk_id": "chunk-2", "score": 20, "reason": "podobny temat"},
            ]
        }
        client = _ScriptedClient(json.dumps(payload))
        reranker = LLMReranker(client=client, model="local")
        candidates = [make_candidate("chunk-1", "A"), make_candidate("chunk-2", "B")]

        result = reranker.rerank(case_brief={}, candidates=candidates)

        assert [(chunk.chunk_id, score) for chunk, _, score in result.selected] == [("chunk-1", 85)]
        assert result.assessed is True

    def test_rerank__falls_back_when_the_model_names_only_unknown_chunks(self) -> None:
        """Verify invented chunk IDs count as a failed answer, not as a judgement of no support."""
        payload = {"selected": [{"chunk_id": "invented", "score": 90, "reason": "x"}]}
        candidates = [make_candidate("chunk-1", "A"), make_candidate("chunk-2", "B")]

        lenient = LLMReranker(client=_ScriptedClient(json.dumps(payload)), model="local")
        result = lenient.rerank(case_brief={}, candidates=candidates)

        assert [chunk.chunk_id for chunk, _, _ in result.selected] == ["chunk-1", "chunk-2"]
        assert result.assessed is False

        strict = LLMReranker(
            client=_ScriptedClient(json.dumps(payload)), model="remote", allow_fallback=False
        )
        with pytest.raises(RuntimeError, match="nieznane chunk_id"):
            strict.rerank(case_brief={}, candidates=candidates)
