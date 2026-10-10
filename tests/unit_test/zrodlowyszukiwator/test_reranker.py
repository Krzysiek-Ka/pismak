from __future__ import annotations

import pytest

from src.Zrodlowyszukiwator.rag_retrieval.models import RetrievedChunk
from src.Zrodlowyszukiwator.rag_retrieval.reranker import (
    ZRODLOWYSZUKIWACZ_RERANKER_MAX_OUTPUT_TOKENS,
    LLMReranker,
    RerankResult,
)


class TestLlmReranker:
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
