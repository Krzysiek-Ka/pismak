from __future__ import annotations

import builtins
import json
from threading import Event, Thread

import pytest

from src._shared.llm_runtime import llm_generation_lock
from src.Zrodlowyszukiwator.brief_to_evidence.models import ResearchPlan
from src.Zrodlowyszukiwator.brief_to_evidence.pipeline import BriefToEvidencePipeline
from src.Zrodlowyszukiwator.rag_retrieval.hybrid import HybridRetriever
from src.Zrodlowyszukiwator.rag_retrieval.models import (
    RetrievalFilters,
    RetrievedChunk,
    SearchIntent,
)
from src.Zrodlowyszukiwator.rag_retrieval.repository import RetrievalRepository
from src.Zrodlowyszukiwator.rag_retrieval.reranker import LLMReranker


def make_chunk(chunk_id: str, document_id: str = "doc-1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=f"Tytuł {chunk_id}",
        document_kind="policy",
        hierarchy_rank=80,
        legal_form="resolution",
        page_start=1,
        page_end=1,
        text=f"Treść {chunk_id}",
    )


class TestRetrievalRepository:
    def test_connect__reports_actionable_error_when_driver_import_is_blocked(
        self, monkeypatch
    ) -> None:
        """Verify a blocked native driver becomes a clear database setup error."""
        original_import = builtins.__import__

        def block_psycopg(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("blocked by application control")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_psycopg)

        with pytest.raises(RuntimeError, match="Nie można załadować sterownika PostgreSQL"):
            RetrievalRepository("postgresql://unused").connect()


class EmptyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeRetrievalRepository:
    def __init__(self, chunk: RetrievedChunk) -> None:
        self.chunk = chunk

    def vector_search(self, **kwargs) -> list[RetrievedChunk]:
        return [self.chunk]

    def lexical_search(self, **kwargs) -> list[RetrievedChunk]:
        return [self.chunk]


class TestHybridRetriever:
    def test_retrieve__returns_empty_for_no_intents(self) -> None:
        """Verify retrieval avoids embedding and database calls when there are no intents."""
        retriever = HybridRetriever(
            repository=FakeRetrievalRepository(make_chunk("E1")),
            embedder=EmptyEmbedder(),
        )

        assert retriever.retrieve([]) == []

    def test_retrieve__merges_duplicate_results_and_marks_intent(self) -> None:
        """Verify the same chunk from vector and lexical search is returned once with RRF score."""
        retriever = HybridRetriever(
            repository=FakeRetrievalRepository(make_chunk("E1")),
            embedder=EmptyEmbedder(),
        )
        intent = SearchIntent(
            semantic_query="bezpieczeństwo",
            lexical_query="bezpieczeństwo",
            purpose="problem",
            priority=3,
        )

        result = retriever.retrieve([intent], top_k=1)

        assert len(result) == 1
        assert result[0].chunk_id == "E1"
        assert result[0].hybrid_score > 0
        assert result[0].matched_intents == ["Q1: problem"]


class TestLLMReranker:
    def test_rerank__waits_for_shared_llm_lock(self) -> None:
        """Verify reranker requests queue before calling the local model."""
        notified = Event()
        called = Event()

        class Client:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        called.set()

                        class Message:
                            content = '{"selected": []}'

                        class Choice:
                            message = Message()

                        class Response:
                            choices = [Choice()]

                        return Response()

                completions = Completions()

            chat = Chat()

        reranker = LLMReranker(client=Client(), model="test", on_wait=notified.set)
        thread = Thread(
            target=lambda: reranker.rerank(case_brief={}, candidates=[make_chunk("E1")])
        )

        with llm_generation_lock():
            thread.start()
            assert notified.wait(timeout=1)
            assert not called.is_set()

        thread.join(timeout=1)

        assert not thread.is_alive()
        assert called.is_set()

    def test_rerank__falls_back_when_model_fails(self) -> None:
        """Verify reranking returns hybrid candidates when structured output fails."""

        class FailingClient:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        raise RuntimeError("LLM unavailable")

                completions = Completions()

            chat = Chat()

        reranker = LLMReranker(client=FailingClient(), model="test", max_selected=1)
        candidates = [make_chunk("E1"), make_chunk("E2")]

        result = reranker.rerank(case_brief={}, candidates=candidates)

        assert [item[0].chunk_id for item in result] == ["E1"]
        assert result[0][2] == 0

    def test_rerank__ignores_unknown_and_duplicate_ids(self) -> None:
        """Verify model output cannot introduce unknown chunks or duplicate selections."""

        class Client:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        payload = {
                            "selected": [
                                {"chunk_id": "unknown", "score": 100, "reason": "x"},
                                {"chunk_id": "E1", "score": 90, "reason": "ok"},
                                {"chunk_id": "E1", "score": 80, "reason": "duplicate"},
                            ]
                        }

                        class Message:
                            content = json.dumps(payload)

                        class Choice:
                            message = Message()

                        class Response:
                            choices = [Choice()]

                        return Response()

                completions = Completions()

            chat = Chat()

        reranker = LLMReranker(client=Client(), model="test")
        result = reranker.rerank(case_brief={}, candidates=[make_chunk("E1")])

        assert [(item[0].chunk_id, item[2]) for item in result] == [("E1", 90)]


class FakePlanner:
    def plan(self, brief: dict) -> ResearchPlan:
        return ResearchPlan(
            case_summary="Sprawa",
            intents=[
                SearchIntent(
                    semantic_query="problem",
                    lexical_query="problem",
                    purpose="problem",
                    priority=2,
                )
            ],
        )


class FakeRetriever:
    def retrieve(
        self, intents: list[SearchIntent], *, filters: RetrievalFilters | None, top_k: int
    ):
        return [make_chunk("E1", "doc-1"), make_chunk("E2", "doc-1"), make_chunk("E3", "doc-2")]


class TestBriefToEvidencePipeline:
    def test_run__limits_evidence_per_document_and_marks_unsupported_topics(self) -> None:
        """Verify the pipeline enforces per-document limits and produces a typed EvidencePack."""
        pipeline = BriefToEvidencePipeline(
            planner=FakePlanner(),
            retriever=FakeRetriever(),
            evidence_k=3,
            max_per_document=1,
        )

        result = pipeline.run({"problem_description": "Problem"})

        assert [e.chunk_id for e in result.evidence] == ["E1", "E3"]
        assert result.evidence[0].evidence_id == "E1"
        assert result.unsupported_topics == ["problem"]
