from __future__ import annotations

from threading import Event, Thread

from src._shared.llm_runtime import llm_generation_lock
from src.Zrodlowyszukiwator.brief_to_evidence.formatting import (
    _truncate_at_boundary,
    evidence_pack_to_markdown,
)
from src.Zrodlowyszukiwator.brief_to_evidence.models import Evidence, EvidencePack, ResearchPlan
from src.Zrodlowyszukiwator.brief_to_evidence.planner import LLMQueryPlanner


def make_pack(excerpt: str = "Fragment źródłowy.") -> EvidencePack:
    return EvidencePack(
        query_summary="Opis sprawy",
        research_plan=ResearchPlan(case_summary="Opis sprawy", intents=[]),
        evidence=[
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="doc-1",
                document_title="Polityka miejska",
                document_kind="policy",
                hierarchy_rank=80,
                legal_form="resolution",
                page_start=3,
                page_end=3,
                section_path=["Rozdział 1"],
                excerpt=excerpt,
                retrieval_score=0.9,
                relevance_reason="Pasuje do sprawy.",
            )
        ],
        unsupported_topics=["finansowanie"],
    )


class TestEvidenceFormatting:
    def test_truncate_at_boundary__adds_ellipsis_when_no_boundary_exists(self) -> None:
        """Verify long text is bounded even when no sentence boundary is available."""
        assert _truncate_at_boundary("abcdefghij", 5) == "abcde…"

    def test_evidence_pack_to_markdown__renders_empty_pack(self) -> None:
        """Verify an empty EvidencePack explicitly reports missing sources."""
        pack = EvidencePack(
            query_summary="Brak wyników",
            research_plan=ResearchPlan(case_summary="Brak wyników", intents=[]),
        )

        result = evidence_pack_to_markdown(pack)

        assert "Brak znalezionych źródeł." in result

    def test_evidence_pack_to_markdown__renders_metadata_and_unsupported_topics(self) -> None:
        """Verify evidence metadata and unsupported topics remain visible to the generator."""
        result = evidence_pack_to_markdown(make_pack())

        assert "[E1] Polityka miejska" in result
        assert "strony PDF: 3" in result
        assert "Rozdział 1" in result
        assert "finansowanie" in result


class FailingClient:
    class Chat:
        class Completions:
            def create(self, **kwargs):
                raise RuntimeError("LLM unavailable")

        completions = Completions()

    chat = Chat()


class EmptyPlanClient:
    class Chat:
        class Completions:
            def create(self, **kwargs):
                class Message:
                    content = '{"case_summary": "", "intents": []}'

                class Choice:
                    message = Message()

                class Response:
                    choices = [Choice()]

                return Response()

        completions = Completions()

    chat = Chat()


class TestLLMQueryPlanner:
    def test_plan__waits_for_shared_llm_lock(self) -> None:
        """Verify planner requests queue behind another local generation."""
        notified = Event()
        called = Event()

        class Client:
            class Chat:
                class Completions:
                    def create(self, **kwargs):
                        called.set()

                        class Message:
                            content = '{"case_summary": "Sprawa", "intents": []}'

                        class Choice:
                            message = Message()

                        class Response:
                            choices = [Choice()]

                        return Response()

                completions = Completions()

            chat = Chat()

        planner = LLMQueryPlanner(client=Client(), model="test", on_wait=notified.set)
        thread = Thread(target=lambda: planner.plan({"problem_description": "Problem"}))

        with llm_generation_lock():
            thread.start()
            assert notified.wait(timeout=1)
            assert not called.is_set()

        thread.join(timeout=1)

        assert not thread.is_alive()
        assert called.is_set()

    def test_plan__uses_fallback_when_llm_fails(self) -> None:
        """Verify retrieval planning remains usable when the external model is unavailable."""
        planner = LLMQueryPlanner(client=FailingClient(), model="test")

        result = planner.plan(
            {
                "problem_description": "Brak przejścia",
                "desired_outcome": "Bezpieczne przejście",
                "requested_actions": ["Zbudowanie przejścia"],
            }
        )

        assert len(result.intents) == 3
        assert result.case_summary == "Bezpieczne przejście"

    def test_plan__uses_fallback_when_llm_returns_empty_intents(self) -> None:
        """Verify an empty structured response does not erase useful fallback queries."""
        planner = LLMQueryPlanner(client=EmptyPlanClient(), model="test")

        result = planner.plan({"problem_description": "Problem", "requested_actions": []})

        assert len(result.intents) == 1
        assert result.intents[0].semantic_query == "Problem"
