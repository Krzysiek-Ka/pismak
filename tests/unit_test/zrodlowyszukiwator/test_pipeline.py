"""Tests for assembling an EvidencePack from a CaseBrief (brief_to_evidence.pipeline)."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import BaseModel

from src.Zrodlowyszukiwator.brief_to_evidence import BriefToEvidencePipeline, ResearchPlan
from src.Zrodlowyszukiwator.rag_retrieval.models import RetrievedChunk, SearchIntent
from src.Zrodlowyszukiwator.rag_retrieval.reranker import RerankOutcome


def intent(purpose: str) -> SearchIntent:
    return SearchIntent(semantic_query=purpose, lexical_query=purpose, purpose=purpose)


def candidate(chunk_id: str, document_id: str, *, intents: list[str] | None = None):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=f"Dokument {document_id}",
        document_kind="policy",
        hierarchy_rank=80,
        legal_form="resolution",
        page_start=1,
        page_end=1,
        text=f"Treść {chunk_id}",
        hybrid_score=0.5,
        matched_intents=intents or [],
    )


class FakePlanner:
    def __init__(self, *purposes: str) -> None:
        self.intents = [intent(purpose) for purpose in purposes]
        self.briefs: list[dict] = []

    def plan(self, brief: dict) -> ResearchPlan:
        self.briefs.append(brief)
        return ResearchPlan(case_summary="Sprawa", intents=self.intents)


class FakeRetriever:
    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        self.candidates = candidates
        self.top_k: int | None = None

    def retrieve(self, intents, *, filters, top_k):
        self.top_k = top_k
        return list(self.candidates)


class FakeReranker:
    def __init__(self, outcome: RerankOutcome | None = None) -> None:
        self.outcome = outcome
        self.calls = 0

    def rerank(self, *, case_brief, candidates):
        self.calls += 1
        if self.outcome is None:
            return RerankOutcome(
                selected=[(c, "Wspiera sprawę.", 3) for c in candidates], assessed=True
            )
        return self.outcome


def pipeline(candidates, *, purposes=("problem",), reranker=None, **limits):
    return BriefToEvidencePipeline(
        planner=FakePlanner(*purposes),
        retriever=FakeRetriever(candidates),
        reranker=reranker,
        **limits,
    )


class Brief(BaseModel):
    title: str
    deadline: date


class TestBriefDict:
    def test_brief_dict__passes_a_dict_through_unchanged(self) -> None:
        """Verify a brief loaded from JSON is planned exactly as stored."""
        brief = {"title": "Sprawa"}

        assert BriefToEvidencePipeline._brief_dict(brief) is brief

    def test_brief_dict__turns_a_model_into_json_compatible_values(self) -> None:
        """Verify dates in the brief do not break the JSON prompt sent to the planner."""
        result = BriefToEvidencePipeline._brief_dict(
            Brief(title="Sprawa", deadline=date(2026, 5, 1))
        )

        assert result == {"title": "Sprawa", "deadline": "2026-05-01"}

    def test_brief_dict__rejects_other_types(self) -> None:
        """Verify a plain string is not silently planned as an empty brief."""
        with pytest.raises(TypeError, match="case_brief musi być dict"):
            BriefToEvidencePipeline._brief_dict("Brak przejścia")


class TestRun:
    def test_run__reports_no_support_and_skips_the_reranker_when_nothing_was_found(self) -> None:
        """Verify an empty search does not spend a slow model call on zero candidates."""
        reranker = FakeReranker()

        pack = pipeline([], purposes=("przejście", "oświetlenie"), reranker=reranker).run({})

        assert reranker.calls == 0
        assert pack.evidence == []
        assert pack.evidence_status == "unsupported"
        assert pack.unsupported_topics == ["przejście", "oświetlenie"]

    def test_run__asks_the_retriever_for_the_configured_number_of_candidates(self) -> None:
        """Verify candidate_k bounds what the reranker has to read."""
        evidence_pipeline = pipeline([], candidate_k=7)

        evidence_pipeline.run({})

        assert evidence_pipeline.retriever.top_k == 7

    def test_run__stops_at_evidence_k_and_numbers_evidence_from_e1(self) -> None:
        """Verify the pack size limit holds and identifiers match the citation scheme."""
        candidates = [candidate(f"c{index}", f"doc-{index}") for index in range(10)]

        pack = pipeline(candidates, evidence_k=4).run({})

        assert [e.evidence_id for e in pack.evidence] == ["E1", "E2", "E3", "E4"]
        assert [e.chunk_id for e in pack.evidence] == ["c0", "c1", "c2", "c3"]

    def test_run__limits_fragments_per_document_while_filling_from_later_candidates(self) -> None:
        """Verify one long document cannot crowd out every other source."""
        candidates = [candidate(f"a{index}", "doc-a") for index in range(4)] + [
            candidate("b0", "doc-b")
        ]

        pack = pipeline(candidates, evidence_k=6, max_per_document=2).run({})

        assert [e.chunk_id for e in pack.evidence] == ["a0", "a1", "b0"]

    def test_run__lists_intents_without_any_matching_evidence_as_unsupported(self) -> None:
        """Verify the letter is told which parts of the case have no source behind them."""
        candidates = [candidate("c1", "doc-1", intents=["Q1: przejście: lokalizacja"])]

        pack = pipeline(candidates, purposes=("przejście: lokalizacja", "oświetlenie")).run({})

        assert pack.unsupported_topics == ["oświetlenie"]

    def test_run__keeps_the_reranker_score_and_drops_a_zero_score(self) -> None:
        """Verify an unassessed fragment is shown without a score instead of a score of 0."""
        candidates = [candidate("c1", "doc-1"), candidate("c2", "doc-2")]
        reranker = FakeReranker(
            RerankOutcome(
                selected=[(candidates[0], "Wspiera.", 3), (candidates[1], "Tło.", 0)],
                assessed=True,
            )
        )

        pack = pipeline(candidates, reranker=reranker).run({})

        assert [e.rerank_score for e in pack.evidence] == [3, None]
        assert pack.evidence[0].relevance_reason == "Wspiera."

    def test_run__summarizes_the_search_in_notes(self) -> None:
        """Verify the saved run explains how many candidates became evidence."""
        candidates = [candidate("c1", "doc-1"), candidate("c2", "doc-1")]

        pack = pipeline(candidates, max_per_document=1).run({})

        assert pack.search_notes[:4] == [
            "Liczba zapytań: 1",
            "Kandydaci po hybrid retrieval: 2",
            "Źródła w EvidencePack: 1",
            "Ocena wsparcia źródeł: unverified",
        ]
