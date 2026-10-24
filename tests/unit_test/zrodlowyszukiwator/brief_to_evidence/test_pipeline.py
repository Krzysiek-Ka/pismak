"""Tests for the CaseBrief-to-EvidencePack pipeline."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import BaseModel

from src.Zrodlowyszukiwator.brief_to_evidence.models import ResearchPlan
from src.Zrodlowyszukiwator.brief_to_evidence.pipeline import BriefToEvidencePipeline
from src.Zrodlowyszukiwator.rag_retrieval.models import (
    RetrievalFilters,
    RetrievedChunk,
    SearchIntent,
)
from src.Zrodlowyszukiwator.rag_retrieval.reranker import RerankOutcome

INTENT = SearchIntent(semantic_query="problem", lexical_query="problem", purpose="problem")


class SingleIntentPlanner:
    def plan(self, brief: dict) -> ResearchPlan:
        return ResearchPlan(case_summary="Sprawa", intents=[INTENT])


class CatalogChunkRetriever:
    def __init__(self) -> None:
        self.filters: RetrievalFilters | None = None

    def retrieve(self, intents, *, filters, top_k):
        self.filters = filters
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                title="Polityka",
                document_kind="policy",
                hierarchy_rank=80,
                legal_form="resolution",
                is_active=True,
                lifecycle_status="current_reference",
                source_url="https://example.org/polityka.pdf",
                page_start=1,
                page_end=2,
                text="Treść fragmentu",
            )
        ]


class FixedOutcomeReranker:
    """Return a fixed reranking outcome for whatever candidates it receives."""

    def __init__(self, *, select: bool, assessed: bool) -> None:
        self.select = select
        self.assessed = assessed

    def rerank(self, *, case_brief: dict, candidates: list[RetrievedChunk]) -> RerankOutcome:
        selected = [(c, "Popiera postulat.", 80) for c in candidates] if self.select else []
        return RerankOutcome(selected=selected, assessed=self.assessed)


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


class ProblemPlanner:
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


class ListRetriever:
    """Return the given candidates in order, like hybrid retrieval would rank them."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, intents, *, filters, top_k):
        return self.chunks[:top_k]


class FixedReranker:
    """Return a fixed outcome: the first `count` candidates, assessed or as a fallback."""

    def __init__(self, count: int, *, assessed: bool) -> None:
        self.count = count
        self.assessed = assessed

    def rerank(self, *, case_brief, candidates):
        selected = [(c, "Popiera postulat.", 80) for c in candidates[: self.count]]
        return RerankOutcome(selected=selected, assessed=self.assessed)


def leading_document_candidates() -> list[RetrievedChunk]:
    """Eight candidates from one document ranked above three from another."""
    return [make_chunk(f"A{i}", "doc-a") for i in range(8)] + [
        make_chunk(f"B{i}", "doc-b") for i in range(3)
    ]


class TestBriefToEvidencePipelineRun:
    def test_run__carries_the_status_and_address_of_the_document_into_the_evidence(self) -> None:
        """Verify what the catalog says about a document survives until the letter is rendered."""
        pipeline = BriefToEvidencePipeline(
            planner=SingleIntentPlanner(), retriever=CatalogChunkRetriever()
        )

        (evidence,) = pipeline.run({"problem_description": "Problem"}).evidence

        assert (evidence.is_active, evidence.lifecycle_status, evidence.source_url) == (
            True,
            "current_reference",
            "https://example.org/polityka.pdf",
        )

    def test_run__hands_the_callers_filters_to_the_retriever(self) -> None:
        """Verify a historical search requested by the caller is not replaced by the default."""
        retriever = CatalogChunkRetriever()
        pipeline = BriefToEvidencePipeline(planner=SingleIntentPlanner(), retriever=retriever)
        filters = RetrievalFilters(exclude_inactive=False)

        pipeline.run({"problem_description": "Problem"}, filters=filters)

        assert retriever.filters is filters

    def test_run__marks_evidence_unverified_without_a_reranker(self) -> None:
        """Verify retrieval alone never claims that the found fragments support the case."""
        pack = BriefToEvidencePipeline(
            planner=SingleIntentPlanner(), retriever=CatalogChunkRetriever()
        ).run({"problem_description": "Problem"})

        assert pack.evidence_status == "unverified"
        assert len(pack.evidence) == 1

    def test_run__marks_evidence_supported_when_the_reranker_selects_fragments(self) -> None:
        """Verify fragments chosen by an assessing reranker are reported as support."""
        pipeline = BriefToEvidencePipeline(
            planner=SingleIntentPlanner(),
            retriever=CatalogChunkRetriever(),
            reranker=FixedOutcomeReranker(select=True, assessed=True),
        )

        pack = pipeline.run({"problem_description": "Problem"})

        assert pack.evidence_status == "supported"
        assert [e.rerank_score for e in pack.evidence] == [80]

    def test_run__returns_no_evidence_when_the_reranker_finds_no_support(self) -> None:
        """Verify a case outside the corpus ends with an explicit abstention, not similar texts."""
        pipeline = BriefToEvidencePipeline(
            planner=SingleIntentPlanner(),
            retriever=CatalogChunkRetriever(),
            reranker=FixedOutcomeReranker(select=False, assessed=True),
        )

        pack = pipeline.run({"problem_description": "Problem"})

        assert pack.evidence_status == "unsupported"
        assert pack.evidence == []
        assert pack.unsupported_topics == ["problem"]

    def test_run__marks_a_fallback_selection_unverified(self) -> None:
        """Verify hybrid fragments kept after a model failure are not presented as assessed."""
        pipeline = BriefToEvidencePipeline(
            planner=SingleIntentPlanner(),
            retriever=CatalogChunkRetriever(),
            reranker=FixedOutcomeReranker(select=True, assessed=False),
        )

        pack = pipeline.run({"problem_description": "Problem"})

        assert pack.evidence_status == "unverified"
        assert len(pack.evidence) == 1

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

    def test_run__fills_the_pack_from_all_candidates_after_a_reranker_fallback(self) -> None:
        """Verify an unassessed fallback is filled like plain retrieval, not cut to the model's limit."""
        pipeline = BriefToEvidencePipeline(
            planner=ProblemPlanner(),
            retriever=ListRetriever(leading_document_candidates()),
            reranker=FixedReranker(6, assessed=False),
            evidence_k=6,
            max_per_document=3,
        )

        result = pipeline.run({"problem_description": "Problem"})

        assert [e.chunk_id for e in result.evidence] == ["A0", "A1", "A2", "B0", "B1", "B2"]
        assert result.evidence_status == "unverified"

    def test_run__does_not_add_unassessed_candidates_to_the_rerankers_selection(self) -> None:
        """Verify a supported pack holds only fragments the reranker judged, even if it is small."""
        pipeline = BriefToEvidencePipeline(
            planner=ProblemPlanner(),
            retriever=ListRetriever(leading_document_candidates()),
            reranker=FixedReranker(5, assessed=True),
            evidence_k=6,
            max_per_document=3,
        )

        result = pipeline.run({"problem_description": "Problem"})

        assert [e.chunk_id for e in result.evidence] == ["A0", "A1", "A2"]
        assert result.evidence_status == "supported"


class TestBriefToEvidencePipelineBriefDict:
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
