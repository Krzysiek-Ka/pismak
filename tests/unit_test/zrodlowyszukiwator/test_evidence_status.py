from __future__ import annotations

from src.Zrodlowyszukiwator.brief_to_evidence import (
    BriefToEvidencePipeline,
    Evidence,
    EvidencePack,
    ResearchPlan,
    evidence_pack_to_markdown,
)
from src.Zrodlowyszukiwator.rag_retrieval.models import (
    RetrievalFilters,
    RetrievedChunk,
    SearchIntent,
)
from src.Zrodlowyszukiwator.rag_retrieval.reranker import RerankOutcome

INTENT = SearchIntent(semantic_query="problem", lexical_query="problem", purpose="problem")


def make_evidence(evidence_id: str, is_active: bool | None) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=f"chunk-{evidence_id}",
        document_id=f"doc-{evidence_id}",
        document_title=f"Dokument {evidence_id}",
        document_kind="strategy",
        hierarchy_rank=100,
        legal_form="resolution",
        is_active=is_active,
        page_start=1,
        page_end=1,
        excerpt="Fragment.",
        retrieval_score=1.0,
        relevance_reason="Pasuje do sprawy.",
    )


class FakePlanner:
    def plan(self, brief: dict) -> ResearchPlan:
        return ResearchPlan(case_summary="Sprawa", intents=[INTENT])


class FakeRetriever:
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


class TestEvidencePackToMarkdown:
    def test_evidence_pack_to_markdown__tells_the_model_whether_each_source_is_in_force(
        self,
    ) -> None:
        """Verify the letter prompt separates binding, historical and unverified documents."""
        pack = EvidencePack(
            query_summary="Sprawa",
            research_plan=ResearchPlan(case_summary="Sprawa", intents=[]),
            evidence=[
                make_evidence("E1", True),
                make_evidence("E2", False),
                make_evidence("E3", None),
            ],
        )

        blocks = evidence_pack_to_markdown(pack).split("## [")[1:]

        assert [
            line.removeprefix("- status obowiązywania: ")
            for block in blocks
            for line in block.splitlines()
            if line.startswith("- status obowiązywania: ")
        ] == ["obowiązujący", "nieobowiązujący (dokument historyczny)", "nieustalony"]


class TestBriefToEvidencePipeline:
    def test_run__carries_the_status_and_address_of_the_document_into_the_evidence(self) -> None:
        """Verify what the catalog says about a document survives until the letter is rendered."""
        pipeline = BriefToEvidencePipeline(planner=FakePlanner(), retriever=FakeRetriever())

        (evidence,) = pipeline.run({"problem_description": "Problem"}).evidence

        assert (evidence.is_active, evidence.lifecycle_status, evidence.source_url) == (
            True,
            "current_reference",
            "https://example.org/polityka.pdf",
        )

    def test_run__hands_the_callers_filters_to_the_retriever(self) -> None:
        """Verify a historical search requested by the caller is not replaced by the default."""
        retriever = FakeRetriever()
        pipeline = BriefToEvidencePipeline(planner=FakePlanner(), retriever=retriever)
        filters = RetrievalFilters(exclude_inactive=False)

        pipeline.run({"problem_description": "Problem"}, filters=filters)

        assert retriever.filters is filters


class FakeReranker:
    """Return a fixed reranking outcome for whatever candidates it receives."""

    def __init__(self, *, select: bool, assessed: bool) -> None:
        self.select = select
        self.assessed = assessed

    def rerank(self, *, case_brief: dict, candidates: list[RetrievedChunk]) -> RerankOutcome:
        selected = [(c, "Popiera postulat.", 80) for c in candidates] if self.select else []
        return RerankOutcome(selected=selected, assessed=self.assessed)


class TestEvidenceSupportStatus:
    def test_run__marks_evidence_unverified_without_a_reranker(self) -> None:
        """Verify retrieval alone never claims that the found fragments support the case."""
        pack = BriefToEvidencePipeline(planner=FakePlanner(), retriever=FakeRetriever()).run(
            {"problem_description": "Problem"}
        )

        assert pack.evidence_status == "unverified"
        assert len(pack.evidence) == 1

    def test_run__marks_evidence_supported_when_the_reranker_selects_fragments(self) -> None:
        """Verify fragments chosen by an assessing reranker are reported as support."""
        pipeline = BriefToEvidencePipeline(
            planner=FakePlanner(),
            retriever=FakeRetriever(),
            reranker=FakeReranker(select=True, assessed=True),
        )

        pack = pipeline.run({"problem_description": "Problem"})

        assert pack.evidence_status == "supported"
        assert [e.rerank_score for e in pack.evidence] == [80]

    def test_run__returns_no_evidence_when_the_reranker_finds_no_support(self) -> None:
        """Verify a case outside the corpus ends with an explicit abstention, not similar texts."""
        pipeline = BriefToEvidencePipeline(
            planner=FakePlanner(),
            retriever=FakeRetriever(),
            reranker=FakeReranker(select=False, assessed=True),
        )

        pack = pipeline.run({"problem_description": "Problem"})

        assert pack.evidence_status == "unsupported"
        assert pack.evidence == []
        assert pack.unsupported_topics == ["problem"]

    def test_run__marks_a_fallback_selection_unverified(self) -> None:
        """Verify hybrid fragments kept after a model failure are not presented as assessed."""
        pipeline = BriefToEvidencePipeline(
            planner=FakePlanner(),
            retriever=FakeRetriever(),
            reranker=FakeReranker(select=True, assessed=False),
        )

        pack = pipeline.run({"problem_description": "Problem"})

        assert pack.evidence_status == "unverified"
        assert len(pack.evidence) == 1

    def test_evidence_pack_to_markdown__tells_the_model_how_far_the_sources_were_checked(
        self,
    ) -> None:
        """Verify the letter prompt says when fragments were not assessed or support nothing."""
        plan = ResearchPlan(case_summary="Sprawa", intents=[])
        unverified = EvidencePack(
            query_summary="Sprawa", research_plan=plan, evidence=[make_evidence("E1", True)]
        )
        supported = unverified.model_copy(update={"evidence_status": "supported"})
        unsupported = EvidencePack(
            query_summary="Sprawa", research_plan=plan, evidence_status="unsupported"
        )

        assert "trafność fragmentów nie została oceniona" in evidence_pack_to_markdown(unverified)
        assert "nie została oceniona" not in evidence_pack_to_markdown(supported)
        assert "nie odpowiada na sprawę" in evidence_pack_to_markdown(unsupported)
