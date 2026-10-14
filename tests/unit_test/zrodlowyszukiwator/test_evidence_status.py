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
