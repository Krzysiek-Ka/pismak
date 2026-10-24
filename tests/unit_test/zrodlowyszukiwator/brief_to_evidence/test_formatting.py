"""Tests for the EvidencePack Markdown sent to the Pismogenerator."""

from __future__ import annotations

from src.Zrodlowyszukiwator.brief_to_evidence.formatting import (
    _truncate_at_boundary,
    evidence_pack_to_markdown,
)
from src.Zrodlowyszukiwator.brief_to_evidence.models import Evidence, EvidencePack, ResearchPlan


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


class TestTruncateAtBoundary:
    def test_truncate_at_boundary__text_without_a_boundary_is_cut_with_an_ellipsis(self) -> None:
        """Verify long text is bounded even when no sentence boundary is available."""
        assert _truncate_at_boundary("abcdefghij", 5) == "abcde…"


class TestEvidencePackToMarkdown:
    def test_evidence_pack_to_markdown__empty_pack_says_no_sources_were_found(self) -> None:
        """Verify an empty EvidencePack explicitly reports missing sources."""
        pack = EvidencePack(
            query_summary="Brak wyników",
            research_plan=ResearchPlan(case_summary="Brak wyników", intents=[]),
        )

        result = evidence_pack_to_markdown(pack)

        assert "Brak znalezionych źródeł." in result

    def test_evidence_pack_to_markdown__evidence_shows_metadata_and_unsupported_topics(
        self,
    ) -> None:
        """Verify evidence metadata and unsupported topics remain visible to the generator."""
        result = evidence_pack_to_markdown(make_pack())

        assert "[E1] Polityka miejska" in result
        assert "strony PDF: 3" in result
        assert "Rozdział 1" in result
        assert "finansowanie" in result

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
