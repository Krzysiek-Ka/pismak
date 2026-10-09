"""Tests for the Pismogenerator helper functions."""

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator_helper_functions, structures
from src.Zrodlowyszukiwator.brief_to_evidence.models import Evidence, EvidencePack, ResearchPlan


def make_evidence_pack() -> EvidencePack:
    return EvidencePack(
        query_summary="Sprawa testowa",
        research_plan=ResearchPlan(case_summary="Sprawa testowa", intents=[]),
        evidence=[
            Evidence(
                evidence_id="E1",
                chunk_id="chunk-1",
                document_id="document-1",
                document_title="Raport miejski",
                document_kind="report",
                hierarchy_rank=1,
                legal_form="report",
                page_start=4,
                page_end=5,
                excerpt="Fragment raportu.",
                retrieval_score=1.0,
                relevance_reason="Pasuje do sprawy.",
            )
        ],
    )


class TestValidateDraft:
    def test_validate_draft__returns_errors_for_invalid_draft(self) -> None:
        """Verify the validator catches empty recipient, short argumentation, and illegal sources."""
        draft = structures.DraftLetter(
            recipient="",
            sender="",
            request_summary="wniosek o dostęp",
            argumentation_markdown="Za krótko.",
            sources=["E99"],
            claims_requiring_verification=[],
            missing_fields=[],
            assumptions_used=[],
        )
        brief = CaseBrief(
            title="Brak dostępu",
            problem_description="Nie ma dostępu do dokumentów.",
            desired_outcome="Przywrócenie dostępu.",
            requested_actions=["Sprawdzenie dokumentacji"],
            facts_to_verify=[],
        )

        errors = pismogenerator_helper_functions.validate_draft(
            draft=draft,
            case_brief=brief,
            evidence_pack=make_evidence_pack(),
            user_evidence=[],
        )

        assert any("adresata" in error.lower() for error in errors)
        assert any("krótka" in error.lower() for error in errors)
        assert any("niedozwolone" in error.lower() for error in errors)


class TestRenderListAndSources:
    def test_render_list__returns_bullet_list_for_items(self) -> None:
        """Verify list rendering keeps a consistent bullet format."""
        assert pismogenerator_helper_functions.render_list(["A", "B"]) == "- A\n- B"

    def test_render_sources__uses_reference_numbers(self) -> None:
        """Verify sources are rendered in numbered format for later reference."""
        assert (
            pismogenerator_helper_functions.render_sources(
                ["E1", "U1"],
                make_evidence_pack(),
                ["Zdjęcie"],
            )
            == "[1] E1: Raport miejski, s. 4-5\n[2] U1: Zdjęcie"
        )

    def test_render_sources__returns_fallback_for_empty_sources(self) -> None:
        """Verify an empty source list renders a safe fallback instead of blank output."""
        assert (
            pismogenerator_helper_functions.render_sources(
                [],
                make_evidence_pack(),
                [],
            )
            == "Brak źródeł wskazanych przez użytkownika."
        )

    def test_render_sources__does_not_crash_on_malformed_user_source_id(self) -> None:
        """Verify malformed user identifiers remain visible instead of crashing the renderer."""
        assert (
            pismogenerator_helper_functions.render_sources(
                ["U99", "Ubad"],
                make_evidence_pack(),
                ["Zdjęcie"],
            )
            == "[1] U99\n[2] Ubad"
        )


class TestRenderLetter:
    def test_render_letter__includes_all_sections(self) -> None:
        """Verify every required letter section is present in the final document."""
        draft = structures.DraftLetter(
            recipient="Urząd",
            sender="",
            request_summary="dostęp do dokumentacji",
            argumentation_markdown="Wnoszę o szybkie rozpatrzenie sprawy.",
            sources=["E1"],
            claims_requiring_verification=["Czy to jest poprawne?"],
            missing_fields=["Brak adresu"],
            assumptions_used=["Założenie neutralne"],
        )

        result = pismogenerator_helper_functions.render_letter(
            draft,
            make_evidence_pack(),
            [],
        )

        assert "**Adresat:** Urząd" in result
        assert "## Źródła" in result
        assert "## Sekcja kontrolna" in result
        assert "### Informacje wymagające weryfikacji przed wysłaniem" in result
