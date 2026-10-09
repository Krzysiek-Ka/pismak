"""Tests for the Pismogenerator orchestration functions."""

from __future__ import annotations

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator, structures
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


class TestExtractSourcesFromBrief:
    def test_extract_sources_from_brief__uses_available_evidence_when_no_sources_field_exists(
        self,
    ) -> None:
        """Verify source candidates are derived from the brief when no explicit sources field is present."""
        brief = CaseBrief(
            title="Brak dostępu",
            problem_description="Problem z dostępem do dokumentów.",
            desired_outcome="Przywrócenie dostępu.",
            requested_actions=["Sprawdzenie dokumentacji"],
            available_evidence=["Zdjęcie 1", "Zdjęcie 1", "Wniosek"],
        )

        result = pismogenerator.extract_sources_from_brief(brief)

        assert result == ["Zdjęcie 1", "Wniosek"]


class TestBuildDraft:
    def test_build_draft__clears_sender_returned_by_model(self) -> None:
        """Verify the model cannot populate the Phase 0 sender field and block draft creation."""

        class FakeLLM:
            def generate_structured(self, **kwargs):
                return structures.DraftLetter(
                    recipient="Urząd",
                    sender="Towarzystwo Upiększania Miasta Wrocławia",
                    request_summary="dostęp do dokumentacji",
                    argumentation_markdown="A" * 200,
                    sources=["E1"],
                )

        brief = CaseBrief(
            title="Brak dostępu",
            problem_description="Problem z dostępem do dokumentów.",
            desired_outcome="Przywrócenie dostępu.",
            requested_actions=["Sprawdzenie dokumentacji"],
            facts_to_verify=[],
        )

        result = pismogenerator.build_draft(brief, make_evidence_pack(), FakeLLM())

        assert result.sender == ""

    def test_build_draft__returns_draft_when_validation_passes(self, monkeypatch) -> None:
        """Verify the draft is generated and returned when validation does not raise errors."""

        class FakeLLM:
            def generate_structured(self, **kwargs):
                return structures.DraftLetter(
                    recipient="Urząd",
                    sender="",
                    request_summary="wniosek o dostęp",
                    argumentation_markdown="A" * 200,
                    sources=["E1"],
                    claims_requiring_verification=[],
                    missing_fields=[],
                    assumptions_used=[],
                )

        brief = CaseBrief(
            title="Brak dostępu",
            problem_description="Problem z dostępem do dokumentów.",
            desired_outcome="Przywrócenie dostępu.",
            requested_actions=["Sprawdzenie dokumentacji"],
            available_evidence=["Zdjęcie 1"],
            facts_to_verify=[],
        )
        evidence_pack = make_evidence_pack()
        monkeypatch.setattr(
            pismogenerator.pismogenerator_helper_functions, "validate_draft", lambda **kwargs: []
        )

        result = pismogenerator.build_draft(
            brief,
            evidence_pack,
            FakeLLM(),
            llm_temperature=0.1,
        )

        assert isinstance(result, structures.DraftLetter)
        assert result.recipient == "Urząd"


class TestBuildLetter:
    def test_build_letter__renders_document_markup(self) -> None:
        """Verify the final letter includes address, summary, sources, and verification blocks."""
        draft = structures.DraftLetter(
            recipient="Dyrektor Urzędu",
            sender="",
            request_summary="dostęp do dokumentacji",
            argumentation_markdown="Dostęp do dokumentacji jest kluczowy.",
            sources=["E1"],
            claims_requiring_verification=["Czy dokument istnieje?"],
            missing_fields=["Brak danych osobowych"],
            assumptions_used=[],
        )

        result = pismogenerator.build_letter(draft, make_evidence_pack())

        assert "**Adresat:** Dyrektor Urzędu" in result
        assert "W imieniu Towarzystwa Upiększania Miasta Wrocławia" in result
        assert "## Źródła" in result
        assert "### Informacje wymagające weryfikacji przed wysłaniem" in result
        assert "Raport miejski" in result


class TestValidateDraft:
    def test_validate_draft__rejects_source_id_outside_evidence_pack(self) -> None:
        """Verify citations cannot reference evidence absent from the supplied pack."""
        draft = structures.DraftLetter(
            recipient="Urząd",
            request_summary="dostęp do dokumentacji",
            argumentation_markdown="A" * 200,
            sources=["E99"],
        )
        brief = CaseBrief(
            title="Sprawa",
            problem_description="Problem.",
            desired_outcome="Rozwiązanie.",
            requested_actions=["Sprawdzenie"],
        )

        errors = pismogenerator.pismogenerator_helper_functions.validate_draft(
            draft=draft,
            case_brief=brief,
            evidence_pack=make_evidence_pack(),
            user_evidence=[],
        )

        assert any("niedozwolone źródła" in error for error in errors)
