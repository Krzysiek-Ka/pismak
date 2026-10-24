"""Tests for the Pismogenerator orchestration functions."""

from __future__ import annotations

import pytest

from src._shared.context_budget import input_char_budget
from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator
from src.Pismogenerator.constants import (
    PISMOGENERATOR_MAX_OUTPUT_TOKENS,
    PISMOGENERATOR_RESERVED_OUTPUT_TOKENS,
)
from src.Pismogenerator.pismogenerator import DraftValidationError
from src.Pismogenerator.prompts import PISMOGENERATOR_SYSTEM_PROMPT, draft_user_prompt
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.brief_to_evidence.models import Evidence, EvidencePack, ResearchPlan


def make_pack(excerpts: list[str]) -> EvidencePack:
    return EvidencePack(
        query_summary="Sprawa testowa",
        research_plan=ResearchPlan(case_summary="Sprawa testowa", intents=[]),
        evidence=[
            Evidence(
                evidence_id=f"E{index}",
                chunk_id=f"chunk-{index}",
                document_id="document-1",
                document_title="Dokument miejski",
                document_kind="strategy",
                hierarchy_rank=100,
                legal_form="resolution",
                page_start=index,
                page_end=index,
                excerpt=excerpt,
                retrieval_score=1.0,
                relevance_reason="Pasuje do sprawy.",
            )
            for index, excerpt in enumerate(excerpts, start=1)
        ],
    )


def make_brief() -> CaseBrief:
    return CaseBrief(
        title="Sprawa",
        problem_description="Problem",
        desired_outcome="Rozwiązanie",
        requested_actions=["Sprawdzenie"],
    )


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


class TestFitDraftUserPrompt:
    def test_fit_draft_user_prompt__shortens_excerpts_until_the_prompt_fits_the_window(
        self, monkeypatch
    ) -> None:
        """Verify six full-length sources are trimmed so the letter still has room to be written."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        pack = make_pack(["Zdanie źródłowe. " * 200] * 6)

        prompt = pismogenerator.fit_draft_user_prompt(make_brief(), pack, [], "ollama")

        budget = input_char_budget(
            "ollama", reserved_output_tokens=PISMOGENERATOR_RESERVED_OUTPUT_TOKENS
        )
        assert len(PISMOGENERATOR_SYSTEM_PROMPT) + len(prompt) <= budget
        assert all(f"## [E{index}]" in prompt for index in range(1, 7))

    def test_fit_draft_user_prompt__keeps_full_excerpt_when_it_fits(self, monkeypatch) -> None:
        """Verify a prompt already inside the window reaches the model with the source intact."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        excerpt = "Zdanie źródłowe. " * 100

        prompt = pismogenerator.fit_draft_user_prompt(
            make_brief(), make_pack([excerpt]), [], "ollama"
        )

        assert excerpt.strip() in prompt

    def test_fit_draft_user_prompt__sends_longer_excerpts_to_a_larger_remote_window(
        self, monkeypatch
    ) -> None:
        """Verify trimming follows the provider window instead of always using the local one."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        monkeypatch.delenv("PISMAK_OPENROUTER_CONTEXT_TOKENS", raising=False)
        pack = make_pack(["Zdanie źródłowe. " * 200] * 6)

        local = pismogenerator.fit_draft_user_prompt(make_brief(), pack, [], "ollama")
        remote = pismogenerator.fit_draft_user_prompt(make_brief(), pack, [], "openrouter")

        assert len(remote) > len(local)

    def test_fit_draft_user_prompt__keeps_full_excerpts_for_a_large_model_window(
        self, monkeypatch
    ) -> None:
        """Verify a local model with a large window gets the same prompt as an unlimited one."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        pack = make_pack(["Zdanie źródłowe. " * 200] * 6)

        prompt = pismogenerator.fit_draft_user_prompt(make_brief(), pack, [], "ollama", 131072)

        assert prompt == draft_user_prompt(make_brief(), pack, [])


class TestBuildDraft:
    def test_build_draft__clears_sender_returned_by_model(self) -> None:
        """Verify the model cannot populate the Phase 0 sender field and block draft creation."""

        class FakeLLM:
            def generate_structured(self, **kwargs):
                return DraftLetter(
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

    def test_build_draft__valid_answer_returns_the_draft(self, monkeypatch) -> None:
        """Verify the draft is generated and returned when validation does not raise errors."""

        class FakeLLM:
            def generate_structured(self, **kwargs):
                return DraftLetter(
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

        assert isinstance(result, DraftLetter)
        assert result.recipient == "Urząd"

    def test_build_draft__model_call_uses_the_module_output_token_limit(self, monkeypatch) -> None:
        """Verify letter generation uses its larger, dedicated output token budget."""
        captured: dict[str, object] = {}

        class FakeLLM:
            def generate_structured(self, **kwargs):
                captured.update(kwargs)
                return DraftLetter(
                    recipient="Urząd",
                    request_summary="Wniosek",
                    argumentation_markdown="A" * 200,
                    sources=[],
                )

        monkeypatch.setattr(
            pismogenerator.pismogenerator_helper_functions,
            "validate_draft",
            lambda **kwargs: [],
        )
        brief = CaseBrief(
            title="Sprawa",
            problem_description="Problem",
            desired_outcome="Rozwiązanie",
            requested_actions=["Sprawdzenie"],
        )
        evidence = EvidencePack(
            query_summary="Brak wyników",
            research_plan=ResearchPlan(case_summary="Brak wyników", intents=[]),
        )

        pismogenerator.build_draft(brief, evidence, FakeLLM())

        assert captured["max_output_tokens"] == PISMOGENERATOR_MAX_OUTPUT_TOKENS

    def test_build_draft__budgets_the_prompt_for_the_window_of_its_llm_client(
        self, monkeypatch
    ) -> None:
        """Verify the draft prompt follows the model window carried by the client that sends it."""
        monkeypatch.delenv("PISMAK_CONTEXT_TOKENS", raising=False)
        captured: dict[str, object] = {}

        class FakeLLM:
            provider = "ollama"
            context_tokens = 131072

            def generate_structured(self, **kwargs):
                captured.update(kwargs)
                return DraftLetter(
                    recipient="Urząd",
                    request_summary="Wniosek",
                    argumentation_markdown="A" * 200,
                    sources=[],
                )

        monkeypatch.setattr(
            pismogenerator.pismogenerator_helper_functions,
            "validate_draft",
            lambda **kwargs: [],
        )
        pack = make_pack(["Zdanie źródłowe. " * 200] * 6)

        pismogenerator.build_draft(make_brief(), pack, FakeLLM())

        assert captured["user_prompt"] == draft_user_prompt(make_brief(), pack, [])

    def test_build_draft__invalid_draft_raises_with_the_errors_and_the_draft(
        self, monkeypatch
    ) -> None:
        """Verify rejected model sources are preserved for the UI and diagnostic log."""

        class FakeLLM:
            def generate_structured(self, **kwargs):
                return DraftLetter(
                    recipient="Urząd",
                    request_summary="Wniosek",
                    argumentation_markdown="A" * 200,
                    sources=["U1", "U2"],
                )

        monkeypatch.setattr(
            pismogenerator.pismogenerator_helper_functions,
            "validate_draft",
            lambda **kwargs: ["Model dodał niedozwolone źródła: U1; U2"],
        )
        brief = CaseBrief(
            title="Sprawa",
            problem_description="Problem",
            desired_outcome="Rozwiązanie",
            requested_actions=["Sprawdzenie"],
        )
        evidence = EvidencePack(
            query_summary="Brak wyników",
            research_plan=ResearchPlan(case_summary="Brak wyników", intents=[]),
        )

        with pytest.raises(DraftValidationError) as error:
            pismogenerator.build_draft(brief, evidence, FakeLLM())

        assert "DraftLetter nie przeszedł walidacji" in str(error.value)
        assert "Model dodał niedozwolone źródła: U1; U2" in str(error.value)
        assert error.value.draft is not None
        assert error.value.draft.sources == ["U1", "U2"]

    def test_build_draft__asks_for_a_schema_with_the_ids_of_this_case(self) -> None:
        """Verify build_draft passes the evidence and user-material IDs and returns DraftLetter."""
        captured: dict[str, object] = {}

        class FakeLLM:
            def generate_structured(self, **kwargs):
                captured.update(kwargs)
                return kwargs["response_model"](
                    recipient="Urząd",
                    request_summary="dostęp do dokumentacji",
                    argumentation_markdown="A" * 200,
                    sources=["E1", "U1"],
                )

        brief = make_brief().model_copy(update={"available_evidence": ["Zdjęcie", "  "]})

        result = pismogenerator.build_draft(brief, make_evidence_pack(), FakeLLM())

        schema = captured["response_model"].model_json_schema()
        assert schema["properties"]["sources"]["items"]["enum"] == ["E1", "U1"]
        assert type(result) is DraftLetter
        assert result.sources == ["E1", "U1"]


class TestBuildLetter:
    def test_build_letter__valid_draft_renders_every_section(self) -> None:
        """Verify the final letter includes address, summary, sources, and verification blocks."""
        draft = DraftLetter(
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
        assert (
            "Jako wieloletni mieszkaniec naszego kochanego miasta, wnioskuję o "
            "dostęp do dokumentacji"
        ) in result
        assert "[imię i nazwisko]" in result
        assert "## Źródła" in result
        assert "### Informacje wymagające weryfikacji przed wysłaniem" in result
        assert "Raport miejski" in result

    def test_build_letter__numbers_user_materials_like_build_draft(self) -> None:
        """Verify a blank material does not shift U1 to the wrong text in the notebook path."""
        draft = DraftLetter(
            recipient="Urząd",
            request_summary="dostęp do dokumentacji",
            argumentation_markdown="A" * 200,
            sources=["U1"],
        )

        letter = pismogenerator.build_letter(draft, make_evidence_pack(), ["  ", " Zdjęcie "])

        assert "- [1] U1: Zdjęcie" in letter


class TestDraftLetterModel:
    def test_draft_letter_model__limits_sources_to_the_allowed_ids_in_the_schema(self) -> None:
        """Verify the schema sent to the model lists only E#/U# IDs, so "1" cannot be produced."""
        from openai.lib._parsing import type_to_response_format_param

        model = pismogenerator.draft_letter_model(["E1", "U1"])
        response_format = type_to_response_format_param(model)

        assert response_format["json_schema"]["name"] == "DraftLetter"
        sources = response_format["json_schema"]["schema"]["properties"]["sources"]
        assert sources["items"]["enum"] == ["E1", "U1"]
        with pytest.raises(ValueError):
            model.model_validate(
                {
                    "recipient": "Urząd",
                    "request_summary": "Wniosek",
                    "argumentation_markdown": "A" * 200,
                    "sources": ["1"],
                }
            )

    def test_draft_letter_model__allows_no_sources_when_nothing_was_found(self) -> None:
        """Verify a case without evidence or user materials gets an empty source list only."""
        model = pismogenerator.draft_letter_model([])

        schema = model.model_json_schema()

        assert schema["properties"]["sources"]["maxItems"] == 0
