"""Tests for the Pismogenerator helper functions."""

import pytest

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


ARGUMENTATION = (
    "Dzieci codziennie przechodzą przez ruchliwą ulicę w miejscu bez oznakowania. "
    "Dokument miejski wskazuje bezpieczeństwo pieszych jako priorytet [1]."
)


def make_brief(**overrides: object) -> CaseBrief:
    values: dict[str, object] = {
        "title": "Przejście przy szkole",
        "problem_description": "Brak przejścia.",
        "desired_outcome": "Wyznaczenie przejścia.",
        "requested_actions": ["Wyznaczenie przejścia"],
    }
    values.update(overrides)
    return CaseBrief.model_validate(values)


def make_draft(**overrides: object) -> structures.DraftLetter:
    values: dict[str, object] = {
        "recipient": "Zarząd Dróg",
        "request_summary": "wyznaczenie przejścia dla pieszych",
        "argumentation_markdown": ARGUMENTATION,
        "sources": ["E1"],
    }
    values.update(overrides)
    return structures.DraftLetter.model_validate(values)


def validate(
    draft: structures.DraftLetter,
    *,
    case_brief: CaseBrief | None = None,
    user_evidence: list[str] | None = None,
) -> list[str]:
    return pismogenerator_helper_functions.validate_draft(
        draft=draft,
        case_brief=case_brief or make_brief(),
        evidence_pack=make_evidence_pack(),
        user_evidence=user_evidence or [],
    )


class TestValidateDraft:
    def test_validate_draft__returns_errors_for_invalid_draft(self) -> None:
        """Verify the validator catches short argumentation and illegal sources."""
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

        assert not any("adresata" in error.lower() for error in errors)
        assert any("krótka" in error.lower() for error in errors)
        assert any("niedozwolone" in error.lower() for error in errors)

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

        errors = pismogenerator_helper_functions.validate_draft(
            draft=draft,
            case_brief=brief,
            evidence_pack=make_evidence_pack(),
            user_evidence=[],
        )

        assert any("niedozwolone źródła" in error for error in errors)

    def test_validate_draft__accepts_a_complete_draft(self) -> None:
        """Verify a correct draft passes, so every other test isolates exactly one problem."""
        assert validate(make_draft()) == []

    def test_validate_draft__rejects_a_sender_filled_in_by_the_model(self) -> None:
        """Verify the model cannot sign the letter on someone's behalf in Phase 0."""
        assert validate(make_draft(sender="Jan Kowalski")) == [
            "Pole nadawcy powinno pozostać puste w Phase 0"
        ]

    def test_validate_draft__rejects_a_blank_request_summary(self) -> None:
        """Verify the fixed 'wnioskuję o …' formula is never left without its object."""
        assert validate(make_draft(request_summary="   ")) == [
            "Brak krótkiego podsumowania wniosku"
        ]

    @pytest.mark.parametrize("summary", ["o wyznaczenie przejścia", "O wyznaczenie przejścia"])
    def test_validate_draft__rejects_a_summary_that_would_double_the_preposition(
        self, summary: str
    ) -> None:
        """Verify the renderer cannot produce 'wnioskuję o o …' whatever the letter case."""
        errors = validate(make_draft(request_summary=summary))

        assert errors == [
            "request_summary nie powinno zaczynać się od 'o' — renderer dodaje 'wnioskuję o'"
        ]

    def test_validate_draft__accepts_a_summary_that_only_starts_with_the_letter_o(self) -> None:
        """Verify words such as 'oznakowanie' are not mistaken for the preposition."""
        assert validate(make_draft(request_summary="oznakowanie przejścia")) == []

    @pytest.mark.parametrize(
        "sentence",
        [
            "Zgodnie z art. 241 wnoszę o zmianę.",
            "Na podstawie KPA wnoszę o zmianę.",
            "Wnoszę o dostęp do informacji publicznej.",
            "Program jest gotowy na start.",
            "Raport za I kwart.",
        ],
    )
    def test_validate_draft__does_not_check_legal_phrases(self, sentence: str) -> None:
        """Verify the legal-phrase check is off, so neither real nor false matches block a draft."""
        assert validate(make_draft(argumentation_markdown=ARGUMENTATION + " " + sentence)) == []

    def test_validate_draft__accepts_argumentation_of_exactly_one_hundred_characters(
        self,
    ) -> None:
        """Verify the minimum length is inclusive and surrounding whitespace does not count."""
        assert validate(make_draft(argumentation_markdown="A" * 100, sources=[])) == []
        assert validate(make_draft(argumentation_markdown="  " + "A" * 99 + "  ", sources=[])) == [
            "Argumentacja projektu jest podejrzanie krótka"
        ]

    def test_validate_draft__accepts_user_materials_as_numbered_sources(self) -> None:
        """Verify U1…Un refer to the user's materials and nothing beyond them."""
        draft = make_draft(sources=["E1", "U1", "U2", "U3"])

        errors = validate(draft, user_evidence=["Zdjęcie", "Pismo z urzędu"])

        assert errors == ["Model dodał niedozwolone źródła: U3"]

    def test_validate_draft__rejects_footnotes_without_a_source_list(self) -> None:
        """Verify a reader is never sent to a footnote that has no source behind it."""
        errors = validate(make_draft(sources=[]))

        assert errors == [
            "Argumentacja zawiera przypisy, ale lista sources jest pusta",
            "Argumentacja odwołuje się do numeru źródła, którego nie ma na liście sources",
        ]

    def test_validate_draft__rejects_a_footnote_number_beyond_the_source_list(self) -> None:
        """Verify [2] with a single source is caught instead of rendering a dangling reference."""
        errors = validate(make_draft(argumentation_markdown=ARGUMENTATION + " Zob. też [2]."))

        assert errors == [
            "Argumentacja odwołuje się do numeru źródła, którego nie ma na liście sources"
        ]

    def test_validate_draft__does_not_block_or_print_facts_missing_from_the_control_section(
        self, capsys
    ) -> None:
        """Verify unchecked facts do not block the draft and no case data reaches the console."""
        brief = make_brief(facts_to_verify=["Data wypadku", "Liczba uczniów"])
        draft = make_draft(claims_requiring_verification=["Data wypadku"])

        errors = validate(draft, case_brief=brief)

        assert errors == []
        assert capsys.readouterr().out == ""


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
            == "- [1] E1: Raport miejski, s. 4-5\n- [2] U1: Zdjęcie"
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
            == "- [1] U99\n- [2] Ubad"
        )

    def test_render_sources__adds_the_document_address_and_a_single_page(self) -> None:
        """Verify a reader can open the cited document and finds one page written once."""
        pack = make_evidence_pack()
        pack.evidence[0] = pack.evidence[0].model_copy(
            update={"page_end": 4, "source_url": "https://bip.um.wroc.pl/dokument.pdf"}
        )

        assert pismogenerator_helper_functions.render_sources(["E1"], pack, []) == (
            "- [1] E1: Raport miejski, s. 4, https://bip.um.wroc.pl/dokument.pdf"
        )

    def test_render_sources__treats_u0_as_unknown_instead_of_the_last_material(self) -> None:
        """Verify U0 is not resolved through index -1 to the user's last material."""
        assert (
            pismogenerator_helper_functions.render_sources(
                ["U0"], make_evidence_pack(), ["Zdjęcie", "Pismo"]
            )
            == "- [1] U0"
        )

    def test_render_list__uses_the_placeholder_for_an_empty_section(self) -> None:
        """Verify an empty control section still says explicitly that there is nothing to check."""
        assert pismogenerator_helper_functions.render_list([]) == "- Brak."
        assert pismogenerator_helper_functions.render_list([], empty_text="Nic.") == "- Nic."


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

    def test_render_letter__keeps_header_fields_and_sources_on_separate_lines_in_markdown(
        self,
    ) -> None:
        """Verify Markdown viewers do not merge the header fields or the sources into one line."""
        draft = structures.DraftLetter(
            recipient="Urząd",
            request_summary="dostęp do dokumentacji",
            argumentation_markdown=ARGUMENTATION,
            sources=["E1", "U1"],
        )

        result = pismogenerator_helper_functions.render_letter(
            draft, make_evidence_pack(), ["Zdjęcie"]
        )

        # Pusta linia rozdziela akapity; sam znak nowej linii Markdown traktuje jak spację.
        assert "**Adresat:** Urząd" + chr(10) * 2 + "**Nadawca:**" in result
        sources = result.split("## Źródła")[1].split("---")[0].strip().splitlines()
        assert sources == ["- [1] E1: Raport miejski, s. 4-5", "- [2] U1: Zdjęcie"]

    def test_render_letter__marks_a_missing_recipient_for_the_user_to_fill_in(self) -> None:
        """Verify an unknown addressee becomes a visible placeholder instead of a blank field."""
        draft = structures.DraftLetter(
            recipient="  ", request_summary="dostęp", argumentation_markdown=ARGUMENTATION
        )

        result = pismogenerator_helper_functions.render_letter(draft, make_evidence_pack(), [])

        assert "**Adresat:** [DO UZUPEŁNIENIA: adresat]" in result

    def test_render_letter__puts_verification_notes_from_the_model_in_square_brackets(
        self,
    ) -> None:
        """Verify a note the model wrote in parentheses reaches the letter as a [ ] marker."""
        draft = make_draft(
            argumentation_markdown=ARGUMENTATION + " Ruch jest duży (wymaga weryfikacji)."
        )

        result = pismogenerator_helper_functions.render_letter(draft, make_evidence_pack(), [])

        assert "Ruch jest duży [wymaga weryfikacji]." in result
        assert "### Informacje wymagające weryfikacji przed wysłaniem" in result


class TestBracketVerificationMarkers:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Ruch jest duży (wymaga weryfikacji).", "Ruch jest duży [wymaga weryfikacji]."),
            ("Ruch jest duży ( wymaga weryfikacji ).", "Ruch jest duży [wymaga weryfikacji]."),
            ("Ruch jest duży — wymaga weryfikacji.", "Ruch jest duży — [wymaga weryfikacji]."),
            ("Wymaga weryfikacji, czy ruch rośnie.", "[Wymaga weryfikacji], czy ruch rośnie."),
            ("Ruch jest duży [wymaga weryfikacji].", "Ruch jest duży [wymaga weryfikacji]."),
        ],
    )
    def test_bracket_verification_markers__brackets_every_unbracketed_note(
        self, text: str, expected: str
    ) -> None:
        """Verify each spelling of the note ends up as one bracketed marker, never doubled."""
        assert pismogenerator_helper_functions.bracket_verification_markers(text) == expected

    def test_bracket_verification_markers__leaves_notes_inside_other_markers(self) -> None:
        """Verify a placeholder that mentions verification is not split by a nested bracket."""
        text = "[DO UZUPEŁNIENIA: liczba, która wymaga weryfikacji]"

        assert pismogenerator_helper_functions.bracket_verification_markers(text) == text

    def test_bracket_verification_markers__leaves_a_negated_phrase_as_plain_text(self) -> None:
        """Verify "nie wymaga weryfikacji" is not turned into a marker that says the opposite."""
        text = "Ten fakt nie wymaga weryfikacji."

        assert pismogenerator_helper_functions.bracket_verification_markers(text) == text

    def test_bracket_verification_markers__ignores_other_forms_of_the_word(self) -> None:
        """Verify headings such as "wymagające weryfikacji" stay as they are."""
        text = "Informacje wymagające weryfikacji przed wysłaniem"

        assert pismogenerator_helper_functions.bracket_verification_markers(text) == text
