"""Tests for the deterministic part of Pismopoprawiator: splitting, guards and applying."""

from __future__ import annotations

import pytest

from src.Pismogenerator.pismogenerator_helper_functions import (
    PHASE0_OPENING_FORMULA,
    render_letter,
)
from src.Pismogenerator.structures import DraftLetter
from src.Pismopoprawiator import pismopoprawiator_helper_functions as helpers
from src.Pismopoprawiator.structures import Correction, Paragraph
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack, ResearchPlan

ARGUMENTATION = (
    "Mieszkańcy ul. Kazimierza Wielkiego zgłaszają korki, ktore spowalniają autobusy.\n\n"
    "Według obserwacji wnioskodawcy przejazd trwa  ponad 20 minut [wymaga weryfikacji] [1]."
)

TEXT = (
    "Mieszkańcy ul. Kazimierza Wielkiego od 12 maja 2026 r. zgłaszają korki, ktore "
    "spowalniają autobusy linii 145. Według obserwacji wnioskodawcy przejazd trwa  ponad "
    "20 minut [wymaga weryfikacji]. Strategia „Wrocław 2050” wskazuje transport publiczny "
    "jako priorytet [1]. Nie postuluję zamknięcia ulicy, ale miasto powinno rozważyć buspas "
    "zgodnie z uchwałą nr XLVIII/1169/13 oraz art. 241 [2]. Dane są na stronie "
    "https://www.wroclaw.pl/mobilnosc. Wogóle nie ma tam pomiarów."
)


def make_letter(argumentation: str = ARGUMENTATION) -> str:
    draft = DraftLetter(
        recipient="Zarząd Dróg i Utrzymania Miasta",
        request_summary="wprowadzenie buspasa na ulicy Kazimierza Wielkiego",
        argumentation_markdown=argumentation,
        sources=["U1"],
        claims_requiring_verification=["Czas przejazdu autobusów."],
    )
    pack = EvidencePack(
        query_summary="Buspas",
        research_plan=ResearchPlan(case_summary="Buspas", intents=[]),
    )
    return render_letter(draft, pack, ["Obserwacje wnioskodawcy"])


def paragraph(text: str = TEXT, *, id: str = "A5", editable: bool = True) -> Paragraph:
    return Paragraph(id=id, text=text, editable=editable, part_index=0)


def correction(
    oryginal: str, poprawka: str, kategoria: str = "ortografia", akapit: str = "A5"
) -> Correction:
    return Correction(akapit=akapit, oryginal=oryginal, poprawka=poprawka, kategoria=kategoria)


def check(item: Correction, target: Paragraph | None = None) -> str | None:
    target = target or paragraph()
    _, reason = helpers.check_correction(item, target, {target.id})
    return reason


class TestSplitLetter:
    def test_split_letter__keeps_sources_and_control_section_out_of_the_corrected_part(
        self,
    ) -> None:
        """Verify the model never sees the source list or the control section."""
        letter = make_letter()

        body, tail = helpers.split_letter(letter)

        assert body + tail == letter
        assert body.endswith("[imię i nazwisko]")
        assert tail.lstrip().startswith("---")
        assert "## Źródła" not in body
        assert "## Sekcja kontrolna" in tail

    def test_split_letter__cuts_at_the_renderer_heading_when_the_model_wrote_its_own(
        self,
    ) -> None:
        """Verify a "## Źródła" heading inside the argumentation stays in the corrected part."""
        letter = make_letter(ARGUMENTATION + "\n\n---\n\n## Źródła\n\nWłasna lista modelu.")

        body, tail = helpers.split_letter(letter)

        assert "Własna lista modelu." in body
        assert "Obserwacje wnioskodawcy" in tail

    def test_split_letter__treats_text_without_sources_as_the_whole_body(self) -> None:
        """Verify a letter that is not from the renderer is still corrected as a whole."""
        assert helpers.split_letter("Jedno zdanie.") == ("Jedno zdanie.", "")


class TestSplitParagraphs:
    def test_split_paragraphs__marks_renderer_blocks_as_read_only(self) -> None:
        """Verify the addressee, sender, type, closing and signature are only context."""
        body, _ = helpers.split_letter(make_letter())

        _, paragraphs = helpers.split_paragraphs(body)

        assert [(p.id, p.editable) for p in paragraphs] == [
            ("A1", False),
            ("A2", False),
            ("A3", True),
            ("A4", False),
            ("A5", True),
            ("A6", True),
            ("A7", False),
            ("A8", False),
        ]
        assert paragraphs[0].text.startswith("**Adresat:**")
        assert paragraphs[2].text.startswith(PHASE0_OPENING_FORMULA)

    def test_split_paragraphs__rebuilds_the_text_byte_for_byte(self) -> None:
        """Verify unusual separators survive, so untouched parts of the letter never change."""
        body = "Pierwszy.\n  \n\nDrugi\\\nz łamaniem.\n\n\n- lista\n- dalej"

        parts, paragraphs = helpers.split_paragraphs(body)

        assert helpers.join_paragraphs(parts, paragraphs) == body
        assert [p.text for p in paragraphs] == [
            "Pierwszy.",
            "Drugi\\\nz łamaniem.",
            "- lista\n- dalej",
        ]


class TestStartsSentence:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Wrocław", True),
            ("Korki. Wrocław", True),
            ("1. **Wrocław", True),
            ("Pismo:\n- Wrocław", True),
            ("ulica we Wrocław", False),
            ("ul. Wrocław", False),
            ("J. Wrocław", False),
            ("priorytet [1] Wrocław", False),
        ],
    )
    def test_starts_sentence__tells_sentence_starts_from_names(
        self, text: str, expected: bool
    ) -> None:
        """Verify an abbreviation dot does not hide a proper name from the names guard."""
        assert helpers.starts_sentence(text, text.rindex("Wrocław")) is expected


class TestCheckCorrection:
    @pytest.mark.parametrize(
        ("oryginal", "poprawka", "kategoria"),
        [
            ("korki, ktore", "korki, które", "ortografia"),
            ("trwa  ponad", "trwa ponad", "spacje"),
            ("Wogóle nie", "W ogóle nie", "ortografia"),
            # Rozwinięcie skrótu nie zmienia treści.
            ("ul. Kazimierza Wielkiego", "ulicy Kazimierza Wielkiego", "ortografia"),
            ("2026 r. zgłaszają", "2026 roku zgłaszają", "ortografia"),
        ],
    )
    def test_check_correction__accepts_language_fixes(
        self, oryginal: str, poprawka: str, kategoria: str
    ) -> None:
        """Verify spelling, spacing, word-splitting fixes and expanded abbreviations pass."""
        start, reason = helpers.check_correction(
            correction(oryginal, poprawka, kategoria), paragraph(), {"A5"}
        )

        assert reason is None
        assert start == TEXT.index(oryginal)

    @pytest.mark.parametrize(
        ("oryginal", "poprawka", "expected"),
        [
            ("od 12 maja", "od 13 maja", "zmienia liczby i daty"),
            ("2026 r.", "2025 r.", "zmienia liczby i daty"),
            ("XLVIII/1169/13", "XLVII/1169/13", "zmienia numery uchwał"),
            ("art. 241", "ust. 241", "zmienia oznaczenia przepisów"),
            ("[wymaga weryfikacji]", "[wymaga sprawdzenia]", "nawiasach kwadratowych"),
            ("https://www.wroclaw.pl/mobilnosc", "https://wroclaw.pl/mobilnosc", "adresy URL"),
            ("Strategia „Wrocław 2050”", "Strategia „Wrocławska 2050”", "zmienia cytaty"),
            ("Według obserwacji wnioskodawcy", "Według obserwacjii wnioskodawcy", "frazy"),
            ("ul. Kazimierza Wielkiego", "ul. Kazimierza Wielkiej", "nazwy własne"),
            ("Nie postuluję", "Postuluję", "zmienia przeczenia: 2 → 1"),
            ("miasto powinno rozważyć", "miasto rozważy", "zmienia modalność"),
            ("korki, ktore", "korki,\nktore", "zmienia formatowanie"),
        ],
    )
    def test_check_correction__rejects_a_change_of_protected_content(
        self, oryginal: str, poprawka: str, expected: str
    ) -> None:
        """Verify each guarded element blocks a correction that changes it, with a clear reason."""
        reason = check(correction(oryginal, poprawka))

        assert reason is not None
        assert expected in reason

    @pytest.mark.parametrize(
        ("oryginal", "poprawka"),
        [
            ("„Wrocław", "„Wrocławia"),
            ("weryfikacji].", "weryfikacji]"),
            ("https://www.wroclaw.pl", "http://www.wroclaw.pl"),
        ],
    )
    def test_check_correction__rejects_a_fragment_that_cuts_through_a_protected_element(
        self, oryginal: str, poprawka: str
    ) -> None:
        """Verify a quote, marker or address can only be touched as a whole and unchanged."""
        reason = check(correction(oryginal, poprawka))

        assert reason is not None
        assert "przecina chroniony element" in reason

    def test_check_correction__allows_a_fragment_that_contains_a_protected_element_whole(
        self,
    ) -> None:
        """Verify spacing next to a footnote can be fixed as long as the footnote is unchanged."""
        target = paragraph("Autobusy stoją w korkach [1] .")

        start, reason = helpers.check_correction(
            correction("korkach [1] .", "korkach [1].", "spacje"), target, {"A5"}
        )

        assert reason is None
        assert start == target.text.index("korkach")

    def test_check_correction__protects_the_opening_formula(self) -> None:
        """Verify "wnioskuję o" and the fixed opening stay intact while the summary is fixed."""
        target = paragraph(f"{PHASE0_OPENING_FORMULA} analize wariantów buspasa", id="A3")

        cut, _ = helpers.check_correction(
            correction("wnioskuję o analize", "wnioskuję o analizę", "fleksja", "A3"),
            target,
            {"A3"},
        )
        _, reason = helpers.check_correction(
            correction("o analize", "o analizę", "fleksja", "A3"), target, {"A3"}
        )
        start, accepted = helpers.check_correction(
            correction("analize wariantów", "analizę wariantów", "fleksja", "A3"),
            target,
            {"A3"},
        )

        assert cut is None
        assert "przecina chroniony element" in reason
        assert accepted is None and start == target.text.index("analize")

    @pytest.mark.parametrize(
        ("item", "expected"),
        [
            (correction("tramwaje", "tramwaj"), "fragmentu nie ma w akapicie A5"),
            (correction("ul", "ulica"), "występuje w akapicie A5 3 razy"),
            (correction("korki", "korki"), "nie różni się od oryginału"),
            (correction("korki", "zatory"), "zbyt odległa od oryginału"),
            (correction("", "korki"), "pusty fragment"),
            (correction(TEXT[:201], TEXT[:200]), "201 znaków (limit 200)"),
            (correction("korki", "korek", "składnia"), "nieznana kategoria „składnia”"),
        ],
    )
    def test_check_correction__rejects_a_correction_that_cannot_be_placed_safely(
        self, item: Correction, expected: str
    ) -> None:
        """Verify missing, ambiguous, empty, too long, too different or unknown fixes are dropped."""
        reason = check(item)

        assert reason is not None
        assert expected in reason

    def test_check_correction__rejects_read_only_unknown_and_context_paragraphs(self) -> None:
        """Verify the code, not the model, decides which paragraphs may change."""
        item = correction("korki, ktore", "korki, które")

        _, read_only = helpers.check_correction(item, paragraph(editable=False), {"A5"})
        _, unknown = helpers.check_correction(item, None, {"A5"})
        _, context = helpers.check_correction(item, paragraph(), {"A6"})

        assert read_only == "akapit A5 jest tylko do odczytu"
        assert unknown == "nie ma akapitu A5"
        assert context == "akapit A5 nie był edytowalny w tym wywołaniu"


class TestReviewCorrections:
    def test_review_corrections__keeps_style_remarks_as_suggestions_only(self) -> None:
        """Verify a style remark is never applied, even when it would pass the guards."""
        paragraphs = [paragraph(id="A1")]
        style = correction("korki, ktore", "zatory, które", "styl", "A1")
        fix = correction("korki, ktore", "korki, które", "ortografia", "A1")

        accepted, rejected, suggestions = helpers.review_corrections(
            paragraphs, [style, fix], {"A1"}
        )

        assert [item.correction for item in accepted] == [fix]
        assert rejected == []
        assert suggestions == [style]


class TestApplyCorrections:
    def test_apply_corrections__applies_from_the_end_so_earlier_positions_stay_valid(
        self,
    ) -> None:
        """Verify fixes that change length do not shift the place of the next fix."""
        paragraphs = [
            paragraph("Korki ktore trwają  długo i autobusy ktore stoją.", id="A1"),
            paragraph("Wogóle brak danych.", id="A2"),
        ]
        corrections = [
            correction("autobusy ktore", "autobusy, które", "interpunkcja", "A1"),
            correction("Wogóle", "W ogóle", akapit="A2"),
            correction("Korki ktore", "Korki, które", "interpunkcja", "A1"),
            correction("trwają  długo", "trwają długo", "spacje", "A1"),
        ]
        accepted, rejected, _ = helpers.review_corrections(paragraphs, corrections, {"A1", "A2"})

        result, applied, overlapping = helpers.apply_corrections(paragraphs, accepted)

        assert rejected == overlapping == []
        assert [p.text for p in result] == [
            "Korki, które trwają długo i autobusy, które stoją.",
            "W ogóle brak danych.",
        ]
        # Kolejność w wyniku odpowiada kolejności w tekście.
        assert [c.oryginal for c in applied] == [
            "Korki ktore",
            "trwają  długo",
            "autobusy ktore",
            "Wogóle",
        ]

    def test_apply_corrections__rejects_overlapping_and_repeated_corrections(self) -> None:
        """Verify two fixes of the same words do not both apply and garble the text."""
        paragraphs = [paragraph("Korki ktore trwają długo.", id="A1")]
        first = correction("Korki ktore", "Korki, które", "interpunkcja", "A1")
        overlapping = correction("ktore trwają", "które trwają", "ortografia", "A1")
        accepted, _, _ = helpers.review_corrections(paragraphs, [first, overlapping, first], {"A1"})

        result, applied, rejected = helpers.apply_corrections(paragraphs, accepted)

        assert result[0].text == "Korki, które trwają długo."
        assert applied == [first]
        assert sorted(item.reason for item in rejected) == [
            "nakłada się na inną poprawkę w akapicie A1",
            "powtórzona poprawka",
        ]


class TestContentViolations:
    def test_content_violations__finds_nothing_in_an_unchanged_letter(self) -> None:
        """Verify the whole-letter guard does not block a letter without corrections."""
        letter = make_letter()

        assert helpers.content_violations(letter, letter) == []

    def test_content_violations__accepts_a_letter_with_language_fixes(self) -> None:
        """Verify ordinary fixes keep the letter within every whole-letter limit."""
        letter = make_letter()
        corrected = letter.replace("ktore", "które").replace("trwa  ponad", "trwa ponad")

        assert helpers.content_violations(letter, corrected) == []

    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            (("Obserwacje wnioskodawcy", "Pomiary wnioskodawcy"), "listę źródeł"),
            (("Zarząd Dróg", "Zarządu Dróg"), "akapit tylko do odczytu A1"),
            (("autobusy.\n\nWedług", "autobusy. Według"), "liczba akapitów"),
            (("20 minut", "20 minut" + " i jeszcze dłużej" * 6), "długość treści"),
            (("20 minut", "25 minut"), "zmienia liczby i daty"),
        ],
    )
    def test_content_violations__reports_every_change_beyond_language(
        self, change: tuple[str, str], expected: str
    ) -> None:
        """Verify the guard catches changes to the tail, fixed blocks, structure, length and data."""
        letter = make_letter()
        corrected = letter.replace(*change, 1)

        violations = helpers.content_violations(letter, corrected)

        assert any(expected in violation for violation in violations), violations
