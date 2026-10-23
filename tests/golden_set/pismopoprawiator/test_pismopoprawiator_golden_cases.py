"""
Kontrola złotego zbioru Pismopoprawiatora bez modeli i sieci (ręcznie, `-m golden_set`).

Sprawdza, że każdy wstawiony błąd i każdy napis chroniony naprawdę jest w piśmie, że
idealna korekta i dozwolone rozwinięcia skrótów przechodzą strażników i nie dają naruszeń,
oraz że ocena wykrywa zmianę treści.
"""

from __future__ import annotations

import pytest
from pismopoprawiator_cases import (
    CASES,
    MODELE,
    TRYBY,
    GoldenCase,
    correct_forms,
    selected_models,
)
from pismopoprawiator_support import (
    extra_corrections,
    fixed_errors,
    resolve_model,
    violations,
)

from src.Pismopoprawiator import pismopoprawiator_helper_functions as helpers
from src.Pismopoprawiator.constants import MAX_CALLS
from src.Pismopoprawiator.structures import Correction, CorrectionResult


def body_paragraphs(case: GoldenCase):
    body, _ = helpers.split_letter(case.letter())
    return body, helpers.split_paragraphs(body)[1]


def perfect_correction(case: GoldenCase, form: int = 0) -> str:
    """Fix every seeded error with its first (form=0) or last (form=-1) accepted form."""
    letter = case.letter()
    for wrong, right in case.bledy:
        letter = letter.replace(wrong, correct_forms(right)[form], 1)
    return letter


def expanded(case: GoldenCase) -> str:
    """Apply every allowed abbreviation expansion of the case."""
    letter = case.letter()
    for fragment, expansion in case.rozwiniecia:
        letter = letter.replace(fragment, expansion, 1)
    return letter


class TestGoldenCases:
    def test_cases__have_unique_ids_and_known_modes(self) -> None:
        """Verify each run of the golden set has its own identifier in the report."""
        assert len({case.id for case in CASES}) == len(CASES)
        assert set(TRYBY) == {"calosc", "po_akapitach"}

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__seeded_errors_occur_once_in_the_corrected_part(self, case: GoldenCase) -> None:
        """Verify every error can be found and fixed, so a missed fix is the model's miss."""
        body, _ = body_paragraphs(case)

        for wrong, right in case.bledy:
            assert body.count(wrong) == 1, wrong
            for form in correct_forms(right):
                assert body.count(form) == 0, form

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__protected_strings_are_in_the_corrected_part(self, case: GoldenCase) -> None:
        """Verify each trap is sent to the model, so leaving it intact is a real result."""
        body, _ = body_paragraphs(case)

        for text in case.chronione:
            assert text in body, text

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__every_seeded_fix_passes_the_guards(self, case: GoldenCase) -> None:
        """Verify the guards never block the fixes the golden set expects."""
        _, paragraphs = body_paragraphs(case)
        editable = {p.id for p in paragraphs if p.editable}

        for wrong, right in case.bledy:
            [paragraph] = [p for p in paragraphs if wrong in p.text]
            for form in correct_forms(right):
                item = Correction(
                    akapit=paragraph.id, oryginal=wrong, poprawka=form, kategoria="ortografia"
                )
                _, reason = helpers.check_correction(item, paragraph, editable)
                assert reason is None, (wrong, form, reason)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__a_perfect_correction_has_no_violations_and_fixes_everything(
        self, case: GoldenCase
    ) -> None:
        """Verify the scoring gives full marks to a correction that fixes exactly the errors."""
        letter = case.letter()

        for form in (0, -1):
            corrected = perfect_correction(case, form)
            fixed, missed = fixed_errors(case, letter, corrected)

            assert violations(case, letter, corrected) == []
            assert missed == []
            assert len(fixed) == len(case.bledy)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__expanding_abbreviations_is_not_a_content_violation(
        self, case: GoldenCase
    ) -> None:
        """Verify "ul." → "ulicy" or "mln zł" → "miliona złotych" does not fail a golden run."""
        letter = case.letter()
        body, _ = body_paragraphs(case)

        for fragment, _ in case.rozwiniecia:
            assert body.count(fragment) == 1, fragment
        assert violations(case, letter, expanded(case)) == []

    @pytest.mark.parametrize(
        ("case_id", "fragment", "expansion"),
        [
            (
                "czyste_pismo",
                "ul. Legnickiej i ul. Na Ostatnim Groszu",
                "ulic Legnickiej i Na Ostatnim Groszu",
            ),
            ("liczby_i_daty", "1,5 mln zł", "1,5 miliona złotych"),
            ("twarde_lamania", "śmieci na pl. Grunwaldzkim", "śmieci na placu Grunwaldzkim"),
        ],
    )
    def test_cases__the_guards_accept_the_expansions_seen_in_real_runs(
        self, case_id: str, fragment: str, expansion: str
    ) -> None:
        """Verify the expansions Bielik made on 2026-10-23 stay allowed and listed in the set."""
        case = next(case for case in CASES if case.id == case_id)
        _, paragraphs = body_paragraphs(case)
        [paragraph] = [p for p in paragraphs if fragment in p.text]
        item = Correction(
            akapit=paragraph.id, oryginal=fragment, poprawka=expansion, kategoria="ortografia"
        )

        _, reason = helpers.check_correction(item, paragraph, {paragraph.id})

        assert reason is None
        assert (fragment, expansion) in case.rozwiniecia

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__fit_the_paragraph_mode_without_skipping(self, case: GoldenCase) -> None:
        """Verify the paragraph-by-paragraph mode can reach every editable paragraph."""
        _, paragraphs = body_paragraphs(case)

        assert sum(p.editable for p in paragraphs) <= MAX_CALLS

    def test_cases__include_a_letter_without_errors(self) -> None:
        """Verify the set measures whether a model leaves a correct letter alone."""
        assert any(not case.bledy for case in CASES)


class TestEvaluation:
    def test_violations__report_a_changed_protected_string(self) -> None:
        """Verify a changed name counts as a content violation even if the guards missed it."""
        case = CASES[0]
        letter = case.letter()
        corrected = letter.replace("ul. Kazimierza Wielkiego", "ul. Kazimierza Wielkiej", 1)

        found = violations(case, letter, corrected)

        assert any("„Kazimierza Wielkiego”" in item for item in found)

    def test_fixed_errors__counts_only_errors_replaced_by_their_correct_form(self) -> None:
        """Verify removing an error without the right form is not counted as a fix."""
        case = CASES[0]
        letter = case.letter()
        corrected = letter.replace("napewno", "pewnie", 1)

        fixed, missed = fixed_errors(case, letter, corrected)

        assert fixed == []
        assert "napewno" in missed

    def test_fixed_errors__accepts_any_listed_correct_form(self) -> None:
        """Verify "2025 roku" counts as a fix of "2025r." just like "2025 r."."""
        case = next(case for case in CASES if case.id == "liczby_i_daty")
        letter = case.letter()
        corrected = letter.replace("2025r.", "2025 roku", 1)

        fixed, _ = fixed_errors(case, letter, corrected)

        assert fixed == ["2025r."]

    def test_extra_corrections__lists_fixes_outside_the_seeded_errors(self) -> None:
        """Verify unexpected changes are surfaced for manual review."""
        case = CASES[0]
        result = CorrectionResult(
            original_text="",
            corrected_text="",
            status="corrected",
            applied=[
                Correction(akapit="A5", oryginal="napewno", poprawka="na pewno", kategoria="x"),
                Correction(akapit="A6", oryginal="stoją w", poprawka="stoją na", kategoria="x"),
            ],
        )

        assert extra_corrections(case, result) == ["A6 „stoją w” → „stoją na” (x)"]


class TestModels:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("bielik", ("ollama", "pismak-bielik:latest")),
            ("openrouter", ("openrouter", "openrouter/free")),
            ("openrouter:meta-llama/llama-3.3-70b-instruct:free", ("openrouter", None)),
            ("qwen2.5", ("ollama", "qwen2.5:latest")),
        ],
    )
    def test_resolve_model__understands_the_e2e_shortcuts(
        self, value: str, expected: tuple[str, str | None]
    ) -> None:
        """Verify the model list uses the same notation as the E2E tests."""
        model = resolve_model(value)

        assert model.provider == expected[0]
        assert model.model == (expected[1] or "meta-llama/llama-3.3-70b-instruct:free")

    def test_selected_models__can_be_narrowed_from_the_environment(self, monkeypatch) -> None:
        """Verify a run can target chosen models without editing the file."""
        monkeypatch.setenv("PISMAK_GOLDEN_MODELE", " bielik , openrouter ")
        assert selected_models() == ["bielik", "openrouter"]
        monkeypatch.delenv("PISMAK_GOLDEN_MODELE")
        assert selected_models() == MODELE
