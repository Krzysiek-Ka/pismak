"""
Złoty zbiór Pismopoprawiatora: korekta językowa gotowego pisma na prawdziwym modelu.

Wejście przypadku to pismo złożone rendererem aplikacji, z wstawionymi błędami językowymi obok
elementów chronionych. Model poprawia każde pismo w dwóch trybach (MODES): całe pismo w jednym
wywołaniu i każdy akapit osobno. Klienta z limitem czasu etapu ustawia
`pismopoprawiator.make_llm`, okno kontekstu `detect_context_tokens`, jak w services.py.

Test nie przechodzi tylko przy naruszeniu treści w którymkolwiek trybie: różnicy wykrytej
przez content_violations albo zmianie liczby wystąpień napisu z `protected_strings`.
Rozwinięcie skrótu („ul.” → „ulicy”) nie jest naruszeniem. Naprawione błędy, poprawki
nadmiarowe i odrzucenia trafiają tylko do raportu. Obsługiwane pola `expected`
(SUPPORTED_EXPECTED): seeded_errors (błędny fragment i poprawne formy), protected_strings,
allowed_expansions (fragment, rozwinięcie i czy widziano je w prawdziwym przebiegu).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src._shared.model_context import detect_context_tokens
from src.Pismopoprawiator import pismopoprawiator
from src.Pismopoprawiator import pismopoprawiator_helper_functions as helpers
from src.Pismopoprawiator.constants import MAX_CALLS
from src.Pismopoprawiator.structures import Correction, CorrectionResult
from tests.golden_set.golden_support import (
    GoldenCase,
    Provider,
    assert_passed,
    golden_run,
    llm_client,
    load_cases,
    provider_skip_reason,
    runs,
    skip_run,
)

pytestmark = pytest.mark.golden

MODULE = "pismopoprawiator"
MODULE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXPECTED = {"seeded_errors", "protected_strings", "allowed_expansions"}
CASES = load_cases(MODULE_DIR, supported_expected=SUPPORTED_EXPECTED)
# Tryby porównywane na każdym piśmie (argumenty correct_letter).
MODES = {
    "calosc": {"mode": "whole"},
    "po_akapitach": {"mode": "paragraphs", "chunk_chars": 1},
}


# ---------------------------------------------------------------------------
# Ocena


def violations(case: GoldenCase, original: str, corrected: str) -> list[str]:
    """List every change of content: guard findings and changed protected strings of the case."""
    found = helpers.content_violations(original, corrected)
    for text in case.expected["protected_strings"]:
        before, after = original.count(text), corrected.count(text)
        if before != after:
            found.append(f"napis chroniony „{text}”: {before} → {after} wystąpień")
    return found


def fixed_errors(case: GoldenCase, original: str, corrected: str) -> tuple[list[str], list[str]]:
    """Split the seeded errors into the fixed ones and the ones still in the letter."""
    fixed, missed = [], []
    for error in case.expected["seeded_errors"]:
        wrong = error["wrong"]
        # Błąd jest naprawiony, gdy zniknął, a pojawiła się któraś z poprawnych form.
        removed = corrected.count(wrong) < original.count(wrong)
        if removed and any(corrected.count(f) > original.count(f) for f in error["right"]):
            fixed.append(wrong)
        else:
            missed.append(wrong)
    return fixed, missed


def extra_corrections(case: GoldenCase, result: CorrectionResult) -> list[str]:
    """Return applied corrections that touch none of the seeded errors, for manual review."""
    wrongs = [error["wrong"] for error in case.expected["seeded_errors"]]
    return [
        f"{item.akapit} „{item.oryginal}” → „{item.poprawka}” ({item.kategoria})"
        for item in result.applied
        if not any(w in item.oryginal or item.oryginal in w for w in wrongs)
    ]


def reason_kind(reason: str) -> str:
    """Group rejection reasons by their kind, without paragraph numbers and values."""
    return re.sub(r"A\d+", "A#", reason.split(":")[0]).strip()


# ---------------------------------------------------------------------------
# Kontrola zbioru (bez modeli)


def body_paragraphs(case: GoldenCase):
    body, _ = helpers.split_letter(case.input)
    return body, helpers.split_paragraphs(body)[1]


def perfect_correction(case: GoldenCase, form: int = 0) -> str:
    """Fix every seeded error with its first (form=0) or last (form=-1) accepted form."""
    letter = case.input
    for error in case.expected["seeded_errors"]:
        letter = letter.replace(error["wrong"], error["right"][form], 1)
    return letter


def expanded(case: GoldenCase) -> str:
    """Apply every allowed abbreviation expansion of the case."""
    letter = case.input
    for item in case.expected["allowed_expansions"]:
        letter = letter.replace(item["fragment"], item["expansion"], 1)
    return letter


class TestGoldenCases:
    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__seeded_errors_occur_once_in_the_corrected_part(self, case: GoldenCase) -> None:
        """Verify every error can be found and fixed, so a missed fix is the model's miss."""
        body, _ = body_paragraphs(case)

        for error in case.expected["seeded_errors"]:
            assert body.count(error["wrong"]) == 1, error["wrong"]
            for form in error["right"]:
                assert body.count(form) == 0, form

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__protected_strings_are_in_the_corrected_part(self, case: GoldenCase) -> None:
        """Verify each trap is sent to the model, so leaving it intact is a real result."""
        body, _ = body_paragraphs(case)

        for text in case.expected["protected_strings"]:
            assert text in body, text

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__every_seeded_fix_passes_the_guards(self, case: GoldenCase) -> None:
        """Verify the guards never block the fixes the golden set expects."""
        _, paragraphs = body_paragraphs(case)
        editable = {p.id for p in paragraphs if p.editable}

        for error in case.expected["seeded_errors"]:
            [paragraph] = [p for p in paragraphs if error["wrong"] in p.text]
            for form in error["right"]:
                item = Correction(
                    akapit=paragraph.id,
                    oryginal=error["wrong"],
                    poprawka=form,
                    kategoria="ortografia",
                )
                _, reason = helpers.check_correction(item, paragraph, editable)
                assert reason is None, (error["wrong"], form, reason)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__a_perfect_correction_has_no_violations_and_fixes_everything(
        self, case: GoldenCase
    ) -> None:
        """Verify the scoring gives full marks to a correction that fixes exactly the errors."""
        for form in (0, -1):
            corrected = perfect_correction(case, form)
            fixed, missed = fixed_errors(case, case.input, corrected)

            assert violations(case, case.input, corrected) == []
            assert missed == []
            assert len(fixed) == len(case.expected["seeded_errors"])

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__expanding_abbreviations_is_not_a_content_violation(
        self, case: GoldenCase
    ) -> None:
        """Verify "ul." → "ulicy" or "mln zł" → "miliona złotych" does not fail a golden run."""
        body, _ = body_paragraphs(case)

        for item in case.expected["allowed_expansions"]:
            assert body.count(item["fragment"]) == 1, item["fragment"]
        assert violations(case, case.input, expanded(case)) == []

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__expansions_seen_in_real_runs_pass_the_guards(self, case: GoldenCase) -> None:
        """Verify the expansions models really made stay allowed by the guards."""
        _, paragraphs = body_paragraphs(case)
        for item in case.expected["allowed_expansions"]:
            if not item["seen_in_real_run"]:
                continue
            [paragraph] = [p for p in paragraphs if item["fragment"] in p.text]
            correction = Correction(
                akapit=paragraph.id,
                oryginal=item["fragment"],
                poprawka=item["expansion"],
                kategoria="ortografia",
            )
            _, reason = helpers.check_correction(correction, paragraph, {paragraph.id})
            assert reason is None, (item["fragment"], reason)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__fit_the_paragraph_mode_without_skipping(self, case: GoldenCase) -> None:
        """Verify the paragraph-by-paragraph mode can reach every editable paragraph."""
        _, paragraphs = body_paragraphs(case)

        assert sum(p.editable for p in paragraphs) <= MAX_CALLS

    def test_cases__include_a_letter_without_errors(self) -> None:
        """Verify the set measures whether a model leaves a correct letter alone."""
        assert any(not case.expected["seeded_errors"] for case in CASES)


class TestEvaluation:
    @staticmethod
    def case(case_id: str) -> GoldenCase:
        return next(case for case in CASES if case.id == case_id)

    def test_violations__changed_protected_string_is_reported(self) -> None:
        """Verify a changed name counts as a content violation even if the guards missed it."""
        case = self.case("buspas")
        corrected = case.input.replace("ul. Kazimierza Wielkiego", "ul. Kazimierza Wielkiej", 1)

        found = violations(case, case.input, corrected)

        assert any("„Kazimierza Wielkiego”" in item for item in found)

    def test_fixed_errors__error_removed_without_its_correct_form_is_missed(self) -> None:
        """Verify removing an error without the right form is not counted as a fix."""
        case = self.case("buspas")
        corrected = case.input.replace("napewno", "pewnie", 1)

        fixed, missed = fixed_errors(case, case.input, corrected)

        assert fixed == []
        assert "napewno" in missed

    def test_fixed_errors__any_listed_correct_form_counts_as_a_fix(self) -> None:
        """Verify "2025 roku" counts as a fix of "2025r." just like "2025 r."."""
        case = self.case("liczby_i_daty")
        corrected = case.input.replace("2025r.", "2025 roku", 1)

        fixed, _ = fixed_errors(case, case.input, corrected)

        assert fixed == ["2025r."]

    def test_extra_corrections__fix_outside_the_seeded_errors_is_listed(self) -> None:
        """Verify unexpected changes are surfaced for manual review."""
        result = CorrectionResult(
            original_text="",
            corrected_text="",
            status="corrected",
            applied=[
                Correction(akapit="A5", oryginal="napewno", poprawka="na pewno", kategoria="x"),
                Correction(akapit="A6", oryginal="stoją w", poprawka="stoją na", kategoria="x"),
            ],
        )

        assert extra_corrections(self.case("buspas"), result) == ["A6 „stoją w” → „stoją na” (x)"]


# ---------------------------------------------------------------------------
# Przebieg na modelu


def correct(
    provider: Provider, case: GoldenCase, session_id: str, settings: dict
) -> CorrectionResult:
    """Correct the case letter the way services.run_pismopoprawiator does."""
    llm = pismopoprawiator.make_llm(
        llm_client(provider),
        provider.model,
        provider.provider,
        session_id=session_id,
        context_tokens=detect_context_tokens(
            provider.provider, provider.model, session_id=session_id
        ),
    )
    return pismopoprawiator.correct_letter(case.input, llm, session_id=session_id, **settings)


@pytest.mark.parametrize(("provider", "case"), runs(CASES))
def test_golden_pismopoprawiator(provider: Provider, case: GoldenCase) -> None:
    """Verify a real model corrects the letter's language without changing its content.

    Both modes run; the edge scenario is the case description (seeded errors next to protected
    numbers, names, quotes and markers).
    """
    reason = provider_skip_reason(provider)
    if reason:
        skip_run(MODULE, MODULE_DIR, provider, case, reason)

    with golden_run(MODULE, MODULE_DIR, provider, case) as run:
        models = []
        for mode, settings in MODES.items():
            result = correct(provider, case, run.session_id, settings)
            if result.model:
                models.append(result.model)
            found = violations(case, case.input, result.corrected_text)
            fixed, missed = fixed_errors(case, case.input, result.corrected_text)
            run.details[mode] = {
                "correction_status": result.status,
                "model": result.model,
                "violations": found,
                "fixed": fixed,
                "missed": missed,
                "extra": extra_corrections(case, result),
                "rejected": [
                    f"{item.correction.akapit} „{item.correction.oryginal}” → "
                    f"„{item.correction.poprawka}”: {item.reason}"
                    for item in result.rejected
                ],
                "rejection_kinds": sorted({reason_kind(item.reason) for item in result.rejected}),
                "suggestions": len(result.suggestions),
                "calls": result.calls,
                "elapsed_seconds": result.elapsed_seconds,
                "errors": result.errors,
                "corrected_text": result.corrected_text,
            }
            run.check(f"no_content_violations:{mode}", not found, " | ".join(found))
            if result.status == "failed":
                run.errors.append(f"{mode}: korekta nie powiodła się ({'; '.join(result.errors)})")
        run.model = ", ".join(dict.fromkeys(models)) or None

    assert_passed(run)
