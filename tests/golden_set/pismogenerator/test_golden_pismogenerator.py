"""
Złoty zbiór Pismogeneratora: projekt pisma na prawdziwym modelu z CaseBrief i EvidencePack.

Wejście przypadku to CaseBrief i EvidencePack z prawdziwymi fragmentami dokumentów (skopiowanymi
z bazy RAG, żeby przypadek nie zależał od ponownego wczytania dokumentów). Ścieżka jest ta sama
co w src/Wyswietlator/app.py: LLMClient z oknem kontekstu modelu, `build_draft` z temperaturą
0.2 i `render_letter` z materiałami użytkownika z briefu.

Zawsze sprawdzane: projekt przechodzi walidację aplikacji (DraftLetter, validate_draft), pismo ma
wszystkie sekcje, źródła należą do dozwolonych identyfikatorów, każdy przypis [n] wskazuje
pozycję listy źródeł, przypisy mają formę [n] (nie [E1]), a w treści nie ma dosłownych
znaków odwrotnego ukośnika. Obsługiwane pola `expected` (SUPPORTED_EXPECTED):
- must_mention: napis albo lista wariantów obecna we wniosku lub argumentacji,
- recipient_must_mention: napis w polu adresata,
- expect_recipient_placeholder: pismo ma znacznik [DO UZUPEŁNIENIA: adresat],
- expect_no_sources: brak źródeł i przypisów,
- allowed_sources_subset: źródła tylko z tej listy,
- min_sources: najmniejsza liczba źródeł,
- no_new_numbers: każda liczba z wniosku i argumentacji jest w briefie albo w źródłach,
- no_legal_basis: brak przepisów i formuł prawnych, których nie ma w danych wejściowych,
- quotes_from_inputs: każdy cytat w cudzysłowie pochodzi z briefu albo ze źródeł.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator
from src.Pismogenerator.pismogenerator import DraftValidationError
from src.Pismogenerator.pismogenerator_helper_functions import (
    CONTROL_SECTION_HEADING,
    PHASE0_CLOSING,
    PHASE0_DOCUMENT_TYPE,
    PHASE0_OPENING_FORMULA,
    PHASE0_SIGNATURE,
    RECIPIENT_LABEL,
    RECIPIENT_PLACEHOLDER,
    SENDER_LABEL,
    SOURCES_HEADING,
    normalize_user_evidence,
)
from src.Wyswietlator import services
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack
from tests.golden_set.golden_support import (
    GoldenCase,
    Provider,
    as_text,
    assert_passed,
    golden_run,
    load_cases,
    make_llm,
    mentions,
    normalize,
    numbers_in,
    provider_skip_reason,
    runs,
    skip_run,
)

pytestmark = pytest.mark.golden

MODULE = "pismogenerator"
MODULE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXPECTED = {
    "must_mention",
    "recipient_must_mention",
    "expect_recipient_placeholder",
    "expect_no_sources",
    "allowed_sources_subset",
    "min_sources",
    "no_new_numbers",
    "no_legal_basis",
    "quotes_from_inputs",
}
CASES = load_cases(MODULE_DIR, supported_expected=SUPPORTED_EXPECTED)
LETTER_SECTIONS = (
    RECIPIENT_LABEL,
    SENDER_LABEL,
    PHASE0_OPENING_FORMULA,
    f"# {PHASE0_DOCUMENT_TYPE}",
    PHASE0_CLOSING,
    PHASE0_SIGNATURE,
    SOURCES_HEADING,
    CONTROL_SECTION_HEADING,
    "### Informacje wymagające weryfikacji przed wysłaniem",
    "### Brakujące pola",
    "### Założenia redakcyjne modelu",
)
LEGAL_BASIS = (
    r"\bart\.\s*\d",
    r"§\s*\d",
    r"\bDz\.\s*U\b",
    r"\bKPA\b",
    r"\bkodeks\w*",
    r"\bustaw\w*\s+z\s+dnia",
    r"działając w trybie",
    r"\bna podstawie\s+(art|przepis|ustaw|rozporz)",
)
FOOTNOTE = re.compile(r"\[(\d+)\]")
# Identyfikatory źródeł (E1, U2) i numeracja list („1. ”, także w środku wiersza po „\n”)
# nie są liczbami w treści pisma.
SOURCE_ID = re.compile(r"\b[EU]\d+\b")
LIST_MARKER = re.compile(r"(?:^|(?<=\s)|(?<=\\n))[1-9][.)](?=\s)", re.MULTILINE)
BRACKETED_SOURCE_ID = re.compile(r"\[[EU]\d+\]")
QUOTE = re.compile(r"„([^”]{12,})”|\"([^\"]{12,})\"")


def case_inputs(case: GoldenCase) -> tuple[CaseBrief, EvidencePack]:
    return (
        CaseBrief.model_validate(case.input["case_brief"]),
        EvidencePack.model_validate(case.input["evidence_pack"]),
    )


class TestGoldenCases:
    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__inputs_match_the_app_schemas(self, case: GoldenCase) -> None:
        """Verify every case can be fed to build_draft exactly as the app would feed it."""
        brief, pack = case_inputs(case)

        assert [e.evidence_id for e in pack.evidence] == [
            f"E{i}" for i in range(1, len(pack.evidence) + 1)
        ]
        assert brief.requested_actions

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__allowed_sources_subset_names_existing_ids(self, case: GoldenCase) -> None:
        """Verify a subset check cannot pass by naming an identifier the case does not offer."""
        brief, pack = case_inputs(case)
        offered = {e.evidence_id for e in pack.evidence} | {
            f"U{i}" for i in range(1, len(normalize_user_evidence(brief.available_evidence)) + 1)
        }

        assert set(case.expected.get("allowed_sources_subset", [])) <= offered


def input_text(brief: CaseBrief, pack: EvidencePack) -> str:
    """Everything the model was allowed to use, as one searchable text."""
    parts = [as_text(brief)]
    for e in pack.evidence:
        parts += [e.document_title, e.excerpt, str(e.page_start), str(e.page_end)]
    return "\n".join(parts)


def body_text(draft) -> str:
    """The model-written request summary and argumentation."""
    return f"{draft.request_summary}\n{draft.argumentation_markdown}"


def letter_body(draft) -> str:
    """The model-written part of the letter, without footnotes and list numbering."""
    text = body_text(draft)
    return LIST_MARKER.sub(" ", SOURCE_ID.sub(" ", FOOTNOTE.sub(" ", text)))


def check_draft(run, case: GoldenCase, brief: CaseBrief, pack: EvidencePack, draft, letter: str):
    """Record the contract checks and the case's expected fields for a valid draft."""
    expected = case.expected
    missing = [s for s in LETTER_SECTIONS if s not in letter]
    run.check("letter_sections", not missing, f"brak: {missing}" if missing else "")

    user_ids = [
        f"U{i}" for i in range(1, len(normalize_user_evidence(brief.available_evidence)) + 1)
    ]
    allowed = {e.evidence_id for e in pack.evidence} | set(user_ids)
    run.check(
        "sources_allowed", set(draft.sources) <= allowed, f"{draft.sources} ⊄ {sorted(allowed)}"
    )
    refs = [
        int(n) for n in FOOTNOTE.findall(f"{draft.request_summary}\n{draft.argumentation_markdown}")
    ]
    bad = [n for n in refs if not 1 <= n <= len(draft.sources)]
    run.check("footnotes_resolvable", not bad, f"przypisy {bad} przy {len(draft.sources)} źródłach")
    # Prompt każe oznaczać źródła numerami [1], [2]; renderer numeruje tylko listę sources.
    raw_ids = sorted(set(BRACKETED_SOURCE_ID.findall(body_text(draft))))
    run.check("citations_as_footnote_numbers", not raw_ids, f"przypisy w formie {raw_ids}")
    run.check(
        "no_stray_backslashes",
        "\\" not in body_text(draft),
        "dosłowne znaki \\ (np. podwójnie escapowane nowe linie) w treści pisma",
    )

    body = f"{draft.request_summary}\n{draft.argumentation_markdown}"
    for item in expected.get("must_mention", []):
        variants = [item] if isinstance(item, str) else item
        found = any(mentions(body, v) for v in variants)
        run.check(f"must_mention:{variants[0]}", found, "" if found else "brak w piśmie")
    if "recipient_must_mention" in expected:
        term = expected["recipient_must_mention"]
        run.check("recipient", mentions(draft.recipient, term), repr(draft.recipient))
    if expected.get("expect_recipient_placeholder"):
        run.check("recipient_placeholder", RECIPIENT_PLACEHOLDER in letter, repr(draft.recipient))
    if expected.get("expect_no_sources"):
        run.check("no_sources", not draft.sources and not refs, f"{draft.sources}, przypisy {refs}")
    if "allowed_sources_subset" in expected:
        subset = set(expected["allowed_sources_subset"])
        run.check("allowed_sources_subset", set(draft.sources) <= subset, str(draft.sources))
    if "min_sources" in expected:
        run.check("min_sources", len(draft.sources) >= expected["min_sources"], str(draft.sources))

    inputs = input_text(brief, pack)
    if expected.get("no_new_numbers"):
        new = sorted(numbers_in(letter_body(draft)) - numbers_in(inputs))
        run.check("no_new_numbers", not new, f"liczby spoza danych: {new}" if new else "")
    if expected.get("no_legal_basis"):
        found = [
            p for p in LEGAL_BASIS if re.search(p, body, re.I) and not re.search(p, inputs, re.I)
        ]
        run.check("no_legal_basis", not found, f"wzorce: {found}" if found else "")
    if expected.get("quotes_from_inputs"):
        quotes = [a or b for a, b in QUOTE.findall(draft.argumentation_markdown)]
        invented = [q for q in quotes if normalize(q) not in normalize(inputs)]
        run.check("quotes_from_inputs", not invented, f"cytaty spoza danych: {invented}")


@pytest.mark.parametrize(("provider", "case"), runs(CASES))
def test_golden_pismogenerator(provider: Provider, case: GoldenCase) -> None:
    """Verify a real model writes a complete, valid letter using only the brief and the sources.

    The case file describes the edge scenario; the checks are listed in the module docstring.
    """
    reason = provider_skip_reason(provider)
    if reason:
        skip_run(MODULE, MODULE_DIR, provider, case, reason)
    brief, pack = case_inputs(case)

    with golden_run(MODULE, MODULE_DIR, provider, case) as run:
        llm = make_llm(
            provider,
            stage="pismogenerator",
            session_id=run.session_id,
            timeout_seconds=case.limits.get("timeout_seconds"),
        )
        try:
            draft = pismogenerator.build_draft(
                case_brief=brief, evidence_pack=pack, llm=llm, llm_temperature=0.2
            )
        except DraftValidationError as exc:
            draft = None
            run.details["draft_with_error"] = (
                exc.draft.model_dump(mode="json") if exc.draft else None
            )
            run.check("draft_valid", False, "; ".join(exc.validation_errors))
        run.model = llm.last_model
        if draft is not None:
            run.check("draft_valid", True)
            letter = services.render_letter(draft, pack, brief.available_evidence)
            run.details.update(draft=draft.model_dump(mode="json"), letter=letter)
            check_draft(run, case, brief, pack, draft, letter)

    assert_passed(run)
