"""
Złoty zbiór Dopytywatora: wywiad i CaseBrief na prawdziwym modelu, bez atrap.

Rozmowa idzie tą samą ścieżką co czat w src/Wyswietlator/app.py: po każdej wiadomości
użytkownika `interview_step` zadaje kolejne pytanie albo buduje brief, a odpowiedzi
użytkownika pochodzą z `user_answers` przypadku, po kolei. Atrapą jest tylko interfejs
(Streamlit), bo test podaje odpowiedzi zamiast czatu.

Obsługiwane pola `expected` (SUPPORTED_EXPECTED):
- brief_ready: brief przechodzi validate_brief_readiness (tytuł, problem, rezultat, postulat),
- min_requested_actions: najmniejsza liczba niepustych postulatów,
- must_mention: każda pozycja (napis albo lista wariantów) występuje w briefie,
- must_not_mention: żaden napis nie występuje w briefie,
- requested_actions_must_not_mention: żaden napis nie występuje w postulatach,
- null_fields: pola briefu puste (null albo pusty tekst/lista); available_evidence, gdy
  użytkownik nie wspomniał o żadnych materiałach (wpisy stają się źródłami U1, U2 w piśmie),
- non_empty_fields: pola briefu niepuste,
- forbidden_claim_statuses: statusy twierdzeń, których brief nie może użyć,
- no_new_numbers: każda liczba w briefie padła w rozmowie.
Zawsze sprawdzane: schemat CaseBrief, limit pytań `limits.max_questions`, starczające odpowiedzi.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.Dopytywator import dopytywator
from src.Dopytywator.dopytywator_helper_functions import validate_brief_readiness
from src.Dopytywator.structures import CaseBrief, ConversationMessage
from tests.golden_set.golden_support import (
    GoldenCase,
    Provider,
    as_text,
    assert_passed,
    golden_run,
    load_cases,
    make_llm,
    mentions,
    numbers_in,
    provider_skip_reason,
    runs,
    skip_run,
)

pytestmark = pytest.mark.golden

MODULE = "dopytywator"
MODULE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXPECTED = {
    "brief_ready",
    "min_requested_actions",
    "must_mention",
    "must_not_mention",
    "requested_actions_must_not_mention",
    "null_fields",
    "non_empty_fields",
    "forbidden_claim_statuses",
    "no_new_numbers",
}
CASES = load_cases(
    MODULE_DIR,
    supported_expected=SUPPORTED_EXPECTED,
    required_limits={"max_questions"},
)


class TestGoldenCases:
    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__answers_cover_every_allowed_question(self, case: GoldenCase) -> None:
        """Verify the interview can never run out of user answers before its question limit."""
        assert len(case.user_answers) >= case.limits["max_questions"]

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__field_checks_name_existing_case_brief_fields(self, case: GoldenCase) -> None:
        """Verify null_fields and non_empty_fields cannot pass by naming a field that is not there."""
        fields = set(CaseBrief.model_fields)
        named = set(case.expected.get("null_fields", [])) | set(
            case.expected.get("non_empty_fields", [])
        )
        assert named <= fields, sorted(named - fields)


def interview(
    case: GoldenCase, llm
) -> tuple[CaseBrief | None, list[ConversationMessage], int, str]:
    """Talk to the model the way the chat in app.py does; return brief, messages, questions, end."""
    messages = [ConversationMessage(role="user", content=case.input)]
    answers = list(case.user_answers)
    asked = 0
    while True:
        step = dopytywator.interview_step(
            messages,
            llm,
            questions_asked=asked,
            max_interview_questions=case.limits["max_questions"],
        )
        if step.case_brief is not None:
            return step.case_brief, messages, asked, step.end_reason or "-"
        asked += 1
        messages.append(ConversationMessage(role="assistant", content=step.question))
        if not answers:
            return None, messages, asked, "brak odpowiedzi użytkownika"
        messages.append(ConversationMessage(role="user", content=answers.pop(0)))


def is_empty(value: object) -> bool:
    return value is None or (isinstance(value, str | list) and not value)


def check_brief(run, case: GoldenCase, brief: CaseBrief, messages: list) -> None:
    """Record every check of the case's expected fields on the finished brief."""
    run.check(
        "schema_valid",
        CaseBrief.model_validate_json(brief.model_dump_json()) == brief,
        "CaseBrief",
    )
    expected = case.expected
    text = as_text(brief)
    conversation = "\n".join(m.content for m in messages)

    if expected.get("brief_ready"):
        errors = validate_brief_readiness(brief)
        run.check("brief_ready", not errors, "; ".join(errors))
    if "min_requested_actions" in expected:
        actions = [a for a in brief.requested_actions if a.strip()]
        run.check(
            "min_requested_actions",
            len(actions) >= expected["min_requested_actions"],
            f"{len(actions)} postulatów",
        )
    for item in expected.get("must_mention", []):
        variants = [item] if isinstance(item, str) else item
        run.check(
            f"must_mention:{variants[0]}",
            any(mentions(text, v) for v in variants),
            "brak w briefie" if not any(mentions(text, v) for v in variants) else "",
        )
    for term in expected.get("must_not_mention", []):
        run.check(f"must_not_mention:{term}", not mentions(text, term), "jest w briefie")
    actions_text = " ".join(brief.requested_actions)
    for term in expected.get("requested_actions_must_not_mention", []):
        run.check(
            f"requested_actions_must_not_mention:{term}",
            not mentions(actions_text, term),
            actions_text if mentions(actions_text, term) else "",
        )
    for name in expected.get("null_fields", []):
        value = getattr(brief, name)
        run.check(f"null_field:{name}", is_empty(value), repr(value))
    for name in expected.get("non_empty_fields", []):
        value = getattr(brief, name)
        run.check(f"non_empty_field:{name}", not is_empty(value), repr(value))
    statuses = [claim.status.value for claim in brief.claims]
    for status in expected.get("forbidden_claim_statuses", []):
        run.check(f"forbidden_claim_status:{status}", status not in statuses, str(statuses))
    if expected.get("no_new_numbers"):
        new = sorted(numbers_in(text) - numbers_in(conversation))
        run.check("no_new_numbers", not new, f"liczby spoza rozmowy: {new}" if new else "")


@pytest.mark.parametrize(("provider", "case"), runs(CASES))
def test_golden_dopytywator(provider: Provider, case: GoldenCase) -> None:
    """Verify a real model ends the interview within the limit with a brief that keeps the facts.

    The case file describes the edge scenario; the checks are listed in the module docstring.
    """
    reason = provider_skip_reason(provider)
    if reason:
        skip_run(MODULE, MODULE_DIR, provider, case, reason)

    with golden_run(MODULE, MODULE_DIR, provider, case) as run:
        llm = make_llm(
            provider,
            stage="dopytywator",
            session_id=run.session_id,
            timeout_seconds=case.limits.get("timeout_seconds"),
        )
        brief, messages, asked, end_reason = interview(case, llm)
        run.model = llm.last_model
        run.details.update(
            questions_asked=asked,
            end_reason=end_reason,
            conversation=[m.model_dump(mode="json", exclude={"created_at"}) for m in messages],
            brief=brief.model_dump(mode="json") if brief else None,
        )
        limit = case.limits["max_questions"]
        run.check("questions_within_limit", asked <= limit, f"{asked}/{limit} pytań")
        if run.check("brief_created", brief is not None, end_reason):
            check_brief(run, case, brief, messages)

    assert_passed(run)
