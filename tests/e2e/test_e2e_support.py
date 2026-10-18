"""
Kontrola ewaluatorów testu E2E przed długim przebiegiem: bez modeli i bazy.

Dane poniżej opisują sytuacje do oceny przez kontrole testu; nie zastępują modeli w E2E.
"""

from __future__ import annotations

import re

import pytest
from e2e_cases import BIELIK, CASES, OPENROUTER, PLLUM, SETUPS, build_setups, resolve_model
from e2e_support import (
    FAIL,
    MAX_QUESTIONS,
    NOT_RUN,
    PASS,
    PASSED,
    REDACTED,
    Check,
    PhaseError,
    analyze_log,
    contains_any,
    interview_invariants,
    legal_basis_outside,
    markdown_summary,
    new_result,
    repeated_questions,
    scrub_secrets,
    text_summary,
    unknown_numbers,
    words_outside,
)

from src.Dopytywator.structures import ConversationMessage

CASE = CASES[0]
CASE_IDS = [case.id for case in CASES]


def entry(name: str = "proba", **overrides: object) -> dict:
    """Return one valid URUCHOMIENIA entry with the given fields changed."""
    return {
        "nazwa": name,
        "dopytywator": "pllum",
        "zrodlowyszukiwacz": "bielik",
        "pismogenerator": "openrouter",
        **overrides,
    }


def interview_checks(
    *, questions: int, calls: int, end_reason: str, ready: bool = True
) -> dict[str, str]:
    """Evaluate the interview invariants of a finished interview with the given counters."""
    result = new_result(SETUPS[0], CASE)
    result.questions_asked = result.answers_used = questions
    result.interview_calls = calls
    result.end_reason = end_reason
    if end_reason == "model_ready":
        result.final_turn = {"ready_to_build_brief": ready, "next_question": None}
    messages = [ConversationMessage(role="user", content=CASE.description)]
    interview_invariants(result, CASE, messages)
    return {check.id: check.status for check in result.checks}


def test_cases_have_an_answer_for_every_allowed_question() -> None:
    assert len({case.id for case in CASES}) == len(CASES)
    for case in CASES:
        assert case.description.strip()
        assert len(case.answers) == MAX_QUESTIONS
        assert all(answer.strip() for answer in case.answers)


def test_run_list_entries_become_setups_with_a_model_per_stage() -> None:
    [setup] = build_setups([entry("mix", sprawy=["buspas"])], CASE_IDS)

    assert (setup.id, setup.dopytywator, setup.zrodla, setup.pismo) == (
        "mix",
        PLLUM,
        BIELIK,
        OPENROUTER,
    )
    assert setup.case_ids == ("buspas",)


def test_models_are_shortcuts_ollama_names_or_openrouter_ids() -> None:
    assert resolve_model("bielik") == BIELIK
    assert (resolve_model("qwen2.5").provider, resolve_model("qwen2.5").model) == (
        "ollama",
        "qwen2.5:latest",
    )
    assert resolve_model("llama3.1:8b").model == "llama3.1:8b"
    custom = resolve_model("openrouter:meta-llama/llama-3.3-70b-instruct:free")
    assert (custom.provider, custom.model) == (
        "openrouter",
        "meta-llama/llama-3.3-70b-instruct:free",
    )


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([entry(pismogenerator="")], "brak pól: pismogenerator"),
        ([entry(zrodla="bielik")], "nieznane pola ['zrodla']"),
        ([entry("a b")], "nazwa może zawierać tylko"),
        ([entry("x"), entry("x")], "„x” już występuje"),
        ([entry(sprawy=["kot"])], "nieznane sprawy ['kot']"),
        ([entry(dopytywator="openrouter:")], "brak identyfikatora modelu"),
        ([], "lista jest pusta"),
    ],
)
def test_run_list_mistakes_are_reported_with_the_entry_to_fix(entries, message) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        build_setups(entries, CASE_IDS)


def test_the_configured_run_list_is_valid() -> None:
    assert SETUPS


def test_result_names_every_stage_and_reproduces_exactly_one_run() -> None:
    [mixed] = build_setups(
        [
            entry(
                "mix_pllum_openrouter_bielik",
                zrodlowyszukiwacz="openrouter",
                pismogenerator="bielik",
            )
        ],
        CASE_IDS,
    )

    result = new_result(mixed, CASE)

    assert result.stages == {
        "Dopytywator": {"name": "pllum", "provider": "ollama", "model": PLLUM.model},
        "Źródłowyszukiwacz": {
            "name": "openrouter",
            "provider": "openrouter",
            "model": OPENROUTER.model,
        },
        "Pismogenerator": {"name": "bielik", "provider": "ollama", "model": BIELIK.model},
    }
    # `-k bielik-kot_wrocek` wybrałby też ten przebieg; polecenie podaje pełny identyfikator.
    assert f"test_pismak_end_to_end[{result.run_id}]" in result.reproduce
    assert " -k " not in result.reproduce


def test_interview_invariants_accept_both_ways_of_ending() -> None:
    assert interview_checks(questions=12, calls=12, end_reason="question_limit")["H3"] == PASS
    checks = interview_checks(questions=4, calls=5, end_reason="model_ready")
    assert checks["H1"] == PASS
    assert checks["H3"] == PASS


def test_interview_invariants_reject_a_broken_limit_or_signal() -> None:
    assert interview_checks(questions=13, calls=13, end_reason="question_limit")["H3"] == FAIL
    assert interview_checks(questions=12, calls=13, end_reason="question_limit")["H3"] == FAIL
    assert interview_checks(questions=12, calls=13, end_reason="model_ready")["H3"] == FAIL
    not_ready = interview_checks(questions=3, calls=4, end_reason="model_ready", ready=False)
    assert not_ready["H3"] == FAIL


def test_keyword_checks_ignore_case_and_polish_characters() -> None:
    assert contains_any("Happening we WROCŁAWIU", ["wroclaw"])
    assert contains_any("Akcja satyryczna", CASE.humor_markers)
    assert not contains_any("Prezydent ma ustąpić", CASE.humor_markers)


def test_invention_heuristics_flag_only_what_the_inputs_do_not_contain() -> None:
    assert unknown_numbers("Badanie z 2025 r. pokazało 75% [1] (E1)", "rok 2025") == ["75"]
    assert words_outside("Prezydent Wrocławia", "prezydent Wrocławia, Jacek Sutryk") == []
    assert words_outside("Urząd Miejski", "żeby miasto przeanalizowało") == ["urzad", "miejski"]
    assert legal_basis_outside("Na podstawie art. 61 Konstytucji", "brak") == ["art."]
    assert legal_basis_outside("Zgodnie z uchwałą nr XV/1", "Uchwała nr XV/1 Rady") == []


def test_list_numbering_is_not_taken_for_a_fact() -> None:
    letter = "Postulaty:\n1. Buspas\n2) Analiza\\n3. **Dane** z 2019 r."
    assert unknown_numbers(letter, "rozmowa bez liczb") == ["2019"]


def test_log_analysis_keeps_the_duration_of_failed_calls(tmp_path) -> None:
    (tmp_path / "pismak_1.log").write_text(
        "2026-10-16 18:00:00,000 ERROR llm_request_failed stage=dopytywator session_id=x "
        "provider=ollama model=m response_model=InterviewTurnResult max_output_tokens=5936 "
        "attempt=1 retry_scheduled=true error_type=ValidationError error_detail=odpowiedź: "
        "Value error elapsed_seconds=3113.660\n",
        "utf-8",
    )

    _, stats = analyze_log(tmp_path)

    assert stats["failed_calls"] == [
        {
            "stage": "dopytywator",
            "attempt": 1,
            "error_type": "ValidationError",
            "elapsed_seconds": 3113.66,
        }
    ]


def test_repeated_questions_are_found() -> None:
    questions = ["Gdzie ma się odbyć happening?", "Kiedy?", "Gdzie ma sie odbyc happening?"]
    assert repeated_questions(questions) == [(1, 3)]


def test_scrub_secrets_redacts_keys_in_written_files(tmp_path) -> None:
    log = tmp_path / "logs" / "pismak.log"
    log.parent.mkdir()
    log.write_text("klucz=sk-or-v1-abcdefghijklmnopqrstuvwxyz0123 haslo=tajne_haslo", "utf-8")
    (tmp_path / "ok.json").write_text("{}", "utf-8")

    leaked = scrub_secrets(tmp_path, ["tajne_haslo"])

    assert leaked == [log]
    assert log.read_text("utf-8") == f"klucz={REDACTED} haslo={REDACTED}"


def test_summary_table_shows_each_stage_model_and_why_runs_did_not_pass() -> None:
    [setup] = build_setups([entry("mix")], CASE_IDS)
    passed = new_result(setup, CASES[1])
    passed.status = PASSED
    passed.questions_asked = 12
    passed.end_reason = "question_limit"
    passed.timings = {"razem": 1575.0}
    failed = new_result(setup, CASE)
    failed.errors = [
        PhaseError("wywiad", "APITimeoutError", "Przekroczono czas", None, "timed out")
    ]
    failed.checks = [Check("H2", "x", "hard", FAIL, "timed out")]
    skipped = new_result(setup, CASE)
    skipped.status = NOT_RUN
    skipped.not_run_reason = "Dopytywator: model nie jest zarejestrowany w Ollamie"

    table = text_summary([passed, failed, skipped])
    markdown = "\n".join(markdown_summary([passed, failed, skipped]))

    header = table[1]
    for column in ("uruchomienie", "Dopytywator", "Źródłowyszukiwacz", "Pismogenerator", "wynik"):
        assert column in header
    assert "| mix " in table[3] and "| pllum " in table[3] and "| limit 12 " in table[3]
    assert "26 min 15 s" in table[3]
    assert len({len(line) for line in table[:7]}) == 1  # wszystkie wiersze tabeli równej szerokości
    assert any("mix-kot_wrocek: wywiad: APITimeoutError — timed out" in line for line in table)
    assert any("model nie jest zarejestrowany" in line for line in table)
    assert "| `mix` | buspas | pllum | bielik | openrouter |" in markdown
    assert "Przyczyny niepowodzeń" in markdown
