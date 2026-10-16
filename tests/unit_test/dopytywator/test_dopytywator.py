"""Tests for the Dopytywator orchestration functions."""

from __future__ import annotations

import pytest

from src.Dopytywator import dopytywator, structures
from src.Dopytywator.constants import DOPYTYWATOR_MAX_OUTPUT_TOKENS


def question(text: str) -> structures.InterviewTurnResult:
    return structures.InterviewTurnResult(
        short_summary="Podsumowanie",
        next_question=text,
        reason_for_question="Brakuje szczegółów.",
        ready_to_build_brief=False,
    )


READY = structures.InterviewTurnResult(
    short_summary="Sprawa jasna",
    next_question=None,
    reason_for_question=None,
    ready_to_build_brief=True,
)


class ScriptedLLM:
    """Return scripted interview turns and remember the prompts that requested them."""

    def __init__(self, *turns: structures.InterviewTurnResult) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, object]] = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.turns.pop(0)


def answer_with(monkeypatch, *answers: str) -> list[str]:
    """Feed console answers in order and return the list of prompts that asked for them."""
    remaining = list(answers)
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return remaining.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


class TestGetProblem:
    def test_get_problem__reads_user_input(self, monkeypatch) -> None:
        """Verify that the text prompt accepts a non-empty problem description."""
        monkeypatch.setattr("builtins.input", lambda prompt: "Brak dostępu do informacji")

        result = dopytywator.get_problem()

        assert result == "Brak dostępu do informacji"

    def test_get_problem__retries_after_empty_input(self, monkeypatch, capsys) -> None:
        """Verify empty initial input is rejected before a valid problem is returned."""
        answers = iter(["   ", "Brak dostępu"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

        result = dopytywator.get_problem()

        assert result == "Brak dostępu"
        assert "Opis nie może być pusty." in capsys.readouterr().out


class TestAskQuestions:
    def test_ask_questions__stops_when_brief_is_ready(self, monkeypatch, capsys) -> None:
        """Verify the interview ends without extra questions once the brief is ready."""

        class FakeLLM:
            def __init__(self) -> None:
                self.calls = 0

            def generate_structured(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return structures.InterviewTurnResult(
                        short_summary="Sytuacja zrozumiała",
                        next_question=None,
                        reason_for_question=None,
                        missing_topics=[],
                        ready_to_build_brief=True,
                    )
                raise AssertionError("Should not be called again")

        llm = FakeLLM()
        monkeypatch.setattr("builtins.input", lambda prompt: "Dostępne są zdjęcia i wniosek")

        result = dopytywator.ask_questions("Brak dostępu", llm, max_interview_questions=3)

        captured = capsys.readouterr().out
        assert "Mam wystarczające informacje" in captured
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "Brak dostępu"

    def test_ask_questions__returns_empty_for_exit_command(self, capsys) -> None:
        """Verify /exit stops the interview without creating a user message."""
        result = dopytywator.ask_questions("/exit", object())

        assert result == []
        assert "Przerwano." in capsys.readouterr().out

    def test_ask_questions__asks_for_a_description_first_when_started_with_done(
        self, capsys
    ) -> None:
        """Verify /done without any description does not build a brief from nothing."""
        llm = ScriptedLLM()

        result = dopytywator.ask_questions("/DONE", llm)

        assert result == []
        assert llm.calls == []
        assert "Najpierw podaj przynajmniej krótki opis sprawy." in capsys.readouterr().out

    def test_ask_questions__records_questions_and_answers_until_the_limit(
        self, monkeypatch, capsys
    ) -> None:
        """Verify the transcript alternates roles and the model sees the current question number."""
        llm = ScriptedLLM(question("Gdzie?"), question("Od kiedy?"))
        answer_with(monkeypatch, "  Przy SP 1  ", "Od września")

        result = dopytywator.ask_questions("Brak przejścia", llm, max_interview_questions=2)

        assert [(m.role, m.content) for m in result] == [
            ("user", "Brak przejścia"),
            ("assistant", "Gdzie?"),
            ("user", "Przy SP 1"),
            ("assistant", "Od kiedy?"),
            ("user", "Od września"),
        ]
        assert len(llm.calls) == 2
        assert "pytaniem pomocniczym nr 1 z maksymalnie 2" in llm.calls[0]["user_prompt"]
        assert "pytaniem pomocniczym nr 2 z maksymalnie 2" in llm.calls[1]["user_prompt"]
        out = capsys.readouterr().out
        assert "Pismak [2/2]: Od kiedy?" in out
        assert "Osiągnięto limit 2 pytań." in out

    def test_ask_questions__repeats_the_prompt_until_the_answer_is_not_blank(
        self, monkeypatch, capsys
    ) -> None:
        """Verify an accidental Enter does not store an empty answer in the transcript."""
        llm = ScriptedLLM(question("Gdzie?"))
        prompts = answer_with(monkeypatch, "", "   ", "Przy SP 1")

        result = dopytywator.ask_questions("Brak przejścia", llm, max_interview_questions=1)

        assert result[-1].content == "Przy SP 1"
        assert len(prompts) == 3
        assert capsys.readouterr().out.count("Wpisz odpowiedź albo użyj /done lub /exit.") == 2

    def test_ask_questions__exit_during_the_interview_stops_without_storing_the_command(
        self, monkeypatch, capsys
    ) -> None:
        """Verify /exit ends the questions at once and is not treated as an answer."""
        llm = ScriptedLLM(question("Gdzie?"), question("Nie powinno paść"))
        answer_with(monkeypatch, "/Exit")

        result = dopytywator.ask_questions("Brak przejścia", llm, max_interview_questions=3)

        assert [m.content for m in result] == ["Brak przejścia", "Gdzie?"]
        assert len(llm.calls) == 1
        out = capsys.readouterr().out
        assert "Przerwano. CaseBrief nie został utworzony." in out
        assert "Osiągnięto limit" not in out

    def test_ask_questions__done_during_the_interview_keeps_the_answers_given_so_far(
        self, monkeypatch, capsys
    ) -> None:
        """Verify /done finishes early and the brief is built from the collected answers."""
        llm = ScriptedLLM(question("Gdzie?"), question("Od kiedy?"))
        answer_with(monkeypatch, "Przy SP 1", "/done")

        result = dopytywator.ask_questions("Brak przejścia", llm, max_interview_questions=5)

        assert [m.content for m in result] == ["Brak przejścia", "Gdzie?", "Przy SP 1", "Od kiedy?"]
        assert len(llm.calls) == 2
        out = capsys.readouterr().out
        assert "Kończę pytania i tworzę brief" in out
        assert "Osiągnięto limit" not in out

    def test_ask_questions__zero_question_limit_skips_the_model(self) -> None:
        """Verify a limit of zero goes straight to the brief with only the description."""
        llm = ScriptedLLM()

        result = dopytywator.ask_questions("Brak przejścia", llm, max_interview_questions=0)

        assert [m.content for m in result] == ["Brak przejścia"]
        assert llm.calls == []

    def test_ask_questions__debug_mode_prints_the_model_reasoning(self, capsys) -> None:
        """Verify DEBUG shows why the model wants to stop or ask, to help tune prompts."""
        dopytywator.ask_questions("Brak przejścia", ScriptedLLM(READY), DEBUG=True)

        out = capsys.readouterr().out
        assert "[DEBUG] summary: Sprawa jasna" in out
        assert "[DEBUG] ready: True" in out

    def test_ask_questions__uses_the_interview_output_budget_and_temperature(self) -> None:
        """Verify each interview turn keeps its short output cap and the caller's temperature."""
        llm = ScriptedLLM(READY)

        dopytywator.ask_questions("Brak przejścia", llm, temperature=0.3)

        assert llm.calls[0]["max_output_tokens"] == DOPYTYWATOR_MAX_OUTPUT_TOKENS
        assert llm.calls[0]["temperature"] == 0.3
        assert llm.calls[0]["response_model"] is structures.InterviewTurnResult


class TestBuildBrief:
    def test_build_brief__returns_valid_case_brief(self, monkeypatch) -> None:
        """Verify that a valid generated response is returned and checked for readiness."""

        class FakeLLM:
            def generate_structured(self, **kwargs):
                return structures.CaseBrief(
                    title="Brak dostępu",
                    problem_description="Nie ma dostępu do dokumentów.",
                    desired_outcome="Usunięcie problemu.",
                    requested_actions=["Złożenie wniosku"],
                )

        llm = FakeLLM()
        monkeypatch.setattr(
            dopytywator.dopytywator_helper_functions, "print_case_brief", lambda brief: None
        )

        case_brief = dopytywator.build_brief([], llm)

        assert isinstance(case_brief, structures.CaseBrief)
        assert case_brief.title == "Brak dostępu"

    def test_build_brief__passes_stage_output_token_limit(self, monkeypatch) -> None:
        """Verify brief generation explicitly uses the Dopytywator's short-output budget."""
        captured: dict[str, object] = {}

        class FakeLLM:
            def generate_structured(self, **kwargs):
                captured.update(kwargs)
                return structures.CaseBrief(
                    title="Sprawa",
                    problem_description="Problem",
                    desired_outcome="Rozwiązanie",
                    requested_actions=["Sprawdzenie"],
                )

        monkeypatch.setattr(
            dopytywator.dopytywator_helper_functions,
            "print_case_brief",
            lambda brief: None,
        )

        dopytywator.build_brief([], FakeLLM())

        assert captured["max_output_tokens"] == DOPYTYWATOR_MAX_OUTPUT_TOKENS

    def test_build_brief__warns_about_missing_required_fields_but_still_returns_the_brief(
        self, monkeypatch, capsys
    ) -> None:
        """Verify gaps are reported to the user instead of being filled in by the application."""

        class FakeLLM:
            def generate_structured(self, **kwargs):
                return structures.CaseBrief(
                    title="Sprawa",
                    problem_description="Problem",
                    desired_outcome="",
                    requested_actions=[],
                )

        monkeypatch.setattr(
            dopytywator.dopytywator_helper_functions, "print_case_brief", lambda brief: None
        )

        case_brief = dopytywator.build_brief([], FakeLLM())

        out = capsys.readouterr().out
        assert case_brief.title == "Sprawa"
        assert "BRIEF MA BRAKI W POLACH OBOWIĄZKOWYCH" in out
        assert "Brak oczekiwanego rezultatu" in out
        assert "Brak co najmniej jednego konkretnego postulatu" in out
        assert "spełnia minimalne kryteria" not in out


BRIEF = structures.CaseBrief(
    title="Przejście przy szkole",
    problem_description="Brak przejścia.",
    desired_outcome="Wyznaczenie przejścia.",
    requested_actions=["Wyznaczenie przejścia"],
)
CONVERSATION = [structures.ConversationMessage(role="user", content="Brak przejścia")]


class TestInterviewStep:
    def test_interview_step__question_is_returned_stripped_without_a_brief(self) -> None:
        """Verify a model question is passed on trimmed and the interview continues."""
        llm = ScriptedLLM(question("  Gdzie?  "))

        step = dopytywator.interview_step(
            CONVERSATION, llm, questions_asked=1, max_interview_questions=3
        )

        assert step.question == "Gdzie?"
        assert step.case_brief is None
        assert step.end_reason is None
        assert step.turn.reason_for_question == "Brakuje szczegółów."
        assert "pytaniem pomocniczym nr 2 z maksymalnie 3" in llm.calls[0]["user_prompt"]
        assert llm.calls[0]["temperature"] == 0.1

    def test_interview_step__ready_model_builds_the_brief_in_the_same_step(self) -> None:
        """Verify a ready verdict is followed at once by the brief built from the conversation."""
        llm = ScriptedLLM(READY, BRIEF)

        step = dopytywator.interview_step(
            CONVERSATION, llm, questions_asked=0, max_interview_questions=3
        )

        assert step.end_reason == "model_ready"
        assert step.case_brief == BRIEF
        assert step.question is None
        assert step.turn == READY
        assert [c["response_model"] for c in llm.calls] == [
            structures.InterviewTurnResult,
            structures.CaseBrief,
        ]
        assert llm.calls[1]["temperature"] == 0.0
        assert llm.calls[1]["max_output_tokens"] == DOPYTYWATOR_MAX_OUTPUT_TOKENS

    def test_interview_step__reached_limit_builds_the_brief_without_asking_the_model(
        self,
    ) -> None:
        """Verify the answer to the last allowed question goes straight to the brief."""
        llm = ScriptedLLM(BRIEF)

        step = dopytywator.interview_step(
            CONVERSATION, llm, questions_asked=3, max_interview_questions=3
        )

        assert step.end_reason == "question_limit"
        assert step.case_brief == BRIEF
        assert step.turn is None
        assert [c["response_model"] for c in llm.calls] == [structures.CaseBrief]

    def test_interview_step__reports_every_successful_model_call(self) -> None:
        """Verify the caller can record model usage after each call, as the UI does."""
        llm = ScriptedLLM(READY, BRIEF)
        calls_seen: list[int] = []

        dopytywator.interview_step(
            CONVERSATION,
            llm,
            questions_asked=0,
            max_interview_questions=3,
            after_llm_call=lambda: calls_seen.append(len(llm.calls)),
        )

        assert calls_seen == [1, 2]

    def test_interview_step__failed_brief_keeps_the_report_of_the_interview_call(self) -> None:
        """Verify a brief failure still leaves the earlier successful call recorded."""

        class BriefFails(ScriptedLLM):
            def generate_structured(self, **kwargs):
                if kwargs["response_model"] is structures.CaseBrief:
                    raise RuntimeError("model niedostępny")
                return super().generate_structured(**kwargs)

        llm = BriefFails(READY)
        calls_seen: list[int] = []

        with pytest.raises(RuntimeError):
            dopytywator.interview_step(
                CONVERSATION,
                llm,
                questions_asked=0,
                max_interview_questions=3,
                after_llm_call=lambda: calls_seen.append(1),
            )

        assert calls_seen == [1]
