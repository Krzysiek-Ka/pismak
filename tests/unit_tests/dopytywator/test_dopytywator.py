"""Tests for the Dopytywator orchestration functions."""

from __future__ import annotations

from src.Dopytywator import dopytywator, structures


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
