"""Tests for the Dopytywator prompt builders."""

from src.Dopytywator import prompts, structures


class TestInterviewerUserPrompt:
    def test_interviewer_user_prompt__includes_question_context(self) -> None:
        """Verify the prompt includes the current question order and the conversation transcript."""
        messages = [
            structures.ConversationMessage(role="user", content="Mam problem z dostępem."),
            structures.ConversationMessage(role="assistant", content="Jakiego wyniku oczekujesz?"),
        ]

        prompt = prompts.interviewer_user_prompt(
            messages,
            question_number=2,
            max_interview_questions=3,
        )

        assert "Pełna historia rozmowy:" in prompt
        assert "To jest decyzja przed pytaniem pomocniczym nr 2" in prompt
        assert "ready_to_build_brief=true" in prompt
        assert "Mam problem z dostępem." in prompt


class TestBriefUserPrompt:
    def test_brief_user_prompt__includes_conversation_history(self) -> None:
        """Verify the brief prompt includes the full conversation history for synthesis."""
        messages = [
            structures.ConversationMessage(role="user", content="Chcę przejrzeć dokumenty."),
        ]

        prompt = prompts.brief_user_prompt(messages)

        assert "Pełna historia rozmowy:" in prompt
        assert "Chcę przejrzeć dokumenty." in prompt
        assert "Utwórz końcowy CaseBrief" in prompt
