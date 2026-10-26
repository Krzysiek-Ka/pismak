"""Tests for the Dopytywator prompt builders."""

from src.Dopytywator import prompts, structures


class TestInterviewerUserPrompt:
    def test_interviewer_user_prompt__conversation_and_question_number_are_included(self) -> None:
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
    def test_brief_user_prompt__whole_conversation_is_included(self) -> None:
        """Verify the brief prompt includes the full conversation history for synthesis."""
        messages = [
            structures.ConversationMessage(role="user", content="Chcę przejrzeć dokumenty."),
        ]

        prompt = prompts.brief_user_prompt(messages)

        assert "Pełna historia rozmowy:" in prompt
        assert "Chcę przejrzeć dokumenty." in prompt
        assert "Utwórz końcowy CaseBrief" in prompt


class TestBuildBriefSystemPrompt:
    def test_build_brief_system_prompt__keeps_observations_out_of_available_evidence(
        self,
    ) -> None:
        """Verify the brief rules say what counts as a material, with a counter-example."""
        prompt = prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT

        assert "available_evidence zawiera wyłącznie materiały, które użytkownik ma" in prompt
        assert "„Brak pomiarów prędkości” nie są materiałami" in prompt

    def test_build_brief_system_prompt__unknown_institution_stays_null(self) -> None:
        """Verify the rules forbid guessing the institution from the topic of the case."""
        prompt = prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT

        assert "nie zgaduj ich z tematu sprawy" in prompt
        assert "institution=null, addressee=null" in prompt

    def test_build_brief_system_prompt__rules_are_numbered_in_order(self) -> None:
        """Verify no rule number is skipped or repeated after rules were added."""
        numbers = [
            int(line.split(".", 1)[0])
            for line in prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT.splitlines()
            if line[:1].isdigit()
        ]

        assert numbers == list(range(1, len(numbers) + 1))
