from src.Dopytywator import structures
from src.Dopytywator.dopytywator_helper_functions import (
    conversation_to_text,
    print_case_brief,
    validate_brief_readiness,
)


class TestConversationToText:
    def test_conversation_to_text__user_and_assistant_messages_get_speaker_labels(self) -> None:
        """Verify each message gets its speaker label and a blank line, so the model can tell the user's words from its own questions."""
        messages = [
            structures.ConversationMessage(role="user", content="Cześć"),
            structures.ConversationMessage(role="assistant", content="Dzień dobry"),
        ]

        result = conversation_to_text(messages)

        assert result == "UŻYTKOWNIK: Cześć\n\nAgent: Dzień dobry"


class TestValidateBriefReadiness:
    def test_validate_brief_readiness__blank_required_fields_return_one_error_each(self) -> None:
        """Verify every empty required field is named, including a list holding only blank actions."""
        brief = structures.CaseBrief(
            title="",
            problem_description="",
            desired_outcome="",
            requested_actions=[""],
        )

        errors = validate_brief_readiness(brief)

        assert errors == [
            "Brak tytułu sprawy",
            "Brak opisu problemu",
            "Brak oczekiwanego rezultatu",
            "Brak co najmniej jednego konkretnego postulatu",
        ]

    def test_validate_brief_readiness__complete_brief_returns_no_errors(self) -> None:
        """Verify a brief with every required field passes, so the UI shows no false gap warning."""
        brief = structures.CaseBrief(
            title="Prawo do dostępu do informacji",
            problem_description="Wniosek o informacje nie został rozpatrzony.",
            desired_outcome="Uzyskanie informacji i wyjaśnień.",
            requested_actions=["Złożenie ponownego wniosku", "Wskazanie procedury"],
        )

        assert validate_brief_readiness(brief) == []


class TestPrintCaseBrief:
    def test_print_case_brief__full_brief_prints_every_section_and_claim(self, capsys) -> None:
        """Verify the console summary shows each field, list and claim with its status and declared evidence."""
        brief = structures.CaseBrief(
            title="Brak dostępu do pomieszczeń",
            problem_description="Nie można korzystać z pomieszczeń rekreacyjnych.",
            location="Warszawa",
            time_context="2026",
            desired_outcome="Przywrócenie dostępu.",
            requested_actions=["Umożliwienie wejścia", "Rozważenie usprawnień"],
            institution="Uczelnia",
            addressee="Dyrektor",
            preferred_tone="stanowczy",
            affected_groups=["Studenci"],
            public_interest_arguments=["Dostęp do przestrzeni wspólnej"],
            available_evidence=["Zdjęcia"],
            previous_actions=["Wniosek z maja"],
            likely_counterarguments=["Brak finansowania"],
            missing_information=["Termin rozwiązania"],
            facts_to_verify=["Czy budynek jest dostępny"],
            user_notes=["Ważne dla studentów"],
            claims=[
                structures.Claim(
                    statement="Problem jest systemowy.",
                    status=structures.ClaimStatus.REQUIRES_VERIFICATION,
                    evidence_description="Zgłoszenia od studentów",
                )
            ],
        )

        print_case_brief(brief)

        captured = capsys.readouterr().out
        assert "CASE BRIEF" in captured
        assert "Tytuł:\nBrak dostępu do pomieszczeń" in captured
        assert "Problem:\nNie można korzystać z pomieszczeń rekreacyjnych." in captured
        assert "Lokalizacja:\nWarszawa" in captured
        assert "Oczekiwany rezultat:\nPrzywrócenie dostępu." in captured
        assert "1. Umożliwienie wejścia" in captured
        assert "Grupy dotknięte problemem" in captured
        assert "Dostęp do przestrzeni wspólnej" in captured
        assert "[requires_verification] Problem jest systemowy." in captured
        assert "deklarowany dowód: Zgłoszenia od studentów" in captured
