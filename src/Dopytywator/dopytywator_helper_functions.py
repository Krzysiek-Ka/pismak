from src.Dopytywator import structures


def conversation_to_text(messages: list[structures.ConversationMessage]) -> str:
    """Convert a list of conversation messages to a plain-text transcript.

    Args:
        messages: A sequence of user/assistant messages in chronological order.

    Returns:
        A text transcript with each message prefixed by the speaker label.
    """
    parts: list[str] = []
    for msg in messages:
        speaker = "UŻYTKOWNIK" if msg.role == "user" else "Agent"
        parts.append(f"{speaker}: {msg.content}")
    return "\n\n".join(parts)


def validate_brief_readiness(brief: structures.CaseBrief) -> list[str]:
    """Validate whether a case brief contains the required core information.

    Args:
        brief: The case brief to validate.

    Returns:
        A list of human-readable validation errors. Empty list means the brief is ready.
    """
    errors: list[str] = []
    if not brief.title.strip():
        errors.append("Brak tytułu sprawy")
    if not brief.problem_description.strip():
        errors.append("Brak opisu problemu")
    if not brief.desired_outcome.strip():
        errors.append("Brak oczekiwanego rezultatu")
    if not [x for x in brief.requested_actions if x.strip()]:
        errors.append("Brak co najmniej jednego konkretnego postulatu")
    return errors


def print_case_brief(brief: structures.CaseBrief) -> None:
    """Print a human-friendly summary of a case brief to stdout.

    Args:
        brief: The case brief that should be displayed.
    """
    print("\n" + "=" * 72)
    print("CASE BRIEF")
    print("=" * 72)
    print(f"\nTytuł:\n{brief.title or '[brak]'}")
    print(f"\nProblem:\n{brief.problem_description or '[brak]'}")
    print(f"\nLokalizacja:\n{brief.location or '[nieustalona]'}")
    print(f"\nKontekst czasowy:\n{brief.time_context or '[nieustalony]'}")
    print(f"\nOczekiwany rezultat:\n{brief.desired_outcome or '[brak]'}")
    print("\nPostulaty:")
    if brief.requested_actions:
        for i, action in enumerate(brief.requested_actions, 1):
            print(f"  {i}. {action}")
    else:
        print("  [brak]")
    print(f"\nInstytucja:\n{brief.institution or '[nieustalona]'}")
    print(f"\nAdresat:\n{brief.addressee or '[nieustalony]'}")
    print(f"\nPreferowany ton:\n{brief.preferred_tone}")

    def show_list(label: str, values: list[str]) -> None:
        """Print a labelled list section, including a fallback placeholder for empty values."""
        print(f"\n{label}:")
        if values:
            for value in values:
                print(f"  - {value}")
        else:
            print("  - brak")

    show_list("Grupy dotknięte problemem", brief.affected_groups)
    show_list("Argumenty interesu publicznego", brief.public_interest_arguments)
    show_list("Dostępne dowody", brief.available_evidence)
    show_list("Wcześniejsze działania", brief.previous_actions)
    show_list("Potencjalne kontrargumenty / słabe punkty", brief.likely_counterarguments)
    show_list("Brakujące informacje", brief.missing_information)
    show_list("Fakty do weryfikacji", brief.facts_to_verify)
    show_list("Notatki użytkownika", brief.user_notes)

    print("\nTwierdzenia:")
    if not brief.claims:
        print("  - brak")
    else:
        for claim in brief.claims:
            ev = (
                f" | deklarowany dowód: {claim.evidence_description}"
                if claim.evidence_description
                else ""
            )
            print(f"  - [{claim.status.value}] {claim.statement}{ev}")
