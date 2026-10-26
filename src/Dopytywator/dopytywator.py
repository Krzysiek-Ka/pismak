from __future__ import annotations

from collections.abc import Callable

from src.Dopytywator import dopytywator_helper_functions, prompts, structures
from src.Dopytywator.constants import DOPYTYWATOR_MAX_OUTPUT_TOKENS


def get_problem() -> str:
    """Collect and validate a brief problem statement from the user."""
    print("Pismak — Agent Dopytywator")
    print("-" * 72)
    print("Nie sprawdzam prawa, faktów ani kompetencji urzędu. Próbuję uszczegółowić Twoj problem.")
    print("Komendy: /done, /exit\n")

    while True:
        initial_problem = input("Opisz sprawę własnymi słowami:\n> ").strip()
        if initial_problem:
            break
        print("Opis nie może być pusty.")

    return initial_problem


def ask_questions(
    initial_problem: str,
    llm: object,
    max_interview_questions: int = 3,
    temperature: float = 0.1,
    DEBUG: bool = False,
) -> list[structures.ConversationMessage]:
    """Guide the user through a short interview and return the transcript."""
    messages: list[structures.ConversationMessage] = []

    if initial_problem.lower() == "/exit":
        print("Przerwano.")
    elif initial_problem.lower() == "/done":
        print("Najpierw podaj przynajmniej krótki opis sprawy.")
    else:
        messages.append(structures.ConversationMessage(role="user", content=initial_problem))
        questions_asked = 0

        while questions_asked < max_interview_questions:
            result = llm.generate_structured(
                system_prompt=prompts.DOPYTYWATOR_INTERVIEW_SYSTEM_PROMPT,
                user_prompt=prompts.interviewer_user_prompt(
                    messages,
                    question_number=questions_asked + 1,
                    max_interview_questions=max_interview_questions,
                ),
                response_model=structures.InterviewTurnResult,
                temperature=temperature,
                max_output_tokens=DOPYTYWATOR_MAX_OUTPUT_TOKENS,
            )

            if DEBUG:
                print("\n")
                print("[DEBUG] summary:", result.short_summary)
                print("[DEBUG] missing_topics:", result.missing_topics)
                print("[DEBUG] reason:", result.reason_for_question)
                print("[DEBUG] ready:", result.ready_to_build_brief)
                print("\n")

            if result.ready_to_build_brief:
                print("\nDopytywator: Mam wystarczające informacje, aby przygotować brief.")
                break

            # dopisz pytanie do listy
            question = result.next_question.strip()
            questions_asked += 1
            messages.append(structures.ConversationMessage(role="assistant", content=question))
            print(f"\nPismak [{questions_asked}/{max_interview_questions}]: {question}")

            # dopisz odpowiedź do listy
            answer = input("Ty: ").strip()
            while not answer:
                print("Wpisz odpowiedź albo użyj /done lub /exit.")
                answer = input("Ty: ").strip()

            if answer.lower() == "/exit":
                print("\nPrzerwano. CaseBrief nie został utworzony.")
                # Pusta rozmowa, jak po /exit na starcie: build_brief nie zbuduje z niej briefu.
                return []
            if answer.lower() == "/done":
                print("\nKończę pytania i tworzę brief z dostępnych danych.")
                break
            messages.append(structures.ConversationMessage(role="user", content=answer))
        else:
            # Limit wyczerpała tylko pętla bez break; /exit, /done i gotowość modelu go nie osiągają.
            print(f"\nOsiągnięto limit {max_interview_questions} pytań.")
        return messages

    return messages


def generate_brief(
    messages: list[structures.ConversationMessage],
    llm: object,
) -> structures.CaseBrief:
    """Ask the model for the final CaseBrief of a conversation."""
    return llm.generate_structured(
        system_prompt=prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT,
        user_prompt=prompts.brief_user_prompt(messages),
        response_model=structures.CaseBrief,
        temperature=0.0,
        max_output_tokens=DOPYTYWATOR_MAX_OUTPUT_TOKENS,
    )


def interview_step(
    messages: list[structures.ConversationMessage],
    llm: object,
    *,
    questions_asked: int,
    max_interview_questions: int,
    temperature: float = 0.1,
    after_llm_call: Callable[[], None] | None = None,
) -> structures.InterviewStep:
    """Run one interview step after the latest user message: ask the next question or build the brief."""

    def called() -> None:
        if after_llm_call is not None:
            after_llm_call()

    # Odpowiedź na ostatnie dozwolone pytanie trafia od razu do briefu.
    if questions_asked >= max_interview_questions:
        case_brief = generate_brief(messages, llm)
        called()
        return structures.InterviewStep(case_brief=case_brief, end_reason="question_limit")

    turn = llm.generate_structured(
        system_prompt=prompts.DOPYTYWATOR_INTERVIEW_SYSTEM_PROMPT,
        user_prompt=prompts.interviewer_user_prompt(
            messages,
            question_number=questions_asked + 1,
            max_interview_questions=max_interview_questions,
        ),
        response_model=structures.InterviewTurnResult,
        temperature=temperature,
        max_output_tokens=DOPYTYWATOR_MAX_OUTPUT_TOKENS,
    )
    called()

    if turn.ready_to_build_brief:
        # Model uznał, że informacje są wystarczające do zbudowania briefu.
        case_brief = generate_brief(messages, llm)
        called()
        return structures.InterviewStep(turn=turn, case_brief=case_brief, end_reason="model_ready")

    return structures.InterviewStep(turn=turn, question=turn.next_question.strip())


def build_brief(
    messages: list[structures.ConversationMessage],
    llm: object,
) -> structures.CaseBrief | None:
    """Build a case brief from a complete conversation and print readiness warnings."""
    # Przerwany (/exit) albo nierozpoczęty wywiad nie ma opisu sprawy, z którego model
    # mógłby zbudować brief bez wymyślania.
    if not any(message.role == "user" for message in messages):
        print("Brak opisu sprawy — CaseBrief nie został utworzony.")
        return None
    case_brief = generate_brief(messages, llm)

    readiness_errors = dopytywator_helper_functions.validate_brief_readiness(case_brief)
    dopytywator_helper_functions.print_case_brief(case_brief)

    if readiness_errors:
        print("\n" + "!" * 72)
        print("BRIEF MA BRAKI W POLACH OBOWIĄZKOWYCH:")
        for error in readiness_errors:
            print(" -", error)
        print("Nie uzupełniam ich wiedzą własną; sprawdź missing_information.")
    else:
        print("\nBrief spełnia minimalne kryteria gotowości Phase 0.")

    return case_brief
