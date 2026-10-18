# 3. Modele danych Phase 0
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=datetime.now)


class ClaimStatus(StrEnum):
    USER_STATEMENT = "user_statement"
    USER_OPINION = "user_opinion"
    USER_ASSUMPTION = "user_assumption"
    SUPPORTED_BY_USER_EVIDENCE = "supported_by_user_evidence"
    REQUIRES_VERIFICATION = "requires_verification"


class Claim(BaseModel):
    statement: str
    status: ClaimStatus
    evidence_description: str | None = None


class CaseBrief(BaseModel):
    title: str
    problem_description: str
    location: str | None = None
    time_context: str | None = None

    desired_outcome: str
    requested_actions: list[str]

    institution: str | None = None
    addressee: str | None = None

    affected_groups: list[str] = Field(default_factory=list)
    public_interest_arguments: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    available_evidence: list[str] = Field(default_factory=list)
    previous_actions: list[str] = Field(default_factory=list)
    likely_counterarguments: list[str] = Field(default_factory=list)

    missing_information: list[str] = Field(default_factory=list)
    facts_to_verify: list[str] = Field(default_factory=list)

    preferred_tone: Literal["neutralny", "stanowczy", "koncyliacyjny", "ekspercki"] = "neutralny"

    user_notes: list[str] = Field(default_factory=list)


class InterviewTurnResult(BaseModel):
    short_summary: str
    next_question: str | None
    reason_for_question: str | None
    missing_topics: list[str] = Field(default_factory=list)
    ready_to_build_brief: bool

    @model_validator(mode="before")
    @classmethod
    def normalize_question_state(cls, data: Any) -> Any:
        """Resolve the contradictions small models often return instead of asking them again."""
        # Schemat JSON nie wyraża zależności między polami, więc gramatyka modelu jej nie
        # wymusi. Każde ponowne zapytanie to na CPU kolejne pół minuty.
        if not isinstance(data, dict):
            return data
        question = data.get("next_question")
        if isinstance(question, str) and question.strip():
            return data
        data = dict(data)
        if data.get("ready_to_build_brief") is False:
            # Model nie ma o co zapytać, więc wywiad może się zakończyć.
            data["ready_to_build_brief"] = True
        if data.get("ready_to_build_brief") is True:
            # Pusty tekst zamiast null przy zakończonym wywiadzie.
            data["next_question"] = None
        return data

    @model_validator(mode="after")
    def validate_question_state(self) -> "InterviewTurnResult":
        """Ensure the next-question fields match whether the interview is complete."""
        if self.ready_to_build_brief and self.next_question is not None:
            raise ValueError("next_question musi być None, gdy ready_to_build_brief=True")
        if not self.ready_to_build_brief:
            if self.next_question is None or not self.next_question.strip():
                raise ValueError(
                    "next_question musi zawierać pytanie, gdy ready_to_build_brief=False"
                )
        return self


class InterviewStep(BaseModel):
    """Outcome of one interview step: the next question, or the brief and why the interview ended."""

    # None, gdy brief powstał po osiągnięciu limitu pytań bez pytania modelu o kolejną turę.
    turn: InterviewTurnResult | None = None
    question: str | None = None
    case_brief: CaseBrief | None = None
    end_reason: Literal["model_ready", "question_limit"] | None = None
