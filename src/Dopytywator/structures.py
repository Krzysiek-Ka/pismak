# 3. Modele danych Phase 0
from datetime import datetime
from enum import StrEnum
from typing import Literal

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
