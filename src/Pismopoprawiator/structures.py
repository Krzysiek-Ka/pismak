from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Correction(BaseModel):
    """One language correction proposed by the model; keys follow the format in the prompt."""

    akapit: str
    oryginal: str
    poprawka: str
    kategoria: str
    uzasadnienie: str = ""

    @field_validator("akapit", mode="before")
    @classmethod
    def normalize_paragraph_id(cls, value: object) -> object:
        """Accept "[A3]", "a3" or "3" for paragraph A3 instead of rejecting the whole answer."""
        if not isinstance(value, str | int):
            return value
        text = str(value).strip().strip("[]").strip().upper()
        return f"A{text}" if text.isdigit() else text

    @field_validator("kategoria", mode="before")
    @classmethod
    def normalize_category(cls, value: object) -> object:
        """Compare categories case-insensitively; an unknown one is rejected later by the code."""
        if not isinstance(value, str):
            return value
        category = value.strip().lower()
        return "literówka" if category == "literowka" else category


class CorrectionResponse(BaseModel):
    """Structured answer of the model: the corrections only, never the rewritten letter."""

    poprawki: list[Correction] = Field(default_factory=list)


class Paragraph(BaseModel):
    """One numbered block of the letter body; read-only blocks are only context for the model."""

    id: str
    text: str
    editable: bool
    # Pozycja tekstu akapitu na liście części treści (akapity przeplatane separatorami).
    part_index: int


class RejectedCorrection(BaseModel):
    correction: Correction
    reason: str


class CorrectionResult(BaseModel):
    """Outcome of the language correction; the letter text is always usable."""

    original_text: str
    corrected_text: str
    applied: list[Correction] = Field(default_factory=list)
    rejected: list[RejectedCorrection] = Field(default_factory=list)
    suggestions: list[Correction] = Field(default_factory=list)
    status: Literal["corrected", "unchanged", "reverted", "failed"]
    mode: Literal["whole", "paragraphs"] | None = None
    provider: str | None = None
    model: str | None = None
    context_tokens: int | None = None
    calls: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)
