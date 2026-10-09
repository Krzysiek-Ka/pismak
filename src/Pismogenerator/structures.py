from __future__ import annotations

from pydantic import BaseModel, Field


class DraftLetter(BaseModel):
    """Minimal draft letter structure used by the Pismogenerator stage."""

    recipient: str
    sender: str = ""

    request_summary: str
    argumentation_markdown: str

    sources: list[str] = Field(default_factory=list)

    claims_requiring_verification: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    assumptions_used: list[str] = Field(default_factory=list)
