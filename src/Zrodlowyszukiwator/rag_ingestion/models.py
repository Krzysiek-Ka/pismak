from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentKind(StrEnum):
    """
    Hierarchia MERYTORYCZNA dokumentu.
    Nie mylić z formą prawną (uchwała / zarządzenie).
    """

    STRATEGY = "strategy"
    POLICY = "policy"
    PROGRAM = "program"
    PLAN = "plan"
    STANDARD = "standard"
    PROCEDURE = "procedure"
    OTHER = "other"


class LegalForm(StrEnum):
    """
    Forma przyjęcia / publikacji dokumentu.
    Zarządzenie jest tutaj, a nie w hierarchii merytorycznej.
    """

    RESOLUTION = "resolution"  # uchwała
    ORDINANCE = "ordinance"  # zarządzenie
    REPORT = "report"
    ANNEX = "annex"
    STANDALONE = "standalone"
    UNKNOWN = "unknown"


class LifecycleStatus(StrEnum):
    """
    Miejsce dokumentu w obiegu miejskich dokumentów.
    Uzupełnia is_active: mówi, w jakiej roli dokument jest dziś używany.
    """

    CURRENT = "current"
    CURRENT_REFERENCE = "current_reference"
    REFERENCE = "reference"
    SUPERSEDED = "superseded"  # zastąpiony nowszym dokumentem


HIERARCHY_RANK: dict[DocumentKind, int] = {
    DocumentKind.STRATEGY: 100,
    DocumentKind.POLICY: 80,
    DocumentKind.PROGRAM: 70,
    DocumentKind.PLAN: 70,
    DocumentKind.STANDARD: 60,
    DocumentKind.PROCEDURE: 50,
    DocumentKind.OTHER: 40,
}


class DocumentRelation(BaseModel):
    target_document_id: str
    relation: str
    # np. "implements", "references", "operationalizes", "supersedes"
    note: str | None = None


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    document_kind: DocumentKind
    city: str = "Wrocław"
    issuer: str | None = None
    adopting_body: str | None = None

    legal_form: LegalForm = LegalForm.UNKNOWN
    adoption_act_number: str | None = None
    adoption_date: date | None = None

    valid_from: date | None = None
    valid_to: date | None = None
    is_active: bool | None = None
    lifecycle_status: LifecycleStatus | None = None
    # False wyłącza dokument z wyszukiwania bez usuwania go z bazy.
    retrieval_enabled: bool = True

    source_url: str | None = None
    metadata_url: str | None = None
    language: str = "pl"
    is_annex: bool = False

    tags: list[str] = Field(default_factory=list)
    relations: list[DocumentRelation] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def validate_superseded_is_inactive(self) -> DocumentMetadata:
        """Reject a superseded document that is not also marked as no longer in force."""
        # Wyszukiwanie pomija dokumenty nieobowiązujące na podstawie is_active,
        # więc sprzeczny wpis przepuściłby zastąpiony dokument do pism.
        if self.lifecycle_status is LifecycleStatus.SUPERSEDED and self.is_active is not False:
            raise ValueError("lifecycle_status=superseded wymaga is_active=false")
        return self

    @property
    def hierarchy_rank(self) -> int:
        """Return the configured content hierarchy rank for this document kind."""
        return HIERARCHY_RANK[self.document_kind]


class ExtractedPage(BaseModel):
    page_number: int
    text: str
    char_count: int
    low_text_quality: bool = False
    # True, gdy na stronie przywrócono „ż/Ż” zepsute przez mapowanie fontu (pdf_extractor).
    legacy_glyphs_repaired: bool = False
    # Powtórzenia nagłówków i stopek stron usunięte z tekstu (pdf_extractor).
    removed_running_lines: list[str] = Field(default_factory=list)


class TextUnit(BaseModel):
    page_number: int
    section_path: list[str] = Field(default_factory=list)
    text: str


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    page_start: int
    page_end: int
    section_path: list[str] = Field(default_factory=list)
    text: str
    embedding_text: str
    text_sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)
