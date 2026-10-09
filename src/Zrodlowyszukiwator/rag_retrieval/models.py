from __future__ import annotations

from pydantic import BaseModel, Field


class SearchIntent(BaseModel):
    """Jedno zapytanie do RAG wygenerowane na podstawie CaseBrief."""

    semantic_query: str
    lexical_query: str
    purpose: str
    priority: int = Field(default=2, ge=1, le=3)


class RetrievalFilters(BaseModel):
    """Opcjonalne filtry. Domyślnie retrieval niczego agresywnie nie odcina."""

    city: str | None = None
    document_kinds: list[str] = Field(default_factory=list)
    active_only: bool = False


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    document_kind: str
    hierarchy_rank: int
    legal_form: str
    adoption_act_number: str | None = None
    adoption_date: str | None = None
    is_active: bool | None = None
    source_url: str | None = None
    source_filename: str | None = None

    page_start: int
    page_end: int
    section_path: list[str] = Field(default_factory=list)
    text: str

    semantic_score: float | None = None
    lexical_score: float | None = None
    hybrid_score: float = 0.0
    matched_intents: list[str] = Field(default_factory=list)
