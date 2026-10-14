from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..rag_retrieval.models import SearchIntent

# Czy znalezione fragmenty rzeczywiście popierają sprawę:
# - supported: reranker ocenił kandydatów i wybrał wspierające fragmenty,
# - unsupported: reranker ocenił kandydatów i żaden nie odpowiada na sprawę (albo nic nie znaleziono),
# - unverified: nikt nie ocenił trafności (brak rerankera albo jego fallback po błędzie modelu).
# Wyszukiwanie wektorowe zawsze zwraca podobne fragmenty, więc sama obecność wyników
# nie jest dowodem, że korpus odpowiada na sprawę.
EvidenceStatus = Literal["supported", "unsupported", "unverified"]


class ResearchPlan(BaseModel):
    case_summary: str
    intents: list[SearchIntent] = Field(default_factory=list)


class Evidence(BaseModel):
    evidence_id: str
    chunk_id: str
    document_id: str
    document_title: str
    document_kind: str
    hierarchy_rank: int
    legal_form: str
    adoption_act_number: str | None = None
    adoption_date: str | None = None
    is_active: bool | None = None
    lifecycle_status: str | None = None

    page_start: int
    page_end: int
    section_path: list[str] = Field(default_factory=list)
    excerpt: str

    source_url: str | None = None
    source_filename: str | None = None

    retrieval_score: float
    rerank_score: int | None = None
    relevance_reason: str
    matched_intents: list[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    query_summary: str
    research_plan: ResearchPlan
    evidence: list[Evidence] = Field(default_factory=list)
    evidence_status: EvidenceStatus = "unverified"
    unsupported_topics: list[str] = Field(default_factory=list)
    search_notes: list[str] = Field(default_factory=list)
