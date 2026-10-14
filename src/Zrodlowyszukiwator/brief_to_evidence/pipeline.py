from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel

from ..rag_retrieval.hybrid import HybridRetriever
from ..rag_retrieval.models import RetrievalFilters
from ..rag_retrieval.reranker import FALLBACK_REASON, LLMReranker
from .models import Evidence, EvidencePack, EvidenceStatus
from .planner import LLMQueryPlanner


class BriefToEvidencePipeline:
    """
    Główny element do użycia w pipeline Pismaka:

        CaseBrief -> ResearchPlan -> hybrid retrieval -> reranking -> EvidencePack
    """

    def __init__(
        self,
        *,
        planner: LLMQueryPlanner,
        retriever: HybridRetriever,
        reranker: LLMReranker | None = None,
        candidate_k: int = 20,
        evidence_k: int = 6,
        max_per_document: int = 3,
    ) -> None:
        """Initialize the CaseBrief-to-EvidencePack pipeline and its retrieval limits."""
        self.planner = planner
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.evidence_k = evidence_k
        self.max_per_document = max_per_document

    @staticmethod
    def _brief_dict(case_brief: Any) -> dict[str, Any]:
        """Convert a CaseBrief-like value into a JSON-compatible dictionary."""
        if isinstance(case_brief, dict):
            return case_brief
        if isinstance(case_brief, BaseModel):
            return case_brief.model_dump(mode="json")
        raise TypeError("case_brief musi być dict albo Pydantic BaseModel")

    def run(
        self,
        case_brief: Any,
        *,
        filters: RetrievalFilters | None = None,
    ) -> EvidencePack:
        """Plan retrieval, select ranked chunks, and assemble the resulting EvidencePack."""
        brief = self._brief_dict(case_brief)
        plan = self.planner.plan(brief)

        candidates = self.retriever.retrieve(
            plan.intents,
            filters=filters,
            top_k=self.candidate_k,
        )

        status: EvidenceStatus
        if not candidates:
            ranked, status = [], "unsupported"
        elif self.reranker is not None:
            outcome = self.reranker.rerank(
                case_brief=brief,
                candidates=candidates,
            )
            if outcome.assessed:
                # Tylko fragmenty ocenione przez model; nieocenionych nie dopisujemy.
                ranked = outcome.selected
                status = "supported" if ranked else "unsupported"
            else:
                ranked = [(c, FALLBACK_REASON, 0) for c in candidates]
                status = "unverified"
        else:
            # Bez rerankera nikt nie ocenia, czy fragmenty odpowiadają na sprawę: wyszukiwanie
            # zawsze zwraca najbardziej podobne fragmenty, także dla pytań spoza korpusu.
            ranked = [(c, "Wysoka pozycja w hybrid retrieval.", 0) for c in candidates]
            status = "unverified"

        # Limit na dokument działa podczas wybierania ze wszystkich kandydatów. Obcięcie listy
        # do evidence_k przed limitem zmniejszało pakiet (np. 3 z 6 miejsc), gdy czołówkę
        # zajmował jeden dokument, choć dalsi kandydaci z innych dokumentów byli dostępni.
        selected: list[tuple[Any, str, int]] = []
        per_doc: Counter[str] = Counter()
        for chunk, reason, rerank_score in ranked:
            if per_doc[chunk.document_id] >= self.max_per_document:
                continue
            selected.append((chunk, reason, rerank_score))
            per_doc[chunk.document_id] += 1
            if len(selected) >= self.evidence_k:
                break

        evidence: list[Evidence] = []
        matched_purposes: set[str] = set()

        for idx, (chunk, reason, rerank_score) in enumerate(selected, start=1):
            for matched in chunk.matched_intents:
                # "Q1: purpose" -> purpose
                matched_purposes.add(matched.split(": ", 1)[-1])

            evidence.append(
                Evidence(
                    evidence_id=f"E{idx}",
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.title,
                    document_kind=chunk.document_kind,
                    hierarchy_rank=chunk.hierarchy_rank,
                    legal_form=chunk.legal_form,
                    adoption_act_number=chunk.adoption_act_number,
                    adoption_date=chunk.adoption_date,
                    is_active=chunk.is_active,
                    lifecycle_status=chunk.lifecycle_status,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=chunk.section_path,
                    excerpt=chunk.text,
                    source_url=chunk.source_url,
                    source_filename=chunk.source_filename,
                    retrieval_score=chunk.hybrid_score,
                    rerank_score=rerank_score or None,
                    relevance_reason=reason,
                    matched_intents=chunk.matched_intents,
                )
            )

        unsupported = [
            intent.purpose for intent in plan.intents if intent.purpose not in matched_purposes
        ]

        notes = [
            f"Liczba zapytań: {len(plan.intents)}",
            f"Kandydaci po hybrid retrieval: {len(candidates)}",
            f"Źródła w EvidencePack: {len(evidence)}",
            f"Ocena wsparcia źródeł: {status}",
            "hierarchy_rank jest metadanym/tie-breakerem, a nie oceną mocy prawnej.",
        ]

        return EvidencePack(
            query_summary=plan.case_summary,
            research_plan=plan,
            evidence=evidence,
            evidence_status=status,
            unsupported_topics=unsupported,
            search_notes=notes,
        )
