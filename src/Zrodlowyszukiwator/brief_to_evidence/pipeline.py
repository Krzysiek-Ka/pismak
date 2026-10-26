from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from ..rag_retrieval.hybrid import HybridRetriever
from ..rag_retrieval.models import RetrievalFilters
from ..rag_retrieval.reranker import FALLBACK_REASON, LLMReranker
from .models import Evidence, EvidencePack, EvidenceStatus, ResearchPlan
from .planner import LLMQueryPlanner

T = TypeVar("T")
LOGGER = logging.getLogger("pismak")
# Fraza przedmiotu postulatu obecna w większej części fragmentów korpusu nie jest kotwicą.
MAX_CONCEPT_SHARE = 0.05


def limit_per_document(
    items: list[T], *, per_document: int, limit: int, document: Callable[[T], str]
) -> list[T]:
    """Take items in order, at most per_document from one document, until limit are taken."""
    taken: list[T] = []
    per_doc: Counter[str] = Counter()
    for item in items:
        if per_doc[document(item)] >= per_document:
            continue
        taken.append(item)
        per_doc[document(item)] += 1
        if len(taken) >= limit:
            break
    return taken


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
        max_candidates_per_document: int = 8,
    ) -> None:
        """Initialize the CaseBrief-to-EvidencePack pipeline and its retrieval limits."""
        self.planner = planner
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.evidence_k = evidence_k
        self.max_per_document = max_per_document
        # Pakiet bierze najwyżej max_per_document fragmentów z dokumentu, więc reranker nie
        # potrzebuje ich więcej niż kilka razy tyle; resztę miejsc dostają inne dokumenty.
        self.max_candidates_per_document = max_candidates_per_document

    def _support_concepts(self, plan: ResearchPlan) -> dict[int, list[str]]:
        """Return each postulate's concept phrases, without those common across the corpus."""
        # Fraza obecna w wielu fragmentach („rower”, „transport zbiorowy”) jest tematem, a nie
        # przedmiotem postulatu: przepuściłaby sprawy spoza korpusu (plan, punkt 4.1).
        repository = getattr(self.retriever, "repository", None)
        concepts: dict[int, list[str]] = {}
        for item in plan.key_concepts:
            for phrase in dict.fromkeys(p.strip() for p in item.concepts if p.strip()):
                if repository is not None:
                    try:
                        share = repository.concept_share(phrase)
                    except Exception:
                        LOGGER.warning("concept_share_failed phrase_chars=%d", len(phrase))
                        share = 0.0
                    if share is None or share > MAX_CONCEPT_SHARE:
                        continue
                concepts.setdefault(item.postulate, []).append(phrase)
        return concepts

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

        # Cały ranking, żeby limit na dokument mógł sięgnąć po kandydatów z dalszych pozycji.
        # Strategia 2050 (437 fragmentów) zajmowała 13 z 20 miejsc i wypierała Politykę
        # mobilności (45) z listy dla rerankera (złoty zbiór, 2026-10-24).
        ranked_chunks = self.retriever.retrieve(plan.intents, filters=filters, top_k=None)
        candidates = limit_per_document(
            ranked_chunks,
            per_document=self.max_candidates_per_document,
            limit=self.candidate_k,
            document=lambda chunk: chunk.document_id,
        )
        key_concepts = self._support_concepts(plan) if self.reranker is not None else {}

        status: EvidenceStatus
        if not candidates:
            ranked, status = [], "unsupported"
        elif self.reranker is not None:
            outcome = self.reranker.rerank(
                case_brief=brief,
                candidates=candidates,
                key_concepts=key_concepts,
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
        selected = limit_per_document(
            ranked,
            per_document=self.max_per_document,
            limit=self.evidence_k,
            document=lambda item: item[0].document_id,
        )

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
        if key_concepts:
            notes.append(
                "Przedmiot postulatów: "
                + "; ".join(f"{n}: {', '.join(p)}" for n, p in sorted(key_concepts.items()))
            )

        return EvidencePack(
            query_summary=plan.case_summary,
            research_plan=plan,
            evidence=evidence,
            evidence_status=status,
            unsupported_topics=unsupported,
            search_notes=notes,
        )
