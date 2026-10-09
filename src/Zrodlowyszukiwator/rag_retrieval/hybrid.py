from __future__ import annotations

import logging
from collections import defaultdict
from time import perf_counter
from typing import Any

from .models import RetrievalFilters, RetrievedChunk, SearchIntent
from .repository import RetrievalRepository

LOGGER = logging.getLogger("pismak")


class HybridRetriever:
    """
    Łączy semantic search i PostgreSQL full-text search przez RRF.

    RRF jest celowo użyty zamiast ręcznego normalizowania similarity i ts_rank,
    ponieważ te dwa wyniki mają różne skale.
    """

    def __init__(
        self,
        *,
        repository: RetrievalRepository,
        embedder: Any,
        vector_limit: int = 20,
        lexical_limit: int = 20,
        rrf_k: int = 60,
        session_id: str | None = None,
    ) -> None:
        """Initialize the hybrid retriever with its repository and ranking limits."""
        self.repository = repository
        self.embedder = embedder
        self.vector_limit = vector_limit
        self.lexical_limit = lexical_limit
        self.rrf_k = rrf_k
        self.session_id = session_id

    def retrieve(
        self,
        intents: list[SearchIntent],
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Merge weighted semantic and lexical results with reciprocal rank fusion."""
        if not intents:
            return []

        started = perf_counter()
        semantic_queries = [i.semantic_query for i in intents]
        embeddings = self.embedder.embed(semantic_queries)

        by_id: dict[str, RetrievedChunk] = {}
        rrf_scores: dict[str, float] = defaultdict(float)
        matched: dict[str, set[str]] = defaultdict(set)

        for intent_idx, (intent, emb) in enumerate(zip(intents, embeddings)):
            weight = {1: 0.8, 2: 1.0, 3: 1.2}[intent.priority]
            intent_label = f"Q{intent_idx + 1}: {intent.purpose}"

            semantic = self.repository.vector_search(
                query_embedding=emb,
                limit=self.vector_limit,
                filters=filters,
            )
            lexical = self.repository.lexical_search(
                query=intent.lexical_query,
                limit=self.lexical_limit,
                filters=filters,
            )

            for rank, item in enumerate(semantic, start=1):
                existing = by_id.get(item.chunk_id)
                if existing is None:
                    by_id[item.chunk_id] = item
                else:
                    if item.semantic_score is not None:
                        existing.semantic_score = max(
                            existing.semantic_score or float("-inf"),
                            item.semantic_score,
                        )
                rrf_scores[item.chunk_id] += weight / (self.rrf_k + rank)
                matched[item.chunk_id].add(intent_label)

            for rank, item in enumerate(lexical, start=1):
                existing = by_id.get(item.chunk_id)
                if existing is None:
                    by_id[item.chunk_id] = item
                else:
                    if item.lexical_score is not None:
                        existing.lexical_score = max(
                            existing.lexical_score or 0.0,
                            item.lexical_score,
                        )
                rrf_scores[item.chunk_id] += weight / (self.rrf_k + rank)
                matched[item.chunk_id].add(intent_label)

        for chunk_id, item in by_id.items():
            item.hybrid_score = rrf_scores[chunk_id]
            item.matched_intents = sorted(matched[chunk_id])

        # hierarchy_rank jest tylko tie-breakerem. Strategia nie powinna automatycznie
        # wygrywać z bardziej konkretnym standardem czy programem.
        ranked = sorted(
            by_id.values(),
            key=lambda x: (x.hybrid_score, x.hierarchy_rank, x.chunk_id),
            reverse=True,
        )
        LOGGER.info(
            "stage_completed stage=retrieval session_id=%s intents=%d candidates=%d elapsed_seconds=%.3f",
            self.session_id or "-",
            len(intents),
            len(ranked),
            perf_counter() - started,
        )
        return ranked[:top_k]
