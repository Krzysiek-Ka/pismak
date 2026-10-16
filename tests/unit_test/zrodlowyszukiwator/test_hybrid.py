"""Tests for merging vector and full-text results with reciprocal rank fusion."""

from __future__ import annotations

import pytest

from src.Zrodlowyszukiwator.rag_retrieval.hybrid import HybridRetriever
from src.Zrodlowyszukiwator.rag_retrieval.models import (
    RetrievalFilters,
    RetrievedChunk,
    SearchIntent,
)


def chunk(
    chunk_id: str,
    *,
    hierarchy_rank: int = 80,
    semantic: float | None = None,
    lexical: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title="Dokument",
        document_kind="policy",
        hierarchy_rank=hierarchy_rank,
        legal_form="resolution",
        page_start=1,
        page_end=1,
        text="Tekst",
        semantic_score=semantic,
        lexical_score=lexical,
    )


def intent(purpose: str, *, priority: int = 2) -> SearchIntent:
    return SearchIntent(
        semantic_query=f"semantyczne {purpose}",
        lexical_query=f"leksykalne {purpose}",
        purpose=purpose,
        priority=priority,
    )


class IndexEmbedder:
    """Embed the n-th query as [n], so the repository can tell which intent is searching."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(index)] for index in range(len(texts))]


class ScriptedRepository:
    """Return per-intent result lists and remember the filters each search received."""

    def __init__(
        self,
        semantic: list[list[RetrievedChunk]],
        lexical: list[list[RetrievedChunk]] | None = None,
    ) -> None:
        self.semantic = semantic
        self.lexical = lexical or [[] for _ in semantic]
        self.lexical_queries: list[str] = []
        self.filters: list[RetrievalFilters | None] = []

    def vector_search(self, *, query_embedding, limit, filters):
        self.filters.append(filters)
        return [item.model_copy() for item in self.semantic[int(query_embedding[0])]]

    def lexical_search(self, *, query, limit, filters):
        self.filters.append(filters)
        self.lexical_queries.append(query)
        return [item.model_copy() for item in self.lexical[len(self.lexical_queries) - 1]]


def retrieve(repository: ScriptedRepository, intents: list[SearchIntent], **kwargs):
    retriever = HybridRetriever(repository=repository, embedder=IndexEmbedder())
    return retriever.retrieve(intents, **kwargs)


class TestRetrieve:
    def test_retrieve__embeds_all_semantic_queries_in_one_call_in_intent_order(self) -> None:
        """Verify one batch request to Ollama serves every intent, keeping vectors aligned."""
        embedder = IndexEmbedder()
        retriever = HybridRetriever(
            repository=ScriptedRepository([[], []]),
            embedder=embedder,
        )

        retriever.retrieve([intent("a"), intent("b")])

        assert embedder.calls == [["semantyczne a", "semantyczne b"]]

    def test_retrieve__ranks_a_high_priority_intent_above_a_low_priority_one(self) -> None:
        """Verify the planner's priority changes the order when both hits rank first."""
        repository = ScriptedRepository([[chunk("low")], [chunk("high")]])

        result = retrieve(repository, [intent("tło", priority=1), intent("sedno", priority=3)])

        assert [item.chunk_id for item in result] == ["high", "low"]

    def test_retrieve__rewards_a_chunk_found_by_both_searches(self) -> None:
        """Verify agreement between vector and full-text search lifts a fragment to the top."""
        repository = ScriptedRepository(
            semantic=[[chunk("only-vector"), chunk("both")]],
            lexical=[[chunk("both", lexical=0.4)]],
        )

        result = retrieve(repository, [intent("sprawa")])

        assert [item.chunk_id for item in result] == ["both", "only-vector"]
        assert result[0].lexical_score == 0.4

    def test_retrieve__breaks_score_ties_by_hierarchy_and_then_chunk_id(self) -> None:
        """Verify equal scores give a stable order with the higher-ranked document first."""
        repository = ScriptedRepository(
            [[chunk("b", hierarchy_rank=60)], [chunk("a", hierarchy_rank=100)], [chunk("c")]]
        )

        result = retrieve(repository, [intent("x"), intent("y"), intent("z")])

        assert [item.chunk_id for item in result] == ["a", "c", "b"]
        equal = retrieve(
            ScriptedRepository([[chunk("a")], [chunk("b")]]), [intent("x"), intent("y")]
        )
        assert [item.chunk_id for item in equal] == ["b", "a"]

    def test_retrieve__keeps_the_best_scores_of_a_chunk_found_by_several_intents(self) -> None:
        """Verify a fragment matching two intents keeps its strongest similarity and both labels."""
        repository = ScriptedRepository(
            semantic=[[chunk("shared", semantic=0.4)], [chunk("shared", semantic=0.9)]],
            lexical=[[chunk("shared", lexical=0.7)], [chunk("shared", lexical=0.2)]],
        )

        [result] = retrieve(repository, [intent("przejście"), intent("szkoła")])

        assert result.semantic_score == 0.9
        assert result.lexical_score == 0.7
        assert result.matched_intents == ["Q1: przejście", "Q2: szkoła"]
        single = retrieve(ScriptedRepository([[chunk("shared")]]), [intent("przejście")])
        assert result.hybrid_score > single[0].hybrid_score

    def test_retrieve__keeps_no_semantic_score_for_a_chunk_found_only_by_text(self) -> None:
        """Verify a missing similarity is not reported as a real score of zero."""
        repository = ScriptedRepository(semantic=[[]], lexical=[[chunk("text-only", lexical=0.3)]])

        [result] = retrieve(repository, [intent("sprawa")])

        assert result.semantic_score is None
        assert result.lexical_score == 0.3

    def test_retrieve__returns_only_the_best_top_k_chunks(self) -> None:
        """Verify the candidate list sent to the reranker respects its size limit."""
        repository = ScriptedRepository([[chunk(f"c{index}") for index in range(5)]])

        result = retrieve(repository, [intent("sprawa")], top_k=2)

        assert [item.chunk_id for item in result] == ["c0", "c1"]

    def test_retrieve__passes_the_same_filters_to_both_searches(self) -> None:
        """Verify an out-of-force document cannot return through the full-text path."""
        filters = RetrievalFilters(active_only=True)
        repository = ScriptedRepository([[]])

        retrieve(repository, [intent("sprawa")], filters=filters)

        assert repository.filters == [filters, filters]
        assert repository.lexical_queries == ["leksykalne sprawa"]

    def test_retrieve__returns_nothing_when_no_search_finds_anything(self) -> None:
        """Verify an empty corpus answer is an empty list, not an error."""
        assert retrieve(ScriptedRepository([[], []]), [intent("a"), intent("b")]) == []

    @pytest.mark.parametrize("priority", [0, 4])
    def test_retrieve__intent_priority_outside_one_to_three_is_rejected_by_the_model(
        self, priority: int
    ) -> None:
        """Verify a planner answer with an unknown priority cannot reach the weight table."""
        with pytest.raises(ValueError, match="priority"):
            intent("sprawa", priority=priority)
