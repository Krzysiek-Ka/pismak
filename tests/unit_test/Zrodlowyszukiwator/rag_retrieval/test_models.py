"""Tests for the retrieval data models."""

from __future__ import annotations

import pytest

from src.Zrodlowyszukiwator.rag_retrieval.models import (
    SearchIntent,
)


def intent(purpose: str, *, priority: int = 2) -> SearchIntent:
    return SearchIntent(
        semantic_query=f"semantyczne {purpose}",
        lexical_query=f"leksykalne {purpose}",
        purpose=purpose,
        priority=priority,
    )


class TestSearchIntent:
    @pytest.mark.parametrize("priority", [0, 4])
    def test_search_intent__priority_outside_one_to_three_is_rejected(self, priority: int) -> None:
        """Verify a planner answer with an unknown priority cannot reach the weight table."""
        with pytest.raises(ValueError, match="priority"):
            intent("sprawa", priority=priority)
