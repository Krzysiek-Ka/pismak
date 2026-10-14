from __future__ import annotations

from src.Zrodlowyszukiwator.rag_retrieval import repository as retrieval_repository
from src.Zrodlowyszukiwator.rag_retrieval.repository import RetrievalRepository, lexical_tsquery


class TestLexicalTsquery:
    def test_lexical_tsquery__joins_words_with_or_so_one_missing_word_does_not_empty_the_result(
        self,
    ) -> None:
        """Verify a multi-word query matches chunks that contain only some of its words."""
        assert lexical_tsquery("bezpieczeństwo pieszych przejścia szkoła") == (
            "bezpieczeńs:* | piesz:* | przejś:* | szkoł:*"
        )

    def test_lexical_tsquery__matches_other_inflected_forms_of_long_words_by_prefix(
        self,
    ) -> None:
        """Verify the query built for one form also covers forms with a different ending."""
        query = lexical_tsquery("rowerowe")

        assert query == "rower:*"
        assert "rowerowych".startswith(query.removesuffix(":*"))

    def test_lexical_tsquery__keeps_short_words_and_numbers_exact(self) -> None:
        """Verify short names and act or year numbers are not widened into prefixes."""
        assert lexical_tsquery("drogi Wrocław 2050 uchwała XXIV/463/25") == (
            "drogi | wrocł:* | 2050 | uchwa:* | xxiv | 463 | 25"
        )

    def test_lexical_tsquery__drops_function_words_that_would_match_almost_every_chunk(
        self,
    ) -> None:
        """Verify Polish connectives and one- or two-letter words never become search terms."""
        assert lexical_tsquery(
            "brak przejścia dla pieszych przy szkole i w pobliżu oraz parku"
        ) == ("brak | przejś:* | piesz:* | szkol:* | pobli:* | parku")

    def test_lexical_tsquery__neutralizes_full_text_operators_in_the_input(self) -> None:
        """Verify text containing tsquery syntax cannot inject operators or break the query."""
        query = lexical_tsquery("rower & !piesi | (tramwaj) <-> 'autobus':*")

        assert query == "rower | piesi | tramw:* | autob:*"

    def test_lexical_tsquery__merges_forms_of_one_word_into_a_single_term(self) -> None:
        """Verify repeated words and different forms sharing a prefix appear only once."""
        assert lexical_tsquery("rowerowe Rowerowe rowerowa trasy trasy") == "rower:* | trasy"

    def test_lexical_tsquery__caps_the_number_of_terms(self) -> None:
        """Verify a long description such as the fallback plan's yields a bounded query."""
        query = lexical_tsquery(" ".join(str(number) for number in range(100, 140)))

        terms = query.split(" | ")
        assert terms[0] == "100"
        assert len(terms) == retrieval_repository.MAX_LEXICAL_TERMS

    def test_lexical_tsquery__returns_nothing_for_text_without_significant_words(self) -> None:
        """Verify a query made only of function words is recognized as empty."""
        assert lexical_tsquery("i w na oraz przez — ...") == ""


class TestLexicalSearch:
    def test_lexical_search__does_not_query_the_database_without_search_terms(
        self, monkeypatch
    ) -> None:
        """Verify an empty query returns nothing instead of sending an empty tsquery."""
        repository = RetrievalRepository("postgresql://unused")

        def fail_to_connect():
            raise AssertionError("the database must not be queried")

        monkeypatch.setattr(repository, "connect", fail_to_connect)

        assert repository.lexical_search(query="i w na oraz") == []
