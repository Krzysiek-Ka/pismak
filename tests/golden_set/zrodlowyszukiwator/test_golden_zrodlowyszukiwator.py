"""
Złoty zbiór Źródłowyszukiwacza: CaseBrief → plan zapytań → wyszukiwanie hybrydowe → reranker.

Model (planner i reranker), baza RAG, embeddingi i filtry są prawdziwe; pipeline składa
`services.build_zrodlowyszukiwator`, czyli ten sam kod, który wywołuje src/Wyswietlator/app.py.
Na każdy dokument korpusu przypadają trzy przypadki: odpowiedź wprost w dokumencie, odpowiedź
z połączenia kilku jego fragmentów i sprawa spoza korpusu. Nieobowiązująca Strategia 2030 ma
przypadki, w których nie może trafić do pakietu. Każde pytanie ewaluacji retrievalu
(tests/golden_set/rag_eval) ma tu jeden odpowiednik: tę samą sprawę zapisaną jako CaseBrief
(`rag_eval_<id>`), ocenianą tymi samymi dowodami co w rag_eval. Na modelu uruchamia się
tylko 10 z nich (RUN_CASE_IDS); pozostałe przypadki przechodzą jedynie kontrole zbioru.

Zawsze sprawdzane: schemat EvidencePack, numeracja E1…En, limity pakietu (6 fragmentów, 3 z
dokumentu), fragmenty identyczne z tekstem w bazie i brak dokumentów jawnie nieobowiązujących.
Obsługiwane pola `expected` (SUPPORTED_EXPECTED):
- expected_source_ids: każdy dokument jest w pakiecie, a pakiet ma ocenę `supported`,
- min_fragments_from_expected: najmniejsza liczba fragmentów z tych dokumentów,
- expect_no_relevant_source: pakiet pusty z oceną `unsupported`,
- forbidden_source_ids: żaden z tych dokumentów nie jest w pakiecie,
- rag_eval_question: pytanie rag_eval, z którego pochodzi sprawa; pakiet zawiera co najmniej
  jeden jego dowód (fragment z wszystkimi frazami `all_of`, jak w rag_eval).
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from src.Dopytywator.structures import CaseBrief
from src.Wyswietlator import services
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack
from src.Zrodlowyszukiwator.rag_retrieval.repository import RetrievalRepository
from tests.golden_set.golden_support import (
    GoldenCase,
    Provider,
    assert_passed,
    database_skip_reason,
    embeddings_skip_reason,
    golden_run,
    llm_client,
    load_cases,
    provider_skip_reason,
    runs,
    skip_run,
)
from tests.golden_set.rag_eval.rag_eval_support import load_questions, supported_items

pytestmark = pytest.mark.golden

MODULE = "zrodlowyszukiwator"
MODULE_DIR = Path(__file__).resolve().parent
SUPPORTED_EXPECTED = {
    "expected_source_ids",
    "min_fragments_from_expected",
    "expect_no_relevant_source",
    "forbidden_source_ids",
    "rag_eval_question",
}
CASES = load_cases(MODULE_DIR, supported_expected=SUPPORTED_EXPECTED)
RAG_EVAL_QUESTIONS = {question.id: question for question in load_questions()}
CASE_KINDS = ("bezpośrednio", "łączenie informacji", "poza korpusem")
RAG_EVAL_KIND = "pytanie rag_eval"
# Na modelu uruchamiane jest tylko 10 odpowiedników pytań rag_eval (decyzja 2026-10-25): 4 bez
# odpowiedzi w korpusie i 6 z odpowiedzią, w tym znane chybienia Polityki mobilności. Pozostałe
# przypadki zostają w cases/ i przechodzą kontrole zbioru; uruchamia je dopiero
# PISMAK_ZRODLOWYSZUKIWATOR_ALL_CASES=1.
RUN_CASE_IDS = (
    "rag_eval_bike_01",
    "rag_eval_bike_02",
    "rag_eval_bike_03",
    "rag_eval_bike_07",
    "rag_eval_bike_08",
    "rag_eval_mob_02",
    "rag_eval_mob_03",
    "rag_eval_mob_06",
    "rag_eval_mob_07",
    "rag_eval_mob_08",
)
RUN_CASES = (
    CASES
    if os.getenv("PISMAK_ZRODLOWYSZUKIWATOR_ALL_CASES") == "1"
    else [case for case in CASES if case.id in RUN_CASE_IDS]
)
EVIDENCE_K = 6
MAX_PER_DOCUMENT = 3


class TestGoldenCases:
    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__input_is_a_valid_case_brief(self, case: GoldenCase) -> None:
        """Verify every case is a brief the Dopytywator could have produced."""
        assert CaseBrief.model_validate(case.input).requested_actions

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
    def test_cases__expect_either_sources_or_abstention(self, case: GoldenCase) -> None:
        """Verify no case asks for both a document and an empty pack."""
        expected = case.expected
        assert expected, "przypadek bez oczekiwań"
        assert not (
            expected.get("expected_source_ids") and expected.get("expect_no_relevant_source")
        )

    def test_cases__every_catalog_document_has_its_three_cases(self) -> None:
        """Verify the standard's three cases per corpus document: direct, combined, not covered."""
        from src.Zrodlowyszukiwator.rag_ingestion.catalog import load_catalog
        from src.Zrodlowyszukiwator.rag_ingestion.cli import DEFAULT_CATALOG_PATH

        kinds = Counter(case_label(case) for case in CASES)
        assert {kind for kind, _ in kinds} <= {*CASE_KINDS, RAG_EVAL_KIND}
        for metadata in load_catalog(DEFAULT_CATALOG_PATH).values():
            for kind in CASE_KINDS:
                assert kinds[(kind, metadata.document_id)] == 1, (metadata.document_id, kind)

    def test_cases__run_cases_are_ten_existing_rag_eval_cases(self) -> None:
        """Verify the model runs exactly the chosen rag_eval mirrors, with and without answers."""
        by_id = {case.id: case for case in CASES}
        chosen = [by_id[case_id] for case_id in RUN_CASE_IDS]

        assert len(set(RUN_CASE_IDS)) == 10
        assert all(case_label(case)[0] == RAG_EVAL_KIND for case in chosen)
        assert any(case.expected.get("expect_no_relevant_source") for case in chosen)
        assert any(case.expected.get("expected_source_ids") for case in chosen)

    def test_cases__every_rag_eval_question_has_one_case(self) -> None:
        """Verify each rag_eval question is mirrored once, by a case labelled as its mirror."""
        mirrored = Counter(case.expected.get("rag_eval_question") for case in CASES)
        mirrored.pop(None, None)
        assert mirrored == dict.fromkeys(RAG_EVAL_QUESTIONS, 1)
        for case in CASES:
            assert (case_label(case)[0] == RAG_EVAL_KIND) == ("rag_eval_question" in case.expected)

    @pytest.mark.parametrize(
        "case", [c for c in CASES if "rag_eval_question" in c.expected], ids=lambda case: case.id
    )
    def test_cases__rag_eval_case_expects_what_its_question_expects(self, case: GoldenCase) -> None:
        """Verify a mirror wants the question's document, or abstention when it has no answer."""
        question = RAG_EVAL_QUESTIONS[case.expected["rag_eval_question"]]

        assert case_label(case)[1] == question.document_id
        if question.answerable:
            assert set(case.expected["expected_source_ids"]) <= question.source_document_ids
            assert not case.expected.get("expect_no_relevant_source")
        else:
            assert case.expected.get("expect_no_relevant_source") is True


def case_label(case: GoldenCase) -> tuple[str, str]:
    """Return the (kind, document id) from the „[kind, document]” start of the description."""
    label = case.description.split("]")[0].strip("[")
    kind, document = (part.strip() for part in label.split(","))
    return kind, document


def indexed_chunks(chunk_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Return document id and text of the given chunks, read from the RAG database."""
    repository = RetrievalRepository(os.environ["DATABASE_URL"])
    with repository.connect() as conn:
        rows = conn.execute(
            "SELECT chunk_id, document_id, text FROM rag_chunks WHERE chunk_id = ANY(%s)",
            [chunk_ids],
        ).fetchall()
    return {chunk_id: (document_id, text) for chunk_id, document_id, text in rows}


def indexed_documents() -> set[str]:
    repository = RetrievalRepository(os.environ["DATABASE_URL"])
    with repository.connect() as conn:
        rows = conn.execute(
            "SELECT document_id FROM rag_documents WHERE retrieval_enabled"
        ).fetchall()
    return {row[0] for row in rows}


def check_pack(run, case: GoldenCase, pack: EvidencePack) -> None:
    """Record the contract checks of a pack and the case's expected fields."""
    evidence = pack.evidence
    documents = [e.document_id for e in evidence]
    run.check("schema_valid", EvidencePack.model_validate_json(pack.model_dump_json()) == pack)
    ids = [e.evidence_id for e in evidence]
    run.check("evidence_numbering", ids == [f"E{i}" for i in range(1, len(ids) + 1)], str(ids))
    per_document = Counter(documents)
    run.check(
        "pack_limits",
        len(evidence) <= EVIDENCE_K and all(n <= MAX_PER_DOCUMENT for n in per_document.values()),
        f"{len(evidence)} fragmentów, {dict(per_document)}",
    )
    stored = indexed_chunks([e.chunk_id for e in evidence])
    changed = [
        e.evidence_id for e in evidence if stored.get(e.chunk_id) != (e.document_id, e.excerpt)
    ]
    run.check("excerpts_match_index", not changed, f"inne niż w bazie: {changed}")
    inactive = [e.document_id for e in evidence if e.is_active is False]
    run.check("no_inactive_documents", not inactive, str(inactive))

    expected = case.expected
    for document_id in expected.get("expected_source_ids", []):
        run.check(f"expected_source:{document_id}", document_id in documents, str(documents))
    if expected.get("expected_source_ids"):
        run.check("status_supported", pack.evidence_status == "supported", pack.evidence_status)
    if "min_fragments_from_expected" in expected:
        wanted = set(expected.get("expected_source_ids", []))
        count = sum(document in wanted for document in documents)
        run.check(
            "min_fragments_from_expected",
            count >= expected["min_fragments_from_expected"],
            f"{count} fragmentów",
        )
    if expected.get("expect_no_relevant_source"):
        run.check(
            "no_relevant_source",
            not evidence and pack.evidence_status == "unsupported",
            f"status {pack.evidence_status}, dokumenty {documents}",
        )
    for document_id in expected.get("forbidden_source_ids", []):
        run.check(f"forbidden_source:{document_id}", document_id not in documents, str(documents))
    question = RAG_EVAL_QUESTIONS.get(expected.get("rag_eval_question"))
    if question is not None and question.evidence:
        covered = sorted(
            {
                label
                for e in evidence
                for label in supported_items(question, e.document_id, e.excerpt)
            }
        )
        labels = [item.label for item in question.evidence]
        run.details["rag_eval_evidence"] = {
            "question": question.id,
            "covered": covered,
            "recall": round(len(covered) / len(labels), 3),
        }
        run.check("rag_eval_evidence", bool(covered), f"pokryte {covered} z {labels}")


@pytest.mark.parametrize(("provider", "case"), runs(RUN_CASES))
def test_golden_zrodlowyszukiwator(provider: Provider, case: GoldenCase) -> None:
    """Verify the real pipeline finds the case's document, or abstains when the corpus is silent.

    The case file describes the edge scenario; the checks are listed in the module docstring.
    """
    reason = provider_skip_reason(provider) or database_skip_reason() or embeddings_skip_reason()
    if not reason:
        missing = set(case.expected.get("expected_source_ids", [])) - indexed_documents()
        reason = f"dokumenty spoza indeksu: {sorted(missing)}" if missing else None
    if reason:
        skip_run(MODULE, MODULE_DIR, provider, case, reason)

    with golden_run(MODULE, MODULE_DIR, provider, case) as run:
        pipeline, documents, chunks = services.build_zrodlowyszukiwator(
            client=llm_client(provider, timeout_seconds=case.limits.get("timeout_seconds")),
            model=provider.model,
            session_id=run.session_id,
            provider=provider.provider,
        )
        try:
            pack = pipeline.run(CaseBrief.model_validate(case.input))
        finally:
            run.model = (
                ", ".join(
                    dict.fromkeys(
                        m for m in (pipeline.planner.last_model, pipeline.reranker.last_model) if m
                    )
                )
                or None
            )
        run.details.update(
            index={"documents": documents, "chunks": chunks},
            research_plan=pack.research_plan.model_dump(mode="json"),
            evidence_status=pack.evidence_status,
            evidence=[
                {
                    "evidence_id": e.evidence_id,
                    "document_id": e.document_id,
                    "pages": f"{e.page_start}-{e.page_end}",
                    "rerank_score": e.rerank_score,
                    "relevance_reason": e.relevance_reason,
                    "excerpt_start": e.excerpt[:160],
                }
                for e in pack.evidence
            ],
            unsupported_topics=pack.unsupported_topics,
            search_notes=pack.search_notes,
        )
        check_pack(run, case, pack)

    assert_passed(run)
