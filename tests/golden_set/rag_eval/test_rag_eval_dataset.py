"""Testy zbioru pytań ewaluacyjnych i funkcji oceny (marker `golden`). Nie wymagają bazy ani Ollamy."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from src.Zrodlowyszukiwator.rag_ingestion.catalog import load_catalog
from src.Zrodlowyszukiwator.rag_ingestion.cli import DEFAULT_CATALOG_PATH
from src.Zrodlowyszukiwator.rag_ingestion.pdf_extractor import extract_pdf_pages
from tests.golden_set.rag_eval.rag_eval_support import (
    PROJECT_ROOT,
    RAW_DIR,
    EvalQuestion,
    EvalRun,
    Hit,
    QuestionResult,
    catalog_by_document_id,
    evidence_precision,
    evidence_rank,
    evidence_recall,
    headline_rows,
    load_questions,
    metadata_problems,
    normalize_for_matching,
    page_attribution_problems,
    reciprocal_rank,
    source_rank,
    strip_diacritics,
)

pytestmark = pytest.mark.golden

QUESTIONS = load_questions()
CATALOG = load_catalog(DEFAULT_CATALOG_PATH)
FILE_BY_DOCUMENT = {meta.document_id: filename for filename, meta in CATALOG.items()}
ANSWERABLE = [q for q in QUESTIONS if q.answerable]
UNANSWERABLE = [q for q in QUESTIONS if not q.answerable]
_PAGES: dict[str, list[str]] = {}


def committed_document_ids() -> set[str]:
    """Return catalog documents whose PDF is tracked by git in the RAW directory."""
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z", "--", "src/Zrodlowyszukiwator/RAG/RAW"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git ls-files jest niedostępne: {exc}")
    names = {Path(path).name for path in listing.split("\0") if path.endswith(".pdf")}
    return {CATALOG[name].document_id for name in names if name in CATALOG}


def page_text(document_id: str, page: int) -> str:
    """Return the normalized text of one PDF page, extracted like during ingestion."""
    if document_id not in _PAGES:
        path = RAW_DIR / FILE_BY_DOCUMENT[document_id]
        if not path.exists():
            pytest.skip(f"Brak pliku {path.name} w RAW")
        _PAGES[document_id] = [normalize_for_matching(p.text) for p in extract_pdf_pages(path)]
    return _PAGES[document_id][page - 1]


def make_hit(rank: int, document_id: str = "doc-a", supports: list[str] | None = None) -> Hit:
    return Hit(
        rank=rank,
        chunk_id=f"{document_id}:{rank:05d}:abc",
        document_id=document_id,
        title="Tytuł",
        page_start=2,
        page_end=3,
        section_path=[],
        text="tekst",
        semantic_score=None,
        lexical_score=None,
        hybrid_score=None,
        supports=supports or [],
        metadata={},
    )


def make_question(evidence_labels: list[str]) -> EvalQuestion:
    return EvalQuestion.model_validate(
        {
            "id": "q",
            "document_id": "doc-a",
            "question": "Pytanie?",
            "answerable": True,
            "expected_answer": "Odpowiedź.",
            "rationale": "Uzasadnienie.",
            "expected_sources": [
                {"document_id": "doc-a", "pages": [3], "quote": "cytat z dokumentu"}
            ],
            "evidence": [
                {"label": label, "document_id": "doc-a", "pages": [3], "all_of": [label]}
                for label in evidence_labels
            ],
        }
    )


class TestDatasetShape:
    def test_dataset__has_unique_question_ids(self) -> None:
        """Verify every question can be referenced unambiguously in reports."""
        assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)

    def test_dataset__covers_each_committed_document_with_answerable_and_unanswerable_questions(
        self,
    ) -> None:
        """Verify >=2 answerable and >=1 unanswerable question for exactly the committed PDFs."""
        committed = committed_document_ids()
        per_document = Counter((q.document_id, q.answerable) for q in QUESTIONS)

        assert committed, "Brak zakomitowanych PDF-ów z wpisem w katalogu"
        assert {q.document_id for q in QUESTIONS} == committed
        for document_id in committed:
            assert per_document[(document_id, True)] >= 2, document_id
            assert per_document[(document_id, False)] >= 1, document_id

    def test_dataset__refers_only_to_catalog_documents(self) -> None:
        """Verify sources, evidence and absence checks name documents from the catalog."""
        referenced = set()
        for q in QUESTIONS:
            referenced |= {q.document_id, *q.acceptable_document_ids}
            referenced |= {s.document_id for s in q.expected_sources}
            referenced |= {e.document_id for e in q.evidence}
            if q.absence_check:
                referenced |= set(q.absence_check.searched_documents)

        assert referenced <= set(FILE_BY_DOCUMENT)

    def test_dataset__absence_checks_cover_the_whole_catalog(self) -> None:
        """Verify unanswerable questions were checked against every catalog document."""
        for q in UNANSWERABLE:
            assert set(q.absence_check.searched_documents) == set(FILE_BY_DOCUMENT), q.id

    @pytest.mark.parametrize("question", UNANSWERABLE, ids=lambda q: q.id)
    def test_dataset__answer_patterns_are_valid_regexes(self, question: EvalQuestion) -> None:
        """Verify every answer pattern compiles, so the corpus check cannot silently pass."""
        for pattern in question.absence_check.answer_patterns:
            re.compile(pattern)


class TestDatasetGrounding:
    @pytest.mark.parametrize("question", ANSWERABLE, ids=lambda q: q.id)
    def test_dataset__quotes_are_on_the_declared_pdf_pages(self, question: EvalQuestion) -> None:
        """Verify each quoted source fragment really appears on the page it is attributed to."""
        for source in question.expected_sources:
            quote = normalize_for_matching(source.quote)
            assert any(quote in page_text(source.document_id, p) for p in source.pages), (
                f"{question.id}: cytatu nie ma na s. {source.pages}: {source.quote!r}"
            )

    @pytest.mark.parametrize("question", ANSWERABLE, ids=lambda q: q.id)
    def test_dataset__evidence_phrases_are_on_the_declared_pdf_pages(
        self, question: EvalQuestion
    ) -> None:
        """Verify the phrases that define supporting evidence are present on the declared pages."""
        for item in question.evidence:
            for phrase in item.all_of:
                normalized = normalize_for_matching(phrase)
                assert any(normalized in page_text(item.document_id, p) for p in item.pages), (
                    f"{question.id}/{item.label}: frazy {phrase!r} nie ma na s. {item.pages}"
                )


class TestEvaluationFunctions:
    def test_normalize_for_matching__ignores_extraction_artifacts(self) -> None:
        """Verify hyphenated line breaks and the broken font do not hide evidence."""
        assert (
            normalize_for_matching("Udział trans- portu i podróŜy") == "udział transportu i podróży"
        )
        assert normalize_for_matching("ul. Grota- Roweckiego") == "ul. grota- roweckiego"

    def test_strip_diacritics__removes_all_polish_marks(self) -> None:
        """Verify the derived query variant contains no Polish characters."""
        assert strip_diacritics("Źródło łąki, żółć") == "Zrodlo laki, zolc"

    def test_metrics__rank_recall_precision_and_mrr(self) -> None:
        """Verify the ranking metrics on a hand-checked result list."""
        question = make_question(["e1", "e2"])
        hits = [
            make_hit(1, "doc-b"),
            make_hit(2, "doc-a"),
            make_hit(3, "doc-a", ["e1"]),
            make_hit(4, "doc-a", ["e2"]),
        ]

        assert source_rank(question, hits) == 2
        assert evidence_rank(hits) == 3
        assert evidence_recall(question, hits, 3) == 0.5
        assert evidence_recall(question, hits, 4) == 1.0
        assert evidence_precision(hits, 4) == 0.5
        assert reciprocal_rank(3) == pytest.approx(1 / 3)
        assert reciprocal_rank(None) == 0.0
        assert reciprocal_rank(25) == 0.0

    def test_headline_rows__averages_top6_precision_and_recall_per_configuration(self) -> None:
        """Verify the report headline gives comparable precision/recall and skips ASCII variants."""
        question = make_question(["e1", "e2"])
        hybrid = [
            make_hit(1, "doc-a", ["e1"]),
            make_hit(2, "doc-b"),
            make_hit(3, "doc-a"),
            make_hit(4, "doc-a", ["e2"]),
        ]
        results = [
            QuestionResult("q", "hybrid", "Pytanie?", hits=hybrid),
            QuestionResult("q", "hybrid-ascii", "Pytanie?", hits=hybrid[:1]),
            QuestionResult("q", "pipeline", "Pytanie?", hits=[make_hit(1, "doc-a", ["e1"])]),
        ]
        run = EvalRun("t", "", "", {}, [], [question], results, {}, {}, [], 0, [], [])

        assert headline_rows(run) == {
            "hybrid": {"precision@6": 0.5, "recall@6": 1.0, "n": 1.0},
            "pipeline": {"precision@6": 1.0, "recall@6": 0.5, "n": 1.0},
        }

    def test_page_attribution_problems__reports_a_chunk_outside_the_evidence_page(self) -> None:
        """Verify evidence found on the wrong pages is reported as a metadata problem."""
        question = make_question(["e1"])

        assert page_attribution_problems(question, make_hit(1, supports=["e1"])) == []
        hit = make_hit(1, supports=["e1"])
        hit.page_start = hit.page_end = 7
        assert len(page_attribution_problems(question, hit)) == 1

    def test_metadata_problems__compares_hit_with_catalog(self) -> None:
        """Verify a chunk carrying a wrong title or out-of-range page is reported."""
        catalog = catalog_by_document_id(CATALOG)
        filename, meta = catalog["wroclaw-bicycle-policy-2010"]
        hit = make_hit(1, "wroclaw-bicycle-policy-2010")
        hit.metadata = {
            "title": meta.title,
            "document_kind": meta.document_kind.value,
            "hierarchy_rank": meta.hierarchy_rank,
            "legal_form": meta.legal_form.value,
            "adoption_act_number": meta.adoption_act_number,
            "adoption_date": meta.adoption_date.isoformat(),
            "is_active": meta.is_active,
            "lifecycle_status": meta.lifecycle_status.value,
            "source_url": meta.source_url,
            "source_filename": filename,
        }

        assert metadata_problems(hit, catalog, {"wroclaw-bicycle-policy-2010": 9}) == []
        hit.metadata["title"] = "Inny tytuł"
        hit.page_end = 12
        assert len(metadata_problems(hit, catalog, {"wroclaw-bicycle-policy-2010": 9})) == 2
