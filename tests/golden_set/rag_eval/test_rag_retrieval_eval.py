"""
Ewaluacja retrievalu na żywym indeksie RAG (PostgreSQL/pgvector + Ollama z bge-m3).

Złoty zbiór: tylko ręcznie, przez `-m golden`; zwykłe `pytest` i CI go pomijają
(addopts w pyproject.toml). Bez bazy albo Ollamy testy są pomijane z powodem.
Opcjonalnie: PISMAK_RAG_EVAL_LABEL (np. baseline/after) nazywa przebieg, a
PISMAK_RAG_EVAL_RERANKER_MODEL (np. pismak-pllum:latest) dodaje konfigurację z rerankerem LLM.
Raport trafia do results/<czas>_<etykieta>/ (report.md, results.json), a wynik każdego pytania
do results/<czas>_<etykieta>_<id>.json; oba są zapisywane przed asercjami.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from typing import Any

import pymupdf
import pytest
from dotenv import load_dotenv

from src._shared.model_context import detect_context_tokens
from src._shared.serwer import set_up_client
from src.Zrodlowyszukiwator.brief_to_evidence.pipeline import BriefToEvidencePipeline
from src.Zrodlowyszukiwator.brief_to_evidence.planner import LLMQueryPlanner
from src.Zrodlowyszukiwator.rag_ingestion.catalog import load_catalog
from src.Zrodlowyszukiwator.rag_ingestion.cli import DEFAULT_CATALOG_PATH
from src.Zrodlowyszukiwator.rag_ingestion.embedder import OllamaEmbedder
from src.Zrodlowyszukiwator.rag_ingestion.pipeline import file_sha256
from src.Zrodlowyszukiwator.rag_retrieval import (
    DatabaseConnectionError,
    HybridRetriever,
    LLMReranker,
    RetrievalRepository,
)
from src.Zrodlowyszukiwator.rag_retrieval.models import RetrievalFilters
from tests.golden_set.rag_eval.rag_eval_support import (
    CANDIDATE_K,
    CASES_DIR,
    EVIDENCE_K,
    K_VALUES,
    MAX_PER_DOCUMENT,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
    EvalQuestion,
    EvalRun,
    QuestionResult,
    answer_pattern_matches,
    catalog_by_document_id,
    evidence_rank,
    evidence_recall,
    headline_rows,
    hit_from_chunk,
    hit_from_evidence,
    load_questions,
    metadata_problems,
    page_attribution_problems,
    render_markdown,
    source_rank,
    strip_diacritics,
    summary_rows,
    supported_items,
)

pytestmark = pytest.mark.golden

QUESTIONS = load_questions()
ANSWERABLE = [q for q in QUESTIONS if q.answerable]
UNANSWERABLE = [q for q in QUESTIONS if not q.answerable]


class FallbackPlanner:
    """Plan zapytań bez LLM: ten sam, którego Pismak używa, gdy planer modelowy zawiedzie."""

    def plan(self, case_brief: dict[str, Any]) -> Any:
        """Return the deterministic fallback plan for a brief."""
        return LLMQueryPlanner._fallback_plan(case_brief)


class CachingEmbedder:
    """Liczy embedding każdego tekstu raz, bo to samo pytanie trafia do kilku konfiguracji."""

    def __init__(self, embedder: OllamaEmbedder) -> None:
        self.embedder = embedder
        self.cache: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings in input order, computing only the texts not seen before."""
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            self.cache.update(zip(missing, self.embedder.embed(missing)))
        return [self.cache[t] for t in texts]


def _git_commit() -> str:
    """Return the short commit hash with a marker for uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "nieznany"
    return f"{commit}{' + niezacommitowane zmiany' if dirty else ''}"


def _index_snapshot(repository: RetrievalRepository) -> tuple[list[dict[str, Any]], list[tuple]]:
    """Read the indexed documents and all chunk texts (read-only queries)."""
    with repository.connect() as conn:
        documents = conn.execute(
            """
            SELECT d.document_id, d.is_active, d.retrieval_enabled, d.ingestion_version,
                   d.content_sha256, d.source_filename, count(c.chunk_id)
            FROM rag_documents d LEFT JOIN rag_chunks c USING (document_id)
            GROUP BY d.document_id ORDER BY d.document_id
            """
        ).fetchall()
        chunks = conn.execute(
            "SELECT chunk_id, document_id, page_start, page_end, text FROM rag_chunks"
        ).fetchall()
    snapshot = []
    for document_id, active, enabled, version, sha, filename, count in documents:
        path = RAW_DIR / filename
        snapshot.append(
            {
                "document_id": document_id,
                "is_active": active,
                "retrieval_enabled": enabled,
                "ingestion_version": version,
                "chunks": count,
                "source_filename": filename,
                "sha_matches_file": file_sha256(path) == sha if path.exists() else None,
            }
        )
    return snapshot, chunks


def _run(config: str, question: EvalQuestion, query: str, search: Any) -> QuestionResult:
    """Run one search and convert its ranked chunks, keeping errors instead of raising."""
    result = QuestionResult(question_id=question.id, config=config, query=query)
    try:
        chunks = search()
    except Exception as exc:  # błąd jednej konfiguracji trafia do raportu, nie przerywa przebiegu
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    result.hits = [hit_from_chunk(rank, chunk, question) for rank, chunk in enumerate(chunks, 1)]
    return result


def _run_pipeline(
    config: str, question: EvalQuestion, pipeline: BriefToEvidencePipeline
) -> QuestionResult:
    """Run the EvidencePack pipeline and record its evidence and abstention signal."""
    result = QuestionResult(question_id=question.id, config=config, query=question.question)
    try:
        pack = pipeline.run({"problem_description": question.question}, filters=RetrievalFilters())
    except Exception as exc:  # błąd jednej konfiguracji trafia do raportu, nie przerywa przebiegu
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    result.hits = [
        hit_from_evidence(rank, ev, question) for rank, ev in enumerate(pack.evidence, 1)
    ]
    result.unsupported_topics = list(pack.unsupported_topics)
    # Sygnałem braku podstaw w kontrakcie RAG jest evidence_status="unsupported". Samo
    # wyszukiwanie zawsze zwraca podobne fragmenty, więc pusta lista nie jest wymagana.
    result.evidence_status = pack.evidence_status
    result.abstained = pack.evidence_status == "unsupported"
    return result


def _diagnostics(
    question: EvalQuestion,
    run_results: dict[str, QuestionResult],
    repository: RetrievalRepository,
    embedder: CachingEmbedder,
    chunks: list[tuple],
) -> dict[str, Any]:
    """Collect scores and likely causes of failure for one question."""
    vector = run_results.get("vector")
    lexical = run_results.get("lexical")
    hybrid = run_results.get("hybrid")
    diag: dict[str, Any] = {
        "max_cosine": vector.hits[0].semantic_score if vector and vector.hits else None,
        "max_lexical": lexical.hits[0].lexical_score if lexical and lexical.hits else None,
        "hints": [],
    }
    hints = diag["hints"]
    if not question.answerable:
        pipeline = run_results.get("pipeline")
        if pipeline and pipeline.hits:
            hints.append(
                f"EvidencePack przedstawia {len(pipeline.hits)} fragmentów jako dowody, choć korpus "
                "nie odpowiada na pytanie."
            )
        return diag

    # Czy dowód w ogóle jest w indeksie i na której pozycji pełnego rankingu leży.
    in_index = {
        item.label: [
            chunk_id
            for chunk_id, document_id, _, _, text in chunks
            if item.label in supported_items(question, document_id, text)
        ]
        for item in question.evidence
    }
    diag["evidence_chunks_in_index"] = in_index
    for label, chunk_ids in in_index.items():
        if not chunk_ids:
            hints.append(
                f"Dowodu „{label}” nie ma w żadnym chunku indeksu: fakt utracony przy ingestion "
                "albo zapisany inaczej niż w PDF."
            )
    filters = RetrievalFilters()
    total = len(chunks)
    embedding = embedder.embed([question.question])[0]
    full_vector = repository.vector_search(query_embedding=embedding, limit=total, filters=filters)
    full_lexical = repository.lexical_search(query=question.question, limit=total, filters=filters)
    for key, name, ranking in (
        ("full_rank_vector", "wektorowym", full_vector),
        ("full_rank_lexical", "leksykalnym", full_lexical),
    ):
        rank = next(
            (
                i
                for i, chunk in enumerate(ranking, 1)
                if supported_items(question, chunk.document_id, chunk.text)
            ),
            None,
        )
        diag[key] = rank
        if hybrid and (evidence_rank(hybrid.hits) or CANDIDATE_K + 1) > EVIDENCE_K:
            where = f"poz. {rank} z {len(ranking)}" if rank else "brak (żaden zwrócony chunk)"
            hints.append(f"Pierwszy wspierający chunk w pełnym rankingu {name}: {where}.")
    if hybrid and hybrid.hits:
        top = Counter(h.document_id for h in hybrid.hits[:EVIDENCE_K])
        document_id, count = top.most_common(1)[0]
        if document_id not in question.source_document_ids and count >= 3:
            hints.append(f"Top-6 `hybrid` zdominowane przez {document_id} ({count}/6).")
    if vector and lexical:
        v, lx = evidence_rank(vector.hits), evidence_rank(lexical.hits)
        if v and not lx:
            hints.append("Dowód w top-20 tylko w wyszukiwaniu wektorowym.")
        elif lx and not v:
            hints.append("Dowód w top-20 tylko w wyszukiwaniu leksykalnym.")
    return diag


def run_evaluation(
    repository: RetrievalRepository,
    embedder: CachingEmbedder,
    label: str,
    reranker_model: str | None,
) -> EvalRun:
    """Run every configuration for every question and collect the evaluation data."""
    started_at = datetime.now().isoformat(timespec="seconds")
    snapshot, chunks = _index_snapshot(repository)
    filters = RetrievalFilters()
    retriever = HybridRetriever(
        repository=repository,
        embedder=embedder,
        vector_limit=CANDIDATE_K,
        lexical_limit=CANDIDATE_K,
    )
    pipeline = BriefToEvidencePipeline(
        planner=FallbackPlanner(),
        retriever=retriever,
        reranker=None,
        candidate_k=CANDIDATE_K,
        evidence_k=EVIDENCE_K,
        max_per_document=MAX_PER_DOCUMENT,
    )
    not_run = [
        "Planer LLM (LLMQueryPlanner z modelem): zapytania są budowane deterministycznie, "
        "tak jak w produkcyjnym fallbacku (pytanie = semantic_query = lexical_query)."
    ]
    rerank_pipeline = None
    if reranker_model:
        reranker = LLMReranker(
            client=set_up_client("local"),
            model=reranker_model,
            max_selected=EVIDENCE_K,
            allow_fallback=False,
            provider="ollama",
            context_tokens=detect_context_tokens("ollama", reranker_model),
        )
        rerank_pipeline = BriefToEvidencePipeline(
            planner=FallbackPlanner(),
            retriever=retriever,
            reranker=reranker,
            candidate_k=CANDIDATE_K,
            evidence_k=EVIDENCE_K,
            max_per_document=MAX_PER_DOCUMENT,
        )
    else:
        not_run.append(
            "`pipeline+reranker`: nie ustawiono PISMAK_RAG_EVAL_RERANKER_MODEL (reranker LLM jest opcjonalny)."
        )

    results: list[QuestionResult] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for question in QUESTIONS:
        variants = [("", question.question)]
        if question.answerable:
            variants.append(("-ascii", strip_diacritics(question.question)))
        per_config: dict[str, QuestionResult] = {}
        for suffix, text in variants:
            intents = LLMQueryPlanner._fallback_plan({"problem_description": text}).intents
            intent = intents[0]
            searches = {
                "vector": lambda: repository.vector_search(
                    query_embedding=embedder.embed([intent.semantic_query])[0],
                    limit=CANDIDATE_K,
                    filters=filters,
                ),
                "lexical": lambda: repository.lexical_search(
                    query=intent.lexical_query, limit=CANDIDATE_K, filters=filters
                ),
                "hybrid": lambda: retriever.retrieve(intents, filters=filters, top_k=CANDIDATE_K),
            }
            for name, search in searches.items():
                per_config[name + suffix] = _run(name + suffix, question, text, search)
        per_config["pipeline"] = _run_pipeline("pipeline", question, pipeline)
        if rerank_pipeline is not None:
            per_config["pipeline+reranker"] = _run_pipeline(
                "pipeline+reranker", question, rerank_pipeline
            )
        results.extend(per_config.values())
        diagnostics[question.id] = _diagnostics(question, per_config, repository, embedder, chunks)

    corpus_absence = {
        question.id: {
            pattern: [
                chunk_id
                for chunk_id, _, _, _, text in chunks
                if pattern in answer_pattern_matches(question, text)
            ]
            for pattern in question.absence_check.answer_patterns
        }
        for question in UNANSWERABLE
    }

    catalog = catalog_by_document_id(load_catalog(DEFAULT_CATALOG_PATH))
    page_counts = {}
    for document_id, (filename, _) in catalog.items():
        if (RAW_DIR / filename).exists():
            with pymupdf.open(RAW_DIR / filename) as doc:
                page_counts[document_id] = len(doc)
    questions_by_id = {q.id: q for q in QUESTIONS}
    meta_problems: list[str] = []
    page_problems: list[str] = []
    checked = 0
    for result in results:
        for hit in result.hits:
            checked += 1
            meta_problems.extend(metadata_problems(hit, catalog, page_counts))
            page_problems.extend(
                page_attribution_problems(questions_by_id[result.question_id], hit)
            )

    return EvalRun(
        label=label,
        started_at=started_at,
        git_commit=_git_commit(),
        settings={
            "zbiór": f"{CASES_DIR.relative_to(PROJECT_ROOT).as_posix()} ({len(QUESTIONS)} pytań)",
            "model embeddingów": os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"),
            "filtry": "RetrievalFilters() — domyślne produkcyjne (bez dokumentów jawnie nieaktywnych)",
            "limity": f"vector/lexical {CANDIDATE_K}, candidate_k {CANDIDATE_K}, evidence_k {EVIDENCE_K}, "
            f"max_per_document {MAX_PER_DOCUMENT}",
            "k": ", ".join(map(str, K_VALUES)),
            "reranker": reranker_model or "nie uruchomiono",
        },
        index_snapshot=snapshot,
        questions=QUESTIONS,
        results=results,
        diagnostics=diagnostics,
        corpus_absence=corpus_absence,
        metadata_problems=list(dict.fromkeys(meta_problems)),
        metadata_checked=checked,
        page_problems=list(dict.fromkeys(page_problems)),
        not_run=not_run,
    )


def question_status(run: EvalRun, question: EvalQuestion) -> tuple[str, list[str]]:
    """Return passed/failed_check for one question, by the same rules as the tests below."""
    problems = []
    pipeline = run.result(question.id, "pipeline")
    reranked = run.result(question.id, "pipeline+reranker")
    if question.answerable:
        if pipeline is None or pipeline.error or source_rank(question, pipeline.hits) is None:
            problems.append("brak właściwego źródła w EvidencePack")
        elif evidence_rank(pipeline.hits) is None:
            problems.append("żaden fragment EvidencePack nie zawiera dowodu")
        if reranked is not None and reranked.evidence_status != "supported":
            problems.append(f"po rerankerze: {reranked.evidence_status}")
    else:
        if any(run.corpus_absence[question.id].values()):
            problems.append("wzorce odpowiedzi występują w indeksie")
        if reranked is not None and not reranked.abstained:
            problems.append(f"brak abstencji po rerankerze ({reranked.evidence_status})")
    return ("failed_check" if problems else "passed"), problems


def write_question_results(run: EvalRun, out_dir: Any, prefix: str) -> None:
    """Write results/<prefix>_<question_id>.json with every configuration's top hits."""
    for question in run.questions:
        status, problems = question_status(run, question)
        configs = {}
        for result in run.results:
            if result.question_id != question.id:
                continue
            configs[result.config] = {
                "error": result.error,
                "evidence_status": result.evidence_status,
                "abstained": result.abstained,
                "source_rank": source_rank(question, result.hits),
                "evidence_rank": evidence_rank(result.hits),
                "top_hits": [
                    {
                        "rank": h.rank,
                        "document_id": h.document_id,
                        "pages": f"{h.page_start}-{h.page_end}",
                        "supports": h.supports,
                    }
                    for h in result.hits[:EVIDENCE_K]
                ],
            }
        payload = {
            "module": "rag_eval",
            "case_id": question.id,
            "status": status,
            "checks": problems,
            "embedding_model": run.settings.get("model embeddingów"),
            "reranker_model": run.settings.get("reranker"),
            "answerable": question.answerable,
            "configs": configs,
            "manual_review": [f"Oczekiwana odpowiedź: {question.expected_answer}"],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{prefix}_{question.id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


def write_report(run: EvalRun) -> Any:
    """Write report.md and results.json, plus one result file per question, under results/."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_DIR / f"{stamp}_{run.label}"
    write_question_results(run, REPORTS_DIR, f"{stamp}_{run.label}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(render_markdown(run), encoding="utf-8")
    payload = {"headline": headline_rows(run)} | {
        key: value for key, value in run.__dict__.items() if key not in ("questions", "results")
    }
    payload["questions"] = [q.model_dump() for q in run.questions]
    payload["results"] = [asdict(r) for r in run.results]
    payload["summary"] = summary_rows(run)
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return out_dir


@pytest.fixture(scope="module")
def evaluation() -> EvalRun:
    """Run the evaluation once against the live index and write the report."""
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("Brak DATABASE_URL — ewaluacja nie została uruchomiona")
    repository = RetrievalRepository(database_url)
    try:
        repository.counts()
    except (DatabaseConnectionError, RuntimeError) as exc:
        pytest.skip(f"Baza RAG niedostępna — ewaluacja nie została uruchomiona: {exc}")
    embedder = CachingEmbedder(
        OllamaEmbedder(
            model=os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"),
            base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        )
    )
    try:
        embedder.embed(["test połączenia"])
    except RuntimeError as exc:
        pytest.skip(f"Ollama niedostępna — ewaluacja nie została uruchomiona: {exc}")

    run = run_evaluation(
        repository,
        embedder,
        label=os.getenv("PISMAK_RAG_EVAL_LABEL", "run"),
        reranker_model=os.getenv("PISMAK_RAG_EVAL_RERANKER_MODEL") or None,
    )
    run.settings["raport"] = str(write_report(run).relative_to(PROJECT_ROOT))
    return run


def test_report_is_written(evaluation: EvalRun) -> None:
    """Show where the report of this run was written."""
    print(f"Raport ewaluacji: {evaluation.settings['raport']}")
    assert (PROJECT_ROOT / evaluation.settings["raport"] / "report.md").exists()


def test_documents_are_indexed_from_the_current_files(evaluation: EvalRun) -> None:
    """Verify every document of the question set is searchable and indexed from the same PDF."""
    snapshot = {doc["document_id"]: doc for doc in evaluation.index_snapshot}
    for document_id in {q.document_id for q in QUESTIONS}:
        doc = snapshot.get(document_id)
        assert doc is not None, f"{document_id}: brak w indeksie — uruchom rag_ingestion"
        assert doc["retrieval_enabled"] and doc["chunks"] > 0, document_id
        assert doc["sha_matches_file"], f"{document_id}: indeks zbudowano z innej wersji PDF"


@pytest.mark.parametrize("question", UNANSWERABLE, ids=lambda q: q.id)
def test_unanswerable_question_has_no_answer_in_the_indexed_corpus(
    evaluation: EvalRun, question: EvalQuestion
) -> None:
    """Verify the 'no answer' label still holds for every chunk of the index."""
    found = {p: ids for p, ids in evaluation.corpus_absence[question.id].items() if ids}
    assert not found, f"{question.id}: wzorce odpowiedzi występują w indeksie: {found}"


@pytest.mark.parametrize("question", ANSWERABLE, ids=lambda q: q.id)
def test_answerable_question_is_supported_by_retrieved_evidence(
    evaluation: EvalRun, question: EvalQuestion
) -> None:
    """Verify the right source and supporting fragments reach the EvidencePack and candidates."""
    pipeline = evaluation.result(question.id, "pipeline")
    hybrid = evaluation.result(question.id, "hybrid")
    assert pipeline.error is None and hybrid.error is None, (pipeline.error, hybrid.error)

    failures = []
    if source_rank(question, pipeline.hits) is None:
        top = Counter(h.document_id for h in pipeline.hits)
        failures.append(f"brak właściwego źródła w EvidencePack (są: {dict(top)})")
    if evidence_rank(pipeline.hits) is None:
        failures.append("żaden fragment EvidencePack nie zawiera dowodu")
    recall = evidence_recall(question, hybrid.hits, CANDIDATE_K)
    if recall < 1:
        covered = {label for hit in hybrid.hits for label in hit.supports}
        missing = [item.label for item in question.evidence if item.label not in covered]
        failures.append(f"evidence_recall@{CANDIDATE_K} = {recall:.2f}, brakuje: {missing}")
    hints = evaluation.diagnostics[question.id]["hints"]
    assert not failures, f"{question.id}: " + "; ".join(failures) + " | " + " ".join(hints)


def _reranked(evaluation: EvalRun, question: EvalQuestion) -> QuestionResult:
    """Return the reranker result of a question, skipping when support was not assessed."""
    result = evaluation.result(question.id, "pipeline+reranker")
    if result is None:
        pytest.skip(
            "Ocena wsparcia wymaga rerankera: ustaw PISMAK_RAG_EVAL_RERANKER_MODEL "
            "(bez niego EvidencePack ma status unverified)"
        )
    assert result.error is None, f"{question.id}: reranker nie zadziałał: {result.error}"
    return result


def test_pipeline_without_reranker_does_not_claim_support(evaluation: EvalRun) -> None:
    """Verify retrieval alone marks its fragments as unverified instead of as evidence."""
    claims = [
        f"{r.question_id}: {r.evidence_status}"
        for r in evaluation.results
        if r.config == "pipeline" and r.evidence_status != "unverified"
    ]
    assert not claims, claims


@pytest.mark.parametrize("question", UNANSWERABLE, ids=lambda q: q.id)
def test_unanswerable_question_makes_the_system_abstain(
    evaluation: EvalRun, question: EvalQuestion
) -> None:
    """Verify the EvidencePack signals missing support instead of presenting unrelated fragments."""
    result = _reranked(evaluation, question)
    presented = [f"{h.document_id} s. {h.page_start}-{h.page_end}" for h in result.hits]
    assert result.abstained, (
        f"{question.id}: brak abstencji (evidence_status={result.evidence_status}) — EvidencePack "
        f"przedstawia {len(presented)} fragmentów jako dowody ({'; '.join(presented)})"
    )


@pytest.mark.parametrize("question", ANSWERABLE, ids=lambda q: q.id)
def test_answerable_question_keeps_support_after_reranking(
    evaluation: EvalRun, question: EvalQuestion
) -> None:
    """Verify the support check does not reject a question the corpus answers."""
    result = _reranked(evaluation, question)
    assert result.evidence_status == "supported", (
        f"{question.id}: fałszywa abstencja (evidence_status={result.evidence_status})"
    )
    assert evidence_rank(result.hits) is not None, (
        f"{question.id}: reranker wybrał fragmenty bez dowodu: "
        + "; ".join(f"{h.document_id} s. {h.page_start}-{h.page_end}" for h in result.hits)
    )


def test_returned_fragments_match_the_document_catalog(evaluation: EvalRun) -> None:
    """Verify title, kind, act, dates, status, URL, file, identifier and pages of every hit."""
    assert not evaluation.metadata_problems, "\n".join(evaluation.metadata_problems[:20])


def test_supporting_fragments_point_to_the_right_pages(evaluation: EvalRun) -> None:
    """Verify a fragment containing the evidence is attributed to the page the evidence is on."""
    assert not evaluation.page_problems, "\n".join(evaluation.page_problems[:20])


def test_reranker_configuration_completed(evaluation: EvalRun) -> None:
    """Verify the optional LLM reranker produced results for every question."""
    results = [r for r in evaluation.results if r.config == "pipeline+reranker"]
    if not results:
        pytest.skip("Reranker nie był uruchamiany (PISMAK_RAG_EVAL_RERANKER_MODEL)")
    errors = [f"{r.question_id}: {r.error}" for r in results if r.error]
    assert not errors, "\n".join(errors)
