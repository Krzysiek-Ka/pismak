"""
Ewaluacja retrievalu RAG Pismaka: zbiór pytań, ocena dowodów, metryki i raport.

Moduł nie łączy się z bazą. Korzystają z niego testy zbioru i ewaluacja na żywym indeksie
w test_rag_retrieval_eval.py (oba tylko ręcznie, marker `golden`). Przypadki leżą w
cases/<id>.json w układzie złotych zbiorów (.github/copilot-instructions.md): pytanie w
`input`, uzasadnienie w `description`, sprawdzalne pola w `expected`, oczekiwana odpowiedź
(do oceny przez człowieka) w `manual_review`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.Zrodlowyszukiwator.rag_ingestion.models import DocumentMetadata
from src.Zrodlowyszukiwator.rag_ingestion.pdf_extractor import repair_legacy_polish_glyphs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASES_DIR = Path(__file__).resolve().parent / "cases"
RAW_DIR = PROJECT_ROOT / "src" / "Zrodlowyszukiwator" / "RAG" / "RAW"
REPORTS_DIR = Path(__file__).resolve().parent / "results"
EXPECTED_ANSWER_PREFIX = "Oczekiwana odpowiedź: "

# Wartości z konfiguracji produkcyjnej (Wyswietlator.services.build_zrodlowyszukiwator):
# reranker dostaje 20 kandydatów, a EvidencePack zawiera 6 fragmentów, najwyżej 3 z dokumentu.
CANDIDATE_K = 20
EVIDENCE_K = 6
MAX_PER_DOCUMENT = 3
K_VALUES = (1, 3, 6, 10, 20)


# ---------------------------------------------------------------------------
# Zbiór pytań


class ExpectedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    # Numery stron pliku PDF liczone od 1, tak jak page_start/page_end w indeksie.
    pages: list[int] = Field(min_length=1)
    quote: str = Field(min_length=10)


class EvidenceItem(BaseModel):
    """Fakt potrzebny do odpowiedzi; chunk go wspiera, gdy zawiera wszystkie frazy all_of."""

    model_config = ConfigDict(extra="forbid")

    label: str
    document_id: str
    pages: list[int] = Field(min_length=1)
    all_of: list[str] = Field(min_length=1)


class AbsenceCheck(BaseModel):
    """Udokumentowane sprawdzenie, że korpus nie odpowiada na pytanie."""

    model_config = ConfigDict(extra="forbid")

    searched_documents: list[str] = Field(min_length=1)
    searched_terms: list[str] = Field(min_length=1)
    # Wzorce, które wskazywałyby odpowiedź. Ewaluacja sprawdza ich brak w całym indeksie.
    answer_patterns: list[str] = Field(min_length=1)
    related_but_insufficient: list[str] = Field(default_factory=list)


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    question: str
    answerable: bool
    expected_answer: str
    rationale: str
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    acceptable_document_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    absence_check: AbsenceCheck | None = None

    @model_validator(mode="after")
    def validate_answerability(self) -> EvalQuestion:
        """Require sources and evidence for answerable questions and an absence check otherwise."""
        if self.answerable:
            if not self.expected_sources or not self.evidence:
                raise ValueError(f"{self.id}: pytanie z odpowiedzią wymaga źródeł i dowodów")
            if self.absence_check is not None:
                raise ValueError(f"{self.id}: absence_check dotyczy tylko pytań bez odpowiedzi")
        else:
            if self.expected_sources or self.evidence:
                raise ValueError(f"{self.id}: pytanie bez odpowiedzi nie może mieć źródeł")
            if self.absence_check is None:
                raise ValueError(f"{self.id}: pytanie bez odpowiedzi wymaga absence_check")
        return self

    @property
    def source_document_ids(self) -> set[str]:
        """Return the documents that count as a correct source for this question."""
        return {s.document_id for s in self.expected_sources} | set(self.acceptable_document_ids)


def question_from_case(case: dict[str, Any]) -> EvalQuestion:
    """Turn one golden-set case file into the question the evaluation scores."""
    unknown = set(case) - {"id", "description", "input", "expected", "manual_review"}
    if unknown:
        raise ValueError(f"{case.get('id')}: nieznane pola {sorted(unknown)}")
    [answer] = [r for r in case["manual_review"] if r.startswith(EXPECTED_ANSWER_PREFIX)]
    return EvalQuestion.model_validate(
        {
            "id": case["id"],
            "question": case["input"],
            "rationale": case["description"],
            "expected_answer": answer.removeprefix(EXPECTED_ANSWER_PREFIX),
            **case["expected"],
        }
    )


def load_questions(cases_dir: Path = CASES_DIR) -> list[EvalQuestion]:
    """Load and validate cases/<id>.json; the file name must be the case id."""
    questions = []
    for path in sorted(cases_dir.glob("*.json")):
        question = question_from_case(json.loads(path.read_text(encoding="utf-8")))
        if question.id != path.stem:
            raise ValueError(f"{path.name}: id {question.id!r} różni się od nazwy pliku")
        questions.append(question)
    return questions


# ---------------------------------------------------------------------------
# Tekst i dowody

# Przeniesienie wyrazu zostawione w tekście („trans- portu”). Wielka litera po łączniku
# oznacza nazwę („Grota- Roweckiego”), więc nie jest sklejana.
_LINE_BREAK_HYPHEN = re.compile(r"(?<=[^\W\d_])- (?=[a-ząćęłńóśźż])")


def normalize_for_matching(text: str) -> str:
    """
    Normalize text for judging evidence by content rather than by extraction artifacts.

    Sklejanie przeniesień i naprawa „Ŝ” są tu tylko po to, żeby stary indeks był
    oceniany za treść, a nie za formatowanie; same defekty mierzy osobno raport.
    """
    text = unicodedata.normalize("NFKC", text)
    text = repair_legacy_polish_glyphs(text)
    text = " ".join(text.split())
    text = _LINE_BREAK_HYPHEN.sub("", text)
    return text.lower()


def strip_diacritics(text: str) -> str:
    """Return the text as typed without Polish characters (also 'ł' -> 'l')."""
    text = text.replace("ł", "l").replace("Ł", "L")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def supported_items(question: EvalQuestion, document_id: str, text: str) -> list[str]:
    """Return labels of the question's evidence items fully contained in one chunk."""
    normalized = normalize_for_matching(text)
    labels = []
    for item in question.evidence:
        allowed = {item.document_id} | set(question.acceptable_document_ids)
        if document_id in allowed and all(
            normalize_for_matching(phrase) in normalized for phrase in item.all_of
        ):
            labels.append(item.label)
    return labels


def answer_pattern_matches(question: EvalQuestion, text: str) -> list[str]:
    """Return the answer patterns of an unanswerable question that match the text."""
    if question.absence_check is None:
        return []
    normalized = normalize_for_matching(text)
    return [
        pattern
        for pattern in question.absence_check.answer_patterns
        if re.search(pattern, normalized)
    ]


# ---------------------------------------------------------------------------
# Wyniki i metryki


@dataclass
class Hit:
    """One ranked result, from a RetrievedChunk or from an EvidencePack entry."""

    rank: int
    chunk_id: str
    document_id: str
    title: str
    page_start: int
    page_end: int
    section_path: list[str]
    text: str
    semantic_score: float | None
    lexical_score: float | None
    hybrid_score: float | None
    supports: list[str]
    metadata: dict[str, Any]
    # Ocena i uzasadnienie rerankera (tylko fragmenty z EvidencePack po rerankingu).
    rerank_score: int | None = None
    reason: str | None = None


@dataclass
class QuestionResult:
    question_id: str
    config: str
    query: str
    hits: list[Hit] = field(default_factory=list)
    error: str | None = None
    # Tylko konfiguracje zwracające EvidencePack mają kontrakt abstencji.
    abstained: bool | None = None
    evidence_status: str | None = None
    unsupported_topics: list[str] = field(default_factory=list)


def hit_from_chunk(rank: int, chunk: Any, question: EvalQuestion) -> Hit:
    """Build a Hit from a RetrievedChunk."""
    return Hit(
        rank=rank,
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=list(chunk.section_path),
        text=chunk.text,
        semantic_score=chunk.semantic_score,
        lexical_score=chunk.lexical_score,
        hybrid_score=chunk.hybrid_score or None,
        supports=supported_items(question, chunk.document_id, chunk.text),
        metadata={
            "title": chunk.title,
            "document_kind": chunk.document_kind,
            "hierarchy_rank": chunk.hierarchy_rank,
            "legal_form": chunk.legal_form,
            "adoption_act_number": chunk.adoption_act_number,
            "adoption_date": chunk.adoption_date,
            "is_active": chunk.is_active,
            "lifecycle_status": chunk.lifecycle_status,
            "source_url": chunk.source_url,
            "source_filename": chunk.source_filename,
        },
    )


def hit_from_evidence(rank: int, evidence: Any, question: EvalQuestion) -> Hit:
    """Build a Hit from an EvidencePack entry, keeping the fields the pipeline passes on."""
    return Hit(
        rank=rank,
        chunk_id=evidence.chunk_id,
        document_id=evidence.document_id,
        title=evidence.document_title,
        page_start=evidence.page_start,
        page_end=evidence.page_end,
        section_path=list(evidence.section_path),
        text=evidence.excerpt,
        semantic_score=None,
        lexical_score=None,
        hybrid_score=evidence.retrieval_score,
        supports=supported_items(question, evidence.document_id, evidence.excerpt),
        metadata={
            "title": evidence.document_title,
            "document_kind": evidence.document_kind,
            "hierarchy_rank": evidence.hierarchy_rank,
            "legal_form": evidence.legal_form,
            "adoption_act_number": evidence.adoption_act_number,
            "adoption_date": evidence.adoption_date,
            "is_active": evidence.is_active,
            "lifecycle_status": evidence.lifecycle_status,
            "source_url": evidence.source_url,
            "source_filename": evidence.source_filename,
        },
        rerank_score=evidence.rerank_score,
        reason=evidence.relevance_reason,
    )


def first_rank(hits: list[Hit], predicate: Any) -> int | None:
    """Return the 1-based rank of the first hit matching the predicate."""
    for hit in hits:
        if predicate(hit):
            return hit.rank
    return None


def source_rank(question: EvalQuestion, hits: list[Hit]) -> int | None:
    """Return the rank of the first chunk from a correct source document."""
    return first_rank(hits, lambda h: h.document_id in question.source_document_ids)


def evidence_rank(hits: list[Hit]) -> int | None:
    """Return the rank of the first chunk that supports any evidence item."""
    return first_rank(hits, lambda h: bool(h.supports))


def evidence_recall(question: EvalQuestion, hits: list[Hit], k: int) -> float:
    """Return the share of the question's evidence items covered by the top-k chunks."""
    covered = {label for hit in hits[:k] for label in hit.supports}
    return len(covered) / len(question.evidence)


def evidence_precision(hits: list[Hit], k: int) -> float:
    """Return the share of presented top-k chunks that support the answer."""
    top = hits[:k]
    return sum(1 for hit in top if hit.supports) / len(top) if top else 0.0


def reciprocal_rank(rank: int | None, cutoff: int = CANDIDATE_K) -> float:
    """Return 1/rank within the cutoff, or 0 when nothing relevant was found."""
    return 1.0 / rank if rank is not None and rank <= cutoff else 0.0


def dominant_other_document(question: EvalQuestion, hits: list[Hit], k: int) -> str | None:
    """Return the most frequent non-source document among the top-k chunks."""
    counts = Counter(
        h.document_id for h in hits[:k] if h.document_id not in question.source_document_ids
    )
    return counts.most_common(1)[0][0] if counts else None


def question_metrics(
    question: EvalQuestion, result: QuestionResult, k_values: tuple[int, ...]
) -> dict[str, Any]:
    """Compute ranking and evidence metrics of one answerable question in one configuration."""
    hits = result.hits
    s_rank = source_rank(question, hits)
    e_rank = evidence_rank(hits)
    metrics: dict[str, Any] = {"source_rank": s_rank, "evidence_rank": e_rank}
    for k in k_values:
        metrics[f"source_hit@{k}"] = float(s_rank is not None and s_rank <= k)
        metrics[f"evidence_hit@{k}"] = float(e_rank is not None and e_rank <= k)
        metrics[f"evidence_recall@{k}"] = evidence_recall(question, hits, k)
    metrics["mrr_source"] = reciprocal_rank(s_rank)
    metrics["mrr_evidence"] = reciprocal_rank(e_rank)
    metrics[f"evidence_precision@{EVIDENCE_K}"] = evidence_precision(hits, EVIDENCE_K)
    metrics["dominant_other_document"] = dominant_other_document(question, hits, EVIDENCE_K)
    return metrics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average the numeric metrics of several questions."""
    keys = [k for k, v in rows[0].items() if isinstance(v, float)] if rows else []
    return {key: mean(row[key] for row in rows) for key in keys}


# ---------------------------------------------------------------------------
# Metadane


def catalog_by_document_id(
    catalog: dict[str, DocumentMetadata],
) -> dict[str, tuple[str, DocumentMetadata]]:
    """Index the filename-keyed catalog by document_id."""
    return {meta.document_id: (filename, meta) for filename, meta in catalog.items()}


def expected_metadata(filename: str, meta: DocumentMetadata) -> dict[str, Any]:
    """Return the values a retrieved chunk of this document must carry."""
    return {
        "title": meta.title,
        "document_kind": meta.document_kind.value,
        "hierarchy_rank": meta.hierarchy_rank,
        "legal_form": meta.legal_form.value,
        "adoption_act_number": meta.adoption_act_number,
        "adoption_date": meta.adoption_date.isoformat() if meta.adoption_date else None,
        "is_active": meta.is_active,
        "lifecycle_status": meta.lifecycle_status.value if meta.lifecycle_status else None,
        "source_url": meta.source_url,
        "source_filename": filename,
    }


def metadata_problems(
    hit: Hit,
    catalog: dict[str, tuple[str, DocumentMetadata]],
    page_counts: dict[str, int],
) -> list[str]:
    """Compare a hit's source metadata, identifier and pages with the document catalog."""
    where = f"{hit.chunk_id} (s. {hit.page_start}-{hit.page_end})"
    if hit.document_id not in catalog:
        return [f"{where}: dokumentu {hit.document_id} nie ma w katalogu"]
    problems = []
    for key, expected in expected_metadata(*catalog[hit.document_id]).items():
        if hit.metadata.get(key) != expected:
            problems.append(f"{where}: {key}={hit.metadata.get(key)!r}, katalog={expected!r}")
    if not hit.chunk_id.startswith(f"{hit.document_id}:"):
        problems.append(f"{where}: chunk_id nie zaczyna się od document_id")
    if not 1 <= hit.page_start <= hit.page_end:
        problems.append(f"{where}: niepoprawny zakres stron")
    page_count = page_counts.get(hit.document_id)
    if page_count is not None and hit.page_end > page_count:
        problems.append(f"{where}: strona poza dokumentem ({page_count} stron)")
    return problems


def page_attribution_problems(question: EvalQuestion, hit: Hit) -> list[str]:
    """Check that a chunk supporting an evidence item covers the page the item is on."""
    problems = []
    pages = set(range(hit.page_start, hit.page_end + 1))
    for label in hit.supports:
        item = next(i for i in question.evidence if i.label == label)
        if not pages & set(item.pages):
            problems.append(
                f"{question.id}/{label}: {hit.chunk_id} ma strony {hit.page_start}-{hit.page_end}, "
                f"a dowód jest na s. {item.pages}"
            )
    return problems


# ---------------------------------------------------------------------------
# Raport


@dataclass
class EvalRun:
    label: str
    started_at: str
    git_commit: str
    settings: dict[str, Any]
    index_snapshot: list[dict[str, Any]]
    questions: list[EvalQuestion]
    results: list[QuestionResult]
    diagnostics: dict[str, dict[str, Any]]
    corpus_absence: dict[str, dict[str, list[str]]]
    metadata_problems: list[str]
    metadata_checked: int
    page_problems: list[str]
    not_run: list[str]

    def result(self, question_id: str, config: str) -> QuestionResult | None:
        """Return the result of one question in one configuration."""
        for result in self.results:
            if result.question_id == question_id and result.config == config:
                return result
        return None

    @property
    def configs(self) -> list[str]:
        """Return configurations in the order they were run."""
        return list(dict.fromkeys(r.config for r in self.results))


def config_k_values(config: str) -> tuple[int, ...]:
    """Return the cutoffs that make sense for a configuration's result length."""
    return (
        tuple(k for k in K_VALUES if k <= EVIDENCE_K) if config.startswith("pipeline") else K_VALUES
    )


def summary_rows(run: EvalRun) -> dict[str, dict[str, float]]:
    """Aggregate metrics of answerable questions per configuration (without ASCII variants)."""
    summary = {}
    for config in run.configs:
        if config.endswith("-ascii"):
            continue
        rows = []
        for question in run.questions:
            result = run.result(question.id, config)
            if question.answerable and result is not None and result.error is None:
                rows.append(question_metrics(question, result, config_k_values(config)))
        if rows:
            summary[config] = aggregate(rows) | {"n": float(len(rows))}
    return summary


# Uśrednione metryki z początku raportu, do szybkiego porównania przebiegów. Odcięcie 6 to
# liczba fragmentów, które pipeline przekazuje dalej (EvidencePack).
HEADLINE_METRICS = {
    f"precision@{EVIDENCE_K}": f"evidence_precision@{EVIDENCE_K}",
    f"recall@{EVIDENCE_K}": f"evidence_recall@{EVIDENCE_K}",
}


def headline_rows(run: EvalRun) -> dict[str, dict[str, float]]:
    """Return the averaged top-6 evidence precision and recall per configuration."""
    return {
        config: {name: row[key] for name, key in HEADLINE_METRICS.items()} | {"n": row["n"]}
        for config, row in summary_rows(run).items()
    }


def _fmt(value: Any) -> str:
    """Format a metric value for a Markdown table."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _short(text: str, limit: int = 110) -> str:
    """Return a one-line excerpt for the report."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(run: EvalRun) -> str:
    """Render the evaluation run as a Markdown report."""
    out: list[str] = []
    add = out.append
    add(f"# Ewaluacja retrievalu RAG — przebieg `{run.label}`\n")

    add(f"## Podsumowanie — pytania z odpowiedzią, top-{EVIDENCE_K}\n")
    names = list(HEADLINE_METRICS)
    add("| konfiguracja | n | " + " | ".join(names) + " |")
    add("|---|---|" + "---|" * len(names))
    for config, row in headline_rows(run).items():
        add(f"| `{config}` | {int(row['n'])} | " + " | ".join(_fmt(row[n]) for n in names) + " |")
    add(
        f"\nprecision@{EVIDENCE_K}: średni udział fragmentów top-{EVIDENCE_K}, które wspierają "
        f"odpowiedź. recall@{EVIDENCE_K}: średni udział faktów pytania (dowodów) pokrytych przez "
        f"top-{EVIDENCE_K}. Przebiegi porównuj przy tym samym n i zbiorze pytań.\n"
    )

    add("## Przebieg\n")
    add(f"- Start: {run.started_at}, commit: `{run.git_commit}`")
    for key, value in run.settings.items():
        add(f"- {key}: {value}")
    add("")

    add("## Stan indeksu\n")
    add("| document_id | aktywny | retrieval | chunki | ingestion_version | SHA-256 = plik w RAW |")
    add("|---|---|---|---|---|---|")
    for doc in run.index_snapshot:
        add(
            f"| {doc['document_id']} | {_fmt(doc['is_active'])} | {_fmt(doc['retrieval_enabled'])} | "
            f"{doc['chunks']} | {doc['ingestion_version']} | {_fmt(doc['sha_matches_file'])} |"
        )
    add("")

    add("## Zmierzone i nieuruchomione\n")
    add("Zmierzone konfiguracje: " + ", ".join(f"`{c}`" for c in run.configs))
    errors = [r for r in run.results if r.error]
    for result in errors:
        add(f"- BŁĄD `{result.config}` / {result.question_id}: {result.error}")
    if run.not_run:
        add("\nNie uruchomiono:")
        for item in run.not_run:
            add(f"- {item}")
    add("")

    summary = summary_rows(run)
    n_answerable = sum(q.answerable for q in run.questions)
    add(f"## Wyniki zbiorcze — pytania z odpowiedzią (n={n_answerable})\n")
    cols = [
        "source_hit@1",
        "source_hit@3",
        "source_hit@6",
        "source_hit@20",
        "evidence_hit@6",
        "evidence_hit@20",
        "evidence_recall@6",
        "evidence_recall@20",
        "mrr_source",
        "mrr_evidence",
        f"evidence_precision@{EVIDENCE_K}",
    ]
    add("| konfiguracja | n | " + " | ".join(cols) + " |")
    add("|---|---|" + "---|" * len(cols))
    for config, row in summary.items():
        add(
            f"| `{config}` | {int(row['n'])} | " + " | ".join(_fmt(row.get(c)) for c in cols) + " |"
        )
    add(
        "\n`pipeline*` zwraca najwyżej 6 fragmentów (EvidencePack), więc kolumny @10 i @20 "
        "go nie dotyczą. MRR liczony do pozycji 20; brak trafienia = 0. Precyzja dowodów to "
        "udział przedstawionych fragmentów top-6, które faktycznie wspierają odpowiedź.\n"
    )

    add("## Ocena wsparcia i abstencja\n")
    add(
        "`evidence_status` z EvidencePack: `supported`, `unsupported` (abstencja) albo "
        "`unverified` (trafność nieoceniona, np. bez rerankera).\n"
    )
    add(
        "| pytanie | odpowiedź w korpusie | konfiguracja | evidence_status "
        "| fragmentów jako dowody | dokumenty | max cosine | max lexical |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for question in run.questions:
        for config in run.configs:
            result = run.result(question.id, config)
            if result is None or not config.startswith("pipeline"):
                continue
            docs = Counter(h.document_id for h in result.hits)
            diag = run.diagnostics.get(question.id, {})
            status = "błąd" if result.error else _fmt(result.evidence_status)
            add(
                f"| {question.id} | {'tak' if question.answerable else 'nie'} | `{config}` | "
                f"{status} | {len(result.hits)} | "
                f"{', '.join(f'{d}×{n}' for d, n in docs.items()) or '—'} | "
                f"{_fmt(diag.get('max_cosine'))} | {_fmt(diag.get('max_lexical'))} |"
            )
    false_abstentions = [
        r.question_id
        for r in run.results
        if r.abstained and any(q.id == r.question_id and q.answerable for q in run.questions)
    ]
    add(f"\nFałszywe abstencje na pytaniach z odpowiedzią: {false_abstentions or 'brak'}.")
    cos_answerable = [
        run.diagnostics[q.id]["max_cosine"]
        for q in run.questions
        if q.answerable and q.id in run.diagnostics
    ]
    cos_unanswerable = [
        run.diagnostics[q.id]["max_cosine"]
        for q in run.questions
        if not q.answerable and q.id in run.diagnostics
    ]
    if cos_answerable and cos_unanswerable:
        add(
            f"Najwyższy cosine w top-20: pytania z odpowiedzią {min(cos_answerable):.3f}–{max(cos_answerable):.3f}, "
            f"bez odpowiedzi {min(cos_unanswerable):.3f}–{max(cos_unanswerable):.3f} "
            "(opis przy tej liczbie pytań, nie dowód rozdzielności).\n"
        )

    add("## Wyniki per pytanie\n")
    for question in run.questions:
        add(f"### {question.id} — {'z odpowiedzią' if question.answerable else 'bez odpowiedzi'}\n")
        add(f"> {question.question}\n")
        add(f"Oczekiwane: {question.expected_answer}\n")
        if question.answerable:
            add(
                "| konfiguracja | poz. źródła | poz. dowodu | recall@6 | recall@20 | dominujący inny dokument w top-6 |"
            )
            add("|---|---|---|---|---|---|")
            for config in run.configs:
                result = run.result(question.id, config)
                if result is None or result.error:
                    continue
                m = question_metrics(question, result, config_k_values(config))
                recall_20 = m.get("evidence_recall@20")
                add(
                    f"| `{config}` | {_fmt(m['source_rank'])} | {_fmt(m['evidence_rank'])} | "
                    f"{_fmt(m['evidence_recall@6'])} | {_fmt(recall_20)} | "
                    f"{_fmt(m['dominant_other_document'])} |"
                )
            add("")
        diag = run.diagnostics.get(question.id, {})
        for hint in diag.get("hints", []):
            add(f"- {hint}")
        hybrid = run.result(question.id, "hybrid")
        if hybrid is not None and hybrid.hits:
            add("\nTop-5 `hybrid`:\n")
            add("| # | dokument | s. | cosine | lexical | RRF | wspiera | fragment |")
            add("|---|---|---|---|---|---|---|---|")
            for hit in hybrid.hits[:5]:
                add(
                    f"| {hit.rank} | {hit.document_id} | {hit.page_start}-{hit.page_end} | "
                    f"{_fmt(hit.semantic_score)} | {_fmt(hit.lexical_score)} | {_fmt(hit.hybrid_score)} | "
                    f"{', '.join(hit.supports) or '—'} | {_short(hit.text).replace('|', '/')} |"
                )
        reranked = run.result(question.id, "pipeline+reranker")
        if reranked is not None and reranked.error is None:
            add(f"\nWybór `pipeline+reranker` (evidence_status: {reranked.evidence_status}):\n")
            if reranked.hits:
                add("| # | dokument | s. | ocena | wspiera | uzasadnienie modelu |")
                add("|---|---|---|---|---|---|")
                for hit in reranked.hits:
                    add(
                        f"| {hit.rank} | {hit.document_id} | {hit.page_start}-{hit.page_end} | "
                        f"{_fmt(hit.rerank_score)} | {', '.join(hit.supports) or '—'} | "
                        f"{_short(hit.reason or '', 90).replace('|', '/')} |"
                    )
            else:
                add("Reranker nie wybrał żadnego fragmentu.")
        add("")

    add("## Metadane i strony\n")
    add(f"Sprawdzono {run.metadata_checked} zwróconych fragmentów (wszystkie konfiguracje).")
    add(f"- Niezgodności metadanych z katalogiem: {len(run.metadata_problems)}")
    for problem in run.metadata_problems[:30]:
        add(f"  - {problem}")
    add(f"- Niezgodności strony dowodu: {len(run.page_problems)}")
    for problem in run.page_problems[:30]:
        add(f"  - {problem}")
    add("")

    add("## Rozbicie według tagów (pytania z odpowiedzią)\n")
    tags = sorted({t for q in run.questions if q.answerable for t in q.tags})
    add("| tag | n | evidence_hit@6 `pipeline` | mrr_evidence `hybrid` |")
    add("|---|---|---|---|")
    for tag in tags:
        tagged = [q for q in run.questions if q.answerable and tag in q.tags]
        pipeline = [run.result(q.id, "pipeline") for q in tagged]
        hybrid = [run.result(q.id, "hybrid") for q in tagged]
        hit6 = [
            question_metrics(q, r, (6,))["evidence_hit@6"]
            for q, r in zip(tagged, pipeline)
            if r and not r.error
        ]
        mrr = [
            question_metrics(q, r, (6,))["mrr_evidence"]
            for q, r in zip(tagged, hybrid)
            if r and not r.error
        ]
        add(
            f"| {tag} | {len(tagged)} | {_fmt(mean(hit6)) if hit6 else '—'} | {_fmt(mean(mrr)) if mrr else '—'} |"
        )
    add("")

    ascii_configs = [c for c in run.configs if c.endswith("-ascii")]
    if ascii_configs:
        add("## Pytania bez polskich znaków (warianty pochodne)\n")
        add("| pytanie | konfiguracja | poz. źródła: oryg. → ASCII | poz. dowodu: oryg. → ASCII |")
        add("|---|---|---|---|")
        for question in run.questions:
            if not question.answerable:
                continue
            for config in ascii_configs:
                base = run.result(question.id, config.removesuffix("-ascii"))
                variant = run.result(question.id, config)
                if base is None or variant is None or base.error or variant.error:
                    continue
                add(
                    f"| {question.id} | `{config}` | {_fmt(source_rank(question, base.hits))} → "
                    f"{_fmt(source_rank(question, variant.hits))} | {_fmt(evidence_rank(base.hits))} → "
                    f"{_fmt(evidence_rank(variant.hits))} |"
                )
        add("")

    add("## Kontrola korpusu dla pytań bez odpowiedzi\n")
    for question_id, matches in run.corpus_absence.items():
        found = {p: chunks for p, chunks in matches.items() if chunks}
        add(f"- {question_id}: {'brak trafień wzorców odpowiedzi' if not found else found}")
    add("")
    return "\n".join(out)
