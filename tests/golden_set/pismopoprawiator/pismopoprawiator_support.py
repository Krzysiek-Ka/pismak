"""
Złoty zbiór Pismopoprawiatora: uruchomienie na prawdziwym modelu, ocena i raport.

Naruszenie treści to każda różnica wykryta przez content_violations (część pisma poza
korektą, akapity tylko do odczytu, liczba akapitów, długość, elementy chronione) albo zmiana
liczby wystąpień napisu z listy `chronione` sprawy. Rozwinięcie skrótu („ul.” → „ulicy”)
nie jest naruszeniem. Cel: zero naruszeń na każdym modelu i w każdym trybie. Liczba
naprawionych błędów i poprawki nadmiarowe trafiają tylko do raportu.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pismopoprawiator_cases import TRYBY, GoldenCase, correct_forms

from src._shared.model_context import detect_context_tokens
from src._shared.serwer import LLMConfigurationError
from src.Pismopoprawiator import pismopoprawiator
from src.Pismopoprawiator.pismopoprawiator_helper_functions import content_violations
from src.Pismopoprawiator.structures import CorrectionResult
from src.Wyswietlator import services

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "golden_set" / "pismopoprawiator"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
OPENROUTER_PREFIX = "openrouter:"
SHORTCUTS = {
    "bielik": ("ollama", "pismak-bielik:latest"),
    "pllum": ("ollama", "pismak-pllum:latest"),
    "openrouter": ("openrouter", "openrouter/free"),
}
PASSED, FAILED, NOT_RUN = "ZALICZONY", "NIEUDANY", "NIEURUCHOMIONY"


@dataclass(frozen=True)
class GoldenModel:
    name: str
    provider: str
    model: str


def resolve_model(value: str) -> GoldenModel:
    """Turn a model entry (shortcut, Ollama model or `openrouter:<id>`) into a provider."""
    value = value.strip()
    if value in SHORTCUTS:
        return GoldenModel(value, *SHORTCUTS[value])
    if value.startswith(OPENROUTER_PREFIX):
        model = value.removeprefix(OPENROUTER_PREFIX).strip()
        if not model:
            raise ValueError(f"brak identyfikatora modelu po „{OPENROUTER_PREFIX}”")
        return GoldenModel(value, "openrouter", model)
    # Ollama traktuje nazwę bez tagu jak `:latest`.
    model = value if ":" in value.rsplit("/", 1)[-1] else f"{value}:latest"
    return GoldenModel(value, "ollama", model)


def available_ollama_models() -> list[str] | None:
    """Return the generation models registered in Ollama, or None when it does not answer."""
    try:
        return services.list_generation_models(services.get_openai_client())
    except Exception:
        return None


def model_problem(model: GoldenModel, ollama_models: list[str] | None) -> str | None:
    """Return why a model cannot be tested on this machine, or None when it can."""
    if model.provider == "ollama":
        if ollama_models is None:
            return "Ollama nie odpowiada"
        if model.model not in ollama_models:
            return f"model {model.model} nie jest zarejestrowany w Ollamie"
        return None
    try:
        services.get_llm_client("openrouter")
    except LLMConfigurationError as exc:
        return str(exc)
    return None


# ---------------------------------------------------------------------------
# Ocena


def violations(case: GoldenCase, original: str, corrected: str) -> list[str]:
    """List every change of content: guard findings and changed protected strings of the case."""
    found = content_violations(original, corrected)
    for text in case.chronione:
        before, after = original.count(text), corrected.count(text)
        if before != after:
            found.append(f"napis chroniony „{text}”: {before} → {after} wystąpień")
    return found


def fixed_errors(case: GoldenCase, original: str, corrected: str) -> tuple[list[str], list[str]]:
    """Split the seeded errors into the fixed ones and the ones still in the letter."""
    fixed, missed = [], []
    for wrong, right in case.bledy:
        # Błąd jest naprawiony, gdy zniknął, a pojawiła się któraś z poprawnych form.
        removed = corrected.count(wrong) < original.count(wrong)
        if removed and any(
            corrected.count(form) > original.count(form) for form in correct_forms(right)
        ):
            fixed.append(wrong)
        else:
            missed.append(wrong)
    return fixed, missed


def extra_corrections(case: GoldenCase, result: CorrectionResult) -> list[str]:
    """Return applied corrections that touch none of the seeded errors, for manual review."""
    extra = []
    for item in result.applied:
        if not any(wrong in item.oryginal or item.oryginal in wrong for wrong, _ in case.bledy):
            extra.append(f"{item.akapit} „{item.oryginal}” → „{item.poprawka}” ({item.kategoria})")
    return extra


def reason_kind(reason: str) -> str:
    """Group rejection reasons by their kind, without paragraph numbers and values."""
    return re.sub(r"A\d+", "A#", reason.split(":")[0]).strip()


@dataclass
class GoldenRun:
    model: str
    provider: str
    mode: str
    case_id: str
    status: str = NOT_RUN
    not_run_reason: str | None = None
    correction_status: str | None = None
    actual_model: str | None = None
    violations: list[str] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    suggestions: int = 0
    calls: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.model}-{self.mode}-{self.case_id}"


def evaluate(
    model: GoldenModel, mode: str, case: GoldenCase, letter: str, result: CorrectionResult
) -> GoldenRun:
    """Score one correction of a golden letter."""
    fixed, missed = fixed_errors(case, letter, result.corrected_text)
    run = GoldenRun(
        model=model.name,
        provider=model.provider,
        mode=mode,
        case_id=case.id,
        correction_status=result.status,
        actual_model=result.model,
        violations=violations(case, letter, result.corrected_text),
        fixed=fixed,
        missed=missed,
        extra=extra_corrections(case, result),
        rejected=[
            f"{item.correction.akapit} „{item.correction.oryginal}” → "
            f"„{item.correction.poprawka}”: {item.reason}"
            for item in result.rejected
        ],
        rejection_reasons=[item.reason for item in result.rejected],
        suggestions=len(result.suggestions),
        calls=result.calls,
        elapsed_seconds=result.elapsed_seconds,
        errors=list(result.errors),
    )
    run.status = FAILED if run.violations else PASSED
    return run


# ---------------------------------------------------------------------------
# Przebieg i raport


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "-"


class GoldenSession:
    """Run golden cases, keep the results and rewrite the report after every run."""

    KEY = pytest.StashKey["GoldenSession"]()

    def __init__(self, label: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.label = label
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.commit = git_commit()
        self.dir = REPORTS_DIR / f"{stamp}_{label}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs: list[GoldenRun] = []
        self.session_id = uuid.uuid4().hex
        # Logi poprawek (zastosowanych i odrzuconych) całego przebiegu w jednym pliku.
        self._logger = logging.getLogger("pismak")
        self._previous_level = self._logger.level
        self._handler = logging.FileHandler(self.dir / "golden.log", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter(LOG_FORMAT))
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.INFO)

    @property
    def report_path(self) -> Path:
        return self.dir / "report.md"

    def close(self) -> None:
        self._logger.removeHandler(self._handler)
        self._handler.close()
        self._logger.setLevel(self._previous_level)

    def run(self, model: GoldenModel, mode: str, case: GoldenCase) -> GoldenRun:
        """Correct one golden letter with the model in the given mode and score it."""
        settings = TRYBY[mode]
        letter = case.letter()
        llm = pismopoprawiator.make_llm(
            services.get_llm_client(model.provider),
            model.model,
            model.provider,
            session_id=self.session_id,
            context_tokens=detect_context_tokens(
                model.provider, model.model, session_id=self.session_id
            ),
        )
        result = pismopoprawiator.correct_letter(
            letter, llm, session_id=self.session_id, **settings
        )
        run = evaluate(model, mode, case, letter, result)
        self._save_letters(run, letter, result)
        self.add(run)
        return run

    def add_not_run(self, model: GoldenModel, mode: str, case: GoldenCase, reason: str) -> None:
        self.add(
            GoldenRun(
                model=model.name,
                provider=model.provider,
                mode=mode,
                case_id=case.id,
                not_run_reason=reason,
            )
        )

    def add(self, run: GoldenRun) -> None:
        self.runs.append(run)
        self._write()

    def _save_letters(self, run: GoldenRun, letter: str, result: CorrectionResult) -> None:
        run_dir = self.dir / f"{run.model.replace('/', '_').replace(':', '_')}__{run.mode}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"{run.case_id}__bez_korekty.md").write_text(letter, encoding="utf-8")
        (run_dir / f"{run.case_id}__po_korekcie.md").write_text(
            result.corrected_text, encoding="utf-8"
        )
        (run_dir / f"{run.case_id}__korekta.json").write_text(
            result.model_dump_json(indent=2, exclude={"original_text", "corrected_text"}),
            encoding="utf-8",
        )

    def summary_rows(self) -> list[dict[str, Any]]:
        """Aggregate the finished runs per model and mode."""
        groups: dict[tuple[str, str], list[GoldenRun]] = {}
        for run in self.runs:
            if run.not_run_reason is None:
                groups.setdefault((run.model, run.mode), []).append(run)
        rows = []
        for (model, mode), runs in groups.items():
            seeded = sum(len(run.fixed) + len(run.missed) for run in runs)
            fixed = sum(len(run.fixed) for run in runs)
            rows.append(
                {
                    "model": model,
                    "tryb": mode,
                    "pism": len(runs),
                    "naruszenia": sum(len(run.violations) for run in runs),
                    "naprawione": f"{fixed}/{seeded}",
                    "nadmiarowe": sum(len(run.extra) for run in runs),
                    "odrzucone": sum(len(run.rejected) for run in runs),
                    "nieudane": sum(run.correction_status == "failed" for run in runs),
                    "wycofane": sum(run.correction_status == "reverted" for run in runs),
                    "wywołania": sum(run.calls for run in runs),
                    "średni czas [s]": round(
                        sum(run.elapsed_seconds for run in runs) / len(runs), 1
                    ),
                }
            )
        return rows

    def text_summary(self) -> list[str]:
        rows = self.summary_rows()
        if not rows:
            return ["Brak ukończonych uruchomień."]
        headers = list(rows[0])
        widths = [max(len(h), *(len(str(row[h])) for row in rows)) for h in headers]
        lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]
        for row in rows:
            lines.append(
                "  ".join(str(row[h]).ljust(w) for h, w in zip(headers, widths, strict=True))
            )
        return lines

    def _write(self) -> None:
        (self.dir / "results.json").write_text(
            json.dumps([asdict(run) for run in self.runs], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.report_path.write_text(self.render_report(), encoding="utf-8")

    def render_report(self) -> str:
        finished = [run for run in self.runs if run.not_run_reason is None]
        total = sum(len(run.violations) for run in finished)
        lines = [
            f"# Złoty zbiór Pismopoprawiatora — {self.label}",
            "",
            f"- Start: {self.started_at}, commit: `{self.commit}`",
            f"- Naruszenia treści łącznie: **{total}** (cel: 0)",
            f"- Log poprawek: `{(self.dir / 'golden.log').relative_to(PROJECT_ROOT).as_posix()}`",
            "",
            "## Podsumowanie (model × tryb)",
            "",
        ]
        rows = self.summary_rows()
        if rows:
            headers = list(rows[0])
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "---|" * len(headers))
            for row in rows:
                lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        else:
            lines.append("Brak ukończonych uruchomień.")

        reasons = Counter(
            reason_kind(reason) for run in finished for reason in run.rejection_reasons
        )
        if reasons:
            lines += ["", "## Odrzucone poprawki według powodu", ""]
            lines += [f"- {reason}: {count}" for reason, count in reasons.most_common()]

        lines += [
            "",
            "## Uruchomienia",
            "",
            "| model | tryb | pismo | wynik | status korekty | naruszenia | naprawione | "
            "nadmiarowe | odrzucone | czas [s] |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for run in self.runs:
            if run.not_run_reason is not None:
                lines.append(
                    f"| {run.model} | {run.mode} | {run.case_id} | {NOT_RUN} | "
                    f"{run.not_run_reason} | | | | | |"
                )
                continue
            lines.append(
                f"| {run.model} | {run.mode} | {run.case_id} | {run.status} | "
                f"{run.correction_status} | {len(run.violations)} | "
                f"{len(run.fixed)}/{len(run.fixed) + len(run.missed)} | {len(run.extra)} | "
                f"{len(run.rejected)} | {run.elapsed_seconds:.1f} |"
            )

        lines += ["", "## Szczegóły", ""]
        for run in finished:
            lines.append(f"### {run.id} (model: {run.actual_model or '-'})")
            lines.append("")
            for title, items in (
                ("Naruszenia", run.violations),
                ("Nienaprawione błędy", run.missed),
                ("Poprawki nadmiarowe (do ręcznej oceny)", run.extra),
                ("Odrzucone poprawki", run.rejected),
                ("Błędy wywołań", run.errors),
            ):
                if items:
                    lines.append(f"**{title}:**")
                    lines += [f"- {item}" for item in items]
                    lines.append("")
            if not any((run.violations, run.missed, run.extra, run.rejected, run.errors)):
                lines += ["Bez uwag.", ""]
        return "\n".join(lines) + "\n"
