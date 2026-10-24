"""
Konfiguracja testów E2E: uruchomienia (który model na którym etapie) i sprawy.

Sprawy leżą w tests/e2e/cases/<id>.json w układzie złotych zbiorów: opis sprawy w `input`,
odpowiedzi użytkownika w `user_answers` (podawane po kolei, celowo niedopasowane do pytań:
test sprawdza, czy Dopytywator radzi sobie z częściowo nieadekwatnymi odpowiedziami), limit
pytań w `limits`, słowa-klucze i oczekiwania w `expected`, ocena człowieka w `manual_review`.
Grupy słów-kluczy służą wyłącznie do ostrzeżeń w raporcie (WARN), a nie do oceny PASS/FAIL.

Uruchomienia pochodzą z konfiguracji projektu (tests/golden_set/golden_support.py): każdy
dostawca obsługuje wszystkie etapy, jak w standardzie testów. PISMAK_GOLDEN_PROVIDERS zawęża
dostawców. Opcjonalne uruchomienia mieszane podaje się w PISMAK_E2E_MIXED jako listę wpisów
rozdzielonych przecinkami, każdy w formie
`dopytywator/zrodlowyszukiwacz/pismogenerator[/pismopoprawiator]`, np. "pllum/bielik/openrouter".
Model to skrót ("bielik", "pllum", "openrouter"), dowolny model Ollamy albo
"openrouter:<id modelu>"; "brak" jako czwarty element wyłącza korektę.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from tests.golden_set.golden_support import configured_providers, load_cases

E2E_DIR = Path(__file__).resolve().parent
SUPPORTED_EXPECTED = {
    "key_info",
    "humor_markers",
    "literal_demand_markers",
    "user_has_no_measurements",
    "expects_early_end",
    "expected_evidence_status",
}

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    id: str
    provider: str  # "ollama" albo "openrouter", jak w services.get_llm_client
    model: str
    label: str


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    description: str
    answers: tuple[str, ...]
    what_is_tested: str
    # Informacje, których brief nie powinien zgubić: nazwa -> warianty (dowolny wystarcza).
    key_info: dict[str, tuple[str, ...]]
    # Znaczniki satyrycznego charakteru sprawy (tylko Case 1).
    humor_markers: tuple[str, ...] = ()
    # Słowa oznaczające dosłowne żądanie polityczne (tylko Case 1).
    literal_demand_markers: tuple[str, ...] = ()
    # Użytkownik nie ma pomiarów: obserwacje nie mogą stać się dowodami (tylko Case 2).
    user_has_no_measurements: bool = False
    # Czy oczekujemy zakończenia wywiadu przed limitem pytań.
    expects_early_end: bool = False
    expected_evidence_status: tuple[str, ...] = field(default_factory=tuple)
    manual_review: tuple[str, ...] = ()
    max_questions: int = 12


@dataclass(frozen=True)
class Setup:
    """Dostawcy etapów LLM w jednym uruchomieniu — trzy wybory z panelu konfiguracji aplikacji."""

    id: str
    dopytywator: Provider
    zrodla: Provider  # planner i reranker Źródłowyszukiwacza
    pismo: Provider
    # Sprawy, na których uruchomienie jest wykonywane; pusta krotka oznacza wszystkie.
    case_ids: tuple[str, ...] = ()
    # Model korekty językowej; None oznacza korektę wyłączoną.
    poprawiator: Provider | None = None

    @property
    def stages(self) -> tuple[tuple[str, Provider], ...]:
        stages = (
            ("Dopytywator", self.dopytywator),
            ("Źródłowyszukiwacz", self.zrodla),
            ("Pismogenerator", self.pismo),
        )
        if self.poprawiator is not None:
            stages += (("Pismopoprawiator", self.poprawiator),)
        return stages

    @property
    def label(self) -> str:
        providers = {provider for _, provider in self.stages}
        if len(providers) == 1:
            return self.dopytywator.label
        return " → ".join(f"{stage}: {provider.label}" for stage, provider in self.stages)


def _provider(label: str, provider: str, model: str) -> Provider:
    where = "OpenRouter (losowy darmowy model)" if provider == "openrouter" else f"{model} (Ollama)"
    return Provider(label, provider, model, where)


# Skróty modeli: modele projektu z modele/<nazwa>/Modelfile i router OpenRouter.
MODELE = {p.label: _provider(p.label, p.provider, p.model) for p in configured_providers()}
OPENROUTER = MODELE.get("openrouter") or _provider("openrouter", "openrouter", "openrouter/free")
BIELIK = MODELE.get("bielik") or _provider("bielik", "ollama", "pismak-bielik:latest")
PLLUM = MODELE.get("pllum") or _provider("pllum", "ollama", "pismak-pllum:latest")

OPENROUTER_PREFIX = "openrouter:"
NO_CORRECTION = "brak"
_REQUIRED_KEYS = ("nazwa", "dopytywator", "zrodlowyszukiwacz", "pismogenerator")
_STAGE_KEYS = ("dopytywator", "zrodlowyszukiwacz", "pismogenerator", "pismopoprawiator")
_OPTIONAL_KEYS = ("pismopoprawiator", "sprawy")
_NAME = re.compile(r"[A-Za-z0-9_-]+")


def resolve_model(value: str) -> Provider:
    """Turn a model entry (shortcut, Ollama model or `openrouter:<id>`) into a provider."""
    value = value.strip()
    if value in MODELE:
        return MODELE[value]
    if value.startswith(OPENROUTER_PREFIX):
        model = value.removeprefix(OPENROUTER_PREFIX).strip()
        if not model:
            raise ValueError(f"brak identyfikatora modelu po „{OPENROUTER_PREFIX}”")
        return Provider(value, "openrouter", model, f"OpenRouter {model}")
    # Ollama traktuje nazwę bez tagu jak `:latest`; z tagiem łatwiej porównać ją z listą modeli.
    model = value if ":" in value.rsplit("/", 1)[-1] else f"{value}:latest"
    return Provider(value, "ollama", model, f"{model} (Ollama)")


def build_setups(entries: list[dict], case_ids: list[str]) -> tuple[Setup, ...]:
    """Validate the run list and turn every entry into a setup; errors name the entry."""
    setups: list[Setup] = []
    for number, entry in enumerate(entries, start=1):
        where = f"URUCHOMIENIA, wpis {number}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: wpis musi być słownikiem {{...}}")
        where = f"{where} ({entry.get('nazwa', 'bez nazwy')})"
        missing = [key for key in _REQUIRED_KEYS if not str(entry.get(key) or "").strip()]
        if missing:
            raise ValueError(f"{where}: brak pól: {', '.join(missing)}")
        unknown = sorted(set(entry) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
        if unknown:
            allowed = ", ".join(_REQUIRED_KEYS + _OPTIONAL_KEYS)
            raise ValueError(f"{where}: nieznane pola {unknown}; dozwolone: {allowed}")
        name = entry["nazwa"].strip()
        if not _NAME.fullmatch(name):
            raise ValueError(f"{where}: nazwa może zawierać tylko litery, cyfry, „_” i „-”")
        if any(setup.id == name for setup in setups):
            raise ValueError(f"{where}: nazwa „{name}” już występuje na liście")
        cases = tuple(entry.get("sprawy") or ())
        unknown_cases = [case for case in cases if case not in case_ids]
        if unknown_cases:
            raise ValueError(
                f"{where}: nieznane sprawy {unknown_cases}; dostępne: {', '.join(case_ids)}"
            )
        try:
            stages = [resolve_model(str(entry[key])) for key in _REQUIRED_KEYS[1:]]
            correction = str(entry.get("pismopoprawiator") or entry["pismogenerator"])
            poprawiator = None if correction.strip() == NO_CORRECTION else resolve_model(correction)
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        setups.append(Setup(name, *stages, case_ids=cases, poprawiator=poprawiator))
    if not setups:
        raise ValueError("URUCHOMIENIA: lista jest pusta — dodaj co najmniej jeden wpis")
    return tuple(setups)


def _case(raw) -> Case:
    """Turn a case file into the Case the E2E driver and evaluators use."""
    title, _, what = raw.description.partition(": ")
    expected = raw.expected
    return Case(
        id=raw.id,
        title=title,
        description=raw.input,
        answers=tuple(raw.user_answers),
        what_is_tested=what or raw.description,
        key_info={name: tuple(variants) for name, variants in expected.get("key_info", {}).items()},
        humor_markers=tuple(expected.get("humor_markers", ())),
        literal_demand_markers=tuple(expected.get("literal_demand_markers", ())),
        user_has_no_measurements=bool(expected.get("user_has_no_measurements", False)),
        expects_early_end=bool(expected.get("expects_early_end", False)),
        expected_evidence_status=tuple(expected.get("expected_evidence_status", ())),
        manual_review=tuple(raw.manual_review),
        max_questions=int(raw.limits["max_questions"]),
    )


CASES = tuple(
    _case(raw)
    for raw in load_cases(
        E2E_DIR, supported_expected=SUPPORTED_EXPECTED, required_limits={"max_questions"}
    )
)


def mixed_entries(value: str) -> list[dict]:
    """Turn PISMAK_E2E_MIXED ("a/b/c[/d], ...") into URUCHOMIENIA-style entries."""
    entries = []
    for item in (part.strip() for part in value.split(",")):
        if not item:
            continue
        models = [m.strip() for m in item.split("/")]
        if len(models) not in (3, 4):
            raise ValueError(f"PISMAK_E2E_MIXED: „{item}” musi mieć 3 albo 4 modele rozdzielone /")
        entry = dict(zip(_STAGE_KEYS, models))
        entry["nazwa"] = "mix_" + "_".join(re.sub(r"[^A-Za-z0-9]+", "", m) for m in models)
        entries.append(entry)
    return entries


def configured_entries() -> list[dict]:
    """Return one run per configured provider, then the optional mixed runs."""
    entries = [
        {"nazwa": label, "dopytywator": label, "zrodlowyszukiwacz": label, "pismogenerator": label}
        for label in MODELE
    ]
    return entries + mixed_entries(os.getenv("PISMAK_E2E_MIXED", ""))


# Lista sprawdzana przy zbieraniu testów: błąd we wpisie zatrzymuje pytest z opisem, który wpis
# poprawić, zanim ruszy jakikolwiek model.
SETUPS = build_setups(configured_entries(), [case.id for case in CASES])


def run_plan() -> list[tuple[Setup, Case]]:
    """Return every (setup, case) pair to run, in the order of the setups and CASES."""
    return [
        (setup, case)
        for setup in SETUPS
        for case in CASES
        if not setup.case_ids or case.id in setup.case_ids
    ]
