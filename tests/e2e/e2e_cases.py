"""
Konfiguracja testów E2E: lista uruchomień (które modele na których etapach) i sprawy.

Żeby zmienić kombinacje modeli, edytuj tylko listę URUCHOMIENIA poniżej. Każdy wpis jest
uruchamiany na każdej sprawie z CASES (albo na sprawach podanych w "sprawy").

Odpowiedzi w sprawach celowo nie są dopasowane do pytań modelu: test sprawdza, czy
Dopytywator radzi sobie z częściowo nieadekwatnymi odpowiedziami. Grupy słów-kluczy służą
wyłącznie do ostrzeżeń w raporcie (WARN), a nie do oceny PASS/FAIL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# URUCHOMIENIA — tę listę edytuj
#
#   nazwa              nazwa uruchomienia: litery, cyfry, „_” i „-”; musi być unikalna
#   dopytywator        model Dopytywatora (wywiad i CaseBrief)
#   zrodlowyszukiwacz  model Źródłowyszukiwacza (planner i reranker; embeddingi zawsze bge-m3)
#   pismogenerator     model Pismogeneratora
#   pismopoprawiator   opcjonalnie: model korekty językowej; domyślnie model Pismogeneratora,
#                      "brak" wyłącza korektę (jak przełącznik „Korekta językowa” w aplikacji)
#   sprawy             opcjonalnie: lista spraw, np. ["buspas"]; domyślnie wszystkie z CASES
#
# Model to jedno z:
#   "bielik", "pllum", "openrouter"     skróty z MODELE niżej
#   "qwen2.5:7b"                        dowolny inny model zarejestrowany w Ollamie
#   "openrouter:<id modelu>"            konkretny model OpenRouter, np.
#                                       "openrouter:meta-llama/llama-3.3-70b-instruct:free"
#
# Uruchomienia idą w kolejności z listy. Szybkie (OpenRouter) warto mieć na początku.

URUCHOMIENIA = [
    {
        "nazwa": "openrouter",
        "dopytywator": "openrouter",
        "zrodlowyszukiwacz": "openrouter",
        "pismogenerator": "openrouter",
    },
    {
        "nazwa": "bielik",
        "dopytywator": "bielik",
        "zrodlowyszukiwacz": "bielik",
        "pismogenerator": "bielik",
    },
    {
        "nazwa": "pllum",
        "dopytywator": "pllum",
        "zrodlowyszukiwacz": "pllum",
        "pismogenerator": "pllum",
    },
    {
        "nazwa": "mix_pllum_bielik_openrouter",
        "dopytywator": "pllum",
        "zrodlowyszukiwacz": "bielik",
        "pismogenerator": "openrouter",
    },
    {
        "nazwa": "mix_bielik_pllum_openrouter",
        "dopytywator": "bielik",
        "zrodlowyszukiwacz": "pllum",
        "pismogenerator": "openrouter",
    },
    {
        "nazwa": "mix_pllum_openrouter_bielik",
        "dopytywator": "pllum",
        "zrodlowyszukiwacz": "openrouter",
        "pismogenerator": "bielik",
    },
]

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


OPENROUTER = Provider(
    "openrouter", "openrouter", "openrouter/free", "OpenRouter (losowy darmowy model)"
)
BIELIK = Provider("bielik", "ollama", "pismak-bielik:latest", "Lokalny Bielik (Ollama)")
PLLUM = Provider("pllum", "ollama", "pismak-pllum:latest", "Lokalny PLLuM (Ollama)")

# Skróty dostępne w URUCHOMIENIA.
MODELE = {provider.id: provider for provider in (BIELIK, PLLUM, OPENROUTER)}

OPENROUTER_PREFIX = "openrouter:"
NO_CORRECTION = "brak"
_REQUIRED_KEYS = ("nazwa", "dopytywator", "zrodlowyszukiwacz", "pismogenerator")
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


HUMOR_MARKERS = (
    "happening",
    "satyr",
    "humor",
    "żart",
    "ironi",
    "artystyczn",
    "zabaw",
    "przebrani",
    "symboliczn",
)

CASES = (
    Case(
        id="kot_wrocek",
        title="Happening z Kotem Wrockiem",
        description=(
            "Chciałbym, aby prezydent Wrocławia, Jacek Sutryk, abdykował ze stanowiska i "
            "namaścił na swojego następcę Kota Wrocka. Dzięki temu raz na zawsze opanujemy "
            "plagę szczurów!"
        ),
        answers=(
            "Ma to być akcja humorystyczna, happening artystyczny.",
            "Tak, chciałbym, aby Kot Wrocek został prezydentem miasta.",
            "Chodzi przede wszystkim o zwrócenie uwagi na problem szczurów we Wrocławiu.",
            "Nie traktuję tego dosłownie — to satyra i sposób na nagłośnienie problemu.",
            "Chcielibyśmy zorganizować happening w centrum Wrocławia, w miejscu widocznym "
            "dla przechodniów.",
            "Nie mamy jeszcze ustalonego terminu ani konkretnej lokalizacji.",
            "Chcemy, żeby mieszkańcy mogli się przyłączyć i robić sobie zdjęcia.",
            "Ton powinien być ironiczny i zabawny, ale bez obrażania konkretnych osób.",
            "Główny przekaz jest taki, że problem szczurów wymaga skuteczniejszych działań miasta.",
            "Możemy przygotować transparenty, ulotki i przebranie kota.",
            "Organizatorami będą prawdopodobnie mieszkańcy działający nieformalnie.",
            "Chciałbym też wiedzieć, jakie działania miasto faktycznie podejmuje w sprawie "
            "szczurów.",
        ),
        what_is_tested=(
            "Czy Dopytywator zbiera wystarczające informacje i kończy rozmowę przed "
            "wykorzystaniem wszystkich 12 odpowiedzi, nie traktuje żartu jako dosłownego "
            "postulatu politycznego i nie wymyśla faktów."
        ),
        key_info={
            "Kot Wrocek": ("wrocek", "wrocka"),
            "problem szczurów": ("szczur",),
            "Wrocław": ("wrocław",),
            "satyryczny charakter (happening)": HUMOR_MARKERS,
        },
        humor_markers=HUMOR_MARKERS,
        literal_demand_markers=("abdyk", "ustąpi", "rezygnac", "odwoła", "namaszcz", "dymisj"),
        expects_early_end=True,
    ),
    Case(
        id="buspas",
        title="Buspas na ulicy Kazimierza Wielkiego",
        description="Chciałbym wprowadzenia buspasa na ulicy Kazimierza Wielkiego we Wrocławiu.",
        answers=(
            "Aktualnie więcej osób porusza się tam komunikacją miejską niż samochodami.",
            "Są tam korki, przez które komunikacja miejska jedzie wolniej.",
            "To absurd, że tak centralnie położona ulica pełni funkcję drogi tranzytowej.",
            "Chodzi mi o ulicę Kazimierza Wielkiego w centrum Wrocławia.",
            "Zależy mi na buspasie w obu kierunkach, o ile będzie to możliwe.",
            "Najważniejsze jest dla mnie zwiększenie punktualności i przewidywalności "
            "przejazdu komunikacji miejskiej.",
            "Nie mam własnych pomiarów ruchu — to moje obserwacje z codziennego poruszania się "
            "po mieście.",
            "Chciałbym ograniczyć ruch samochodów, które nie muszą przejeżdżać przez centrum.",
            "Nie postuluję całkowitego zamknięcia ulicy dla samochodów.",
            "Należy uwzględnić wpływ zmian na dostawy, mieszkańców, służby i pozostały ruch.",
            "Chciałbym, żeby miasto przeanalizowało możliwe warianty i udostępniło dane "
            "uzasadniające decyzję.",
            "Pismo powinno być konkretne i rzeczowe, bez agresywnego tonu.",
        ),
        what_is_tested=(
            "Czy model doprecyzowuje konkretny postulat, odróżnia obserwacje użytkownika od "
            "potwierdzonych danych i przygotowuje brief nadający się do wyszukiwania źródeł "
            "i wygenerowania pisma."
        ),
        key_info={
            "ulica Kazimierza Wielkiego": ("kazimierza wielkiego",),
            "buspas": (
                "buspas",
                "bus-pas",
                "pas autobus",
                "pas dla autobus",
                "pas dla komunikacji",
                "pas bus",
                "pasa autobus",
                "pasa dla autobus",
                "pasa dla komunikacji",
            ),
            "komunikacja miejska": ("komunikacj", "autobus", "tramwaj", "transport publiczn"),
            "Wrocław": ("wrocław",),
        },
        user_has_no_measurements=True,
        expected_evidence_status=("supported",),
    ),
)

# Lista sprawdzana przy zbieraniu testów: błąd we wpisie zatrzymuje pytest z opisem, który wpis
# poprawić, zanim ruszy jakikolwiek model.
SETUPS = build_setups(URUCHOMIENIA, [case.id for case in CASES])


def run_plan() -> list[tuple[Setup, Case]]:
    """Return every (setup, case) pair to run, in the order of URUCHOMIENIA and CASES."""
    return [
        (setup, case)
        for setup in SETUPS
        for case in CASES
        if not setup.case_ids or case.id in setup.case_ids
    ]
