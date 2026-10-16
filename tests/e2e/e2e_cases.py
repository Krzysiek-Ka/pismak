"""
Przypadki testów E2E: opis sprawy i 12 odpowiedzi podawanych kolejno, niezależnie od pytań.

Odpowiedzi celowo nie są dopasowane do pytań modelu: test sprawdza, czy Dopytywator radzi
sobie z częściowo nieadekwatnymi odpowiedziami. Grupy słów-kluczy służą wyłącznie do
ostrzeżeń w raporcie (WARN), a nie do oceny PASS/FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


# OpenRouter jest pierwszy: odpowiada w minutach, więc ewentualny problem ze środowiskiem
# widać, zanim ruszą wielogodzinne przebiegi lokalne na CPU.
PROVIDERS = (
    Provider("openrouter", "openrouter", "openrouter/free", "OpenRouter (losowy darmowy model)"),
    Provider("bielik", "ollama", "pismak-bielik:latest", "Lokalny Bielik (Ollama)"),
    Provider("pllum", "ollama", "pismak-pllum:latest", "Lokalny PLLuM (Ollama)"),
)

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
