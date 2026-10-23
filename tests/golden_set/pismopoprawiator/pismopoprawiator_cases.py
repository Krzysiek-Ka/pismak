"""
Złoty zbiór Pismopoprawiatora: pisma z wstawionymi błędami językowymi i pułapkami.

Każda sprawa to projekt pisma złożony tym samym rendererem co w aplikacji. Pola sprawy:

- `bledy` — pary (błędny fragment, poprawna forma). Poprawnych form może być kilka, np.
  „2025 r.” albo „2025 roku”. Służą do pomiaru, ile błędów model naprawił.
- `chronione` — napisy, które po korekcie muszą wystąpić dokładnie tyle samo razy co przed
  nią; ich zmiana jest naruszeniem treści. Nie zawierają skrótów, bo rozwinięcie skrótu
  („ul.” → „ulicy”, „mln zł” → „miliona złotych”) nie zmienia treści.
- `rozwiniecia` — przykładowe rozwinięcia skrótów (fragment, rozwinięty fragment), które
  korekta może wprowadzić bez naruszenia treści. Test zbioru sprawdza, że ocena nie liczy
  ich jako naruszeń. Strażnicy mogą część z nich odrzucić (np. „km/h” → „kilometrów na
  godzinę” jest zbyt odległe od oryginału); odrzucona poprawka nie trafia do pisma i nie
  jest błędem.

Listę modeli zmienia się w MODELE albo zmienną PISMAK_GOLDEN_MODELE (np. "bielik,openrouter").
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.Pismogenerator.pismogenerator_helper_functions import render_letter
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.brief_to_evidence.models import Evidence, EvidencePack, ResearchPlan

# ---------------------------------------------------------------------------
# MODELE — skróty jak w testach E2E: "bielik", "pllum", "openrouter" (openrouter/free),
# dowolny model z `ollama list` albo "openrouter:<id modelu>".

MODELE = ["openrouter", "bielik", "pllum"]

# Tryby porównywane na każdym piśmie (argumenty correct_letter): całe pismo w jednym
# wywołaniu i każdy akapit edytowalny w osobnym wywołaniu.
TRYBY = {
    "calosc": {"mode": "whole"},
    "po_akapitach": {"mode": "paragraphs", "chunk_chars": 1},
}

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldenCase:
    id: str
    title: str
    recipient: str
    request_summary: str
    argumentation: str
    sources: tuple[str, ...] = ()
    bledy: tuple[tuple[str, str | tuple[str, ...]], ...] = ()
    chronione: tuple[str, ...] = ()
    rozwiniecia: tuple[tuple[str, str], ...] = ()

    def letter(self) -> str:
        """Render the case the way the app renders a generated letter."""
        draft = DraftLetter(
            recipient=self.recipient,
            request_summary=self.request_summary,
            argumentation_markdown=self.argumentation,
            sources=list(self.sources),
            claims_requiring_verification=["Dane liczbowe podane przez wnioskodawcę."],
        )
        return render_letter(draft, EVIDENCE_PACK, USER_EVIDENCE)


def correct_forms(right: str | tuple[str, ...]) -> tuple[str, ...]:
    """Return every accepted form of a seeded error's fix; the first is the reference one."""
    return (right,) if isinstance(right, str) else right


def _evidence(evidence_id: str, title: str, url: str, page: int) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        chunk_id=f"chunk-{evidence_id}",
        document_id=f"document-{evidence_id}",
        document_title=title,
        document_kind="policy",
        hierarchy_rank=100,
        legal_form="resolution",
        page_start=page,
        page_end=page,
        excerpt="Fragment dokumentu.",
        retrieval_score=1.0,
        relevance_reason="Dotyczy sprawy.",
        source_url=url,
    )


EVIDENCE_PACK = EvidencePack(
    query_summary="Złoty zbiór",
    research_plan=ResearchPlan(case_summary="Złoty zbiór", intents=[]),
    evidence=[
        _evidence(
            "E1",
            "Wrocławska polityka mobilności",
            "https://www.wroclaw.pl/files/Wroclawska-Polityka-Mobilnosci.pdf",
            11,
        ),
        _evidence(
            "E2",
            "Strategia Rozwoju Wrocławia „Wrocław 2050”",
            "https://www.wroclaw.pl/beta2/files/dokumenty/Strategia_Wroclaw_2050.pdf",
            122,
        ),
    ],
)
USER_EVIDENCE = ["Obserwacje wnioskodawcy z codziennych przejazdów."]


CASES = (
    GoldenCase(
        id="buspas",
        title="Buspas: literówki, łączna pisownia, podwójna spacja",
        recipient="Zarząd Dróg i Utrzymania Miasta we Wrocławiu",
        request_summary="wprowadzenie buspasa na ulicy Kazimierza Wielkiego we Wrocławiu",
        argumentation=(
            "Na ul. Kazimierza Wielkiego autobusy stoją w korkach, ktore wydłużają przejazd. "
            "Według obserwacji wnioskodawcy w godzinach szczytu przejazd odcinka trwa ponad "
            "15 minut [wymaga weryfikacji].\n\n"
            "Wrocławska polityka mobilności wskazuje, że transport publiczny powinien mieć "
            "pierwszeństwo przed ruchem samochodowym [1]. Buspas napewno skróciłby czas "
            "przejazdu linii autobusowych.\n\n"
            "Nie postuluję całkowitego zamknięcia ulicy dla samochodów. Proszę o przeanalizowanie "
            "wariantów i udostępnienie danych, które  uzasadnią decyzję."
        ),
        sources=("E1",),
        bledy=(
            ("korkach, ktore", "korkach, które"),
            ("napewno", "na pewno"),
            ("które  uzasadnią", "które uzasadnią"),
        ),
        chronione=(
            "Kazimierza Wielkiego",
            "Według obserwacji",
            "15 minut",
            "[wymaga weryfikacji]",
            "[1]",
            "Nie postuluję",
            "powinien",
            "Proszę o przeanalizowanie",
            "wnioskuję o wprowadzenie",
        ),
        rozwiniecia=(("Na ul. Kazimierza Wielkiego", "Na ulicy Kazimierza Wielkiego"),),
    ),
    GoldenCase(
        id="przejscie_przy_szkole",
        title="Przejście przy szkole: przecinki przed „że” i „który”, spacja przed przecinkiem",
        recipient="Zarząd Dróg i Utrzymania Miasta we Wrocławiu",
        request_summary="wyznaczenie przejścia dla pieszych przy Szkole Podstawowej nr 42",
        argumentation=(
            "Dzieci ze Szkoły Podstawowej nr 42 codziennie przechodzą przez ul. Bardzką "
            "w miejscu bez oznakowania. Uważam że brak przejścia stwarza realne zagrożenie.\n\n"
            "Na odcinku obowiązuje ograniczenie do 50 km/h, a kierowcy często je przekraczają "
            "[wymaga weryfikacji]. Przejście które zostałoby wyznaczone przy wejściu do "
            "szkoły , poprawiłoby bezpieczeństwo uczniów.\n\n"
            "Wniosek dotyczy terenu działki [DO UZUPEŁNIENIA: numer działki]. Zajęcia w szkole "
            "rozpoczynają się 1 września 2026 r."
        ),
        sources=("U1",),
        bledy=(
            ("Uważam że", "Uważam, że"),
            ("Przejście które", "Przejście, które"),
            ("szkoły , poprawiłoby", "szkoły, poprawiłoby"),
        ),
        chronione=(
            "Szkoły Podstawowej",
            "42",
            "Bardzką",
            "50",
            "[wymaga weryfikacji]",
            "[DO UZUPEŁNIENIA: numer działki]",
            "1 września 2026",
        ),
        rozwiniecia=(
            ("Podstawowej nr 42 codziennie", "Podstawowej numer 42 codziennie"),
            ("przez ul. Bardzką", "przez ulicę Bardzką"),
            ("do 50 km/h", "do 50 kilometrów na godzinę"),
            ("1 września 2026 r.", "1 września 2026 roku."),
        ),
    ),
    GoldenCase(
        id="uchwala_fleksja",
        title="Droga rowerowa: fleksja przy uchwale, cytat i adres URL w treści",
        recipient="Prezydent Wrocławia",
        request_summary="rozważenie budowy drogi rowerowej łączącej osiedle Gaj z centrum",
        argumentation=(
            "Zgodnie z uchwała nr XLVIII/1169/13 Rady Miejskiej Wrocławia miasto rozwija sieć "
            "dróg rowerowych [1]. Polityka Rowerowa Wrocławia zakłada wzrost udziału ruchu "
            "rowerowego do 15% [1].\n\n"
            "Brakuje jednak łącznika dla mieszkańcy osiedla Gaj, którzy dojeżdżają do centrum. "
            "Wnioskodawca prosi o rozważenie budowy drogi rowerowej w celu skuteczniej obsługi "
            "tego ruchu.\n\n"
            "Strategia „Wrocław 2050” opisuje rower jako codzienny środek transportu [2]. "
            "Szczegóły są dostępne pod adresem https://www.wroclaw.pl/rowerowy-wroclaw."
        ),
        sources=("E1", "E2"),
        bledy=(
            ("z uchwała nr", ("z uchwałą nr", "z uchwałą numer")),
            ("dla mieszkańcy osiedla", "dla mieszkańców osiedla"),
            ("skuteczniej obsługi", "skuteczniejszej obsługi"),
        ),
        chronione=(
            "XLVIII/1169/13",
            "Rady Miejskiej Wrocławia",
            "15",
            "Polityka Rowerowa Wrocławia",
            "„Wrocław 2050”",
            "https://www.wroclaw.pl/rowerowy-wroclaw",
            "osiedla Gaj",
            "[1]",
            "[2]",
            "prosi o rozważenie",
        ),
        rozwiniecia=(("do 15% [1]", "do 15 procent [1]"),),
    ),
    GoldenCase(
        id="kot_wrocek",
        title="Happening z Kotem Wrockiem: pisownia łączna i rozdzielna, cytat, nazwiska",
        recipient="Prezydent Wrocławia",
        request_summary="zgodę na zorganizowanie happeningu z Kotem Wrockiem w centrum Wrocławia",
        argumentation=(
            "Proponujemy satyryczny happening, który przedewszystkim ma zwrócić uwagę na "
            "problem szczurów we Wrocławiu. Kot Wrocek, maskotka akcji, „obejmie urząd” tylko "
            "symbolicznie.\n\n"
            "Akcja potrwa conajmniej dwie godziny i odbędzie się w miejscu widocznym dla "
            "przechodniów [DO UZUPEŁNIENIA: miejsce i termin]. Nieznamy jeszcze liczby "
            "uczestników.\n\n"
            "Ton akcji będzie ironiczny, ale nie będzie obrażał konkretnych osób, w tym "
            "prezydenta Jacka Sutryka."
        ),
        sources=("U1",),
        bledy=(
            ("przedewszystkim", "przede wszystkim"),
            ("conajmniej", "co najmniej"),
            ("Nieznamy", "Nie znamy"),
        ),
        chronione=(
            "Kot Wrocek",
            "„obejmie urząd”",
            "Jacka Sutryka",
            "[DO UZUPEŁNIENIA: miejsce i termin]",
            "nie będzie obrażał",
            "dwie godziny",
            "satyryczny",
        ),
    ),
    GoldenCase(
        id="czyste_pismo",
        title="Pismo bez błędów: model nie powinien niczego zmieniać",
        recipient="Zarząd Dróg i Utrzymania Miasta we Wrocławiu",
        request_summary="zmianę programu sygnalizacji na skrzyżowaniu ul. Legnickiej",
        argumentation=(
            "Na skrzyżowaniu ul. Legnickiej i ul. Na Ostatnim Groszu brakuje sygnalizacji dla "
            "rowerzystów. Według obserwacji wnioskodawcy rowerzyści czekają tam nawet 3 minuty "
            "[wymaga weryfikacji].\n\n"
            "Wrocławska polityka mobilności wskazuje, że ruch rowerowy powinien być traktowany "
            "na równi z innymi środkami transportu [1]. Nie wnoszę o przebudowę skrzyżowania, "
            "a jedynie o zmianę programu sygnalizacji.\n\n"
            "Proszę o informację, czy taka zmiana jest możliwa i w jakim terminie mogłaby zostać "
            "wprowadzona."
        ),
        sources=("E1",),
        chronione=(
            "Legnickiej",
            "Na Ostatnim Groszu",
            "Według obserwacji",
            "3 minuty",
            "[wymaga weryfikacji]",
            "Nie wnoszę",
            "powinien",
            "mogłaby",
            "[1]",
        ),
        rozwiniecia=(
            # Zmiana, którą wprowadził Bielik w przebiegu z 2026-10-23.
            (
                "ul. Legnickiej i ul. Na Ostatnim Groszu",
                "ulic Legnickiej i Na Ostatnim Groszu",
            ),
            ("na skrzyżowaniu ul. Legnickiej", "na skrzyżowaniu ulicy Legnickiej"),
        ),
    ),
    GoldenCase(
        id="lista_postulatow",
        title="Lista postulatów z pogrubieniami: pisownia rozdzielna i „ó”",
        recipient="Zarząd Zieleni Miejskiej we Wrocławiu",
        request_summary="realizację trzech postulatów dotyczących okolic Hali Targowej",
        argumentation=(
            "Wnioskodawca przedstawia następujące postulaty:\n"
            "1. **Wyznaczenie strefy płatnego parkowania** na ul. Ołbińskiej.\n"
            "2. **Montaż stojaków rowerowych** przy Hali Targowej, gdzie brakuje conajmniej "
            "20 miejsc [wymaga weryfikacji].\n"
            "3. **Przegląd oznakowania** na odcinku od pl. Bema do ul. Jedności Narodowej.\n\n"
            "Postulaty wynikają z rozmów z mieszkańcami, ktorzy codziennie korzystają z tych "
            "miejsc."
        ),
        sources=("U1",),
        bledy=(
            ("conajmniej 20", "co najmniej 20"),
            ("mieszkańcami, ktorzy", "mieszkańcami, którzy"),
        ),
        chronione=(
            "1. **Wyznaczenie strefy płatnego parkowania**",
            "2. **Montaż stojaków rowerowych**",
            "3. **Przegląd oznakowania**",
            "Ołbińskiej",
            "Hali Targowej",
            "Bema",
            "Jedności Narodowej",
            "20 miejsc",
            "[wymaga weryfikacji]",
        ),
        rozwiniecia=(
            ("na ul. Ołbińskiej", "na ulicy Ołbińskiej"),
            ("od pl. Bema do ul. Jedności", "od placu Bema do ulicy Jedności"),
        ),
    ),
    GoldenCase(
        id="liczby_i_daty",
        title="Remont wiaduktu: spacje przy datach i skrótach, kwoty, godziny, paragrafy",
        recipient="Zarząd Inwestycji Miejskich we Wrocławiu",
        request_summary="udostępnienie harmonogramu remontu wiaduktu na ul. Grabiszyńskiej",
        argumentation=(
            "Remont wiaduktu na ul. Grabiszyńskiej trwa od marca 2025r. i według zapowiedzi "
            "potrwa do 30 czerwca 2027 r. Objazd wydłuża dojazd o ok.10 minut.\n\n"
            "Koszt inwestycji wynosi 1,5 mln zł, a utrudnienia obowiązują codziennie "
            "w godzinach 6:00–22:00. Zgodnie z § 3 ust. 2 regulaminu przewoźnik powinien "
            "informować pasażerów o zmianach.\n\n"
            "Proszę o udostępnienie harmonogramu prac oraz informacji, czy termin zakończenia "
            "jest nadal aktualny."
        ),
        sources=("U1",),
        bledy=(
            ("2025r.", ("2025 r.", "2025 roku")),
            ("ok.10 minut", ("ok. 10 minut", "około 10 minut")),
        ),
        chronione=(
            "Grabiszyńskiej",
            "30 czerwca 2027",
            "1,5",
            "6:00–22:00",
            "§ 3",
            "powinien",
        ),
        rozwiniecia=(
            ("wiaduktu na ul. Grabiszyńskiej trwa", "wiaduktu na ulicy Grabiszyńskiej trwa"),
            # Zmiana, którą wprowadził Bielik w przebiegu z 2026-10-23.
            ("1,5 mln zł", "1,5 miliona złotych"),
        ),
    ),
    GoldenCase(
        id="twarde_lamania",
        title="Akapit z twardymi łamaniami wierszy, jak w odpowiedziach Bielika",
        recipient="Ekosystem Wrocław",
        request_summary="ustawienie nowych koszy na śmieci na pl. Grunwaldzkim",
        argumentation=(
            "**Opis problemu:**\\\n"
            "Na pl. Grunwaldzkim brakuje koszy na śmieci, przez co odpady leża na chodnikach.\\\n"
            "**Postulat:**\\\n"
            "Proszę o ustawienie co najmniej 10 nowych koszy [wymaga weryfikacji].\\\n"
            "**Uzasadnienie:**\\\n"
            "Czystość przestrzeni publicznej wpływa na komfort mieszkańców i turystów, ktorzy "
            "odwiedzają tę okolicę."
        ),
        sources=("U1",),
        bledy=(
            ("odpady leża", "odpady leżą"),
            ("turystów, ktorzy", "turystów, którzy"),
        ),
        chronione=(
            "**Opis problemu:**\\\n",
            "**Postulat:**\\\n",
            "Grunwaldzkim",
            "10 nowych koszy",
            "[wymaga weryfikacji]",
        ),
        rozwiniecia=(
            # Zmiana, którą wprowadził Bielik w przebiegu z 2026-10-23.
            ("śmieci na pl. Grunwaldzkim", "śmieci na placu Grunwaldzkim"),
            ("Na pl. Grunwaldzkim brakuje", "Na placu Grunwaldzkim brakuje"),
        ),
    ),
    GoldenCase(
        id="cytat_z_bledem",
        title="Błąd wewnątrz cytatu musi zostać; poprawiane są tylko błędy poza cytatem",
        recipient="Miejskie Przedsiębiorstwo Komunikacyjne we Wrocławiu",
        request_summary="zaktualizowanie informacji dla pasażerów na przystanku Plac Wróblewskiego",
        argumentation=(
            "Na tablicy przy przystanku widnieje napis „Przystanek tymczasowo nieczyny”, choć "
            "autobusy zatrzymują sie tam codziennie.\n\n"
            "W uchwale budżetowej zapisano: „modernizacja przystanków na trasie linii 107 "
            "i 133”. Wnioskodawca prosi o aktualizacje informacji dla pasażerów [1]."
        ),
        sources=("E1",),
        bledy=(
            ("zatrzymują sie", "zatrzymują się"),
            ("o aktualizacje informacji", "o aktualizację informacji"),
        ),
        chronione=(
            "„Przystanek tymczasowo nieczyny”",
            "„modernizacja przystanków na trasie linii 107 i 133”",
            "[1]",
        ),
    ),
    GoldenCase(
        id="modalnosc_i_przeczenia",
        title="Nawierzchnia ul. Krakowskiej: ogonki i spacja obok przeczeń i modalności",
        recipient="Zarząd Dróg i Utrzymania Miasta we Wrocławiu",
        request_summary="przegląd nawierzchni na ul. Krakowskiej",
        argumentation=(
            "Miasto powinno niezwłocznie przeanalizowac stan nawierzchni na ul. Krakowskiej. "
            "Nie twierdzę, że droga jest nieprzejezdna, ale może stwarzać zagrożenie dla "
            "rowerzystów [wymaga weryfikacji].\n\n"
            "Należy rozważyć przynajmniej tymczasowe oznakowanie ubytków. Prawdopodobnie "
            "wystarczyłaby naprawa kilku  odcinków, a nie remont całej ulicy."
        ),
        sources=("U1",),
        bledy=(
            ("przeanalizowac", "przeanalizować"),
            ("kilku  odcinków", "kilku odcinków"),
        ),
        chronione=(
            "powinno",
            "Nie twierdzę",
            "nieprzejezdna",
            "może stwarzać",
            "Należy rozważyć",
            "Prawdopodobnie",
            "a nie remont",
            "Krakowskiej",
            "[wymaga weryfikacji]",
        ),
        rozwiniecia=(("nawierzchni na ul. Krakowskiej.", "nawierzchni na ulicy Krakowskiej."),),
    ),
)


def selected_models() -> list[str]:
    """Return the models to run: PISMAK_GOLDEN_MODELE when set, otherwise MODELE."""
    raw = os.getenv("PISMAK_GOLDEN_MODELE", "").strip()
    return [name.strip() for name in raw.split(",") if name.strip()] if raw else list(MODELE)
