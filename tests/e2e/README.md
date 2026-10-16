# Testy end-to-end

Sprawdzają cały przepływ Pismaka na prawdziwych modelach, bazie RAG i konfiguracji
z `.env`: opis sprawy → wywiad Dopytywatora → CaseBrief → Źródłowyszukiwacz → Pismogenerator.
Nie ma tu atrap modeli, bazy, retrievera ani generatora.

Zakres: trzech dostawców (OpenRouter `openrouter/free`, lokalny `pismak-bielik`,
lokalny `pismak-pllum`) × dwa przypadki = 6 uruchomień. W każdym uruchomieniu ten sam
dostawca obsługuje wszystkie etapy LLM (Dopytywator, planner, reranker, Pismogenerator).
Embeddingi zawsze liczy lokalny `bge-m3`.

## Uruchamianie tylko ręczne

Testy trwają godzinami i wysyłają zapytania do usług, więc **CI ich nie uruchamia**.
Bez `PISMAK_E2E=1` pytest nie zbiera z tego katalogu żadnego testu (`conftest.py`),
więc zwykłe `python -m pytest` przy pushu i pull requeście ich nie uruchamia ani nie
pokazuje jako pominiętych.

```powershell
# Wymagane: baza (docker compose up -d postgres) z zaindeksowanymi dokumentami,
# Ollama z bge-m3, pismak-bielik i pismak-pllum, OPENROUTER_KEY w .env.
$env:PISMAK_E2E = "1"
$env:PISMAK_E2E_LABEL = "po-zmianie"          # opcjonalna nazwa przebiegu

# Szybka kontrola ewaluatorów (sekundy, bez modeli i bazy)
.venv\Scripts\python.exe -m pytest tests/e2e/test_e2e_support.py -q

# Wszystkie 6 uruchomień (na CPU: 3–5 h)
.venv\Scripts\python.exe -m pytest tests/e2e/test_pismak_e2e.py -v -rA -s

# Wybrane uruchomienia, np. tylko OpenRouter (10–20 min) albo jeden przypadek
.venv\Scripts\python.exe -m pytest tests/e2e/test_pismak_e2e.py -k openrouter -v -rA -s
.venv\Scripts\python.exe -m pytest tests/e2e/test_pismak_e2e.py -k "bielik-buspas" -v -rA -s
```

Przebiegi OpenRouter wysyłają oba przypadki do usługi zewnętrznej (zob. ostrzeżenie
w aplikacji). Darmowy plan ma limity zapytań; odrzucenie (429) kończy uruchomienie jako
NIEUDANY z podanym powodem.

## Przypadki

Opisy i listy 12 odpowiedzi są w `e2e_cases.py`. Po każdym pytaniu modelu sterownik
podaje następną odpowiedź z listy, niezależnie od treści pytania: test celowo sprawdza,
czy model radzi sobie z częściowo nieadekwatnymi odpowiedziami. Limit to 12 pytań, ten
sam co w aplikacji (`services.MAX_INTERVIEW_QUESTIONS`). Po 12. pytaniu brief powstaje
z ostatniej odpowiedzi, bez 13. wywołania modelu.

1. `kot_wrocek` — happening z Kotem Wrockiem: czy wywiad kończy się przed limitem i czy
   model rozpoznaje satyrę zamiast dosłownego postulatu politycznego.
2. `buspas` — buspas na ul. Kazimierza Wielkiego: czy postulat jest konkretny, a obserwacje
   użytkownika nie są traktowane jak potwierdzone dane.

## Ocena

Wynik testu (PASS/FAIL) zależy **tylko od niezmienników**:

| id | niezmiennik |
|---|---|
| H1 | Rozmowa zaczyna się od dokładnego opisu sprawy. |
| H2 | Każda tura wywiadu bez błędu, pytania niepuste, odpowiedzi mieszczą się w oknie modelu. |
| H3 | Najwyżej 12 pytań, odpowiedzi i wywołań wywiadu; spójny sygnał gotowości (`model_ready` przed limitem albo `question_limit` po 12 pytaniach). |
| H4 | CaseBrief powstał i ma tytuł, problem, oczekiwany rezultat i co najmniej jeden postulat. |
| H5 | RAG odpytał bazę (wpis `stage_completed stage=retrieval` w logu); każdy dowód istnieje w `rag_chunks` z tym samym tekstem, dokumentem i stronami; brak dowodów oznacza `unsupported`; najwyżej 6 dowodów, 3 z dokumentu. |
| H6 | Pismo przeszło walidację aplikacji (adresat, brak zakazanych podstaw prawnych, tylko dozwolone źródła) i powstał `letter.md`. |
| H7 | Logi i raport nie zawierają klucza OpenRouter ani hasła bazy (znalezione są usuwane). |

Ocena jakości (S1–S9) to heurystyki: słowa-klucze, liczby i nazwy spoza rozmowy, wzorce
podstaw prawnych, fallbacki plannera i rerankera, ucięcia odpowiedzi, ponowienia i czas
wywołań. Trafia do raportu jako WARN i wymaga ręcznej oceny na podstawie pełnego briefu,
dowodów i pisma. Model nie musi kończyć wywiadu po tej samej liczbie pytań co inne modele.

Uruchomienie bez wymaganej usługi (baza, Ollama, model, `bge-m3`, klucz OpenRouter) jest
pomijane jako NIEURUCHOMIONY z powodem. Awaria w trakcie (timeout, brak połączenia,
odrzucenie, dwukrotnie błędna odpowiedź strukturalna) daje NIEUDANY z typem błędu,
komunikatem, jaki pokazałaby aplikacja, i miejscem w kodzie.

## Raport

`outputs/e2e/<czas>_<etykieta>/` (poza repozytorium, jak inne wyniki w `outputs/`):

- `report.md`, `results.json` — przepisywane po każdym uruchomieniu, więc przerwany
  przebieg też zostawia raport. Tabela zbiorcza: dostawca i model (skonfigurowany oraz
  faktyczne), liczba pytań, przyczyna zakończenia, CaseBrief, RAG, pismo, czas, błędy, wynik.
- `<dostawca>__<przypadek>/result.json`, `turns.json` — pytania z powodem i brakującymi
  tematami według modelu, podane odpowiedzi, czasy.
- `<dostawca>__<przypadek>/logs/pismak_*.log` — pełny log sprawy: budżety kontekstu,
  tokeny, odrzucone odpowiedzi modelu, ślady stosu.
- `<dostawca>__<przypadek>/artifacts/` — to samo, co zapisuje aplikacja: `conversation.json`,
  `brief.json`, `evidence.json`, `draft.json` lub `draft_with_error.json`, `letter.md`, `models.json`.

Każda sekcja raportu podaje polecenie, które odtwarza dane uruchomienie.

## Pliki

- `e2e_cases.py` — dostawcy, przypadki, słowa-klucze do ostrzeżeń.
- `e2e_support.py` — sterownik przebiegu (ta sama ścieżka co `src/Wyswietlator/app.py`:
  `dopytywator.interview_step`, `services.build_zrodlowyszukiwator`, `pismogenerator.build_draft`),
  kontrole, analiza logu, raport, usuwanie danych uwierzytelniających.
- `test_pismak_e2e.py` — warunki wstępne i 6 uruchomień.
- `test_e2e_support.py` — kontrola ewaluatorów bez modeli.
- `conftest.py` — bramka `PISMAK_E2E=1`.
