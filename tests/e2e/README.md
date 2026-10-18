# Testy end-to-end

Sprawdzają cały przepływ Pismaka na prawdziwych modelach, bazie RAG i konfiguracji
z `.env`: opis sprawy → wywiad Dopytywatora → CaseBrief → Źródłowyszukiwacz → Pismogenerator.
Nie ma tu atrap modeli, bazy, retrievera ani generatora.

## Jakie modele testować — lista `URUCHOMIENIA`

Kombinacje modeli wpisuje się na początku `e2e_cases.py`, w liście `URUCHOMIENIA`. Każdy
wpis to jedno uruchomienie, wykonywane na każdej sprawie (`kot_wrocek`, `buspas`) albo
tylko na sprawach z pola `sprawy`. Etapy odpowiadają trzem wyborom dostawcy w panelu
aplikacji; Źródłowyszukiwacz obejmuje planner i reranker, embeddingi zawsze liczy lokalny `bge-m3`.

```python
URUCHOMIENIA = [
    {"nazwa": "bielik", "dopytywator": "bielik", "zrodlowyszukiwacz": "bielik", "pismogenerator": "bielik"},
    {"nazwa": "pllum_or", "dopytywator": "pllum", "zrodlowyszukiwacz": "openrouter", "pismogenerator": "bielik"},
    {"nazwa": "qwen_buspas", "dopytywator": "qwen2.5:7b", "zrodlowyszukiwacz": "bielik",
     "pismogenerator": "openrouter:meta-llama/llama-3.3-70b-instruct:free", "sprawy": ["buspas"]},
]
```

| pole | znaczenie |
|---|---|
| `nazwa` | unikalna nazwa uruchomienia (litery, cyfry, `_`, `-`); pojawia się w tabeli i w nazwie katalogu wyników |
| `dopytywator` | model wywiadu i CaseBrief |
| `zrodlowyszukiwacz` | model plannera i rerankera |
| `pismogenerator` | model piszący pismo |
| `sprawy` | opcjonalnie: lista spraw; domyślnie wszystkie |

Model to skrót `bielik` (`pismak-bielik`), `pllum` (`pismak-pllum`) lub `openrouter`
(`openrouter/free`, losowy darmowy model — tak jak w aplikacji), dowolna inna nazwa modelu
z `ollama list` (np. `qwen2.5:7b`; bez tagu oznacza `:latest`) albo konkretny model
OpenRouter z przedrostkiem `openrouter:`. Literówkę w nazwie pola, powtórzoną nazwę albo
nieznaną sprawę pytest zgłasza od razu przy zbieraniu testów, z numerem wpisu do poprawienia.
Model, którego nie ma w Ollamie, oznacza uruchomienie jako NIEURUCHOMIONE z powodem.

## Uruchamianie tylko ręczne

Testy trwają godzinami i wysyłają zapytania do usług, więc **CI ich nie uruchamia**.
Bez `PISMAK_E2E=1` pytest nie zbiera z tego katalogu żadnego testu (`conftest.py`),
więc zwykłe `python -m pytest` przy pushu i pull requeście ich nie uruchamia ani nie
pokazuje jako pominiętych.

```powershell
# Wymagane: baza (docker compose up -d postgres) z zaindeksowanymi dokumentami,
# Ollama z bge-m3 i modelami z URUCHOMIENIA, OPENROUTER_KEY w .env (gdy lista go używa).
$env:PISMAK_E2E = "1"
$env:PISMAK_E2E_LABEL = "po-zmianie"          # opcjonalna nazwa przebiegu

# Szybka kontrola listy URUCHOMIENIA i ewaluatorów (sekundy, bez modeli i bazy)
.venv\Scripts\python.exe -m pytest tests/e2e/test_e2e_support.py -q

# Podgląd, co zostanie uruchomione (nic nie wywołuje modeli)
.venv\Scripts\python.exe -m pytest tests/e2e/test_pismak_e2e.py --collect-only -q

# Wszystkie uruchomienia z listy, po kolei; na końcu tabela podsumowania
.venv\Scripts\python.exe -m pytest tests/e2e/test_pismak_e2e.py -v

# To samo z bieżącym podglądem postępu (wynik każdego uruchomienia zaraz po nim)
.venv\Scripts\python.exe -m pytest tests/e2e/test_pismak_e2e.py -v -s

# Jedno uruchomienie: pełny identyfikator <nazwa>-<sprawa>, bo `-k` dopasowuje fragment nazwy
.venv\Scripts\python.exe -m pytest "tests/e2e/test_pismak_e2e.py::test_pismak_end_to_end[bielik-buspas]" -v
```

Na końcu wyjścia pytest pojawia się sekcja **Podsumowanie E2E**: tabela z jednym wierszem
na uruchomienie (nazwa, sprawa, model każdego etapu, liczba pytań, kto zakończył wywiad,
CaseBrief, wynik RAG, pismo, czas, wynik, liczba ostrzeżeń), pod nią przyczyny niepowodzeń
i ścieżka do pełnego raportu. Ta sama tabela otwiera `report.md`. Czas trwania zależy od
listy: na CPU lokalne uruchomienie zajmuje zwykle 20–40 min, OpenRouter kilka minut.

Każda sekcja raportu podaje gotowe polecenie odtwarzające dane uruchomienie.

Komputer nie może zasnąć w trakcie testu: uśpienie wstrzymuje wywołanie modelu (może ono
zawisnąć do wznowienia systemu) i zawyża zmierzone czasy. Podłącz zasilanie i wyłącz
usypianie na czas przebiegu.

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
  przebieg też zostawia raport. Otwiera go ta sama tabela podsumowania co w terminalu;
  modele, które faktycznie odpowiedziały, są w sekcji każdego uruchomienia.
- `<nazwa>__<sprawa>/result.json`, `turns.json` — pytania z powodem i brakującymi
  tematami według modelu, podane odpowiedzi, czasy.
- `<nazwa>__<sprawa>/logs/pismak_*.log` — pełny log sprawy: budżety kontekstu,
  tokeny, odrzucone odpowiedzi modelu, ślady stosu.
- `<nazwa>__<sprawa>/artifacts/` — to samo, co zapisuje aplikacja: `conversation.json`,
  `brief.json`, `evidence.json`, `draft.json` lub `draft_with_error.json`, `letter.md`, `models.json`.


## Pliki

- `e2e_cases.py` — lista `URUCHOMIENIA` (do edycji), skróty modeli, sprawy, słowa-klucze do ostrzeżeń.
- `e2e_support.py` — sterownik przebiegu (ta sama ścieżka co `src/Wyswietlator/app.py`:
  `dopytywator.interview_step`, `services.build_zrodlowyszukiwator`, `pismogenerator.build_draft`),
  kontrole, analiza logu, raport, usuwanie danych uwierzytelniających.
- `test_pismak_e2e.py` — warunki wstępne i uruchomienia z listy.
- `test_e2e_support.py` — kontrola ewaluatorów bez modeli.
- `conftest.py` — bramka `PISMAK_E2E=1` i tabela podsumowania na końcu wyjścia pytest.
