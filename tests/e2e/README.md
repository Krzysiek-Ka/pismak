# Testy end-to-end

Cały przepływ na prawdziwych modelach, bazie RAG i `.env`, bez atrap: opis sprawy → Dopytywator → CaseBrief → Źródłowyszukiwacz → Pismogenerator → Pismopoprawiator. Ścieżka jest ta sama co w `src/Wyswietlator/app.py`.

```text
tests/e2e/
  test_e2e_pipeline.py     uruchomienia na modelach
  test_e2e_support.py      kontrola ewaluatorów, bez modeli
  e2e_cases.py             wczytanie spraw i lista uruchomień
  e2e_support.py           sterownik przebiegu, kontrole, analiza logu, raport
  cases/<sprawa>.json      sprawy
  results/                 raporty, poza Gitem
```

## Uruchamianie tylko ręczne

Każdy moduł testów ma `pytestmark = pytest.mark.e2e`; zwykłe `python -m pytest` i CI go pomijają. Wymagane są: baza z dokumentami, Ollama z `bge-m3` i modelami projektu oraz `OPENROUTER_KEY` dla uruchomień OpenRouter.

```powershell
.venv\Scripts\python.exe -m pytest tests/e2e/test_e2e_support.py -m e2e -q                    # ewaluatory, bez modeli
.venv\Scripts\python.exe -m pytest tests/e2e/test_e2e_pipeline.py -m e2e -v -s                # wszystkie uruchomienia
.venv\Scripts\python.exe -m pytest "tests/e2e/test_e2e_pipeline.py::test_e2e_pipeline[bielik-buspas]" -m e2e -v
```

- **Uruchomienia:** jeden dostawca z konfiguracji projektu na wszystkich etapach (jak w złotych zbiorach), identyfikator `<dostawca>-<sprawa>`. `PISMAK_GOLDEN_PROVIDERS` zawęża dostawców, a `PISMAK_E2E_MIXED` (np. `pllum/bielik/openrouter`) dodaje uruchomienia mieszane; opis w `e2e_cases.py`.
- **Sprawy:** dwie sprawy z 12 odpowiedziami podawanymi po kolei, niezależnie od pytań.
- **Czas:** na CPU 20–40 min na uruchomienie, OpenRouter kilka minut. Komputer nie może zasnąć.
- **Etykieta przebiegu:** opcjonalnie `PISMAK_E2E_LABEL`.

## Ocena i raport

- **Wynik PASS/FAIL:** zależy tylko od niezmienników H1–H8 (wywiad, CaseBrief, RAG, pismo, brak sekretów w logach, korekta bez zmiany treści).
- **Ostrzeżenia:** S1–S10 to heurystyki (WARN) do ręcznej oceny.
- **Brak usługi:** uruchomienie jest pomijane jako NIEURUCHOMIONY.
- **Awaria w trakcie:** uruchomienie kończy się jako NIEUDANY, z typem błędu i miejscem w kodzie.

Każde uruchomienie zapisuje `results/<czas>_<dostawca>_<sprawa>.json` (modele etapów, status `passed` / `failed_check` / `infra_error` / `skipped`, kontrole, czasy etapów, ponowienia, źródła, błędy, `manual_review`). Raport zbiorczy trafia do `results/<czas>_<etykieta>/` i jest przepisywany po każdym uruchomieniu:
- `report.md` z tabelą podsumowania (ta sama tabela pojawia się na końcu wyjścia pytest) i poleceniem odtwarzającym każde uruchomienie;
- dla każdego uruchomienia logi i artefakty, takie jak zapisuje aplikacja.
