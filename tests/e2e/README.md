# Testy end-to-end

Cały przepływ na prawdziwych modelach, bazie RAG i `.env`, bez atrap: opis sprawy → Dopytywator → CaseBrief → Źródłowyszukiwacz → Pismogenerator → Pismopoprawiator. Ścieżka jest ta sama co w `src/Wyswietlator/app.py`.

## Uruchamianie tylko ręczne

`conftest.py` daje każdemu testowi marker `e2e`, który zwykłe `python -m pytest` i CI pomijają. Wymagane są: baza z dokumentami, Ollama z `bge-m3` i modelami z listy oraz `OPENROUTER_KEY`, gdy lista go używa.

```powershell
.venv\Scripts\python.exe -m pytest tests/e2e/test_e2e_support.py -m e2e -q    # ewaluatory, bez modeli
.venv\Scripts\python.exe -m pytest tests/e2e/test_pismak_e2e.py -m e2e -v -s  # wszystkie uruchomienia
.venv\Scripts\python.exe -m pytest "tests/e2e/test_pismak_e2e.py::test_pismak_end_to_end[bielik-buspas]" -m e2e -v
```

- **Konfiguracje modeli:** lista `URUCHOMIENIA` w `e2e_cases.py`; pola opisuje komentarz nad nią.
- **Przypadki:** dwie sprawy z 12 odpowiedziami podawanymi po kolei, niezależnie od pytań.
- **Czas:** na CPU 20–40 min na uruchomienie, OpenRouter kilka minut. Komputer nie może zasnąć.
- **Etykieta przebiegu:** opcjonalnie `PISMAK_E2E_LABEL`.

## Ocena i raport

- **Wynik PASS/FAIL:** zależy tylko od niezmienników H1–H8 (wywiad, CaseBrief, RAG, pismo, brak sekretów w logach, korekta bez zmiany treści).
- **Ostrzeżenia:** S1–S10 to heurystyki (WARN) do ręcznej oceny.
- **Brak usługi:** uruchomienie jest pomijane jako NIEURUCHOMIONY.
- **Awaria w trakcie:** uruchomienie kończy się jako NIEUDANY, z typem błędu i miejscem w kodzie.

Raport trafia do `outputs/e2e/<czas>_<etykieta>/` i jest przepisywany po każdym uruchomieniu:
- `report.md` z tabelą podsumowania (ta sama tabela pojawia się na końcu wyjścia pytest) i poleceniem odtwarzającym każde uruchomienie;
- dla każdego uruchomienia logi i artefakty, takie jak zapisuje aplikacja.
