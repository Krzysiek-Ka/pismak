# Złote zbiory

Każdy moduł ma tu zbiór najważniejszych przypadków brzegowych, sprawdzanych na prawdziwych modelach, z minimum atrap. Przypadki i oczekiwane wyniki leżą obok testów.

```text
tests/golden_set/<moduł>/
  <moduł>_cases.py                  przypadki, oczekiwane wyniki, lista MODELE
  <moduł>_support.py                przebieg, ocena, raport
  test_<moduł>_golden_cases.py      kontrola zbioru bez modeli (sekundy)
  test_<moduł>_golden_llm.py        przebiegi na modelach (marker llm)
```

Nazwy mają przedrostek modułu, bo katalogi nie są pakietami.

## Uruchamianie tylko ręczne

`conftest.py` daje każdemu testowi z tego katalogu marker `golden_set`. Zwykłe `python -m pytest` i CI pomijają markery `golden_set` i `llm`.

```powershell
.venv\Scripts\python.exe -m pytest tests/golden_set/<moduł> -m golden_set -v -s   # całość
.venv\Scripts\python.exe -m pytest tests/golden_set/<moduł> -m llm -v -s          # tylko modele
```

Model, którego nie ma na maszynie, jest pomijany z powodem. Przebiegi OpenRouter wysyłają przypadki do usługi zewnętrznej. Raport (`report.md`, `results.json`, log) trafia do `outputs/golden_set/<moduł>/<czas>_<etykieta>/` i jest przepisywany po każdym uruchomieniu.

## Moduły

**Ewaluacja RAG** (`rag_eval/`): pytania do dokumentów korpusu na żywym indeksie (PostgreSQL + Ollama z `bge-m3`). Raport zaczyna się od `precision@6` i `recall@6` na konfigurację; trafia do `outputs/rag_eval/<czas>_<etykieta>/`. Szczegóły: [rag_eval/README.md](rag_eval/README.md).

**Pismopoprawiator**: dziesięć pism × tryby „całość” i „po akapitach”.
- **Wynik testu:** test nie przechodzi tylko przy naruszeniu treści, czyli gdy zmienią się elementy chronione albo napisy z listy `chronione`. Rozwinięcie skrótu („ul.” → „ulicy”) nie jest naruszeniem.
- **Tylko w raporcie:** naprawione błędy, poprawki nadmiarowe i odrzucenia.
- **Opcje:** `PISMAK_GOLDEN_MODELE` (np. `bielik,openrouter`), `PISMAK_GOLDEN_LABEL`.
