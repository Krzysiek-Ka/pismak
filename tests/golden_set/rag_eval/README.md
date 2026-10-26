# Ewaluacja retrievalu RAG

Sprawdza, czy Źródłowyszukiwacz znajduje właściwe dokumenty i fragmenty dla pytań mieszkańców oraz czy rozpoznaje pytania bez odpowiedzi w korpusie. Mierzy sam retrieval, bez wywołań LLM; reranker LLM jest opcjonalny.

```powershell
.venv\Scripts\python.exe -m pytest tests/golden_set/rag_eval/test_rag_eval_dataset.py -m golden -q   # testy zbioru, bez bazy

$env:PISMAK_RAG_EVAL_LABEL = "nazwa"                         # opcjonalnie: etykieta przebiegu
$env:PISMAK_RAG_EVAL_RERANKER_MODEL = "pismak-pllum:latest"  # opcjonalnie: reranker
.venv\Scripts\python.exe -m pytest tests/golden_set/rag_eval -m golden -v -rA   # całość, z żywym indeksem
```

Jak cały `tests/golden_set`, testy uruchamia się tylko ręcznie przez `-m golden`; zwykłe `pytest` i CI je pomijają. Bez bazy albo bez Ollamy testy na indeksie są pomijane z powodem. Raport trafia do `results/<czas>_<etykieta>/` (`report.md`, `results.json`), a wynik każdego pytania do `results/<czas>_<etykieta>_<id>.json`, także gdy testy nie przechodzą. Testy nie mają progów dobranych pod wynik ani oznaczeń `xfail`.

## Zbiór

`cases/<id>.json` zawiera po jednym pytaniu na plik, w układzie złotych zbiorów: pytanie w `input`, uzasadnienie w `description`, źródła, dowody i `absence_check` w `expected`, oczekiwana odpowiedź w `manual_review`. Pola opisuje `EvalQuestion` w `rag_eval_support.py`.
- **Liczba pytań:** każdy zakomitowany PDF ma co najmniej 2 pytania z odpowiedzią w tym dokumencie i co najmniej 1 bez odpowiedzi w całym korpusie.
- **Strony:** numery stron pliku PDF liczone od 1, a nie numery wydrukowane na stronie.
- **Pytanie bez odpowiedzi:** dodaje się je dopiero po przeszukaniu wszystkich dokumentów katalogu. Przeszukane hasła i wzorce zapisuje się w `absence_check`. Nowy dokument w katalogu wymaga ponownego sprawdzenia tych pytań.

## Metryki

Porównywane konfiguracje to `vector`, `lexical`, `hybrid`, `pipeline` (EvidencePack: 6 fragmentów, najwyżej 3 z dokumentu, wybrane z 20 kandydatów, w których jeden dokument ma najwyżej 8) i opcjonalnie `pipeline+reranker`; warianty `*-ascii` są bez polskich znaków. Liczone są `source_hit@k`, `evidence_hit@k`, `evidence_recall@k`, MRR i `evidence_precision@6`.

Raport zaczyna się od tabeli `precision@6` i `recall@6` dla każdej konfiguracji; te same liczby są w kluczu `headline` w `results.json`. Służą do szybkiego porównania przebiegów:
- **precision@6:** średni udział fragmentów top-6, które wspierają odpowiedź.
- **recall@6:** średni udział faktów pytania (pozycji `evidence`) pokrytych przez top-6.

Wyniki testów:
- **Pytanie z odpowiedzią** przechodzi, gdy właściwe źródło i dowód są w EvidencePack.
- **Pytanie bez odpowiedzi** przechodzi, gdy pipeline z rerankerem zwraca `unsupported`.

Przy kilkunastu pytaniach metryki są gruboziarniste, więc przy porównaniach liczą się też wyniki dla pojedynczych pytań.
