# Ewaluacja retrievalu RAG

Sprawdza, czy Źródłowyszukiwacz znajduje właściwe dokumenty i fragmenty dla pytań mieszkańców oraz czy rozpoznaje pytania bez odpowiedzi w korpusie. Mierzy sam retrieval, bez wywołań LLM; reranker LLM jest opcjonalny.

```powershell
.venv\Scripts\python.exe -m pytest tests/rag_eval -q       # testy zbioru, bez bazy (też w CI)

$env:PISMAK_RAG_EVAL = "1"                                   # ewaluacja na żywym indeksie
$env:PISMAK_RAG_EVAL_RERANKER_MODEL = "pismak-pllum:latest"  # opcjonalnie: reranker
.venv\Scripts\python.exe -m pytest tests/rag_eval -v -rA
```

Bez `PISMAK_RAG_EVAL=1`, bez bazy albo bez Ollamy testy na indeksie są pomijane z powodem. Raport trafia do `outputs/rag_eval/<czas>_<etykieta>/`, także gdy testy nie przechodzą. Testy nie mają progów dobranych pod wynik ani oznaczeń `xfail`.

## Zbiór

`data/rag_eval_questions.jsonl` zawiera po jednym rekordzie na linię; pola opisuje `EvalQuestion` w `rag_eval_support.py`.
- **Liczba pytań:** każdy zakomitowany PDF ma 3 pytania: 2 z odpowiedzią w tym dokumencie i 1 bez odpowiedzi w całym korpusie.
- **Strony:** numery stron pliku PDF liczone od 1, a nie numery wydrukowane na stronie.
- **Pytanie bez odpowiedzi:** dodaje się je dopiero po przeszukaniu wszystkich dokumentów. Przeszukane hasła i wzorce zapisuje się w `absence_check`.

## Metryki

Porównywane konfiguracje to `vector`, `lexical`, `hybrid`, `pipeline` (EvidencePack: 6 fragmentów, najwyżej 3 z dokumentu) i opcjonalnie `pipeline+reranker`; warianty `*-ascii` są bez polskich znaków. Liczone są `source_hit@k`, `evidence_hit@k`, `evidence_recall@k`, MRR i `evidence_precision@6`.

- **Pytanie z odpowiedzią** przechodzi, gdy właściwe źródło i dowód są w EvidencePack.
- **Pytanie bez odpowiedzi** przechodzi, gdy pipeline z rerankerem zwraca `unsupported`.

Przy kilku pytaniach metryki są gruboziarniste, więc przy porównaniach liczą się wyniki dla pojedynczych pytań.
