# Ewaluacja retrievalu RAG

Sprawdza, czy Źródłowyszukiwacz znajduje właściwe dokumenty i fragmenty dla
realistycznych pytań mieszkańców oraz czy rozpoznaje pytania, na które korpus nie
odpowiada. Mierzy sam retrieval przez istniejący interfejs (`RetrievalRepository`,
`HybridRetriever`, `BriefToEvidencePipeline`), bez wywołań LLM. Reranker LLM można
dołączyć opcjonalnie.

## Pliki

- `data/rag_eval_questions.jsonl` — zbiór pytań, jeden rekord JSON na linię.
- `rag_eval_support.py` — model rekordu, ocena dowodów, metryki, kontrola metadanych, raport.
- `test_rag_eval_dataset.py` — testy zbioru bez bazy (uruchamiane także w CI).
- `test_rag_retrieval_eval.py` — ewaluacja na żywym indeksie (tylko na żądanie).

## Uruchomienie

```powershell
# Testy zbioru (bez bazy) — część zwykłego `python -m pytest`
.venv\Scripts\python.exe -m pytest tests/rag_eval -q

# Ewaluacja na żywym indeksie: działająca baza (docker compose up -d postgres),
# Ollama z bge-m3 i zaindeksowane dokumenty
$env:PISMAK_RAG_EVAL = "1"
$env:PISMAK_RAG_EVAL_LABEL = "po-zmianie"                         # opcjonalna nazwa przebiegu
$env:PISMAK_RAG_EVAL_RERANKER_MODEL = "pismak-pllum:latest"       # opcjonalnie: reranker LLM
.venv\Scripts\python.exe -m pytest tests/rag_eval -v -rA
```

Bez `PISMAK_RAG_EVAL=1` testy na żywym indeksie są pomijane. Gdy baza albo Ollama nie
działa, są pomijane z podanym powodem. Raport trafia do
`outputs/rag_eval/<czas>_<etykieta>/report.md` (wraz z `results.json`) i jest zapisywany,
zanim zostaną sprawdzone asercje, więc powstaje także wtedy, gdy testy nie przechodzą.

Testy nie mają progów dobranych pod wynik ani oznaczeń `xfail`: pytanie, które system
obsługuje źle, jest widoczne jako FAILED wraz z diagnozą.

## Zasady zbioru

- Dla każdego zakomitowanego PDF-a z katalogu: dokładnie 3 pytania — 2 z odpowiedzią
  w tym dokumencie i 1, na które nie odpowiada żaden dokument korpusu. Pilnuje tego
  `test_rag_eval_dataset.py` (lista PDF-ów z `git ls-files`).
- Pytania brzmią jak pytania mieszkańca lub aktywisty: parafrazy, język potoczny,
  pytania o cele, zasady, obowiązki i terminy.
- Strony to numery stron pliku PDF liczone od 1 (jak `page_start` w indeksie), a nie
  numery wydrukowane na stronie. Cytaty są zapisane tak, jak strona wygląda w PDF.
- Pytanie bez odpowiedzi dodajemy dopiero po przeszukaniu **wszystkich** dokumentów
  katalogu. W `absence_check` zapisujemy przeszukane hasła, wzorce, które oznaczałyby
  odpowiedź (`answer_patterns`, sprawdzane w całym indeksie przy każdym przebiegu),
  oraz powiązane fragmenty, które nie odpowiadają na pytanie.

### Pola rekordu

| pole | znaczenie |
|---|---|
| `id`, `document_id` | identyfikator pytania i dokument, dla którego je napisano |
| `question`, `answerable` | treść pytania i to, czy korpus na nie odpowiada |
| `expected_answer`, `rationale` | odpowiedź oparta na źródle i uzasadnienie (parafrazy, dystraktory) |
| `expected_sources[]` | `document_id`, `pages`, dosłowny `quote` |
| `evidence[]` | fakty potrzebne do odpowiedzi: fragment je wspiera, gdy zawiera wszystkie frazy `all_of` |
| `acceptable_document_ids` | inne dokumenty, które też liczą się jako właściwe źródło |
| `tags` | zjawiska, np. `parafraza`, `synonimy`, `daty`, `dystraktor`, `trudny-negatyw` |
| `absence_check` | tylko dla pytań bez odpowiedzi (opis wyżej) |

## Metryki

Konfiguracje: `vector`, `lexical`, `hybrid` (RRF, 20+20), `pipeline` (EvidencePack:
6 fragmentów, najwyżej 3 z dokumentu), opcjonalnie `pipeline+reranker`. Zapytanie
powstaje tak jak w produkcyjnym fallbacku planera, czyli pytanie jest jednocześnie
zapytaniem wektorowym i pełnotekstowym. Warianty `*-ascii` to te same pytania bez
polskich znaków.

- **Dowód**: fragment z właściwego dokumentu zawierający wszystkie frazy jednego elementu
  `evidence`. Ocena ignoruje artefakty ekstrakcji (przeniesienia wyrazów, złe mapowanie
  fontu), żeby oceniać treść, a nie formatowanie.
- `source_hit@k`: właściwy dokument jest w top-k. `evidence_hit@k`: fragment z dowodem jest w top-k.
- `evidence_recall@k`: udział faktów z `evidence` pokrytych przez top-k.
- `mrr_source`, `mrr_evidence`: średnia odwrotność pozycji pierwszego trafienia do pozycji 20.
  MRR premiuje trafienie blisko szczytu listy, a do pisma trafia tylko czoło rankingu
  (reranker dostaje 20 kandydatów, EvidencePack ma 6 fragmentów).
- `evidence_precision@6`: udział przedstawionych fragmentów top-6, które wspierają odpowiedź.
- Abstencja: `EvidencePack.evidence_status == "unsupported"`. Trafność ocenia reranker,
  więc bez niego status to `unverified` (fragmenty nieocenione), a nie abstencja.

Pytanie z odpowiedzią przechodzi, gdy właściwe źródło i dowód są w EvidencePack, a
wszystkie fakty są wśród 20 kandydatów. Pytanie bez odpowiedzi przechodzi, gdy system
abstynuje. Wektorowe wyszukiwanie zawsze zwraca podobne fragmenty, dlatego test nie
wymaga pustej listy wyników, tylko sygnału braku podstaw zgodnego z kontraktem.

Testy abstencji i fałszywej abstencji wymagają rerankera (`PISMAK_RAG_EVAL_RERANKER_MODEL`).
Bez niego są pomijane, a osobny test sprawdza, że pipeline bez rerankera nie przypisuje
fragmentom statusu `supported`.

Przy kilku pytaniach metryki są gruboziarniste: jedno pytanie z odpowiedzią to 25%
wyniku. Do porównań przed i po zmianie liczą się przede wszystkim wyniki per pytanie.
