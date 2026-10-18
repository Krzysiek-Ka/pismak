# Pismak — Źródłowyszukiwacz / rag_ingestion

`rag_ingestion` jest częścią offline / jednorazową. Uruchamiasz ją przy dodawaniu
lub aktualizowaniu dokumentu źródłowego.

Pipeline:

PDF
→ ekstrakcja stron
→ konserwatywne czyszczenie
→ wykrywanie sekcji
→ semantyczne chunkowanie
→ wzbogacenie chunków o metadata
→ embedding przez lokalną Ollamę
→ PostgreSQL + pgvector

## Dwie osobne osie dokumentu

Nie traktujemy `zarządzenia` jako poziomu tej samej hierarchii co `strategia`
czy `standard`.

### Hierarchia merytoryczna

1. Strategia — 100
2. Polityka — 80
3. Program / Plan — 70
4. Standard — 60
5. Procedura / instrukcja — 50
6. Inne — 40

`hierarchy_rank` rozstrzyga remisy w rankingu wyszukiwania (`src/Zrodlowyszukiwator/rag_retrieval/hybrid.py`).
Nie jest oceną mocy prawnej.

### Forma prawna / publikacyjna

Oddzielne pole `legal_form`:

- `resolution` — uchwała,
- `ordinance` — zarządzenie,
- `annex` — załącznik do aktu,
- `report`,
- `standalone`,
- `unknown`.

Przykład: *Polityka Rowerowa Wrocławia* ma `document_kind=policy`, a jednocześnie
jest załącznikiem do uchwały. Standard techniczny może mieć
`document_kind=standard` i zostać wprowadzony zarządzeniem.

## Dlaczego Program jest osobnym poziomem

Wrocławski Program Tramwajowy 2.0 sam określa się jako materiał analityczny
wspierający decyzje inwestycyjne i odwołuje się do celów Wrocławskiej Polityki
Mobilności. Dlatego w RAG warto umieścić `program/plan` pomiędzy polityką
a standardem.

## Przygotowanie lokalne

1. PostgreSQL z rozszerzeniem pgvector (`docker compose up -d`, główne README, rozdział 4).
2. `RAG_INGESTION_DATABASE_URL` w `.env` — adres roli `pismak_ingest`, która jest
   właścicielem tabel. Rola z `DATABASE_URL` ma tylko prawo odczytu i nie wczyta dokumentu.
3. Ollama z modelem embeddingowym:

```bash
ollama pull bge-m3
```


## Dodanie dokumentu

Skopiuj PDF do `src/Zrodlowyszukiwator/RAG/RAW/` i dopisz jego metadane do
`src/Zrodlowyszukiwator/RAG/document_catalog.json` (kluczem wpisu jest nazwa pliku).

Następnie, z katalogu głównego repozytorium:

```cmd
python -m src.Zrodlowyszukiwator.rag_ingestion.cli --file src\Zrodlowyszukiwator\RAG\RAW\nazwa.pdf
```

Polecenie tworzy tabele, jeśli ich jeszcze nie ma. Inny katalog metadanych
wskażesz parametrem `--catalog`.

Jeśli plik o tym samym `document_id` i tym samym SHA-256 jest już w bazie,
treść nie jest przetwarzana ponownie, ale metadane z katalogu są zapisywane od
nowa. Zmiana tytułu, typu dokumentu albo miasta powoduje pełne przetworzenie,
bo te pola wchodzą w skład embeddingów. Pełne przetworzenie następuje też wtedy,
gdy dokument zapisano starszą wersją przetwarzania (`INGESTION_VERSION` w
`pipeline.py`, podnoszona przy każdej zmianie ekstrakcji lub dzielenia tekstu).
`--force` wymusza je zawsze.

## Reguły przygotowania tekstu

- Linie rozpoznane jako nagłówki trafiają do `section_path`, ale zostają też w
  treści fragmentu. Heurystyka nagłówków myli się na liniach łamanych w PDF, a
  usunięcie takiej linii oznaczałoby utratę treści źródła.
- Linie spisu treści (z kropkami prowadzącymi) są pomijane, bo powtarzają tytuły.
- Nagłówki i stopki stron są usuwane od drugiego wystąpienia. Rozpoznajemy je jako linie
  z literami, które w dokumencie mającym co najmniej 4 strony występują wyłącznie wśród
  3 pierwszych lub 3 ostatnich linii strony i to na co najmniej 25% stron. Wynik ingestion
  wypisuje usunięte linie w `removed_running_lines`.
- Znaki sterujące (np. punktory z fontu symbolicznego zapisane jako U+0001) są usuwane.
- Wyraz przeniesiony do następnej linii (`trans-` / `portu`) jest sklejany.
  Łącznik zawieszony przed spójnikiem („jedno- i dwukierunkowe”) zostaje.
- Polskich znaków nie poprawiamy, z jednym wyjątkiem: zepsute mapowanie fontu,
  w którym „ż” trafia do tekstu jako „Ŝ”, a „Ż” jako „ś”. „Ŝ” nie występuje w
  polszczyźnie, więc naprawa jest stosowana tylko w dokumencie, który go zawiera.
  Wynik ingestion podaje strony z naprawą w `repaired_glyph_pages`.

Pola `is_active`, `lifecycle_status` i `retrieval_enabled` oraz ich wpływ na
wyszukiwanie opisuje główne README, punkt 5.2.

## Zasada wersjonowania

`document_id` identyfikuje logiczny dokument/wersję, np.:

- `wroclaw-mobility-policy-2013`
- `wroclaw-strategy-2050-2025`

Jeśli pojawia się nowa oficjalna wersja dokumentu, lepiej utworzyć nowy
`document_id` i później dodać relację `supersedes`, zamiast nadpisywać historię.

## Ważne dla retrieval

Każdy chunk zachowuje:

- `document_id`,
- tytuł,
- typ dokumentu,
- `hierarchy_rank`,
- formę prawną,
- stronę początkową i końcową PDF,
- `section_path`,
- tekst źródłowy,
- SHA-256,
- embedding,
- metadata JSONB,
- `tsvector` do wyszukiwania pełnotekstowego.

Dzięki temu `rag_retrieval` łączy wyszukiwanie wektorowe i pełnotekstowe, a każdy
znaleziony fragment ma precyzyjne odwołanie do strony i dokumentu.
