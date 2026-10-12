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

`hierarchy_rank` będzie później pomocniczym sygnałem w retrieval/rerankingu.
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

1. PostgreSQL z rozszerzeniem pgvector.
2. Ollama.
3. Model embeddingowy:

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
ingestion zostanie pominięty. `--force` wymusza ponowne przetworzenie.

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
- `tsvector` do późniejszego hybrid search.

Dzięki temu `rag_retrieval` będzie mógł robić vector search + lexical search,
a wynik nadal będzie miał precyzyjne cytowanie strony i dokumentu.
