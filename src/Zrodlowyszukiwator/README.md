# Źródłowyszukiwacz

## Cel

Źródłowyszukiwacz dostarcza fragmenty dokumentów pasujące do sprawy. Przygotowuje `EvidencePack` z tekstem źródłowym i metadanymi, z którego korzysta Pismogenerator. Składa się z przygotowania bazy dokumentów oraz wyszukiwania w tej bazie.

## Struktura

- `rag_ingestion/` — przetwarzanie PDF-ów, dzielenie na fragmenty, tworzenie embeddingów i zapis do PostgreSQL/pgvector. Instrukcje znajdują się w `rag_ingestion/README.md`.
- `rag_retrieval/` — wyszukiwanie wektorowe i pełnotekstowe, łączenie rankingów oraz opcjonalne ponowne uszeregowanie wyników.
- `brief_to_evidence/` — planowanie zapytań na podstawie `CaseBrief`, uruchamianie retrieval i budowanie `EvidencePack`.
- `RAG/` — metadane dokumentów (`document_catalog.json`) oraz katalog `RAW/` z plikami PDF. Do repozytorium dołączone są dwa dokumenty; pozostałe pliki z `RAW/` zostają lokalnie.

## Przepływ danych

Przygotowanie źródeł odbywa się osobno: PDF-y są ekstrahowane, dzielone na fragmenty, wzbogacane metadanymi i embeddingami, a następnie zapisywane w bazie. Podczas tworzenia pisma `brief_to_evidence` otrzymuje `CaseBrief` i układa zapytania. `rag_retrieval` wyszukuje pasujące fragmenty wektorowo i pełnotekstowo, scala wyniki, a opcjonalny reranker porządkuje kandydatów. Pipeline wybiera źródła i zwraca `EvidencePack` z niezmienionymi fragmentami tekstu oraz ich metadanymi.

Przepływ: PDF-y → `rag_ingestion` → baza dokumentów → `CaseBrief` → plan zapytań → `rag_retrieval` → `EvidencePack` → Pismogenerator.