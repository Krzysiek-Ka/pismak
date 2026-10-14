# HANDOVER — audyt przedpublikacyjny Pismaka

> **Nie commituj tego pliku do publicznego repozytorium.** Opisuje stan zabezpieczeń komputera właściciela (konfiguracja bazy, działający tunel). Katalog `plans/` jest śledzony przez gita, a ten plik nie jest ignorowany — patrz zadanie T-08.

| | |
|---|---|
| Data audytu | 2026-10-11 |
| Audytowany commit | `dab3384` („add openruter”), `main` == `origin/main`, drzewo czyste |
| Zakres audytu | wyłącznie odczyt; **żaden plik projektu nie został zmieniony** |
| Odbiorca | model AI wdrażający poprawki oraz właściciel projektu |
| Werdykt | projekt **nie jest gotowy** do upublicznienia repo ani do publicznego dema |
| Zmiany po audycie | 2026-10-11: **K1 naprawione** (T-01 wykonane) — zmiana tylko na hoście i w lokalnym `.env`, bez zmian w plikach śledzonych. 2026-10-12: **W1 i W2 naprawione** przez właściciela (commit `483321d`); `print` z `DATABASE_URL` usunięty z notebooka (commit `27ee2f1`); **W3 częściowo** — tunel zweryfikowany lokalnie, README poprawione, Access do ustawienia w panelu; **W4 naprawione** (T-15); naprawiony skutek uboczny K1 — `DATABASE_URL` wskazuje `127.0.0.1` (patrz T-01); okno kontekstu wykrywane dla wybranego modelu i limit czasu 15 minut dla modeli lokalnych (T-15, T-10; commit `656ec68`); **W6 naprawione** — dwa dokumenty PDF i pliki Modelfile dołączone do repozytorium, README uzupełnione (T-19; commit `9c602f9`). 2026-10-13: **S1 naprawione** — interfejs pokazuje tylko ogólne komunikaty, szczegóły i ślad stosu są w logach (T-09; commit `464334c`); **S2 naprawione** — jeden handler logów bez otwartych plików, wpisy bez sprawy trafiają do pliku wspólnego (T-11; commit `7ea21fe`); **S4 uznane za działanie zamierzone** — użytkownik ma móc wybierać dostawcę i model, T-13 nie będzie realizowane; **S5 naprawione** — obsługa nowych pól katalogu, dokumenty nieobowiązujące nie trafiają do pism, baza zsynchronizowana z katalogiem (T-16; commit `d8f8df9`). 2026-10-14: **S6 naprawione** — wyszukiwanie pełnotekstowe łączy znaczące słowa przez OR i dopasowuje odmienione formy prefiksem; zapytania, które dawały 0 trafień, dają pełne 20 (T-18, niezacommitowane); **S9 częściowo** — zależności przypięte do najnowszych wersji, `requirements.txt` jest pełną blokadą generowaną z `requirements.in` (T-20; commit `29b40d3`); **D4 rozstrzygnięte** — licencja projektu zmieniona z MIT na AGPL-3.0-or-later, zgodną z PyMuPDF (T-20, niezacommitowane) |

## Jak czytać ten dokument

- **[FAKT]** — potwierdzone lekturą kodu albo poleceniem; przy każdym podaję, czym.
- **[HIPOTEZA]** — wniosek, którego nie potwierdziłem. Zweryfikuj, zanim na nim zbudujesz poprawkę.
- **[DECYZJA]** — wymaga odpowiedzi właściciela. Nie rozstrzygaj samodzielnie.
- **[OPS]** — czynność na komputerze właściciela, nie zmiana w repozytorium. Wykonuj tylko po wyraźnym potwierdzeniu.
- Identyfikatory ustaleń: `K` krytyczne, `W` wysokie, `S` średnie, `N` niskie. Zadania: `T-01…`.

Numery linii odnoszą się do commita `dab3384`. Po każdej zmianie pliku sprawdź je ponownie.

---

## 1. Co przeczytać przed rozpoczęciem pracy

W tej kolejności. Pozycje 1–4 są obowiązkowe przed jakąkolwiek zmianą.

1. `src/Wyswietlator/app.py` — cały przepływ UI, stan sesji, obsługa błędów (795 linii).
2. `src/Wyswietlator/services.py` — klienci LLM, logowanie per sesja, komunikaty błędów, `build_zrodlowyszukiwator`, `save_run`.
3. `src/_shared/serwer.py` i `src/_shared/llm_runtime.py` — `LLMClient.generate_structured`, ponowienia, globalna blokada.
4. `src/Pismogenerator/pismogenerator_helper_functions.py`, `pismogenerator.py`, `prompts.py`, `structures.py`.
5. `src/Zrodlowyszukiwator/brief_to_evidence/` — `pipeline.py`, `planner.py`, `formatting.py`, `models.py`.
6. `src/Zrodlowyszukiwator/rag_retrieval/` — `repository.py`, `hybrid.py`, `reranker.py`, `models.py`.
7. `src/Dopytywator/` — `structures.py`, `prompts.py`, `constants.py`.
8. `docker-compose.yaml`, `.env_template`, `requirements.txt`, `.github/workflows/ci.yml`, `.gitignore`, `run_pismak.bat`.
9. `README.md` (sekcje 3, 5, 8) oraz `src/*/README.md`.
10. Testy wyznaczające obecne zachowanie: `tests/unit_tests/shared/test_serwer.py`, `tests/unit_tests/pismogenerator/test_pismogenerator.py`, `tests/unit_test/wyswietlator/test_services.py`, `tests/unit_tests/test_release_configuration.py`.

Moduł `rag_ingestion` (offline) czytaj dopiero przy zadaniach T-16, T-20, T-23.

**Nie kieruj się** plikiem `plans/handover_publikacja_pismak.md`. Jest nieaktualny: opisuje katalog `wyswietlator/` z `models.py` i `requirements-ui.txt`, których nie ma. Jego wymagania funkcjonalne (izolacja sesji, blokada, limit wiadomości) są w większości już wdrożone.

---

## 2. Architektura

### 2.1 Topologia uruchomieniowa

```text
Przeglądarka testera
      │ HTTPS
Cloudflare (Access?) ── tunel zarządzany zdalnie ── usługa Windows `cloudflared`
                                                          │
                                              127.0.0.1:8501  Streamlit (run_pismak.bat)
                                                          │
                     ┌────────────────────────────────────┼─────────────────────────────┐
        127.0.0.1:11434 Ollama                 PostgreSQL 17 + pgvector         OpenRouter (opcjonalnie)
        pismak-pllum, pismak-bielik, bge-m3    kontener `pismak-postgres`        model `openrouter/free`
```

Wszystko działa na jednym komputerze z Windows 11. Nie ma osobnego backendu: Streamlit importuje moduły domenowe bezpośrednio.

### 2.2 Przepływ danych

```text
chat (app.py)
  └─ LLMClient.generate_structured(InterviewTurnResult)      ← Dopytywator, pytanie po pytaniu
  └─ LLMClient.generate_structured(CaseBrief)                ← po limicie pytań albo gdy model uzna, że wystarczy
       └─ save_run()                                          → outputs/<run>/conversation.json, brief.json

przycisk „Znajdź źródła i wygeneruj pismo”
  └─ build_zrodlowyszukiwator()                               ← services.py
       └─ BriefToEvidencePipeline.run(case_brief)
            ├─ LLMQueryPlanner.plan()        → ResearchPlan (3–5 SearchIntent)
            ├─ HybridRetriever.retrieve()    → OllamaEmbedder (/api/embed) + vector_search + lexical_search, RRF k=60
            ├─ LLMReranker.rerank()          → do 6 fragmentów, tylko znane chunk_id
            └─ limit 3 fragmentów na dokument → EvidencePack (E1…En)
  └─ pismogenerator.build_draft()            → DraftLetter → validate_draft()
  └─ render_letter()                          → letter.md (składany deterministycznie w kodzie)
  └─ save_run()                               → evidence.json, draft.json, letter.md, models.json
```

### 2.3 Kontrakty danych — nie zmieniaj bez uzasadnienia

| Kontrakt | Definicja | Uwagi |
|---|---|---|
| `ConversationMessage` | `src/Dopytywator/structures.py:9` | duplikat w `src/_shared/utilities.py` (martwy) |
| `InterviewTurnResult` | `structures.py:56` | walidator spójności `next_question` / `ready_to_build_brief` |
| `CaseBrief` | `structures.py:29` | wejście dla RAG i Pismogeneratora; zapisywany jako `brief.json` |
| `SearchIntent`, `RetrievalFilters`, `RetrievedChunk` | `rag_retrieval/models.py` | `priority` 1–3 |
| `ResearchPlan`, `Evidence`, `EvidencePack` | `brief_to_evidence/models.py` | zapisywany jako `evidence.json` |
| `DraftLetter` | `src/Pismogenerator/structures.py:6` | `sender` zawsze czyszczony w `build_draft` |
| Identyfikatory źródeł | `pismogenerator_helper_functions.py:57-62` | `E{n}` = kolejność w `EvidencePack`; `U{n}` = pozycja w `CaseBrief.available_evidence`; przypis `[n]` = pozycja w `DraftLetter.sources` |
| Artefakty uruchomienia | `services.py:252-321` | `outputs/<YYYYMMDD_HHMMSS>_<uuid8>_ui/` z plikami `conversation.json`, `brief.json`, `evidence.json`, `draft.json`, `draft_with_error.json`, `letter.md`, `models.json` |
| Schemat bazy | `rag_ingestion/repository.py` (`ensure_schema`) | `rag_documents`, `rag_chunks` (`embedding vector(1024)`, `search_vector` w konfiguracji `'simple'`). Od 2026-10-13 `rag_documents` ma dodatkowo `lifecycle_status`, `retrieval_enabled`, `metadata_url`, `is_annex` |
| Format logu | `services.py:64-82` | `logs/pismak_<timestamp>.log`, zdarzenia `nazwa klucz=wartość` |

Zmienne środowiskowe faktycznie czytane przez kod: `OPENROUTER_KEY`, `OLLAMA_URL`, `DATABASE_URL`, `RAG_EMBEDDING_MODEL`, `RAG_EMBEDDING_DIM` (tylko CLI ingestion), `PISMAK_PROJECT_ROOT`. Docker Compose czyta `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`.

### 2.4 Stan globalny i stan sesji

- Globalne (współdzielone między sesjami): `_LLM_LOCK` w `llm_runtime.py:11`; słownik `_CASE_LOG_HANDLERS` w `services.py:51`; klient Ollamy w `st.cache_resource` (`app.py:166`); lista modeli w `st.cache_data` z TTL 30 s (`app.py:173`).
- Per sesja (`st.session_state`, `app.py:62-73`): `session_id`, `stage`, `messages`, `question_count`, `case_brief`, `evidence_pack`, `draft`, `letter_markdown`, `run_dir`, `model_usage` oraz klucze widżetów `cfg_*`.
- [FAKT] W kodzie nie znalazłem danych użytkownika trzymanych globalnie.

### 2.5 Limity, timeouty i ponowienia — stan obecny

| Miejsce | Zachowanie |
|---|---|
| `LLMClient.generate_structured` (`serwer.py:303-378`) | 1 wywołanie; przy błędzie walidacji 1 naprawa; przy ucięciu 1 ponowienie z prośbą o krótszą odpowiedź, potem próba odzyskania częściowego JSON |
| Planner, reranker | 1 ponowienie przy ucięciu; dla Ollamy cichy fallback deterministyczny, dla OpenRouter błąd (`services.py:181`) |
| `build_draft` | brak ponowienia po `DraftValidationError` |
| Klient Ollamy w UI (`services.get_openai_client`) | od 2026-10-12: timeout 900 s, `max_retries=0` (w dniu audytu: 300 s i 2 ponowienia) |
| Klient Ollamy w `set_up_client` (`serwer.py`) | od 2026-10-12: timeout 900 s, `max_retries=0` (w dniu audytu: 180 s i 2 ponowienia) |
| Klient OpenRouter (`serwer.py:118-123`) | timeout 60 s, `max_retries=0` |
| Embedder (`rag_ingestion/embedder.py`) | `urllib`, timeout 300 s, bez ponowień, poza blokadą |
| `psycopg.connect` (`rag_retrieval/repository.py:35`) | bez `connect_timeout`, nowe połączenie na każde zapytanie |
| Blokada `_LLM_LOCK` | bez limitu czasu oczekiwania |
| Limity wyjścia | Dopytywator 8200, Pismogenerator 8192, planner 4096, reranker 4096 tokenów |
| Limity wejścia | tylko 5000 znaków na wiadomość i maks. 12 pytań (`services.py:48-49`); brak liczenia tokenów |

### 2.6 Migawka środowiska z dnia audytu

- Python 3.14.2; openai 1.109.1; pydantic 2.13.5; streamlit 1.64.0; psycopg 3.3.6 (implementacja `python`, systemowe libpq 18); pgvector 0.5.0; PyMuPDF 1.28.2; pytest 8.4.2; ruff 0.16.9.
- Ollama 0.33.2: `OLLAMA_HOST=127.0.0.1:11434`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_CONTEXT_LENGTH=262144` (ustawienie lokalne). Modele: `pismak-pllum` (gemma3, 3.9B, kontekst 131072, szablon `{{ .Prompt }}`), `pismak-bielik` (llama, 4.8B, kontekst 8192), `bge-m3` (kontekst 8192, wymiar 1024).
- PostgreSQL 17.11, rozszerzenie `vector` 0.8.6, `listen_addresses=*`. Baza: 6 dokumentów, 1385 chunków, wymiar 1024, indeksy HNSW i GIN obecne.
- cloudflared 2026.10.0 jako usługa Windows.
- `pytest`: 92 passed, 1 failed. `ruff check`: czysto. `ruff format --check`: 3 pliki do przeformatowania. `pip check`: czysto.

---

## 3. Ustalenia potwierdzone

### 3.1 Krytyczne

**K1 — PostgreSQL wystawiony na wszystkie interfejsy z publicznym hasłem domyślnym.**

> **Status: NAPRAWIONE 2026-10-11** (szczegóły i wyniki weryfikacji w T-01). Poniższe punkty opisują stan z dnia audytu i zostają jako uzasadnienie.

- [FAKT] `netstat` pokazuje `0.0.0.0:5432` i `[::]:5432`; `docker ps` pokazuje `0.0.0.0:5432->5432/tcp`; `docker inspect` zwraca `PortBindings` z pustym `HostIp`.
- [FAKT] `docker-compose.yaml:13` wymaga `127.0.0.1:5432:5432`. Kontener utworzono 2026-09-20 i nie odtworzono po zmianie; `docker compose up -d --dry-run` zapowiada `Recreate`.
- [FAKT] Rola `pismak` jest superuserem (`rolsuper = true`) i jedyną rolą. Hasło w lokalnym `.env` równa się wartości z `.env_template:34` i `docker-compose.yaml:9`, czyli jest publiczne.
- [FAKT] `pg_hba`: `scram-sha-256` dla połączeń spoza pętli zwrotnej kontenera, `trust` dla `127.0.0.1`, `::1` i gniazda lokalnego wewnątrz kontenera.
- [FAKT] Zapora Windows jest włączona we wszystkich profilach, sieć Wi‑Fi ma kategorię Private, nie znalazłem reguły zezwalającej zawierającej w nazwie `docker`, `postgres` ani `5432`.
- [HIPOTEZA] Port jest osiągalny z sieci lokalnej. Nie testowałem tego. Jeśli jest, superuser z publicznym hasłem oznacza wykonanie poleceń w kontenerze (`COPY … FROM PROGRAM`).


### 3.2 Wysokie

**W1 — Test nie przechodzi na wypchniętym HEAD.**
> **Status: NAPRAWIONE 2026-10-12**
- [FAKT] `tests/unit_tests/pismogenerator/test_pismogenerator.py:134` oczekuje „W imieniu Towarzystwa Upiększania Miasta Wrocławia”.
- [FAKT] `src/Pismogenerator/pismogenerator_helper_functions.py:96` ma `PHASE0_OPENING_FORMULA = "Jako wieloletni mieszkaniec naszego kochango miasta, wnioskuję o"` (literówka „kochango”). Zmiana weszła w commicie `af587cb`.
- [FAKT] `README.md:93` i `src/Pismogenerator/prompts.py:30-31` nadal opisują formułę „W imieniu (…)”.
- [FAKT] CI (`.github/workflows/ci.yml`) uruchamia `pytest -q`, więc będzie czerwone. Stanu CI na GitHubie nie sprawdzałem.

**W2 — Podpis każdego pisma zawiera na sztywno nazwę konkretnej organizacji.**
> **Status: NAPRAWIONE 2026-10-12**
- [FAKT] `pismogenerator_helper_functions.py:160` wstawia „Towarzystwo Upiększania Miasta Wrocławia” do każdego `letter.md`.
- [FAKT] Pole `**Nadawca:**` jest zawsze puste (`:149`).
- [DECYZJA] Czy organizacja zgadza się, by pisma z publicznego dema nosiły jej nazwę.

**W3 — Tunel Cloudflare działa stale, a jego konfiguracji nie da się sprawdzić z repozytorium.**
> **Status: CZĘŚCIOWO NAPRAWIONE 2026-10-12.** Zweryfikowano lokalnie, że tunel nie publikuje żadnej usługi, i poprawiono README. Otwarte pozostaje wyłącznie ustawienie Access w panelu Cloudflare w chwili dodawania trasy — patrz T-02.
- [FAKT] Usługa `cloudflared`: tryb Auto, stan Running, konto LocalSystem, polecenie `tunnel run --token-file C:\ProgramData\cloudflared\token`. To tunel zarządzany zdalnie — trasy i polityki są w panelu Cloudflare.
- [FAKT, 2026-10-12] Konfigurację, którą usługa otrzymała z Cloudflare, da się odczytać lokalnie: `curl http://127.0.0.1:20241/config`. Wynik: `version: -1`, jedyna reguła `ingress` to `http_status:503`, `warp-routing` puste. `/ready` zgłasza 4 aktywne połączenia. Tunel jest więc połączony, ale **nie ma żadnej trasy publicznej** i nie wystawia Streamlita ani niczego innego.
- [FAKT, 2026-10-12] `app.pismak.pl` (adres z dawnego README i ze starego handoveru) nie istnieje w DNS. `pismak.pl` wskazuje na adres spoza zakresów proxy Cloudflare.
- [HIPOTEZA] Domena `pismak.pl` nie jest podpięta do konta Cloudflare właściciela; adres w dokumentacji był przykładem. README używa teraz neutralnego `<NAZWA_HOSTA>`.
- [FAKT] Licznik `cloudflared_tunnel_total_requests` pokazywał 1 żądanie od startu usługi. Źródła nie ustaliłem; przy obecnej konfiguracji odpowiedzią mogło być tylko 503.
- [FAKT] README opisywał wariant z lokalnym `config.yml`, inny niż wdrożony. Poprawione 2026-10-12: sekcja „Stały Cloudflare Tunnel” opisuje tunel zarządzany zdalnie, kolejność „najpierw Access, potem trasa” i dwa polecenia weryfikacyjne.
- [FAKT] Aplikacja nie ma własnego uwierzytelniania ani limitowania zapytań. Po dodaniu trasy jedyną ochroną będzie Access.
- Hipoteza z audytu („tunel ma publiczną nazwę hosta chronioną Access”) jest obalona w pierwszej części: nazwy hosta nie ma. Istnienia aplikacji Access w koncie nadal nie da się sprawdzić lokalnie.

**W4 — Brak budżetu kontekstu; wymagane ustawienie Ollamy nie jest udokumentowane.**
> **Status: NAPRAWIONE 2026-10-12** (commity `f4730ed`, `656ec68`). Opis rozwiązania, pomiary i ograniczenia w T-15. Poniższe punkty opisują stan z dnia audytu.
- [FAKT] Logi serwera Ollamy zawierają 7 zdarzeń `truncating input prompt`: prompty 10 155–11 540 tokenów obcięte do 4 099. Wszystkie przy wcześniejszym `num_ctx=4096`.
- [FAKT] Dziś Ollama ładuje modele z `num_ctx=262144`; to ustawienie maszyny, nieobecne w README i `.env_template`. Modelfile'e nie ustawiają `num_ctx`.
- [FAKT] `pismak-bielik` ma maksymalny kontekst 8192 — mniej niż zaobserwowane prompty. W logach aplikacji wszystkie 3 uruchomienia RAG szły na `pismak-bielik`.
- [FAKT] `max_tokens` dla Dopytywatora (8200) i Pismogeneratora (8192) jest równy lub większy od całego okna Bielika.
- [FAKT] Kod nie liczy tokenów wejścia. Rozmiar promptu wyznaczają: reranker 20 kandydatów × 1800 znaków (`rag_retrieval/reranker.py:228`, `services.py:211-229`), Pismogenerator 6 × 2400 znaków (`brief_to_evidence/formatting.py:22`), rozmowa do 13 × 5000 znaków.
- [FAKT] Ollama obcina prompt bez błędu po stronie klienta, więc aplikacja nie wie o obcięciu.
- [HIPOTEZA] Prompty 10–11,5 tys. tokenów pochodziły z rerankera (rachunek znaków się zgadza; nie skorelowałem znaczników czasu).
- [HIPOTEZA] `n_ctx=131072` dla PLLuM istotnie zwiększa zużycie pamięci i czas ładowania. Nie mierzyłem.

**W5 — Jedno zawieszone żądanie blokuje wszystkich użytkowników.**
- [FAKT] `llm_runtime.py:24-28`: próba nieblokująca, potem `_LLM_LOCK.acquire()` bez limitu czasu.
- [FAKT] Klient Ollamy z `services.py:103-107` ma timeout 300 s; `openai._constants.DEFAULT_MAX_RETRIES == 2`. Trzy próby po 300 s odbywają się wewnątrz blokady (`serwer.py:204-218`), czyli do ok. 15 minut.
- [FAKT] Blokada obejmuje też żądania do OpenRouter, które nie obciążają lokalnego GPU.
- [FAKT] Nie ma limitu liczby spraw ani zapytań na sesję.

**W6 — Na czystym środowisku demo się nie uruchomi.**
> **Status: NAPRAWIONE 2026-10-12** (commit `9c602f9`; dwa PDF-y i dwa pliki Modelfile są śledzone). Opis w T-19. Poniższe punkty opisują stan z dnia audytu.
- [FAKT] PDF-y źródłowe są ignorowane (`src/Zrodlowyszukiwator/RAG/RAW/`), a `RAG/document_catalog.json` ma `source_url: null` dla wszystkich 6 dokumentów. README nie mówi, skąd je pobrać.
- [FAKT] `rag_ingestion/cli.py:23` domyślnie wskazuje `examples/document_catalog.json`, który nie istnieje. Faktyczny katalog to `src/Zrodlowyszukiwator/RAG/document_catalog.json`.
- [FAKT] `README.md:416-417`: sekcja 5.3 `.env` urywa się w pół zdania i skleja z nagłówkiem 5.4.
- [FAKT] `rag_ingestion/README.md:69,72,76` odwołuje się do `requirements-rag.txt`, `.env.example`, `examples/document_catalog.json`; `Zrodlowyszukiwator/README.md:13-14` do `02_brief_to_evidence.ipynb` i `README_PART2.md`. Żaden z tych plików nie istnieje.
- [FAKT] Modelfile'i dla `pismak-pllum` i `pismak-bielik` nie ma w repo; README (sekcja 4.2) pokazuje tylko linię `FROM`.
- [FAKT] Główne README nie opisuje OpenRouter; robi to tylko `src/Wyswietlator/README.md`.
- [FAKT] `requirements.txt` instaluje `psycopg` bez wariantu `[binary]`, więc na Windows trzeba ręcznie doinstalować libpq (`README.md:296-299`).

### 3.3 Średnie

**S1 — Szczegóły wewnętrzne widoczne dla użytkownika.**
> **Status: NAPRAWIONE 2026-10-13** (commit `464334c`). Opis w T-09. Poniższe punkty opisują stan z dnia audytu.
- [FAKT] `app.py:568-570` i `:764-766` dopisują do komunikatu „Miejsce błędu: `plik:linia in funkcja`” (`services.py:149-158`).
- [FAKT] `DatabaseConnectionError` niesie host, port i polecenia Dockera (`rag_retrieval/repository.py:41-46`) i jest przekazywany do UI bez zmian (`services.py:123-124`).
- [FAKT] Nie ma `.streamlit/config.toml`; `streamlit config show` potwierdza `showErrorDetails = "full"`, `gatherUsageStats = true`, `headless = false`. Wyjątek poza blokami `try` pokaże pełny traceback.

**S2 — Wyciek uchwytów plików logów.**
> **Status: NAPRAWIONE 2026-10-13** (commit `7ea21fe`). Opis w T-11. Poniższe punkty opisują stan z dnia audytu.
- [FAKT] `configure_logging` (`services.py:64-82`) tworzy plik i `FileHandler` dla każdego nowego `session_id`, czyli przy każdym otwarciu lub odświeżeniu strony.
- [FAKT] Handler jest zamykany tylko w `reset_case` (`app.py:148`). Po rozłączeniu sesji zostaje na zawsze.
- [FAKT] Logger ma `propagate = False`, a każdy handler filtruje po `session_id`. Rekordy bez tego pola, np. `LOGGER.exception("model_list_failed")` w `app.py:322`, nie trafiają nigdzie.

**S3 — UI nie informuje o zapisie danych.**
> **Status: NAPRAWIONE 2026-10-13** dodane informacja i wymagana zgoda
- [FAKT] `save_run` zapisuje pełną rozmowę, brief, dowody i pismo bez terminu usunięcia.
- [FAKT] `log_model_response` (`serwer.py:31-59`) zapisuje do 20 000 znaków odpowiedzi modelu, gdy jest niepoprawna lub ucięta.
- [FAKT] Jedyne ostrzeżenie o prywatności w UI dotyczy OpenRouter (`app.py:364-373`). O zapisie lokalnym mówi tylko `README.md:588-597`.

**S4 — Użytkownik dema sam wybiera dostawcę i model.**
> **Status: DZIAŁANIE ZAMIERZONE — decyzja właściciela z 2026-10-13.** Użytkownik ma móc wybierać dostawcę i model. To nie jest usterka i nie wymaga poprawki; nie ograniczaj tego wyboru. T-13 nie będzie realizowane. Poniższe punkty zostają jako opis zachowania.
- Konsekwencja do pamiętania: zużycie klucza OpenRouter i mocy obliczeniowej hosta ogranicza wyłącznie Cloudflare Access (W3, T-02).
- [FAKT] Każdy użytkownik może przełączyć dowolny etap na OpenRouter (`app.py:269-301`) i wtedy używany jest klucz właściciela (`serwer.py:113-123`).
- [FAKT] Może też wybrać dowolny model generatywny z hosta (`services.py:161-167`) i ustawić do 12 pytań.
- [FAKT] W logach jest 8 wywołań OpenRouter: 4 udane, 2 błędy walidacji lub połączenia, 2 ucięcia. Próba jest mała.

**S5 — Retrieval nie odfiltrowuje dokumentów nieobowiązujących.**
> **Status: NAPRAWIONE 2026-10-13** (commit `d8f8df9`; poprawka katalogu właściciela w `e96e4d3`). Opis w T-16. Poniższe punkty opisują stan z dnia audytu.
- [FAKT] `app.py:648` wywołuje `zrodla.run(case_brief)` bez filtrów; `RetrievalFilters.active_only` domyślnie `False` (`rag_retrieval/models.py:20`).
- [FAKT] W bazie `wroclaw-strategy-2030` ma `is_active = false` (203 chunki). Pole `is_active` nie trafia do rerankera (`reranker.py:219-231`) ani do kontekstu Pismogeneratora (`formatting.py:36-49`).
- [FAKT] W `document_catalog.json` Strategia 2030 (`:36-37`) ma ten sam numer i datę aktu co Strategia 2050 (`:12-13`). Jedna z tych par jest błędna.
- [FAKT] `active_only=True` generuje warunek `d.is_active IS TRUE` (`repository.py:71-74`). Cztery z sześciu dokumentów mają `is_active = NULL`, więc ten filtr zostawiłby tylko Strategię 2050.

**S6 — Wyszukiwanie pełnotekstowe nie działa dla zapytań wielowyrazowych.**
> **Status: NAPRAWIONE 2026-10-14** (niezacommitowane). Opis, pomiary i ograniczenia w T-18. Poniższe punkty opisują stan z dnia audytu.
- [FAKT] `repository.py:160` używa `websearch_to_tsquery('simple', …)`: wszystkie słowa łączone przez AND, bez odmiany.
- [FAKT] Test na bazie: wszystkie 3 zapytania leksykalne zapisane w śledzonym `evidence.json` zwracają 0 trafień. Kontrolnie: „bezpieczeństwo pieszych” → 11, „bezpieczeństwo pieszych przejścia szkoła” → 0.
- [FAKT] Plan awaryjny (`planner.py:214`) wysyła jako zapytanie leksykalne pierwsze 220 znaków opisu problemu.
- Skutek: wyszukiwanie hybrydowe jest w praktyce wyłącznie wektorowe.

**S7 — Walidator podstaw prawnych daje fałszywe alarmy.**
- [FAKT] `pismogenerator_helper_functions.py:37-51` szuka podciągu `"art."` w tekście zamienionym na małe litery. Pasują też „na start.” i „kwart.”.
- [FAKT] `DraftValidationError` nie uruchamia ponowienia (`pismogenerator.py:84-85`).
- [FAKT] Ponowne kliknięcie przycisku uruchamia cały RAG od początku (`app.py:600-648`); `evidence_pack` zapisany w sesji nie jest używany.

**S8 — Notebook drukuje `DATABASE_URL`.**
> **Status: NAPRAWIONE 2026-10-13**
- [FAKT] `src/Zrodlowyszukiwator/rag_ingestion/notebooks/01_rag_ingestion.ipynb`, komórka 6: `print(os.environ.get("DATABASE_URL"))` z zapisanym wynikiem zawierającym hasło. Dziś hasło równa się publicznej wartości domyślnej; po jego zmianie i ponownym uruchomieniu wyciekłoby prawdziwe.
- [FAKT] Wszystkie trzy śledzone notebooki mają zapisane wyniki komórek i ścieżki `C:\Krzysiek\…`; `src/main.ipynb` zawiera zapisany błąd `ValueError`.

**S9 — Zależności.**
> **Status: CZĘŚCIOWO NAPRAWIONE 2026-10-14** (wersje i blokada: commit `29b40d3`; licencja: niezacommitowane) — wersje przypięte, plik blokady dodany, granica `openai` rozstrzygnięta. Licencja projektu zmieniona na AGPL-3.0-or-later (D4), więc PyMuPDF nie stwarza już konfliktu licencji. Otwarte: podział zależności. `psycopg` zostaje bez `[binary]` (decyzja właściciela). Opis w T-20. Poniższe punkty opisują stan z dnia audytu.
- [FAKT] Brak pliku blokady; `requirements.txt` ma same zakresy. Jeden plik łączy zależności uruchomieniowe z notebookami, testami i linterem.
- [FAKT] `serwer.py:8` importuje prywatne `openai.lib._parsing.type_to_response_format_param`, a `llm_runtime.py:9` importuje `LengthFinishReasonError`. Dolna granica to `openai>=1.0`.
- [HIPOTEZA] Oba symbole pojawiły się dopiero w okolicach openai 1.40. Sprawdź przed ustawieniem nowej granicy.
- [FAKT] Import `services` ładuje PyMuPDF: `services.py:40` → `rag_ingestion/__init__.py:2` → `pipeline.py` → `pdf_extractor.py:7`. PyMuPDF jest dostępny na licencji AGPL-3.0 albo komercyjnej; projekt deklaruje MIT. To nie jest porada prawna — patrz D4.
- [FAKT] Komunikat w `rag_ingestion/repository.py:24` każe sprawdzić `psycopg[binary]`, choć projekt go nie używa.

**S10 — Testy.**
- [FAKT] Istnieją dwa równoległe drzewa: `tests/unit_test/` (24 testy, z `__init__.py`) i `tests/unit_tests/` (67 testów, bez `__init__.py`). Sześć nazw klas powtarza się w obu.
- [FAKT] `.github/copilot-instructions.md:4` każe umieszczać testy w `tests\unit_test\<moduł>\`.
- [FAKT] `app.py`, `rag_ingestion/cli.py` i `rag_ingestion/catalog.py` nie mają żadnego testu. Izolacja sesji w UI nie jest testowana automatycznie.
- [FAKT] Wszystkie testy używają atrap; żaden nie łączy się z siecią ani z bazą.

### 3.4 Niskie

- **N1** Limit pytań: `README.md:75` mówi 10, `services.py:49` ma 12, `serwer.py:19` ma 3 (stała nieużywana przez UI).
- **N2** `.env_template` definiuje osiem zmiennych nieczytanych przez kod (`APP_ENV`, `APP_NAME`, `LOG_LEVEL`, `PLLLUM_MODEL_PATH`, `BIELIK_MODEL_PATH`, `PLLLUM_API_URL`, `BIELIK_API_URL`, `RAG_LLM_MODEL`); linia 14 ma spację po `=`. `serwer.py:14` ma adres Ollamy na sztywno i ignoruje `OLLAMA_URL`. `app.py:23` ładuje `.env` z `override=True`, więc plik nadpisuje zmienne procesu.
- **N3** ~~YAML tunelu w README ma niespójne wcięcie.~~ Usunięte 2026-10-12 razem z opisem wariantu lokalnego (W3). Drzewo katalogów w sekcji 9 i lokalna ścieżka w `rag_ingestion/README.md` poprawione 2026-10-12 (W6). N3 zamknięte.
- **N4** `ruff format --check` wskazuje 3 pliki; CI uruchamia tylko `ruff check`. `.pre-commit-config.yaml` przypina `v0.11.13`, zainstalowany ruff to 0.16.9.
- **N5** `.gitignore` nie obejmuje `.streamlit/secrets.toml`, plików poświadczeń cloudflared ani tego handoveru. Lokalne refy `refs/agents/*` zawierają 57 MB PDF-ów; nie ma ich na `origin/main`.
- **N6** `pismak-pllum` ma w Ollamie szablon `TEMPLATE {{ .Prompt }}` bez znaczników czatu. [HIPOTEZA] Obniża to jakość odpowiedzi.
- **N7** Martwy kod: `utilities.py`, gałąź `exited` w `dopytywator.py:97-101`, stałe `serwer.py:19-21`, obsługa `LengthFinishReasonError` w `llm_runtime.py:40-48`. `print()` w ścieżce serwera wypisuje dane sprawy na standardowe wyjście (`pismogenerator_helper_functions.py:83-88`, `pismogenerator.py:45-50`).
- **N8** Chunker może przekroczyć `max_chars` i wytworzyć chunk z samego overlapu (`rag_ingestion/chunker.py:185-197`). W bazie najdłuższy chunk ma 6363 znaki przy limicie 3500.
- **N9** Odświeżenie strony gubi całą sprawę; komentarz w `app.py:781` sugeruje inaczej. `st.markdown` renderuje linki i obrazy z treści modelu (HTML jest wyłączony).
- **N10** Metadane commitów zawierają prywatny adres e-mail autora; po upublicznieniu będzie widoczny.
- **N11** Aplikacja łączy się z bazą jako superuser, choć potrzebuje tylko `SELECT` na dwóch tabelach.

### 3.5 Sprawdzone i poprawne

- Skan całej historii i plików śledzonych wzorcami kluczy (`sk-or-v1-`, `sk-`, `ghp_`, `hf_`, `AKIA`, klucze prywatne, JWT) — bez trafień. `.env` nigdy nie był commitowany. Klucz OpenRouter występuje wyłącznie w `.env`.
- Ollama nasłuchuje tylko na `127.0.0.1:11434`. `run_pismak.bat` wiąże Streamlit do `127.0.0.1:8501`.
- Wszystkie zapytania SQL są parametryzowane; fragment `filter_sql` składa się ze stałych.
- Źródła w piśmie są ograniczone do listy dozwolonych identyfikatorów; reranker odrzuca nieznane `chunk_id`.
- `unsafe_allow_html=True` jest użyte tylko ze stałymi tekstami (`app.py:92-142`). XSRF w Streamlit jest domyślnie włączony.
- Katalogi uruchomień mają w nazwie UUID; logi są rozdzielone per sesja i objęte testem.
- Wiadomości powyżej 5000 znaków są odrzucane; wysyłka do OpenRouter wymaga zaznaczenia zgody.

---

## 4. Czego nie zweryfikowano

1. Tunel Cloudflare — częściowo zweryfikowane 2026-10-12: tunel nie ma tras i niczego nie wystawia (lokalny `/config`). Nadal niezweryfikowane: czy w koncie Cloudflare istnieje aplikacja Access i jakie ma polityki.
2. ~~Osiągalność portu 5432 z sieci lokalnej i z internetu.~~ Bezprzedmiotowe po T-01: port jest związany wyłącznie z `127.0.0.1`. Nie wiadomo natomiast, czy ktoś połączył się z bazą w okresie 2026-09-20 – 2026-10-11, gdy port był otwarty; logów połączeń nie sprawdzałem.
3. Stan CI na GitHubie oraz przebieg testów na Linuksie i Pythonie 3.11.
4. Przepływ end-to-end w przeglądarce, dwie sesje równolegle, kolejka generacji, zachowanie przy wyłączonej Ollamie albo bazie. Nie uruchamiałem Streamlita ani generacji.
5. Instalacja na czystym środowisku.
6. Czy pole „Max pytań” (`app.py:304-313`, bez `key`) zachowuje wartość po zablokowaniu po pierwszym pytaniu. Jeśli wraca do 1, wywiad zawsze kończy się po jednym pytaniu.
7. Planner i reranker przez OpenRouter. Wysyłają `strict: True` ze schematem z `model_json_schema()` (`planner.py:100-107`, `reranker.py:106-113`), który nie spełnia wymogów trybu ścisłego. W logach RAG działał tylko na Ollamie.
8. Zużycie pamięci i czas PLLuM przy `n_ctx=131072`; parametry sprzętu.
9. Licencje modeli PLLuM i Bielik oraz dokumentów miejskich (nie są rozpowszechniane w repo).
10. Limity i uprawnienia klucza OpenRouter.
11. Czy w wolumenie bazy są dane, których nie da się odtworzyć z lokalnych PDF-ów.

---

## 5. Decyzje właściciela

| ID | Pytanie | Blokuje |
|---|---|---|
| D1 | Która formuła otwierająca jest docelowa: „W imieniu …” czy „Jako wieloletni mieszkaniec …”? | T-05 |
| D2 | Czy podpis ma zawierać nazwę organizacji? Jeśli tak — czy jest zgoda; jeśli nie — jaki tekst domyślny? | T-06 |
| D3 | Przykład w `outputs/`: zgoda autora maila, anonimizacja czy przykład syntetyczny? Historia: `git filter-repo` z force-push czy nowe repozytorium z jednym commitem? | T-04 |
| D4 | ~~PyMuPDF (AGPL albo licencja komercyjna) w projekcie MIT: zostaje z opisem w README, czy wymiana biblioteki?~~ Rozstrzygnięte 2026-10-14: licencja projektu zmieniona na AGPL-3.0-or-later; PyMuPDF zostaje. | — |
| D5 | ~~Czy publiczne demo pozwala wybrać OpenRouter i dowolny model?~~ Rozstrzygnięte 2026-10-13: tak, wybór dostawcy i modelu jest zamierzony (S4). | — |
| D6 | Które modele są oficjalnie wspierane? Bielik z oknem 8192 wymaga dużo mniejszych promptów. | T-15 |
| D7 | ~~Poprawny numer i data aktu dla Strategii 2030 oraz adresy URL wszystkich dokumentów źródłowych.~~ Dostarczone przez właściciela w katalogu (commit `e96e4d3`). | — |
| D8 | Które drzewo testów zostaje: `tests/unit_test/` czy `tests/unit_tests/`? | T-21 |
| D9 | Czy `plans/` (stary handover, ten plik, plan audytu) ma być w publicznym repo? | T-08 |
| D10 | Czy prywatny e-mail w metadanych commitów może zostać publiczny? | T-04 |

---

## 6. Zadania naprawcze

### Etap 0 — operacyjny, przed wszystkim innym

**T-01 [OPS] Zamknąć port bazy i zmienić hasło (K1). — WYKONANE 2026-10-11**

Co zrobiono, w tej kolejności:
1. `ALTER ROLE pismak PASSWORD …` przez gniazdo lokalne kontenera; nowe hasło to 48 losowych znaków szesnastkowych.
2. W lokalnym `.env` zmieniono dokładnie dwie linie: `POSTGRES_PASSWORD` i hasło w `DATABASE_URL`. Hasło istnieje tylko tam.
3. `docker compose up -d` — kontener odtworzony z `docker-compose.yaml`, wolumen `pismak_pismak_pgdata` podłączony ponownie.

Wyniki weryfikacji:
- `netstat`: jedyny nasłuch to `127.0.0.1:5432`; `PortBindings` ma `HostIp=127.0.0.1`.
- Połączenie TCP z nowym hasłem działa; baza ma nadal 6 dokumentów i 1385 chunków.
- Połączenie z dawnym hasłem domyślnym jest odrzucane (`password authentication failed`).
- Połączenie z tego komputera na jego adresy sieci lokalnej, port 5432: odmowa.
- `docker compose up -d --dry-run` zgłasza `Running`, czyli kontener odpowiada plikowi compose.
- Żaden plik śledzony nie został zmieniony.

Co zostaje bez zmian i dlaczego:
- Wartość domyślna hasła nadal jest w `.env_template:34` i `docker-compose.yaml:9`. To szablon dla nowych instalacji; lokalny `.env` ją nadpisuje. Zmiana `.env` nie zmienia hasła w już zainicjalizowanym wolumenie — do tego służy `ALTER ROLE`.
- ~~Zapisany wynik w `01_rag_ingestion.ipynb` zawiera dawny DSN, a komórka 6 drukuje `DATABASE_URL`.~~ Usunięte przez właściciela 2026-10-12 (commit `27ee2f1`): w notebooku nie ma już ani tego `print`, ani żadnego `postgresql://`. Reszta T-07 (wyczyszczenie wyników w pozostałych notebookach) pozostaje.
- Rola aplikacji to wciąż superuser (N11, T-23).

**Skutek uboczny wykryty i naprawiony 2026-10-12.** Po związaniu portu wyłącznie z `127.0.0.1` adres `localhost` w `DATABASE_URL` zaczął powodować ogromne opóźnienia: `localhost` wskazuje najpierw na `::1`, gdzie nikt już nie nasłuchuje.
- [FAKT] Zmierzone: `psycopg.connect` z hostem `localhost` — 130 s na jedno połączenie; z hostem `127.0.0.1` — 0,05 s. Surowe połączenie TCP z `::1` jest odrzucane po 2 s; resztę dokłada sterownik (implementacja `python`, Windows).
- [FAKT] Etap RAG otwiera nowe połączenie dla każdego zapytania (`rag_retrieval/repository.py`), czyli 7–11 połączeń na sprawę. Od 2026-10-11 do 2026-10-12 RAG praktycznie nie działał. Weryfikacja T-01 tego nie wychwyciła, bo nie mierzyła czasu.
- Naprawa: `DATABASE_URL` wskazuje `127.0.0.1` w lokalnym `.env` i w `.env_template` (z komentarzem). `RetrievalRepository.counts()` trwa teraz 0,43 s.
- Otwarte: `psycopg.connect` nadal nie ma `connect_timeout`, więc podobna pomyłka w konfiguracji znów da ciche, wielominutowe zawieszenie. Warto dodać przy T-10.

Jeśli trzeba to powtórzyć: **nie używaj** `docker compose down -v` (usuwa wolumen z bazą RAG).

**T-02 [OPS] Zabezpieczyć tunel przez Access (W3). — CZĘŚĆ LOKALNA WYKONANA 2026-10-12**

Wykonane:
- Weryfikacja lokalna: tunel połączony, zero tras (szczegóły w W3). Nie ma dziś czego chronić ani czego zamykać.
- `README.md`, sekcja „Stały Cloudflare Tunnel”: opis faktycznego wariantu, usunięty błędny YAML (N3), dodane polecenia weryfikacyjne. `pytest`: 93 passed.

Pozostaje dla właściciela — wymaga logowania do panelu Cloudflare, model tego nie zrobi:
1. Dodaj aplikację Access typu Self-hosted dla docelowej nazwy hosta, z polityką ograniczoną do listy adresów e-mail. **Zrób to przed dodaniem trasy.**
2. Dodaj trasę publiczną: nazwa hosta → `http://127.0.0.1:8501`. Żadnych tras do portów 11434, 5432 ani Docker API.
3. Zdecyduj, czy usługa ma startować automatycznie. Bez tras jest nieszkodliwa.

Akceptacja (obie komendy są w README):
- `curl http://127.0.0.1:20241/config` pokazuje dokładnie jedną trasę do `http://127.0.0.1:8501` oraz regułę domyślną.
- `curl -I https://<NAZWA_HOSTA>` bez logowania zwraca `302` do domeny `cloudflareaccess.com`.
- Konto z listy loguje się, konto spoza listy dostaje odmowę.

**T-03 [DECYZJA] Uzyskać odpowiedzi D1–D10.**

### Etap 1 — blokery publikacji

**T-04 Wymienić przykład i wyczyścić historię (K2).** Zależy od D3, D10.
- Zastąp `outputs/20260926_200658_ui/` przykładem syntetycznym, wygenerowanym po poprawkach T-05 i T-06. Zaktualizuj wyjątek w `.gitignore`.
- Usunięcie z historii wymaga wyraźnego potwierdzenia właściciela (force-push albo nowe repo). Wykonaj je raz, tuż przed upublicznieniem.
- Nie używaj `git push --mirror` — wypchnąłby `refs/agents/*`.
- Akceptacja: `git log origin/main --oneline -- outputs/20260926_200658_ui` jest puste; `git log origin/main -p | grep -c "skrzynki mailow"` zwraca 0.

**T-05 Ujednolicić formułę otwierającą (W1).** Zależy od D1. Wykonaj razem z T-06 (ten sam plik).
- Pliki: `pismogenerator_helper_functions.py:96`, `tests/unit_tests/pismogenerator/test_pismogenerator.py:134`, `README.md:93`, `prompts.py:30-31`.
- Jeśli zostaje nowa formuła, popraw „kochango”.
- Akceptacja: `python -m pytest -q` → 0 niepowodzeń.

**T-06 Usunąć nazwę organizacji z kodu na sztywno (W2).** Zależy od D2.
- Zmiana tylko w rendererze; nie dodawaj pól do `DraftLetter`.
- Akceptacja: domyślny `letter.md` nie zawiera nazwy organizacji, chyba że ją skonfigurowano; testy w `tests/unit_tests/pismogenerator/` zaktualizowane.

**T-07 Oczyścić notebooki (S8).**
- Usuń `print(os.environ.get("DATABASE_URL"))` z komórki 6 w `01_rag_ingestion.ipynb`. Wyczyść wyniki we wszystkich trzech notebookach.
- `tests/unit_tests/test_release_configuration.py` sprawdza treść komórek `src/main.ipynb` — nie usuwaj wymaganych tam fragmentów.
- Akceptacja: żaden śledzony notebook nie zawiera `postgresql://` ani niepustych `outputs`; `pytest` zielony.

**T-08 Uzupełnić `.gitignore` (N5).** Zależy od D9.
- Dodaj `.streamlit/secrets.toml`, pliki poświadczeń cloudflared oraz — zgodnie z D9 — dokumenty audytu w `plans/`.
- Akceptacja: `git check-ignore` potwierdza każdą regułę.

### Etap 2 — utwardzenie dema

Zadania T-09, T-12, T-14 i T-17 zmieniają `app.py`. Wykonuj je kolejno, nie równolegle.

**T-09 Usunąć szczegóły wewnętrzne z UI (S1). — WYKONANE 2026-10-13**

Zasada ustalona przez właściciela: informacja o błędzie i jej szczegóły mają być w logach, nie w interfejsie.

Interfejs:
- `services.llm_error_message` zwraca dla każdego rodzaju błędu ogólny komunikat i nigdy nie przepuszcza treści wyjątku. Zniknęły: host i port bazy, polecenia Dockera, nazwy zmiennych i plików (`OPENROUTER_KEY`, `.env`), nazwy pól schematu, lista błędów walidacji pisma, odsyłacz „sprawdź log”.
- `app.py` nie dopisuje już „Miejsce błędu: plik:linia” ani fragmentu uciętej odpowiedzi modelu.
- Treści wyjątków (`DatabaseConnectionError`, `DraftValidationError` itd.) zostały bez zmian — korzystają z nich logi, notebooki i `test_repository.py`.
- Świadomie zostawione: rozmiar okna kontekstu w komunikacie o zbyt długich danych (pomaga wybrać inny model) oraz komunikaty walidacji wiadomości użytkownika.

Logi:
- Zdarzenia `interview_failed` i `pipeline_failed` zawierają typ błędu, miejsce w kodzie, szczegóły (`error_log_detail`) i — nowość — pełny ślad stosu z łańcuchem przyczyn (`exc_info`). Przy błędzie bazy w logu widać więc także pierwotny wyjątek sterownika.
- `model_list_failed` ma teraz `session_id`; wcześniej ten rekord nie trafiał do żadnego pliku (część S2).
- Uciętą odpowiedź modelu zapisuje jak dotąd klient LLM (`llm_model_response outcome=truncated`).
- Nowy wspólny plik `logs/app_errors.log` zbiera wyjątki, których kod aplikacji nie obsłużył. `configure_logging` podpina go raz na proces do loggera `streamlit.error_util`.

Konfiguracja `.streamlit/config.toml`:
- `client.showErrorDetails = "none"`, `browser.gatherUsageStats = false`, `server.headless = true`.
- `logger.enableRich = false` — konieczne. [FAKT] Gdy zainstalowana jest biblioteka `rich` (wchodzi jako zależność pośrednia, tu wersja 15.0.0), Streamlit domyślnie wypisuje nieobsłużone wyjątki przez nią prosto na konsolę i **nie** wywołuje loggera; plik `app_errors.log` zostawał wtedy pusty.
- Plik jest wczytywany z katalogu roboczego, więc aplikację trzeba uruchamiać z katalogu głównego repozytorium (`run_pismak.bat` to robi).

Weryfikacja:
- [FAKT] Testy z prawdziwym skryptem `app.py` (`tests/unit_test/wyswietlator/test_app.py`, Streamlit `AppTest`): awaria modelu i awaria bazy pokazują ogólny komunikat bez hosta, portu, ścieżki i śladu stosu, a log sprawy zawiera pełną treść, miejsce, ślad stosu i przyczynę. To pierwsze testy obejmujące `app.py` (część S10).
- [FAKT] Błąd nieobsłużony: przeglądarka dostaje ogólny tekst Streamlita, `app_errors.log` typ, treść i ślad stosu. Test ustawia opcje jawnie, więc nie zależy od pliku konfiguracyjnego; osobny test pilnuje wartości w `.streamlit/config.toml`.
- [FAKT] `streamlit config show` z katalogu repozytorium pokazuje wszystkie cztery wartości z pliku.
- 174 testy przechodzą, `ruff check` czysty.

Czego to nie obejmuje:
- Nie sprawdzono w przeglądarce na uruchomionej aplikacji — tylko przez `AppTest`.
- Ogólny komunikat Streamlita przy błędzie nieobsłużonym jest po angielsku.
- Błędna wartość `PISMAK_CONTEXT_TOKENS` nadal pokaże użytkownikowi nazwę tej zmiennej (`validate_user_message` zgłasza ją jako `ValueError`).
- `logs/app_errors.log` rośnie bez ograniczeń, jak pozostałe logi (S2, T-11).

**T-10 Ograniczyć czas trzymania blokady (W5). — CZĘŚCIOWO 2026-10-12**
- Zrobione: `max_retries=0` dla obu klientów Ollamy i jeden limit czasu `OLLAMA_REQUEST_TIMEOUT_SECONDS = 900` w `serwer.py` (decyzja właściciela: 15 minut zamiast 300 s, bo pełne okno modelu wymaga długich wywołań na CPU). Najgorszy przypadek trzymania blokady to nadal 15 minut, ale już jedno żądanie, nie trzy ponawiane.
- [FAKT] Dowód z logu Ollamy (2026-10-12, jeszcze na starych ustawieniach): żądanie rerankera zerwane po dokładnie 5 min 0 s, ponowione przez SDK i zerwane po 4 min 59 s, trzecia próba udana po 4 min 50 s. Jedno wywołanie zajęło więc blokadę na prawie 15 minut i dwa razy policzyło ten sam prompt od zera.
- Uwaga: `app.py` trzyma klienta Ollamy w `st.cache_resource`, więc zmiana limitu wymaga restartu Streamlita.
- ~~Ustaw `max_retries=0` dla klientów Ollamy i rozważ krótszy timeout.~~
- Dodaj limit czasu oczekiwania na `_LLM_LOCK` z osobnym wyjątkiem i czytelnym komunikatem.
- Rozważ pominięcie blokady dla OpenRouter. Uwaga: `tests/unit_tests/zrodlowyszukiwator/test_retrieval.py` i `test_brief_to_evidence.py` sprawdzają kolejkowanie plannera i rerankera.
- Akceptacja: test jednostkowy ścieżki przekroczenia czasu; ręcznie — druga sesja dostaje komunikat zamiast wisieć.

**T-11 Naprawić cykl życia logów (S2). — WYKONANE 2026-10-13**

Jak to działa (`src/Wyswietlator/services.py`):
- Jeden `_CaseLogHandler` na katalog logów zastąpił `FileHandler` tworzony dla każdej sesji. Handler trzyma mapę „sesja → plik” i przy każdym wpisie otwiera plik, dopisuje linię i go zamyka.
- Własność z README zachowana: każda sprawa ma własny plik `pismak_<timestamp>.log`, a wpisy równoległych sesji się nie mieszają. Plik powstaje przy pierwszym wpisie sprawy, nie przy konfiguracji.
- Ten sam handler jest podpięty do loggera `streamlit.error_util`, więc błędy nieobsłużone (T-09) nadal trafiają do `logs/app_errors.log`.
- Wpis bez `session_id`, z nieznaną sesją albo dla sprawy zamkniętej przez „Nowa sprawa” trafia do `logs/app_errors.log` zamiast przepadać.
- Mapa pamięta najwyżej `MAX_TRACKED_CASES = 1000` ostatnio używanych spraw. `configure_logging` jest wołane przy każdym rerunie i odświeża sprawę, więc wypadają tylko sesje porzucone.
- `emit` odtwarza katalog `logs/`, jeśli ktoś go usunął w trakcie pracy.

Pomiar przed i po, 50 otwartych sesji:

| | Przed | Po |
|---|---|---|
| Handlery dodane do loggera | 50 | 1 |
| Otwarte, zablokowane pliki | 50 | 0 |
| Wpis bez identyfikatora sesji | przepada | zapisany |

- [FAKT] Koszt jednego wpisu z otwarciem i zamknięciem pliku: 0,5 ms.
- [FAKT] Testy w `tests/unit_test/wyswietlator/test_services.py`: 50 spraw bez przyrostu handlerów i z plikami dającymi się usunąć; 8 wątków po 100 wpisów bez przemieszania; wpisy bez sprawy; limit pamiętanych spraw; usunięcie katalogu w trakcie pracy; zamknięcie sprawy. 180 testów przechodzi.
- [FAKT] W kodzie `src/` nie ma dziś ani jednego wywołania loggera bez `session_id` (sprawdzone analizą AST), więc plik wspólny jest siatką bezpieczeństwa, nie zwykłą ścieżką.

Czego to nie rozwiązuje:
- Każde otwarcie lub odświeżenie strony nadal tworzy nowy plik logu (wpis `session_started`). Liczba plików w `logs/` rośnie bez ograniczeń i nie ma retencji.
- Sesja nieużywana tak długo, że wyprzedziło ją 1000 nowszych, po powrocie dostanie nowy plik — jej log będzie w dwóch plikach.
- Wspólny plik nazywa się `app_errors.log`, choć może dostać także wpis informacyjny bez sprawy.

**T-12 Dodać w UI informację o zapisie danych (S3).**
- Stały komunikat widoczny przed pierwszą wiadomością: rozmowa i pismo są zapisywane na komputerze operatora, nie podawaj zbędnych danych osobowych.
- Akceptacja: komunikat widoczny przy świeżej sesji.

**T-13 Tryb demo (S4). — NIE REALIZOWAĆ (decyzja właściciela z 2026-10-13)**
- Wybór dostawcy i modelu przez użytkownika jest zamierzony (S4, D5). Nie dodawaj przełącznika, który go ukrywa lub ogranicza.
- Pierwotny zakres, tylko dla kontekstu: ~~przełącznik w zmiennej środowiskowej ukrywający wybór dostawcy i modelu oraz ograniczający liczbę pytań~~.

**T-14 [opcjonalne] Limit użycia na użytkownika (W5).** Zależy od T-02.
- Licznik w `st.session_state` nie wystarczy: odświeżenie strony tworzy nową sesję. Limit musi być przypisany do tożsamości, np. do nagłówka z adresem e-mail dodawanego przez Cloudflare Access.
- [HIPOTEZA] Nagłówek Access jest dostępny w Streamlit przez `st.context.headers`. Sprawdź to przed wdrożeniem.
- Wykonaj tylko wtedy, gdy testy z testerami pokażą nadużycia; Access i T-10 są pierwszą linią obrony.
- Akceptacja: po przekroczeniu limitu użytkownik dostaje komunikat, a odświeżenie strony go nie zeruje.

### Etap 3 — poprawność

**T-15 Budżet kontekstu (W4). — WYKONANE 2026-10-12**

Jak to działa:
- Moduł `src/_shared/context_budget.py` liczy budżet. **Okno pochodzi z modelu wybranego w UI** (decyzja właściciela z 2026-10-12: zawsze pełne okno modelu), a ustala je `src/_shared/model_context.py`:
  - Ollama: `GET /api/ps` podaje okno, z jakim serwer faktycznie załadował model. Gdy model nie jest załadowany, moduł ładuje go pustym żądaniem `/api/generate` pod `llm_generation_lock` i czyta ponownie. Zmierzone: Bielik 8192, PLLuM 131 072 (pierwsze ładowanie PLLuM: 52 s).
  - OpenRouter: publiczna lista `GET /api/v1/models`, pamięć podręczna 1 h. Dla `openrouter/free` najmniejsze okno wśród darmowych modeli obsługujących structured output (zmierzone: 65 536); dla konkretnego identyfikatora jego własne.
  - Wykrywanie wywołują `services.build_zrodlowyszukiwator` i `app.py`; wynik jest przekazywany jawnie jako `context_tokens` do `LLMClient`, plannera, rerankera i `validate_user_message`. Kod biblioteczny i testy nie robią zapytań sieciowych; bez przekazanej wartości obowiązują wartości zapasowe 8192 / 32 768.
  - Nieudane wykrycie: ostrzeżenie `context_window_detection_failed`, ostatnia znana wartość dla modelu albo wartość zapasowa.
- `PISMAK_CONTEXT_TOKENS` i `PISMAK_OPENROUTER_CONTEXT_TOKENS` to już tylko opcjonalny górny limit okna; domyślnie nieustawione. Decyzja D6 nie była potrzebna.
- Tokeny są szacowane ze znaków przelicznikiem 2,5 znaku na token. Zmierzone na prawdziwych promptach Pismaka: Bielik 2,81–4,11, PLLuM 3,29–4,15; najniżej wypada JSON rerankera. Do tego 256 tokenów marginesu.
- `fit_output_tokens` jest wywoływane przed każdym żądaniem (`LLMClient._call_structured`, planner, reranker). Obniża `max_tokens`, żeby prompt i odpowiedź mieściły się w oknie razem. Stałe `*_MAX_OUTPUT_TOKENS` zostały jako górne limity — są potrzebne modelom OpenRouter z tokenami rozumowania.
- Gdy na odpowiedź zostaje mniej niż 1024 tokeny, leci `LLMInputLimitError` i żądanie w ogóle nie jest wysyłane. UI pokazuje komunikat o oknie kontekstu. Planner i reranker na Ollamie przechodzą wtedy na istniejący fallback.
- Reranker (`_fitted_user_prompt`): najpierw skraca tekst kandydatów z 1800 do min. 500 znaków, potem odrzuca kandydatów od końca rankingu.
- Pismogenerator (`fit_draft_user_prompt`): skraca fragmenty źródeł z 2400 do min. 400 znaków. `EvidencePack` i `letter.md` są nietknięte — zmienia się tylko to, ile tekstu trafia do promptu.
- Dopytywator: `validate_user_message` odrzuca wiadomość, która nie zmieści się w pozostałym miejscu rozmowy, i podaje, ile znaków zostało.
- Logi: nowa linia `llm_context_budget` (znaki, szacowane tokeny, efektywny limit wyjścia) oraz rzeczywiste `prompt_tokens` i `completion_tokens` w `llm_request_completed`. Porównanie obu pozwala sprawdzić przelicznik.
- Dokumentacja: `README.md` sekcja 4.4 i `.env_template`.

Test na żywo, Bielik (okno 8192), sprawa z `outputs/20260926_200658_ui/brief.json`, 2026-10-12:

| Etap | Szacunek tokenów promptu | Rzeczywiste | Odpowiedź | Czas |
|---|---|---|---|---|
| Planner | 1503 | 1159 | 404 | 98 s |
| Reranker (15 z 20 kandydatów, po 500 znaków) | 6604 | 6295 | 805 | 376 s |
| Pismogenerator | 5768 | 3995 | 912 | 305 s |

Ten sam test na PLLuM (okno 131 072, pełne prompty):

| Etap | Szacunek tokenów promptu | Rzeczywiste | Odpowiedź | Czas |
|---|---|---|---|---|
| Planner | 1503 | 1152 | 521 | 85 s |
| Reranker (wszystkich 20 kandydatów, pełny tekst) | 12 162 | 9304 | 485 | 245 s |
| Pismogenerator (3 źródła) | 4334 | 3175 | 369 | 104 s |

- [FAKT] PLLuM dostał prompt rerankera 30 405 znaków bez żadnego skracania, a `ollama ps` pokazywał 131 072 przed pierwszym żądaniem i po ostatnim. Cały przebieg trwał ok. 7,5 minuty; na Bieliku ok. 13 minut.
- [FAKT] Pismo z PLLuM przeszło walidację, ale argumentacja ma tylko 497 znaków, a reranker wybrał 3 źródła. To kwestia jakości modelu (por. N6: szablon `{{ .Prompt }}`), nie okna.
- [FAKT] 0 nowych zdarzeń `truncating input prompt` w logu Ollamy w obu przebiegach; pismo z Bielika przeszło `validate_draft` (źródła E1–E5, U1). To zamyka akceptację W4.
- [FAKT] Reranker i Pismogenerator trwały dłużej niż dawny limit 300 s — także przy przyciętych promptach. Limit 900 s jest potrzebny już dla Bielika.
- [FAKT] Najniższy zaobserwowany przelicznik to 2,62 znaku na token (reranker: krótkie teksty, dużo metadanych JSON). Zapas wobec przyjętego 2,5 wynosi ok. 5% plus 256 tokenów marginesu. Jeśli w logach `prompt_tokens` kiedykolwiek przekroczy `estimated_prompt_tokens`, trzeba obniżyć `CHARS_PER_TOKEN`.

Czego to nie rozwiązuje:
- Okno z `/api/ps` jest prawdziwe tylko dla modelu załadowanego z ustawieniami domyślnymi serwera. Jeśli inne narzędzie załaduje ten sam model przez natywne API z własnym `num_ctx`, Pismak odczyta tamtą wartość, a jego własne żądanie przeładuje model z domyślną. Przy większej wartości obcej wróci ciche obcinanie.
- Na czystej instalacji Ollama ładuje modele z małym oknem domyślnym (np. 4096). Pismak to wykryje i zgłosi, że dane się nie mieszczą, podając rozmiar okna — nie obetnie po cichu — ale żeby pipeline działał, trzeba podnieść okno w Ollamie (README 4.4).
- Pełne okno oznacza długie wywołania na CPU: 28 tokenów promptu na sekundę średnio, więc prompt rerankera rzędu 11 tys. tokenów to 6–7 minut. Stąd limit 15 minut (patrz T-10).
- Modelfile'i nadal nie ma w repo (część W6).
- Zrobione przed T-10, wbrew pierwotnej kolejności. T-10 zmienia te same miejsca w `serwer.py`, `planner.py` i `reranker.py` — uważaj przy scalaniu.

**T-16 Dokumenty nieobowiązujące (S5). — WYKONANE 2026-10-13**

Punkt wyjścia: właściciel poprawił katalog (commit `e96e4d3`): numer i data aktu Strategii 2030, `is_active` i adresy źródłowe wszystkich dokumentów oraz cztery nowe pola — `lifecycle_status`, `retrieval_enabled`, `metadata_url`, `is_annex`.
- [FAKT] Model `DocumentMetadata` (`extra="forbid"`) odrzucał te pola, więc po tym commicie katalog w ogóle się nie wczytywał: 3 testy czerwone, polecenie ingestion niesprawne. Naprawa S5 musiała objąć obsługę nowych pól.

Co zrobiono:
- `rag_ingestion/models.py`: cztery nowe pola i enum `LifecycleStatus` (`current`, `current_reference`, `reference`, `superseded`). Walidator odrzuca wpis `superseded`, który nie ma `is_active=false`.
- `rag_ingestion/repository.py`: `ensure_schema` dodaje cztery kolumny przez `ALTER TABLE … ADD COLUMN IF NOT EXISTS` (zmiana addytywna, działa na istniejącej bazie); `upsert_document` je zapisuje.
- `rag_retrieval`: nowy filtr `RetrievalFilters.exclude_inactive` (domyślnie `True`) daje warunek `d.is_active IS NOT FALSE` — odpada tylko dokument jawnie nieobowiązujący, `NULL` zostaje. Warunek `d.retrieval_enabled` obowiązuje zawsze. `RetrievedChunk` i `Evidence` niosą `lifecycle_status`.
- Kontekst Pismogeneratora (`brief_to_evidence/formatting.py`) ma linię „status obowiązywania”, a prompt systemowy zakaz przedstawiania źródła nieobowiązującego jako aktualnego.
- `rag_ingestion/pipeline.py`: przy niezmienionej treści pliku metadane z katalogu są zapisywane do bazy bez ponownego dzielenia i embeddingów (`reason: identical_content_metadata_refreshed`), razem z kopią metadanych w chunkach. Zmiana tytułu, typu albo miasta wymusza pełne przetworzenie, bo te pola są częścią tekstu embeddingu.

Rozstrzygnięcie projektowe, które może wymagać potwierdzenia właściciela:
- Katalog zostawia Strategię 2030 z `retrieval_enabled: true` i notatką „zachować w RAG do zapytań historycznych”. Przyjęto, że generowanie pisma nie jest zapytaniem historycznym: dokument zostaje w bazie i jest dostępny przez `RetrievalFilters(exclude_inactive=False)`, ale domyślnie nie trafia do dowodów pisma. Aplikacja nie ma dziś trybu zapytań historycznych, więc w praktyce UI tego dokumentu nie użyje.

Zastosowanie na bazie właściciela (2026-10-13):
- [FAKT] Polecenie ingestion uruchomione dla 6 lokalnych PDF-ów dodało kolumny i odświeżyło metadane. Liczba chunków (1385) i suma kontrolna embeddingów bez zmian.
- [FAKT] Po odświeżeniu baza zgadza się z katalogiem w każdym porównywanym polu (0 różnic), a kopia metadanych w chunkach nie jest przestarzała.

Pomiar wyszukiwania (plan zapasowy, bez LLM), sprawa „realizacja celów strategii rozwoju”:

| | Przed | Po |
|---|---|---|
| Kandydaci ze Strategii 2030 (uchylonej) | 2 z 20 | 0 z 20 |
| Dowody z adresem źródłowym | 0 | wszystkie |

Weryfikacja na tymczasowej, pustej bazie (usuniętej po teście):
- [FAKT] Nowy schemat powstaje od zera; dokument z `retrieval_enabled: false` i dokument `superseded` nie pojawiają się w domyślnym wyszukiwaniu; przy `exclude_inactive=False` wraca tylko ten drugi; ponowne włączenie w katalogu działa bez embeddingów; zmiana tytułu przetwarza plik ponownie (27 z 27 chunków z nowym tytułem); sprzeczny wpis jest odrzucany.
- 200 testów przechodzi (20 nowych), `ruff check` czysty.

Czego to nie obejmuje:
- Baza utworzona starszą wersją kodu nie ma nowych kolumn i wyszukiwanie zgłosi na niej błąd, dopóki nie zostanie raz uruchomione polecenie ingestion. README, punkt 6.2, o tym mówi.
- `lifecycle_status` nie trafia do promptu ani do UI — jest przechowywany i zapisywany w `evidence.json`. Znaczenia statusów `current_reference` i `reference` nie interpretowano poza treścią katalogu.
- Reranker nie dostaje statusu dokumentu; przy domyślnym filtrze nie ma to znaczenia.
- Nie wykonano pełnego przebiegu z LLM po tej zmianie; zmierzono wyszukiwanie i kontekst.

**T-17 Walidator i ponowne użycie dowodów (S7).** Po T-09.
- Zastąp dopasowanie podciągu wyrażeniem z granicami słów.
- Po `DraftValidationError` użyj `st.session_state.evidence_pack` zamiast ponownie uruchamiać RAG.
- Akceptacja: „na start.” przechodzi, „art. 241 KPA” nie przechodzi; drugie kliknięcie nie wywołuje plannera.

**T-18 Wyszukiwanie pełnotekstowe (S6). — WYKONANE 2026-10-14 (niezacommitowane)**
- Przejdź na semantykę OR. Zostań przy konfiguracji `'simple'` — zmiana wymagałaby migracji kolumny `search_vector` i ponownego ingestion.
- Popraw zapytanie leksykalne w planie awaryjnym (`planner.py:214`).
- Akceptacja: „bezpieczeństwo pieszych przejścia szkoła” zwraca więcej niż 0 trafień na bieżącej bazie; test jednostkowy zapytania.

Co zrobiono (`src/Zrodlowyszukiwator/rag_retrieval/repository.py`):
- Nowa funkcja `lexical_tsquery` buduje zapytanie z tekstu: małe litery, podział na słowa (znaki operatorów tsquery i interpunkcja odpadają, więc nie da się wstrzyknąć składni), pominięcie słów krótszych niż 3 znaki i ok. 70 polskich słów funkcyjnych (`STOPWORDS`), liczby zostają zawsze. Słowa dłuższe niż 5 znaków tracą do 3 ostatnich liter i są dopasowywane prefiksem (`rowerowe` → `rower:*`, pasuje też `rowerowych`); krótsze dopasowywane dokładnie. Terminy łączone przez `|`, bez powtórzeń, najwyżej 16 (`MAX_LEXICAL_TERMS`).
- `lexical_search` używa `to_tsquery('simple', …)` z tym zapytaniem zamiast `websearch_to_tsquery`. Ranking bez zmian (`ts_rank_cd`), więc wyżej są fragmenty z większą liczbą terminów. Zapytanie bez znaczących słów zwraca pustą listę bez łączenia z bazą.
- Planu awaryjnego (`planner.py:287/296/305`, `problem[:220]` itd.) nie zmieniano: tekst trafia do `lexical_tsquery`, która sama wybiera z niego do 16 terminów. Osobna obróbka w plannerze dublowałaby tę logikę.
- Schemat bazy, kolumna `search_vector` i konfiguracja `'simple'` bez zmian — nie trzeba ponownego ingestion.

Wybór wariantu — pomiar na bazie właściciela (tylko odczyt, 7 zapytań, dokumenty aktywne):

| Zapytanie (skrót) | trafienia AND (stary) | wspólne z top-20 wektorowym: AND / OR dokładny / OR z prefiksem |
|---|---|---|
| bezpieczeństwo pieszych przejścia szkoła | 0 | 0 / 12 / 10 |
| drogi rowerowe infrastruktura rowerowa | 2 | 2 / 4 / 9 |
| komunikacja tramwajowa częstotliwość kursów | 0 | 0 / 2 / 4 |
| dostępność przestrzeni publicznej osoby z niepełnosprawnościami | 0 | 0 / 0 / 1 |
| zieleń miejska drzewa nasadzenia | 1 | 0 / 2 / 9 |
| Straż Miejska we Wrocławiu, … ruch drogowy | 0 | 0 / 0 / 0 |
| Brak bezpiecznego przejścia dla pieszych przy szkole podstawowej | 0 | 0 / 11 / 9 |

Wybrano OR z prefiksem: najwięcej zgodności z wyszukiwaniem wektorowym w 4 z 7 zapytań i jedyny wariant, który łapie odmianę („rowerowe” / „rowerowa”, „szkoła” / „szkole”).

Weryfikacja po zmianie (przez prawdziwe `RetrievalRepository` i `HybridRetriever`, tylko odczyt):
- [FAKT] „bezpieczeństwo pieszych przejścia szkoła”: 20 trafień (limit), wcześniej 0. Pierwsze wyniki to strony 28–29 standardu dostępności o przejściach dla pieszych.
- [FAKT] Każde z 7 zapytań zwraca 20 trafień, czas 58–162 ms. W top-20 wyniku hybrydowego 10–16 pozycji ma teraz wkład wyszukiwania pełnotekstowego; przed zmianą najwyżej 2, bo wyszukiwanie zwracało 0–2 fragmenty.
- [FAKT] Trzy zapytania leksykalne z `outputs/20260926_200658_ui/evidence.json` (6–10 słów): wcześniej 0 dopasowań, teraz 367, 478 i 734 dopasowania; każde wyszukiwanie zwraca 20 najlepszych.
- [FAKT] Żaden zwrócony fragment nie pochodzi z dokumentu nieobowiązującego (filtr z T-16 działa także tu).
- 209 testów przechodzi (`tests/unit_test/zrodlowyszukiwator/test_lexical_query.py` — 9 nowych; jedna zmieniona asercja w `test_retrieval_filters.py`), `ruff check` czysty.

Ograniczenia:
- To nie jest prawdziwa lematyzacja. Ucięcie końcówki to heurystyka: krótkie słowa (≤ 5 znaków) nie łapią innych form („drogi” nie znajdzie „droga”), a zbyt krótki prefiks może dopasować niezwiązane słowa. Właściwe rozwiązanie to słownik polski w PostgreSQL (np. Hunspell/`polish`), co wymaga migracji `search_vector` i ponownego ingestion — poza zakresem.
- OR dopasowuje dużą część korpusu (do 62% aktywnych fragmentów przy długich zapytaniach), więc trafność zależy od rankingu `ts_rank_cd`. Słowa obecne prawie wszędzie w tym korpusie (np. „Wrocław” → `wrocła:*`, „miejski” → `miejs:*`) nie są pomijane i słabiej różnicują wyniki. Przy zapytaniu o Straż Miejską (tematu brak w korpusie) wyszukiwanie zwraca ogólne fragmenty Strategii 2050.
- Numery aktów („XXIV/463/25”) rozpadają się na osobne terminy połączone przez OR, więc numer nie jest dopasowywany jako całość. Przed zmianą działał tylko jako fraza, jeśli wszystkie części występowały w tym samym fragmencie.
- Lista słów funkcyjnych jest ręczna i niepełna.
- Nie wykonano pełnego przebiegu z LLM po tej zmianie; zmierzono samo wyszukiwanie.

### Etap 4 — gotowość do publikacji

**T-19 README i szablon konfiguracji (W6, N1, N2, N3). — CZĘŚĆ W6 WYKONANA 2026-10-12**

Decyzja właściciela: do repozytorium trafiają **tylko dwa** dokumenty źródłowe — `01_Polityka-rowerowa-wroclawia.pdf` (111 KB) i `01_Wroclawska_polityka_mobilnosci.pdf` (1,2 MB). Pozostałe cztery PDF-y zostają lokalne.

Co zrobiono:
- `.gitignore`: katalog `RAW/` ignoruje wszystko poza tymi dwoma plikami; katalog `modele/` ignoruje wszystko poza `modele/bielik/Modelfile` i `modele/pllum/Modelfile`. Sprawdzone `git check-ignore` i `git add -A --dry-run`: modele GGUF (12 GB) i cztery pozostałe PDF-y nadal są ignorowane.
- Nowe `modele/bielik/Modelfile` i `modele/pllum/Modelfile`: ten sam szablon co w modelach używanych lokalnie plus jawne `PARAMETER num_ctx` (8192 i 131 072). Sprawdzone przez `ollama create` pod tymczasową nazwą: te same wagi (skrót SHA-256), różnica wyłącznie w `num_ctx`; modele tymczasowe usunięte.
- Źródła plików GGUF zweryfikowane na Hugging Face po rozmiarze i skrócie SHA-256: PLLuM `Jerzman/PLLuM-4B-chat-2512-Q4_K_M-GGUF`, Bielik `second-state/Bielik-4.5B-v3.0-Instruct-GGUF`. Oficjalne `speakleash/Bielik-4.5B-v3.0-Instruct-GGUF` nie zawiera dziś wersji Q4_K_M. Oba repozytoria deklarują licencję apache-2.0.
- `rag_ingestion/cli.py`: `DEFAULT_CATALOG_PATH` wskazuje katalog z repozytorium niezależnie od katalogu roboczego.
- `README.md`: „Szybki start”, sekcja 4.2 (pobranie i rejestracja modeli), 5.2 `.env` przed uruchomieniem bazy, sekcja 6 (polecenia dla dwóch dołączonych dokumentów i dodawanie własnych), sekcja OpenRouter, poprawione drzewo katalogów, wskazówka `--native-tls` dla `uv`.
- `rag_ingestion/README.md` i `Zrodlowyszukiwator/README.md`: usunięte odwołania do nieistniejących plików i lokalna ścieżka. Komunikat w `rag_ingestion/repository.py` nie odsyła już do `psycopg[binary]`.
- Testy: `tests/unit_test/zrodlowyszukiwator/test_cli.py` oraz trzy nowe w `tests/unit_tests/test_release_configuration.py` (dołączone PDF-y istnieją i mają metadane; Modelfile'e ustawiają okno; README nie odwołuje się do nieistniejących plików).

Weryfikacja ścieżki czystego środowiska:
- [FAKT] Polecenia z README 6.1 wykonane na tymczasowej, pustej bazie: tabele utworzone automatycznie, 27 + 36 fragmentów, ponowne uruchomienie pominięte (`identical_content_already_ingested`), wyszukiwanie przez kod aplikacji zwraca kandydatów z obu dokumentów. Baza tymczasowa usunięta; właściwa ma nadal 6 dokumentów i 1385 fragmentów.
- [FAKT] `uv pip install --dry-run -r requirements.txt` na świeżym Pythonie 3.14: 140 pakietów, bez konfliktu. Na tym komputerze `uv` wymaga `--native-tls` (błąd `UnknownIssuer`).
- [FAKT] Świeża instalacja pobrałaby nowsze wersje niż lokalne `.venv` (np. streamlit 1.65.0 wobec 1.64.0, pydantic 2.14.1 wobec 2.13.5, ruff 0.17.0 wobec 0.16.9), bo nie ma pliku blokady (S9, T-20).

Czego nie zweryfikowano i co zostaje:
- Pełnego przebiegu na innym komputerze (T-24) nie było: instalacja zależności tylko jako próba na sucho, pobieranie modeli tylko jako sprawdzenie adresów i skrótów.
- Katalog metadanych nadal zawiera wpisy czterech niedołączonych dokumentów. Adresy źródłowe właściciel uzupełnił później (commit `e96e4d3`, patrz T-16).
- Notebooki ingestion mają zaszytą ścieżkę `C:\Krzysiek\Pismak`; README o tym uprzedza, poprawka należy do T-07.
- `psycopg` zostaje bez `[binary]` — decyzja właściciela z 2026-10-14: wariant `[binary]` sprawiał mu problemy (T-20).
- Szablon PLLuM `{{ .Prompt }}` został przeniesiony do Modelfile bez zmian, żeby odtwarzał testowaną konfigurację (N6).
- Otwarte z pierwotnego zakresu T-19: limit pytań (N1), nieużywane zmienne w `.env_template` (N2).
- `test_release_configuration.py` wymaga, by README zawierało `src\Wyswietlator\app.py` i nie zawierało `requirements-ui.txt` ani `Zrodlowyszukiwacz`.

**T-20 Zależności i licencje (S9). — CZĘŚCIOWO 2026-10-14** D4 rozstrzygnięte.
- Ustal i ustaw faktyczną dolną granicę `openai` albo zastąp import prywatnego API własnym kodem. Dodaj plik blokady.
- ~~Spraw, by import `services` nie ładował PyMuPDF.~~ Po zmianie licencji na AGPL (D4) nie jest to potrzebne ze względu na licencję. Leniwy import może jeszcze skrócić start aplikacji, ale nie blokuje publikacji.
- Opisz licencję PyMuPDF w README (zrobione — sekcja „Licencja”). `psycopg` zostaje bez `[binary]` — patrz niżej.
- Akceptacja: po `import services` w `sys.modules` nie ma `pymupdf` ani `fitz`; świeże środowisko instaluje się i przechodzi testy.

Co zrobiono 2026-10-14 (część „wersje i blokada”, commit `29b40d3`):
- Nowy `requirements.in`: 14 zależności bezpośrednich przypiętych `==` do najnowszych wersji z PyPI z 2026-10-14. Skoki wersji głównych względem `.venv`: openai 1.109.1 → 3.31.0, pandas 2.3.3 → 3.0.6, pytest 8.4.2 → 9.1.1, ipykernel 6.31.0 → 7.4.0. numpy 2.5 wymaga Pythona 3.12+, więc numpy ma dwa piny z markerem: 2.5.4 dla 3.12+, 2.4.6 dla 3.11.
- `requirements.txt` jest teraz plikiem blokady: `uv pip compile --universal --python-version 3.11`, 145 pakietów z dokładnymi wersjami i markerami platform. README 3.2–3.5 i CI instalują z niego bez zmian poleceń.
- Granica `openai` (HIPOTEZA o 1.40) nie jest już potrzebna: wersja jest przypięta, a w 3.31.0 istnieją oba importowane symbole (`openai.lib._parsing.type_to_response_format_param`, `LengthFinishReasonError`) i wszystkie klasy błędów z `services.py`. Import prywatnego API zostaje — przy każdym podniesieniu `openai` sprawdź, czy nadal istnieje.
- `.pre-commit-config.yaml`: ruff `v0.11.13` → `v0.17.0`, równy wersji z blokady (część N4; `ruff format .` z T-22 nie wykonane).
- README 3.1, 3.5, „Struktura UI”, sekcja 9: opis blokady i polecenie jej odtworzenia.
- Lokalne `.venv` zaktualizowane do blokady przez `uv pip install -r requirements.txt` (bez usuwania dodatkowych pakietów).

Weryfikacja:
- [FAKT] Świeże środowiska z blokady na Pythonie 3.11.16 i 3.14.2: 255 testów przechodzi, 18 pominiętych — tyle samo co na starym `.venv`. `ruff check` 0.17.0 czysty. `uv pip check` bez konfliktów.
- [FAKT] Prawdziwe wywołanie `LLMClient.generate_structured` przez `openai 3.31.0` do `pismak-bielik` w Ollamie zwraca poprawny obiekt Pydantic.
- [FAKT] Na bazie właściciela (tylko odczyt): `counts()` = 6 dokumentów, 1450 fragmentów; `vector_search` i `lexical_search` z `active_only=True` zwracają wyniki (psycopg 3.3.6, pgvector 0.5.1, numpy 2.5.4).
- [FAKT] `AppTest` na prawdziwym `app.py` ze Streamlit 1.66.0: pierwszy render bez wyjątków.

Otwarte z T-20:
- Licencja (D4): zrobione 2026-10-14, opis niżej.
- Zależności uruchomieniowe nadal w jednym pliku z notebookami, testami i linterem. `test_release_configuration.py` wymaga `pandas` (zależność notebooków) w `requirements.txt`, więc podział wymaga też zmiany tego testu.
- `psycopg` bez `[binary]` — **rozstrzygnięte 2026-10-14**: decyzja właściciela, bo wariant `[binary]` sprawiał mu problemy (szczegóły nieznane). Nie zmieniaj. Uzasadnienie jest w komentarzu w `requirements.in`; README 3.5 opisuje instalację `libpq`.
- PyMuPDF 1.28.2 ostrzega, że API `fitz` zostanie usunięte; `pdf_extractor.py` i `tests/rag_eval` nadal używają `fitz`. Przypięta wersja działa.
- openai 3.31.0 i streamlit 1.66.0 wydano w dniu przypięcia. Nie było przebiegu pełnej sprawy przez UI (wywiad → RAG → pismo) na nowych wersjach.

Co zrobiono 2026-10-14 (licencja, D4, niezacommitowane):
- Decyzja właściciela: licencja projektu AGPL zamiast MIT. Wybrano `AGPL-3.0-or-later` (standardowe brzmienie zalecane przez FSF). Jeśli właściciel woli `AGPL-3.0-only`, wystarczy zmienić zdanie o wersji w sekcji „Licencja” README.
- [FAKT] Właściciel jest jedynym autorem wszystkich commitów (`git log`), więc zmiana licencji nie wymaga zgody innych osób.
- `LICENSE`: oficjalny tekst z `https://www.gnu.org/licenses/agpl-3.0.txt`, bez zmian (661 linii, SHA-256 `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0`).
- `README.md`: nowa sekcja „Licencja” na końcu — informacja o prawach autorskich, wariant licencji, PyMuPDF, obowiązek z sekcji 13 przy udostępnianiu przez sieć, wyłączenie modeli i dokumentów z `RAW/`.
- Nie dodano nagłówków licencyjnych do plików źródłowych (AGPL ich nie wymaga; FSF je zaleca).
- To nie jest porada prawna.

Otwarte po zmianie licencji:
- Sekcja 13 AGPL: przy publicznym demie interfejs powinien pokazywać link do kodu źródłowego. Adres repozytorium nie jest jeszcze znany (README używa `<URL_REPOZYTORIUM>`). Dodaj link w `app.py` przed upublicznieniem dema.
- Licencje dwóch dołączonych dokumentów miejskich w `RAW/` nie są sprawdzone (pytanie 9 w sekcji 4).

**T-21 Testy (S10).** Zależy od D8.
- Ustal drzewo docelowe już na etapie T-03 i tam umieszczaj wszystkie nowe testy; przeniesienie starych zrób tutaj.
- Dodaj testy `AppTest` dla `app.py` (pierwszy render, za długa wiadomość, wymóg zgody, rozdzielność stanu sesji) oraz testy `catalog.py` i `cli.py`.
- Akceptacja: jedno drzewo; CI zielone na 3.11 i 3.14.

**T-22 Formatowanie (N4).** `ruff format .`; wyrównaj wersję w `.pre-commit-config.yaml`.

**T-23 Porządki (N6, N7, N8, N9, N11).** Martwy kod, `print()` w ścieżce serwera, chunker, osobna rola bazy tylko do odczytu, szablon czatu PLLuM. Każdą pozycję traktuj osobno; żadna nie blokuje publikacji.

### Etap 5 — akceptacja

**T-24** Wykonaj testy z sekcji 8, w tym próbę na czystym środowisku.

---

## 7. Zależności między zadaniami

```text
T-01 ──(wykonane 2026-10-11)
T-02 ──(część lokalna wykonana 2026-10-12; Access w panelu — właściciel, przed udostępnieniem adresu)
T-03 ──► T-04, T-05, T-06, T-08, T-15, T-16, T-19, T-20, T-21
T-05 + T-06 ──(ten sam plik, razem)──► T-04 (nowy przykład powstaje po nich)
T-09 ──► T-12 ──► T-17                   (wszystkie zmieniają app.py)
T-10 ──► T-15                            (serwer.py, services.py)
T-13 ──(nie realizować: wybór dostawcy i modelu jest zamierzony)
T-02 ──► T-14 (opcjonalne)
T-07, T-11, T-16, T-18, T-22, T-23 ──(niezależne)
wszystko ──► T-19 ──► T-04 (przepisanie historii) ──► T-24 ──► upublicznienie
```

---

## 8. Testy akceptacyjne

### Automatyczne — po każdym zadaniu

```cmd
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```

Stan wyjściowy: 92 passed, 1 failed; `ruff check` czysto; 3 pliki do przeformatowania.

### Ręczne — przed udostępnieniem

| Nr | Scenariusz | Oczekiwany wynik |
|---|---|---|
| A1 | Pełny przepływ na Ollamie: opis → pytania → brief → źródła → pismo → pobranie | `letter.md` bez zdublowanych sekcji, źródła z listy `E#`/`U#` |
| A2 | Dwie sesje (okno zwykłe i incognito), dwie różne sprawy | rozmowy, briefy i pisma się nie mieszają; dwa katalogi w `outputs/` |
| A3 | Dwie generacje uruchomione jednocześnie | druga czeka z komunikatem i kończy się poprawnie albo dostaje czytelny komunikat o zajętości |
| A4 | Ollama zatrzymana | komunikat ogólny, bez ścieżek i tracebacku |
| A5 | Kontener bazy zatrzymany | komunikat ogólny, bez hosta i portu |
| A6 | Wiadomość powyżej 5000 znaków | odrzucona z komunikatem |
| A7 | „Max pytań” = 3 | wywiad zadaje do 3 pytań (weryfikuje punkt 6 z sekcji 4) |
| A8 | Po przebiegu A1 na każdym wspieranym modelu | brak nowych `truncating input prompt` w logu Ollamy |
| A9 | `netstat -ano \| findstr :5432` oraz `:11434` | tylko adresy pętli zwrotnej |
| A10 | Wejście na publiczny adres bez logowania i z konta spoza listy | strona Access, odmowa |
| A11 | 50 odświeżeń strony | liczba otwartych plików logów nie rośnie bez ograniczeń |
| A12 | Świeży klon na czystym środowisku, wyłącznie według README | demo działa, RAG zwraca źródła |
| A13 | `git status` po pełnym przebiegu | brak nowych plików śledzonych z `outputs/` i `logs/` |
| A14 | Skan wzorców sekretów na gałęzi do publikacji | 0 trafień |

---

## 9. Ograniczenia dla wdrażającego

1. Nie zmieniaj kontraktów z sekcji 2.3 (pola modeli, identyfikatory `E#`/`U#`, nazwy plików `save_run`, schemat bazy) bez uzasadnienia zapisanego w opisie zmiany. Zmiany addytywne są dopuszczalne.
2. Nie zmieniaj architektury: zostaje Streamlit, lokalna Ollama, lokalny PostgreSQL. Bez FastAPI, Reacta, Redisa, Celery i własnego systemu logowania.
3. Nie dodawaj nowych technologii, jeśli wystarcza biblioteka standardowa albo obecne zależności.
4. Operacje nieodwracalne — usunięcie wolumenu bazy, `git push --force`, przepisanie historii, usuwanie `outputs/` i `logs/` — tylko po wyraźnym potwierdzeniu właściciela.
5. Nie wypisuj wartości sekretów w logach, testach, komunikatach ani dokumentach. Dotyczy `.env`, pliku `C:\ProgramData\cloudflared\token` i nowego hasła bazy.
6. Nie wymyślaj danych: numerów aktów, dat, adresów URL dokumentów, treści licencji. Brakujące dane pochodzą od właściciela (D7).
7. Teksty interfejsu i dokumentacja są po polsku.
8. Konwencja testów z `.github/copilot-instructions.md`: pytest, klasa na funkcję, nazwy `test_<funkcja>__<przypadek>`, krótki docstring w każdym teście.
9. Ruff: długość linii 100, reguły `E`, `F`, `I`, `UP` (`pyproject.toml`).
10. Nie poprawiaj literówek w nazwach publicznych (`drafr_user_prompt`, `PLLLUM_*`, `test_dopytywator_helper_funtcions.py`) przy okazji innych zmian — to osobna, świadoma decyzja.
11. Baza RAG jest odtwarzalna tylko z lokalnych PDF-ów w `src/Zrodlowyszukiwator/RAG/RAW/` (ignorowane, 66 MB). Nie usuwaj ich.

---

## 10. Polecenia użyte do weryfikacji

Wszystkie są tylko do odczytu. Uruchamiaj je z katalogu repozytorium, by sprawdzić, czy stan się zmienił.

```bash
# repozytorium
git status -sb
git ls-files
git for-each-ref
git log origin/main --oneline -- outputs/20260926_200658_ui

# testy i jakość (bez zapisu cache)
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
.venv/Scripts/python.exe -m ruff check . --no-cache
.venv/Scripts/python.exe -m ruff format --check . --no-cache
.venv/Scripts/python.exe -m pip check

# usługi
netstat -ano | grep LISTENING | grep -E ':(5432|11434|8501)\s'
docker ps --format 'table {{.Names}}\t{{.Ports}}\t{{.Status}}'
docker inspect pismak-postgres --format '{{.Created}} {{json .HostConfig.PortBindings}}'
docker compose config
docker compose up -d --dry-run
ollama list
ollama show pismak-bielik
ollama show pismak-pllum --modelfile
grep -ci 'truncating input prompt' "$LOCALAPPDATA/Ollama"/server*.log
.venv/Scripts/python.exe -m streamlit config show
sc query cloudflared
```

Stan bazy sprawdzałem zapytaniami `SELECT` w połączeniu oznaczonym jako tylko do odczytu: liczniki tabel, `pg_roles`, `pg_extension`, `pg_hba_file_rules`, `vector_dims(embedding)`, oraz liczbę trafień `search_vector @@ websearch_to_tsquery('simple', …)`.

---

## 11. Rekomendacja

Stan na 2026-10-14: K1, W1, W2, W4, W6, S1, S2, S5 i S6 są zamknięte, S3 i S8 oznaczone przez właściciela jako naprawione, W3 i W5 częściowo. S4 to działanie zamierzone, nie usterka. Nie upubliczniaj repozytorium, dopóki otwarte jest K2 (przykład z treścią prywatnego maila w `outputs/` i w historii). Nie dodawaj trasy do tunelu i nie udostępniaj adresu dema, dopóki nie ma aplikacji Access (pozostała część W3). Z W5 zostaje brak limitu czasu oczekiwania na blokadę modelu. Pozostałe ustalenia nie blokują publikacji.
