# Audyt przedpublikacyjny Pismaka — raport i plan handoveru

> Aktualizacja 2026-10-11: **K1 naprawione** — szczegóły w `HANDOVER_PRE_PUBLICATION_AUDIT.md`, zadanie T-01. Reszta dokumentu opisuje stan z dnia audytu.

## Kontekst

Pismak (Streamlit + Ollama + PostgreSQL/pgvector, opcjonalnie OpenRouter) ma zostać upubliczniony na GitHubie i udostępniony jako demo przez Cloudflare Tunnel. Zlecenie: zebrać dowody bez zmian w kodzie, przygotować raport i plik `HANDOVER_PRE_PUBLICATION_AUDIT.md` dla modelu wdrażającego poprawki.

Etap dowodowy jest **zakończony** (wyłącznie operacje tylko do odczytu). Audytowany commit: `dab3384` (`main` == `origin/main`, drzewo czyste). Po zatwierdzeniu planu powstanie **jeden** nowy plik — handover. Żaden istniejący plik nie zostanie zmieniony.

## Jak audytowałem (plan audytu — wykonany)

1. Struktura: `git ls-files`, pliki ignorowane, refy, historia, rozmiary blobów.
2. Bezpieczeństwo: skan wzorców sekretów w całej historii i plikach śledzonych (wartości maskowane), `.env` (tylko nazwy kluczy), `.gitignore`, śledzone `outputs/`, notebooki, `logs/` (tylko statystyki zdarzeń).
3. Poprawność: lektura całego `src/` (42 pliki), `pytest` (bez cache), `ruff check`, `ruff format --check`, `pip check`.
4. Konfiguracja: `ollama list/show`, logi serwera Ollamy, `docker inspect`, `docker compose config` i `--dry-run`, zapytania `SELECT` do bazy RAG, `streamlit config show`.
5. Demo: nasłuchujące porty (`netstat`), usługa `cloudflared` (token zamaskowany), profil zapory Windows.

Nie uruchamiałem Streamlita ani generacji LLM, nie instalowałem niczego, nie zmieniałem stanu kontenerów.

## Ustalenia potwierdzone (fakty)

### Krytyczne

- **K1 — PostgreSQL wystawiony na wszystkie interfejsy z publicznym hasłem domyślnym.** Działający kontener publikuje `0.0.0.0:5432` i `[::]:5432` (`docker ps`, `netstat`, `PortBindings HostIp=""`), choć `docker-compose.yaml:13` wymaga `127.0.0.1`. Kontener utworzono 2026-09-20 i nie odtworzono po zmianie compose (`docker compose up -d --dry-run` → `Recreate`). Rola `pismak` jest superuserem, a hasło równa się wartości z `.env_template:34` / `docker-compose.yaml:9`. Zapora Windows jest włączona, sieć Wi‑Fi ma profil Private, reguły zezwalającej po nazwie nie znalazłem — **osiągalności z LAN nie testowałem**, stąd waga warunkowa.

### Wysokie

- **W1 — Test nie przechodzi na wypchniętym HEAD (92 passed, 1 failed).** `tests/unit_tests/pismogenerator/test_pismogenerator.py:134` oczekuje „W imieniu Towarzystwa…”, a `src/Pismogenerator/pismogenerator_helper_functions.py:96` od commita `af587cb` ma „Jako wieloletni mieszkaniec naszego kochango miasta, wnioskuję o” (z literówką). CI z `.github/workflows/ci.yml` będzie czerwone; `README.md:93` i `src/Pismogenerator/prompts.py:30-31` opisują starą formułę.
- **W2 — Każde pismo jest podpisane na sztywno nazwą konkretnej organizacji** (`pismogenerator_helper_functions.py:160`). W publicznym demie dowolny użytkownik wygeneruje pismo sygnowane „Towarzystwo Upiększania Miasta Wrocławia”.
- **W3 — Tunel Cloudflare działa stale, a jego konfiguracji nie da się sprawdzić z repo.** Usługa `cloudflared`: Auto, Running, LocalSystem, `tunnel run --token-file C:\ProgramData\cloudflared\token` (tunel zarządzany zdalnie). `README.md:634-677` opisuje inny wariant (lokalny `config.yml`). Aplikacja nie ma własnego uwierzytelniania ani limitów — cała ochrona zależy od polityki Access w panelu.
- **W4 — Brak budżetu kontekstu; wymagane ustawienie Ollamy jest nieudokumentowane.** Logi Ollamy: 7 zdarzeń `truncating input prompt` (prompty 10 155–11 540 tokenów obcięte do 4 099). Maszyna ma dziś lokalnie `OLLAMA_CONTEXT_LENGTH=262144`, czego nie ma w README ani `.env_template`. `pismak-bielik` ma maksymalny kontekst 8192, czyli mniej niż zaobserwowane prompty, a `max_tokens` wynosi 8192/8200 (`src/Pismogenerator/constants.py:1`, `src/Dopytywator/constants.py:1`). Źródła rozmiaru: reranker 20 × 1800 znaków (`rag_retrieval/reranker.py:228`, `Wyswietlator/services.py:211-229`), pismogenerator 6 × 2400 (`brief_to_evidence/formatting.py:22`), rozmowa do 13 × 5000 znaków (`services.py:48-49`). Obcięcie jest ciche — aplikacja nie zgłasza błędu.
- **W5 — Jedno zawieszone żądanie blokuje wszystkich.** `src/_shared/llm_runtime.py:28` bierze globalną blokadę bez limitu czasu; klient Ollamy (`services.py:103-107`) ma timeout 300 s i domyślne `max_retries=2` SDK (potwierdzone), więc blokada może trwać ok. 15 min. Obejmuje też żądania OpenRouter (`_shared/serwer.py:204`). Brak limitu spraw na sesję.
- **W6 — Czyste środowisko nie uruchomi dema.** PDF-y źródłowe są ignorowane, `RAG/document_catalog.json` ma `source_url: null` dla wszystkich 6 dokumentów, README nie mówi, skąd je wziąć. `rag_ingestion/cli.py:23` wskazuje nieistniejący `examples/document_catalog.json`; `README.md:416` (sekcja `.env`) jest urwana; `rag_ingestion/README.md:69,72,76` i `Zrodlowyszukiwator/README.md:13-14` odwołują się do nieistniejących plików; Modelfile'i `pismak-pllum`/`pismak-bielik` nie ma w repo; OpenRouter nie jest opisany w głównym README.

### Średnie

- **S1 — Szczegóły wewnętrzne w UI.** `Wyswietlator/app.py:568-570,764-766` dopisuje „Miejsce błędu: plik:linia in funkcja”; komunikat z `rag_retrieval/repository.py:41-46` (host:port, polecenia Dockera) trafia do użytkownika przez `services.py:123-124`; brak `.streamlit/config.toml`, więc obowiązuje `client.showErrorDetails="full"`.
- **S2 — Wyciek uchwytów logów.** `services.py:64-82` tworzy plik i `FileHandler` dla każdej sesji (każde otwarcie lub odświeżenie strony), zamykany tylko w `reset_case` (`app.py:148`). Rekordy bez `session_id` (np. `app.py:322`) przepadają.
- **S3 — UI nie informuje o zapisie danych.** Rozmowa, brief i pismo trafiają bezterminowo do `outputs/` (`services.py:252-321`), a odpowiedzi modeli do `logs/` (`serwer.py:31-59`). Ostrzeżenie w UI dotyczy tylko OpenRouter (`app.py:364-373`).
- **S4 — Użytkownik dema wybiera dostawcę i model.** Może przełączyć na OpenRouter (klucz właściciela, `app.py:269-301`), wybrać dowolny model z hosta i 12 pytań. W logach 8 wywołań OpenRouter: 4 udane, 2 błędy, 2 ucięcia.
- **S5 — Retrieval nie odfiltrowuje dokumentów nieobowiązujących.** `app.py:648` nie przekazuje filtrów, `rag_retrieval/models.py:20` ma `active_only=False`; w bazie `wroclaw-strategy-2030` ma `is_active=false` (203 chunki). W `document_catalog.json:36-37` ten dokument ma ten sam numer i datę aktu co Strategia 2050 (`:12-13`).
- **S6 — Ramię pełnotekstowe nie działa dla zapytań wielowyrazowych.** `repository.py:160` używa `websearch_to_tsquery('simple', …)` (AND bez stemmingu). Test na bazie: 3 z 3 zapisanych zapytań plannera → 0 trafień; „bezpieczeństwo pieszych” 11, z dwoma dodatkowymi słowami 0.
- **S7 — Fałszywe alarmy walidatora.** Podciąg `"art."` (`pismogenerator_helper_functions.py:37-51`) łapie „start.”, „kwart.”; `DraftValidationError` nie ma ponowienia, a kolejne kliknięcie uruchamia cały RAG od nowa.
- **S8 — Notebook drukuje `DATABASE_URL`.** `01_rag_ingestion.ipynb` (komórka 6) ma zapisany output z hasłem (dziś równym wartości domyślnej). Wszystkie 3 notebooki mają zapisane outputy i ścieżki lokalne.
- **S9 — Zależności.** Brak lockfile; `serwer.py:8` importuje prywatne API `openai.lib._parsing` przy dolnej granicy `openai>=1.0`; PyMuPDF (AGPL-3.0 lub licencja komercyjna) jest ładowany także przez UI (`services.py:40` → `rag_ingestion/__init__.py:2`), a projekt jest MIT.
- **S10 — Testy.** Dwa równoległe drzewa `tests/unit_test/` (24) i `tests/unit_tests/` (67) z powtórzonymi nazwami klas; `app.py`, `cli.py`, `catalog.py` nie mają testów.

### Niskie

- N1 limit pytań: README 10, `services.py:49` 12, `serwer.py:19` 3. N2 osiem zmiennych z `.env_template` nieczytanych przez kod; `serwer.py:14` ignoruje `OLLAMA_URL`; `app.py:23` `override=True`. N3 YAML tunelu w `README.md:650-653` ma błędne wcięcie; `plans/handover_publikacja_pismak.md` jest nieaktualny. N4 `ruff format --check`: 3 pliki; pre-commit `v0.11.13` wobec ruff 0.16.9. N5 `.gitignore` nie obejmuje `.streamlit/secrets.toml` ani poświadczeń cloudflared; lokalne `refs/agents/*` trzymają 57 MB PDF-ów (nie ma ich na `origin/main`). N6 `pismak-pllum` ma szablon `{{ .Prompt }}`. N7 martwy kod i `print()` z danymi sprawy na stdout. N8 chunker przekracza `max_chars` (w bazie maks. 6363 przy limicie 3500). N9 telemetria Streamlit włączona; F5 gubi sprawę. N10 prywatny e-mail w metadanych commitów. N11 aplikacja łączy się jako superuser.

### Potwierdzone jako poprawne

Brak sekretów w historii i plikach śledzonych; `.env` nigdy nie był commitowany; klucz OpenRouter występuje tylko w `.env`. Ollama słucha wyłącznie na `127.0.0.1:11434`; `run_pismak.bat` wiąże Streamlit do `127.0.0.1`. SQL jest parametryzowany. Źródła w piśmie są ograniczone do dozwolonych identyfikatorów. Stan użytkownika żyje tylko w `st.session_state`; katalogi uruchomień mają UUID. `ruff check` i `pip check` przechodzą. Baza RAG: 6 dokumentów, 1385 chunków, wymiar 1024.

## Nie zweryfikowano

1. Konfiguracja tunelu i polityka Access w panelu Cloudflare; czy tunel jest dziś osiągalny publicznie.
2. Osiągalność portu 5432 z sieci lokalnej i internetu.
3. Stan CI na GitHubie i przebieg na Linuksie / Pythonie 3.11.
4. Przepływ end-to-end, dwie sesje równolegle, kolejka, zachowanie przy wyłączonej Ollamie lub bazie.
5. Instalacja na czystym środowisku.
6. Czy „Max pytań” (`app.py:304-313`, bez `key`) zachowuje wartość po zablokowaniu.
7. Planner i reranker przez OpenRouter (`strict: True` z nieścisłym schematem) — w logach RAG biegł tylko na Ollamie.
8. Pamięć i czas PLLuM przy `n_ctx=131072`; wpływ szablonu `{{ .Prompt }}` na jakość.
9. Zgody: autor maila z przykładu, organizacja z podpisu; licencje modeli i dokumentów miejskich.
10. Limity klucza OpenRouter.

## Kolejność napraw

| Etap | Zakres | Zależności |
|---|---|---|
| 0. Operacyjny (właściciel, bez zmian w repo) | K1: odtworzyć kontener, zmienić hasło przez `ALTER ROLE` (zmiana `.env` nie zmienia hasła w istniejącym wolumenie); W3: sprawdzić panel Cloudflare; decyzje do K2, W1, W2 | blokuje wszystko |
| 1. Blokery publikacji | K2 (przykład syntetyczny + usunięcie z historii), W1, W2, S8 | decyzje z etapu 0; przepisanie historii wymaga potwierdzenia force-push |
| 2. Utwardzenie dema | S1, W5, S2, S3, S4, `.streamlit/config.toml` | S4 zależy od wyboru modeli w W4 |
| 3. Poprawność | W4, S5, S7, S6 | W4 przed S4 |
| 4. Publikacja | W6, S9, S10, N1–N11 | README na końcu, bo opisuje stan po poprawkach |
| 5. Akceptacja | testy automatyczne, E2E, próba czystej instalacji | wszystkie powyższe |

## Plan testów

- Automatyczne po każdym etapie: `python -m pytest -q`, `ruff check .`, `ruff format --check .`.
- Nowe testy jednostkowe do każdej poprawki w kodzie (konwencja z `.github/copilot-instructions.md`).
- Ręczne E2E: pełny przepływ na Ollamie; dwie sesje (zwykła i incognito) bez mieszania danych; dwie generacje naraz; Ollama wyłączona; baza wyłączona; wiadomość ponad 5000 znaków; brak ścieżek i tracebacków w komunikatach; odmowa Access dla obcego konta; `netstat` pokazuje 5432 tylko na `127.0.0.1`; świeży klon według README.

## Rekomendacja

**Projekt nie jest gotowy** ani do upublicznienia repozytorium, ani do publicznego dema. Blokują to K1, K2, W1, W2 i W3; W4–W6 decydują o tym, czy demo zadziała wiarygodnie u testerów i na czystym środowisku. Baza kodu jest zdrowa (brak sekretów w historii, poprawna izolacja sesji, parametryzowany SQL), a etapy 0–1 są małe.

## Co zrobię po zatwierdzeniu

1. Utworzę `plans/HANDOVER_PRE_PUBLICATION_AUDIT.md` (zgodnie z decyzją właściciela: katalog `plans/`, plik nieśledzony, bez commita). Zawartość:
   - status, data, audytowany commit, legenda fakt / hipoteza;
   - pliki do przeczytania przed startem, w kolejności;
   - architektura: komponenty, przepływ danych, kontrakty (`CaseBrief`, `ResearchPlan`, `EvidencePack`, `DraftLetter`, identyfikatory `E#`/`U#`, pliki `save_run`, schemat `rag_documents`/`rag_chunks`), stan globalny, topologia uruchomieniowa;
   - ustalenia potwierdzone z dowodem i sposobem weryfikacji; osobno hipotezy i rzeczy niezweryfikowane;
   - zadania naprawcze `T-01…` (pliki, zakres, zależności, test akceptacyjny, oznaczenie „wymaga decyzji właściciela” lub „operacyjne”);
   - kolejność i graf zależności, testy akceptacyjne, polecenia weryfikacyjne;
   - ograniczenia: bez zmian kontraktów danych i architektury bez uzasadnienia, bez nowych technologii, bez operacji nieodwracalnych (wolumen bazy, force-push) bez potwierdzenia, bez ujawniania sekretów.
2. Przekażę raport końcowy w rozmowie.

Uwaga: handover i ten plan opisują stan zabezpieczeń tego komputera, więc nie powinny trafić do publicznego repozytorium. Katalog `plans/` jest śledzony przez gita, a oba pliki nie są ignorowane — `git add .` lub `git add plans` dodałoby je do commita. Nie zmieniam `.gitignore` (zakaz zmian w istniejących plikach); zadanie T-08 w handoverze obejmuje dopisanie reguły.

## Weryfikacja

- `git status --short` pokazuje wyłącznie dwa nowe pliki w `plans/`: ten plan i `HANDOVER_PRE_PUBLICATION_AUDIT.md`.
- `git diff --stat` jest pusty.
- Skan pliku handoveru wzorcami sekretów (`sk-or-v1-`, hasło z `.env`, zawartość tokenu) nie zwraca trafień poza publiczną wartością domyślną z `.env_template`, przywołaną z nazwy pliku, bez wartości.
- Każde ustalenie w handoverze ma wskazany plik lub polecenie, którym zostało potwierdzone.
