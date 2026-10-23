# Pismak

Pismak pomaga przejść od opisu problemu do projektu pisma do instytucji publicznej.
Dopytuje o sprawę, wyszukuje fragmenty dokumentów miejskich i na ich podstawie
przygotowuje projekt wniosku z listą źródeł.

```text
opis sprawy → Dopytywator → CaseBrief → Źródłowyszukiwacz (RAG) → EvidencePack → Pismogenerator → Pismopoprawiator → letter.md
```

- **Dopytywator** zadaje pytania pomocnicze i buduje `CaseBrief`.
- **Źródłowyszukiwacz** planuje zapytania, łączy wyszukiwanie wektorowe i pełnotekstowe
  (PostgreSQL + pgvector), a reranker LLM wybiera fragmenty do `EvidencePack`.
- **Pismogenerator** pisze projekt wyłącznie na podstawie briefu, znalezionych fragmentów
  i materiałów użytkownika; ostateczny `letter.md` składa kod.
- **Pismopoprawiator** poprawia język pisma (ortografia, interpunkcja, gramatyka, literówki,
  spacje) bez zmiany treści. Model zwraca tylko listę poprawek, a kod sprawdza każdą
  strażnikami i stosuje przyjęte. Przy błędzie zostaje pismo bez korekty, a pipeline biegnie
  dalej. Szczegóły: `src/Pismopoprawiator/README.md`.
- **Wyświetlator** to interfejs Streamlit łączący te etapy.

Modele działają lokalnie w Ollamie (PLLuM lub Bielik oraz `bge-m3` do embeddingów);
każdy etap generatywny może zamiast tego użyć OpenRouter. Projekt jest prototypem
rozwijanym na Windowsie. Planowane: sprawdzanie faktów, wybór adresata i wysyłka.

## 1. Szybki start

1. Zainstaluj narzędzia z rozdziału 2.
2. Pobierz kod i zależności:

   ```cmd
   git clone <URL_REPOZYTORIUM>
   cd Pismak
   uv venv
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. Utwórz konfigurację i uruchom bazę (rozdział 4):

   ```cmd
   copy .env_template .env
   docker compose up -d
   ```

4. Przygotuj modele (rozdział 3): `ollama pull bge-m3` oraz co najmniej jeden model
   generatywny.
5. Wczytaj do bazy dwa dołączone dokumenty (punkt 5.1).
6. Uruchom `run_pismak.bat` i otwórz `http://localhost:8501`.

## 2. Instalacja

Potrzebne są: Git, Python 3.11 lub nowszy, `uv`, Docker Desktop, Ollama dla Windows
(ollama.com) oraz klient PostgreSQL z biblioteką `libpq`.

```cmd
winget install --id=astral-sh.uv -e
winget install -e --id Docker.DockerDesktop
```

- `requirements.txt` to pełna blokada wersji (także zależności pośrednich) dla
  Pythona 3.11+ i wszystkich systemów; instaluj z niego, nie edytuj ręcznie.
- Jeśli `uv` zgłasza `invalid peer certificate: UnknownIssuer` (firmowe proxy,
  antywirus sprawdzający HTTPS), dodaj `--native-tls`.
- `psycopg` korzysta z systemowej `libpq` (bez wariantu `[binary]`). Zainstaluj
  klienta PostgreSQL, dodaj jego katalog `bin` (np. `C:\Program Files\PostgreSQL\17\bin`)
  do `PATH` i uruchom ponownie VS Code/Jupyter.

## 3. Modele (Ollama)

### 3.1. PLLuM i Bielik

Repozytorium zawiera pliki `Modelfile` (szablon rozmowy i okno kontekstu); same modele
pobierz do tych samych katalogów:

| Model w Ollamie | Plik (rozmiar) | Źródło na Hugging Face |
|---|---|---|
| `pismak-pllum` | `modele\pllum\pllum-4b-chat-2512-q4_k_m.gguf` (2,5 GB) | `Jerzman/PLLuM-4B-chat-2512-Q4_K_M-GGUF` |
| `pismak-bielik` | `modele\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf` (2,9 GB) | `second-state/Bielik-4.5B-v3.0-Instruct-GGUF` |

```cmd
curl -L -o modele\pllum\pllum-4b-chat-2512-q4_k_m.gguf https://huggingface.co/Jerzman/PLLuM-4B-chat-2512-Q4_K_M-GGUF/resolve/main/pllum-4b-chat-2512-q4_k_m.gguf
curl -L -o modele\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf https://huggingface.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF/resolve/main/Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf
ollama create pismak-pllum -f modele\pllum\Modelfile
ollama create pismak-bielik -f modele\bielik\Modelfile
```

Wystarczy jeden model; interfejs pokazuje wszystkie modele generatywne z Ollamy. Jeśli
wszystkie etapy mają używać OpenRouter, wystarczy `bge-m3`. Oba pliki GGUF to
kwantyzacje osób trzecich modeli `CYFRAGOVPL/PLLuM-4B-chat-2512` i
`speakleash/Bielik-4.5B-v3.0-Instruct` — sprawdź ich licencje. PLLuM zarejestrowany
dawniej z szablonem `{{ .Prompt }}` zaktualizujesz tym samym `ollama create`.

### 3.2. Embeddingi

```cmd
ollama pull bge-m3
```

### 3.3. Okno kontekstu

Pismak dopasowuje prompt i odpowiedź do okna wybranego modelu: dla Ollamy pyta serwer,
z jakim oknem załadował model; dla OpenRouter korzysta z publicznej listy modeli
(`openrouter/free`: najmniejsze okno wśród darmowych). Gdy danych jest za dużo, skraca
fragmenty źródeł, a zbyt długą rozmowę zgłasza w interfejsie.

Okno modelu lokalnego ustala Ollama, a jej wartość domyślna bywa mała (np. 4096).
Dlatego `Modelfile` ustawia pełne okno: `num_ctx` 8192 dla Bielika i 131 072 dla PLLuM.
Zmiana: popraw tę linię i ponów `ollama create`; bieżące okno pokazuje `ollama ps`.
Większe okno to więcej pamięci i dłuższe odpowiedzi — na CPU pełny prompt potrafi
liczyć się kilkanaście minut, stąd limit 15 minut na żądanie. Opcjonalny górny limit
okna ustawisz w `.env`: `PISMAK_CONTEXT_TOKENS` (Ollama),
`PISMAK_OPENROUTER_CONTEXT_TOKENS` (OpenRouter).

### 3.4. Pamięć

Ollama na CPU trzyma w pamięci do trzech modeli naraz. Ogranicz Ollamę do jednego
modelu naraz i uruchom ją ponownie;

```cmd
setx OLLAMA_MAX_LOADED_MODELS 1
```

## 4. Baza danych

### 4.1. `.env` i role

```cmd
copy .env_template .env
```

Zrób to przed pierwszym uruchomieniem bazy: hasła są zapisywane przy pierwszym starcie
i późniejsza zmiana `.env` ich nie zmieni. Hasła z szablonu są publiczne — zmień je,
zanim komukolwiek udostępnisz aplikację.

| Rola | Hasło w `.env` | Do czego | Adres w `.env` |
|---|---|---|---|
| `pismak` (superuser) | `POSTGRES_PASSWORD` | administracja bazą | — |
| `pismak_ingest` | `PISMAK_INGEST_DB_PASSWORD` | wczytywanie dokumentów, właściciel tabel | `RAG_INGESTION_DATABASE_URL` |
| `pismak_app` | `PISMAK_APP_DB_PASSWORD` | interfejs i wyszukiwanie; tylko odczyt | `DATABASE_URL` |

Zmieniając hasło roli, wpisz je też w jej adres. Zmienne ustawione w systemie mają
pierwszeństwo przed `.env`. `OPENROUTER_KEY` jest potrzebny tylko do OpenRouter.
Nie dodawaj `.env` do repozytorium.

### 4.2. Uruchomienie bazy

```cmd
docker compose up -d
```

Kontener `pismak-postgres` (PostgreSQL 17 + pgvector) nasłuchuje tylko na
`127.0.0.1:5432`. Przy pierwszym starcie, na pustym wolumenie, wykonują się
`docker/postgres/init/001_extensions.sql` (rozszerzenie `vector`) i
`docker/postgres/init/002_roles.sql` (role z punktu 4.1).

Baza utworzona przed rozdzieleniem ról: dopisz do `.env` nowe zmienne z szablonu
(`DATABASE_URL` przestaw na `pismak_app`), odtwórz kontener i uruchom skrypt ról — można
go powtarzać, a istniejące tabele przekaże roli `pismak_ingest`:

```cmd
docker compose up -d
docker exec pismak-postgres psql -U pismak -d pismak -f /docker-entrypoint-initdb.d/002_roles.sql
```

Nie używaj `docker compose down -v` — usuwa wolumen z bazą.

## 5. Dokumenty (RAG)

### 5.1. Dokumenty z repozytorium

Repozytorium zawiera dwa dokumenty: Politykę Rowerową Wrocławia i Wrocławską politykę
mobilności. Przy działającej bazie i `bge-m3` w Ollamie:

```cmd
python -m src.Zrodlowyszukiwator.rag_ingestion.cli --file src\Zrodlowyszukiwator\RAG\RAW\01_Polityka-rowerowa-wroclawia.pdf
python -m src.Zrodlowyszukiwator.rag_ingestion.cli --file src\Zrodlowyszukiwator\RAG\RAW\01_Wroclawska_polityka_mobilnosci.pdf
```

Polecenie łączy się jako `pismak_ingest` (`RAG_INGESTION_DATABASE_URL`; bez tej zmiennej
kończy się błędem), w razie potrzeby tworzy tabele `rag_documents` i `rag_chunks` i
wypisuje liczbę stron oraz fragmentów. To wystarcza do uruchomienia dema.

### 5.2. Własne dokumenty

1. Skopiuj PDF do `src/Zrodlowyszukiwator/RAG/RAW/` (poza dwoma powyższymi pliki z tego
   katalogu nie trafiają do repozytorium).
2. Dopisz metadane do `src/Zrodlowyszukiwator/RAG/document_catalog.json` (kluczem jest
   nazwa pliku; katalog ma już wpisy czterech dokumentów spoza repozytorium).
3. Uruchom polecenie z punktu 5.1 z nazwą swojego pliku; inny katalog wskażesz
   parametrem `--catalog`.

Niezmieniony plik nie jest przetwarzany ponownie — odświeżane są tylko metadane z
katalogu. Zmiana tytułu, typu dokumentu albo miasta, nowsza wersja przetwarzania po
aktualizacji kodu lub `--force` powodują pełne przetworzenie z embeddingami. Po
aktualizacji kodu uruchom polecenie przynajmniej raz. O użyciu dokumentu decydują pola
katalogu: `is_active: false` — dokument nieobowiązujący, nie trafia do pism;
`retrieval_enabled: false` — pomijany w każdym wyszukiwaniu.

## 6. Interfejs

```cmd
run_pismak.bat
```

Skrypt uruchamia Streamlit tylko na `127.0.0.1:8501`. Bez skryptu, z katalogu głównego
repozytorium (stąd wczytywany jest `.streamlit/config.toml`):

```cmd
.venv\Scripts\python.exe -m streamlit run src\Wyswietlator\app.py --server.address 127.0.0.1 --server.port 8501
```

Przeglądarka nie otwiera się sama — wejdź na `http://localhost:8501`. Interfejs wymaga
działającej Ollamy z `bge-m3` i bazy z wczytanymi dokumentami.

- Dla każdego etapu wybierasz dostawcę (Ollama albo OpenRouter) i model oraz limit
  pytań Dopytywatora (domyślnie 1, najwyżej 12).
- Przełącznik „Korekta językowa” (domyślnie włączony) uruchamia Pismopoprawiator po
  Pismogeneratorze. Każde żądanie korekty ma limit 5 minut. Gdy korekta się nie uda, pismo
  zostaje bez niej.
- Po rozmowie widzisz brief, znalezione źródła i projekt pisma do pobrania jako
  `pismo.md`. Po korekcie masz zakładki: wersję poprawioną, wersję bez korekty i
  „Różnice” (zastosowane poprawki, zmiany wiersz po wierszu, odrzucone poprawki z powodem,
  sugestie stylistyczne). Gdy korekta zmieniła pismo, do pobrania są `pismo.md` i
  `pismo_bez_korekty.md`.
- Pismo ma stałą strukturę: adresat, formuła „Jako wieloletni mieszkaniec naszego
  kochanego miasta, wnioskuję o …”, typ `WNIOSEK`, argumentacja z przypisami, podpis do
  uzupełnienia, lista źródeł i sekcja kontrolna z informacjami do weryfikacji. Uwagi dla
  użytkownika stoją w nawiasach kwadratowych: `[DO UZUPEŁNIENIA: …]` i
  `[wymaga weryfikacji]`. Model ma polecenie nie dodawać podstawy prawnej, ale aplikacja
  tego nie sprawdza — przeczytaj pismo przed wysłaniem.
- Ollama obsługuje jedno żądanie naraz: kolejne sprawy czekają z komunikatem, a po
  15 minutach oczekiwania dostają prośbę o ponowienie. Żądania do OpenRouter nie czekają.

### OpenRouter (opcjonalnie)

Każdy etap generatywny może użyć darmowego routera `openrouter/free`. Wpisz klucz jako
`OPENROUTER_KEY` w `.env`, uruchom aplikację ponownie i wybierz dostawcę w interfejsie.
Treść zapytań trafia wtedy do usługi zewnętrznej, więc interfejs wymaga zgody;
embeddingi i baza zostają lokalne. Szczegóły i test połączenia:
`src/Wyswietlator/README.md`.

### Dane, logi i prywatność

- Każda sprawa jest zapisywana w `outputs/<data>_<id>_ui/` (rozmowa, brief, źródła,
  draft, pismo, użyte modele, a po korekcie także `letter_bez_korekty.md` i
  `korekta.json`), a jej log w `logs/pismak_<czas>.log`. Log zawiera każdą poprawkę
  językową, zastosowaną albo odrzuconą, z powodem. Błędy bez
  przypisanej sprawy trafiają do `logs/app_errors.log`.
- Pliki nie są szyfrowane i nie mają terminu usunięcia; ma do nich dostęp każdy z
  dostępem do dysku serwera. Logi mogą zawierać niepoprawne lub ucięte odpowiedzi modelu
  (do 20 000 znaków), czyli dane sprawy. Interfejs wymaga potwierdzenia tej informacji.
  Usuwaj stare sprawy ręcznie, nie wpisuj zbędnych danych osobowych i nie publikuj
  `outputs/` ani `logs/`.
- Przy błędzie interfejs pokazuje ogólny komunikat; szczegóły i ślad stosu są w logu.
- Sprawa żyje tylko w otwartej karcie: odświeżenie strony zaczyna nową (poprzednia
  zostaje w `outputs/`).
- Linki i obrazy z treści modelu są pokazywane jako tekst; pobrane pismo jest
  niezmienione.

## 7. Udostępnianie przez Cloudflare Tunnel

Streamlit nasłuchuje tylko na `127.0.0.1` — nie zmieniaj tego na `0.0.0.0` i nie
przekierowuj portów routera. Tunel ma prowadzić wyłącznie do `http://127.0.0.1:8501`;
nigdy do Ollamy (`11434`), PostgreSQL (`5432`), Docker API ani `outputs/`.

Pismak używa tunelu zarządzanego zdalnie (usługa Windows `cloudflared`, trasy i dostęp w
panelu Cloudflare Zero Trust; potrzebna domena w Cloudflare). Kolejność ma znaczenie —
trasa bez Access jest od razu publiczna:

1. Utwórz tunel w panelu i zainstaluj usługę poleceniem z panelu:
   `cloudflared service install <TOKEN>`.
2. Dodaj aplikację Access typu Self-hosted dla `<NAZWA_HOSTA>` z polityką dla
   wybranych adresów e-mail.
3. Dopiero wtedy dodaj trasę `<NAZWA_HOSTA>` → `http://127.0.0.1:8501`.

Sprawdzenie:

```cmd
curl http://127.0.0.1:20241/config
curl -I https://<NAZWA_HOSTA>
```

Pierwsze polecenie ma pokazać tylko trasę do `http://127.0.0.1:8501` i regułę domyślną
(sama reguła `http_status:503` oznacza, że tunel niczego nie publikuje). Drugie, bez
logowania, ma zwrócić `302` do domeny `cloudflareaccess.com` — każda inna odpowiedź
oznacza brak ochrony: usuń trasę. Sprawdź też logowanie kontem z listy i odmowę dla
konta spoza niej. Quick Tunnel (`cloudflared tunnel --url …`) nie daje ochrony Access —
tylko do krótkich prób bez danych osobowych.

Przed każdym udostępnieniem: baza tylko na `127.0.0.1` (`docker compose config`),
`DATABASE_URL` wskazuje `pismak_app`, działają dwie niezależne sesje, kolejka generacji
i komunikaty przy wyłączonej Ollamie lub bazie, a tunel przechodzi oba powyższe
sprawdzenia. Token tunelu i `.env` nie trafiają do repozytorium.

## 8. Rozwój

```cmd
python -m pytest
python -m ruff check .
python -m ruff format .
```

- GitHub Actions uruchamia testy, `ruff check` i `ruff format --check` na Pythonie 3.11
  i 3.14; `pre-commit` uruchamia Ruff lokalnie przed commitem.
- Testy end-to-end na prawdziwych modelach i bazie uruchamia się tylko ręcznie
  (marker `e2e`: `python -m pytest tests/e2e -m e2e`), zob. `tests/e2e/README.md`.
- Złote zbiory modułów są w `tests/golden_set/<moduł>/` i uruchamia się je tylko ręcznie
  (marker `golden_set`, testy na modelach także `llm`); zwykłe `pytest` i CI je pomijają.
  Pismopoprawiator:
  `python -m pytest tests/golden_set/pismopoprawiator -m golden_set -v -s`
  (zob. `tests/golden_set/README.md`).
- Zależności zmieniasz w `requirements.in`, po czym odtwarzasz blokadę; wersję ruff w
  `.pre-commit-config.yaml` trzymaj równą tej z `requirements.in`:

  ```cmd
  uv pip compile --universal --python-version 3.11 requirements.in -o requirements.txt
  ```

- `jupyter lab`: `src/main.ipynb` pokazuje cały przepływ, notebooki ingestion są w
  `src/Zrodlowyszukiwator/rag_ingestion/notebooks/`. Same znajdują katalog projektu.

Struktura: `src/Dopytywator/`, `src/Zrodlowyszukiwator/` (`rag_ingestion`,
`rag_retrieval`, `brief_to_evidence`, katalog i PDF-y w `RAG/`), `src/Pismogenerator/`,
`src/Pismopoprawiator/`,
`src/Wyswietlator/` (`app.py` — interfejs, `services.py` — adaptery i zapis wyników),
`src/_shared/` (klient LLM, kolejka modelu, budżet kontekstu), `modele/`,
`docker/postgres/init/`, `tests/`.

## Licencja

Copyright (C) 2026 Pismak contributors

Pismak jest wolnym oprogramowaniem na licencji GNU Affero General Public License w
wersji 3 albo (według uznania) dowolnej późniejszej (`AGPL-3.0-or-later`), bez
jakiejkolwiek gwarancji. Pełny tekst: [`LICENSE`](LICENSE). Do odczytu PDF projekt
używa PyMuPDF (AGPL-3.0 albo licencja komercyjna Artifex).

Jeśli udostępniasz zmodyfikowaną wersję przez sieć (np. Streamlit przez tunel), sekcja
13 licencji wymaga, by użytkownicy mogli pobrać jej kod źródłowy, np. przez link do
repozytorium w interfejsie. Licencja nie obejmuje modeli językowych (PLLuM i Bielik mają
własne) ani dokumentów w `src/Zrodlowyszukiwator/RAG/RAW/`, które pozostają na
warunkach wydawców.
