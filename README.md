# Pismak

Pismak jest pomocnikiem aktywisty: aplikacją AI do przygotowywania pism i
wniosków do instytucji publicznych.

Sprawa użytkownika przechodzi przez następujące etapy:

1. użytkownik opisuje problem,
2. Dopytywator zadaje pytania i doprecyzowuje intencje,
3. Źródłowyszukiwacz wyszukuje odpowiednie fragmenty dokumentów,
4. Pismogenerator tworzy projekt pisma na podstawie briefu i dowodów.

Projekt jest obecnie rozwijany lokalnie na Windowsie z użyciem modeli uruchamianych przez Ollamę oraz PostgreSQL + pgvector jako bazy dla RAG.

---

## 1. Cel projektu

Celem Pismaka jest zbudowanie narzędzia typu chat, które pomaga przejść od nieuporządkowanego opisu problemu do gotowego, możliwego do wysłania pisma.

Docelowy przepływ:

```text
Użytkownik
   ↓
Dopytywator
   ↓
CaseBrief
   ↓
Źródłowyszukiwacz
   ↓
EvidencePack
   ↓
Pismogenerator
   ↓
DraftLetter / letter.md
   ↓
weryfikacja, korekta i wysyłka
```

Najważniejsze założenia:

- system nie powinien wymyślać faktów ani źródeł,
- każdy etap ma mieć jasno zdefiniowane wejście i wyjście,
- docelowo lokalne modele mają być wymienne z modelami chmurowymi,
- źródła użyte w piśmie mają być możliwe do wskazania i zweryfikowania,
- RAG ma korzystać z miejskich strategii, polityk, programów, standardów i innych dokumentów,
- generowanie pisma ma być oddzielone od wyszukiwania i weryfikacji źródeł.

---

## 2. Aktualnie zaimplementowane komponenty

### Dopytywator

Dopytywator prowadzi rozmowę z użytkownikiem i zbiera informacje potrzebne do opisania sprawy.

Jego zadania:

- zrozumienie problemu,
- ustalenie oczekiwanego rezultatu,
- ustalenie konkretnych postulatów,
- zebranie informacji o miejscu i czasie,
- zebranie informacji o dowodach,
- zebranie informacji o wcześniejszych działaniach,
- rozpoznanie brakujących informacji,
- oddzielenie twierdzeń, opinii i przypuszczeń.

Wynikiem jest obiekt:

```python
CaseBrief
```

Dopytywator zadaje maksymalnie 10 pytań pomocniczych.

### Pismogenerator

Pismogenerator otrzymuje zaakceptowany `CaseBrief` oraz `EvidencePack` i generuje
projekt pisma. `EvidencePack` zawiera dokładne fragmenty źródeł znalezione przez
Źródłowyszukiwacz; generator nie ma bezpośredniego dostępu do bazy RAG.

W polu `DraftLetter.sources` zapisywane są wyłącznie identyfikatory dozwolonych
materiałów: `E1`, `E2` itd. dla EvidencePack oraz `U1`, `U2` itd. dla materiałów
podanych przez użytkownika. Renderer rozwija te identyfikatory do czytelnej listy
źródeł w `letter.md`.

W Phase 0 obowiązuje stała struktura:

```text
1. Adresat
2. Nadawca
3. Podsumowanie: „Jako wieloletni mieszkaniec naszego kochanego miasta, wnioskuję o ...”
4. Typ pisma: WNIOSEK
5. Rozwinięcie i argumentacja
6. Podpis
7. Źródła
8. Sekcja kontrolna
```

W Phase 0:

- typ pisma jest zawsze `WNIOSEK`,
- Pismogenerator nie wybiera podstawy prawnej,
- nie dodaje art. 241 KPA ani ustawy o dostępie do informacji publicznej,
- nie może wymyślać faktów ani źródeł,
- finalny Markdown jest składany deterministycznie przez kod.

Wynikiem jest:

```python
DraftLetter
```

oraz finalny plik:

```text
letter.md
```

### Źródłowyszukiwacz — RAG

RAG jest podzielony na dwie części.

#### `rag_ingestion`

Część uruchamiana przy dodawaniu lub aktualizowaniu dokumentu.

Pipeline:

```text
PDF
 ↓
ekstrakcja tekstu
 ↓
czyszczenie
 ↓
wykrywanie sekcji
 ↓
chunking
 ↓
metadata
 ↓
embedding
 ↓
PostgreSQL + pgvector
```

Każdy chunk zachowuje m.in.:

- `document_id`,
- tytuł dokumentu,
- rodzaj dokumentu,
- rangę w hierarchii,
- formę prawną,
- numer strony,
- ścieżkę sekcji,
- tekst,
- hash treści,
- embedding,
- metadata JSON.

Hierarchia merytoryczna dokumentów:

```text
Strategia             100
Polityka                80
Program / Plan          70
Standard                60
Procedura / instrukcja  50
Inne                    40
```

Forma prawna jest przechowywana oddzielnie, np.:

```text
uchwała
zarządzenie
załącznik
raport
dokument samodzielny
```

Dzięki temu np. standard może być dokumentem typu `standard`, ale jednocześnie zostać wprowadzony zarządzeniem.

#### `rag_retrieval` / `brief_to_evidence`

Ta część działa jako etap wykonywany przy każdej sprawie:

```text
CaseBrief
 ↓
ResearchPlan
 ↓
hybrid search
 ↓
reranking
 ↓
EvidencePack
```

Wyszukiwanie obejmuje:

- semantic / vector search,
- full-text / keyword search,
- filtrowanie po metadata,
- RRF dla połączenia wyników,
- opcjonalny reranking LLM,
- wybór ograniczonej liczby fragmentów do `EvidencePack`.

Każdy element `EvidencePack` zawiera m.in. dokładny fragment tekstu, identyfikator
dokumentu i chunka, tytuł dokumentu, strony, ścieżkę sekcji, uzasadnienie trafności
oraz opcjonalny URL. Pismogenerator otrzymuje gotowy pakiet dowodów i nie łączy się
bezpośrednio z bazą RAG.

### Lokalne modele

Modele są uruchamiane przez Ollamę.

Planowane / używane role:

```text
PLLuM / Bielik
→ generowanie i analiza tekstu

BGE-M3
→ embeddings dla RAG
```

Kod komunikuje się z modelami przez HTTP API, dzięki czemu później można zastąpić lokalny model modelem chmurowym bez przebudowy całej aplikacji.

---

## 3. Konfiguracja projektu (Windows)

### Szybki start

Kolejność kroków na czystym komputerze; szczegóły każdego są w dalszych punktach.

1. Zainstaluj narzędzia z punktu 3.1.
2. Pobierz kod i zależności (3.2–3.5):

   ```cmd
   git clone <URL_REPOZYTORIUM>
   cd Pismak
   uv venv
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. Utwórz konfigurację i uruchom bazę (5.2–5.3):

   ```cmd
   copy .env_template .env
   docker compose up -d
   ```

4. Przygotuj modele (4.2–4.3): `ollama pull bge-m3`, a potem pobierz i zarejestruj
   co najmniej jeden model generatywny.
5. Wczytaj do bazy dwa dokumenty dołączone do repozytorium (6.1).
6. Uruchom interfejs: `run_pismak.bat` i otwórz `http://localhost:8501`.

### 3.1. Wymagania

Potrzebne są:

- Git,
- Python 3.11 lub nowszy,
- `uv`,
- Docker Desktop,
- Ollama,
- klient PostgreSQL z biblioteką `libpq` (patrz 3.5).

Zależności Pythona są zapisane w `requirements.txt`.

### 3.2. Sklonuj repozytorium

```cmd
git clone <URL_REPOZYTORIUM>
cd Pismak
```

### 3.3. Zainstaluj `uv`

```cmd
winget install --id=astral-sh.uv -e
```

Po instalacji zamknij terminal i otwórz go ponownie.


### 3.4. Utwórz środowisko Python

W katalogu projektu:

```cmd
uv venv
```

Powstanie:

```text
.venv/
```

Aktywacja w CMD:

```cmd
.venv\Scripts\activate
```

Po aktywacji:

```text
(.venv) C:\...\Pismak>
```

### 3.5. Zainstaluj zależności

```cmd
uv pip install -r requirements.txt
```

Jeśli `uv` zgłasza błąd certyfikatu (`invalid peer certificate: UnknownIssuer`),
co zdarza się za firmowym proxy lub przy programie antywirusowym sprawdzającym
ruch HTTPS, dodaj do polecenia `--native-tls`.

`psycopg` używa systemowej biblioteki klienta PostgreSQL `libpq` (bez wariantu
`[binary]`). Na Windowsie zainstaluj klienta PostgreSQL i dodaj jego katalog `bin`
(np. `C:\Program Files\PostgreSQL\17\bin`) do zmiennej `PATH`. Następnie uruchom
ponownie VS Code/Jupyter, aby kernel zobaczył `libpq.dll`.

Testy uruchom:

```cmd
python -m pytest
```


## 4. Ollama i modele lokalne

### 4.1. Zainstaluj Ollamę

Pobierz i zainstaluj Ollamę dla Windows.

Po instalacji sprawdź:

```cmd
ollama --version
```

Lista modeli:

```cmd
ollama list
```

### 4.2. PLLuM i Bielik

Pismak był rozwijany z dwoma modelami generatywnymi w formacie GGUF. Repozytorium
zawiera ich pliki `Modelfile` (szablon rozmowy i okno kontekstu); same modele
pobierz do tych samych katalogów:

| Model w Ollamie | Plik (rozmiar) | Repozytorium na Hugging Face |
|---|---|---|
| `pismak-pllum` | `modele\pllum\pllum-4b-chat-2512-q4_k_m.gguf` (2,5 GB) | `Jerzman/PLLuM-4B-chat-2512-Q4_K_M-GGUF` |
| `pismak-bielik` | `modele\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf` (2,9 GB) | `second-state/Bielik-4.5B-v3.0-Instruct-GGUF` |

```cmd
curl -L -o modele\pllum\pllum-4b-chat-2512-q4_k_m.gguf https://huggingface.co/Jerzman/PLLuM-4B-chat-2512-Q4_K_M-GGUF/resolve/main/pllum-4b-chat-2512-q4_k_m.gguf
curl -L -o modele\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf https://huggingface.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF/resolve/main/Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf
```

Następnie zarejestruj modele w Ollamie:

```cmd
ollama create pismak-pllum -f modele\pllum\Modelfile
ollama create pismak-bielik -f modele\bielik\Modelfile
```

Oba pliki to kwantyzacje przygotowane przez osoby trzecie na podstawie modeli
`CYFRAGOVPL/PLLuM-4B-chat-2512` i `speakleash/Bielik-4.5B-v3.0-Instruct`. Przed
użyciem sprawdź ich licencje na stronach modeli.

Wystarczy jeden z nich. Nazwy też nie są wymagane: interfejs pokazuje wszystkie
modele generatywne dostępne w Ollamie. Modeli lokalnych nie potrzebujesz wcale,
jeśli wszystkie etapy mają korzystać z OpenRouter (punkt 8) — wtedy wystarczy
`bge-m3` z punktu 4.3.

### 4.3. Model embeddingowy

Dla RAG używany jest lokalny model embeddingowy:

```cmd
ollama pull bge-m3
```

Sprawdź:

```cmd
ollama list
```

### 4.4. Okno kontekstu

Pismak mieści prompt i odpowiedź modelu w oknie kontekstu tego modelu, który
wybrano w interfejsie. Rozmiar okna ustala sam, osobno dla każdego etapu:

- **Ollama:** pyta serwer, z jakim oknem faktycznie załadował model. Przy
  pierwszym użyciu modelu najpierw go ładuje, więc pierwsze zapytanie zaczyna się
  z kilkusekundowym opóźnieniem.
- **OpenRouter:** pobiera publiczną listę modeli. Dla routera `openrouter/free`
  przyjmuje najmniejsze okno wśród darmowych modeli, bo router sam wybiera model.

Gdy danych jest więcej, niż mieści okno, aplikacja skraca fragmenty źródeł
wysyłane do modelu i obniża limit długości odpowiedzi, a zbyt długą rozmowę
zgłasza w interfejsie. Model z większym oknem dostaje więc więcej źródeł.

O tym, jakie okno dostaje model lokalny, decyduje Ollama, nie Pismak. Jej wartość
domyślna bywa dużo mniejsza niż maksimum modelu (np. 4096). Pliki `Modelfile` z
repozytorium ustawiają dlatego pełne okno obu modeli:

```text
PARAMETER num_ctx 8192      (modele\bielik\Modelfile)
PARAMETER num_ctx 131072    (modele\pllum\Modelfile)
```

Żeby zmienić okno, popraw tę linię i ponownie wykonaj `ollama create`. Dla modeli
bez własnego ustawienia okno wyznacza zmienna środowiskowa
`OLLAMA_CONTEXT_LENGTH` (wymaga ponownego uruchomienia Ollamy). Bieżącą wartość
pokazuje kolumna `CONTEXT`:

```cmd
ollama ps
```

Większe okno zwiększa zużycie pamięci i wydłuża odpowiedź. Na komputerze bez GPU
pełny prompt potrafi być przetwarzany kilkanaście minut, dlatego żądanie do
modelu lokalnego ma limit 15 minut.

Jeśli chcesz ograniczyć okno mimo większych możliwości modelu, ustaw w `.env`
górny limit: `PISMAK_CONTEXT_TOKENS` dla Ollamy albo
`PISMAK_OPENROUTER_CONTEXT_TOKENS` dla OpenRouter. Domyślnie limitu nie ma.

---

## 5. Docker i baza danych

Pismak używa PostgreSQL + pgvector.

Na obecnym etapie:

```text
Python / src / Jupyter → .venv
Ollama                  → Windows
PostgreSQL + pgvector   → Docker
```

### 5.1. Zainstaluj Docker Desktop

```cmd
winget install -e --id Docker.DockerDesktop
```

Po instalacji uruchom Docker Desktop.

Sprawdź:

```cmd
docker --version
```

oraz:

```cmd
docker compose version
```

### 5.2. `.env`

Utwórz lokalny plik konfiguracyjny na podstawie szablonu:

```cmd
copy .env_template .env
```

Wartości z szablonu wystarczą do pracy lokalnej. Zrób to przed pierwszym
uruchomieniem bazy: `POSTGRES_PASSWORD` jest zapisywane w bazie przy jej
pierwszym starcie, a późniejsza zmiana samego pliku `.env` już go nie zmieni.
Hasło z szablonu jest publiczne, więc zmień je, zanim udostępnisz komukolwiek
aplikację. Nowe hasło wpisz także w `DATABASE_URL`.

`OPENROUTER_KEY` uzupełnij tylko wtedy, gdy chcesz korzystać z OpenRouter.
Pliku `.env` nie dodawaj do repozytorium.

### 5.3. `docker-compose.yaml`

Konfiguracja znajduje się już w repozytorium. Uruchom bazę poleceniem:

```cmd
docker compose up -d
```

Sprawdź stan kontenera:

```cmd
docker ps
```

Rozszerzenie `pgvector` jest włączane przez skrypt:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

zapisany w `docker/postgres/init/001_extensions.sql`.

### 5.4. Uruchom bazę

Powinien działać kontener:

```text
pismak-postgres
```

### 5.5. Sprawdź PostgreSQL

```cmd
docker exec -it pismak-postgres psql -U pismak -d pismak
```

Następnie:

```sql
SELECT version();
```

Sprawdź pgvector:

```sql
SELECT extname, extversion
FROM pg_extension
WHERE extname = 'vector';
```

Wyjście:

```text
\q
```

### 5.6. Utwórz schema RAG

Polecenie z punktu 6.1 tworzy tabele samo. Poniższy przykład jest potrzebny tylko
wtedy, gdy chcesz utworzyć je osobno, np. z notebooka.

Kod `rag_ingestion` zawiera:

```python
PgVectorRepository
```

Przykład:

```python
import os

from dotenv import load_dotenv

from src.Zrodlowyszukiwator.rag_ingestion.repository import (
    PgVectorRepository,
)

load_dotenv()

repo = PgVectorRepository(
    os.environ["DATABASE_URL"],
    embedding_dim=1024,
)

repo.ensure_schema()
```

Tworzone są m.in.:

```text
rag_documents
rag_chunks
```

---

## 6. Dokumenty w RAG

```text
PDF
 ↓
rag_ingestion
 ↓
PostgreSQL / pgvector
```

### 6.1. Dokumenty dołączone do repozytorium

W katalogu `src/Zrodlowyszukiwator/RAG/RAW/` są dwa dokumenty źródłowe:

- `01_Polityka-rowerowa-wroclawia.pdf` — Polityka Rowerowa Wrocławia,
- `01_Wroclawska_polityka_mobilnosci.pdf` — Wrocławska polityka mobilności.

Wczytaj je do bazy. Baza musi działać, a Ollama mieć model `bge-m3`:

```cmd
python -m src.Zrodlowyszukiwator.rag_ingestion.cli --file src\Zrodlowyszukiwator\RAG\RAW\01_Polityka-rowerowa-wroclawia.pdf
python -m src.Zrodlowyszukiwator.rag_ingestion.cli --file src\Zrodlowyszukiwator\RAG\RAW\01_Wroclawska_polityka_mobilnosci.pdf
```

Polecenie tworzy tabele `rag_documents` i `rag_chunks`, jeśli ich jeszcze nie ma,
i wypisuje liczbę stron oraz fragmentów. To wystarcza do uruchomienia dema.

### 6.2. Własne dokumenty

Metadane dokumentów są w `src/Zrodlowyszukiwator/RAG/document_catalog.json`;
kluczem wpisu jest nazwa pliku PDF. Katalog zawiera też wpisy czterech dokumentów,
których plików nie ma w repozytorium — zadziałają dopiero wtedy, gdy umieścisz
PDF o tej samej nazwie w katalogu `RAW/`.

Żeby dodać dokument:

1. Skopiuj PDF do `src/Zrodlowyszukiwator/RAG/RAW/` (pliki w tym katalogu poza
   dwoma wymienionymi wyżej nie trafiają do repozytorium).
2. Dopisz jego metadane do katalogu.
3. Uruchom to samo polecenie z `--file`. Inny katalog metadanych wskażesz
   parametrem `--catalog`.

Gdy SHA-256 pliku się nie zmienił, ponowne uruchomienie nie dzieli go i nie liczy
embeddingów od nowa, ale zapisuje w bazie bieżące metadane z katalogu. Poprawkę
statusu, numeru aktu czy adresu wystarczy więc wprowadzić w katalogu i uruchomić
polecenie jeszcze raz. Zmiana tytułu, typu dokumentu albo miasta powoduje pełne
przetworzenie, bo te pola są częścią embeddingów. Dokument zapisany starszą wersją
przetwarzania tekstu jest przetwarzany od nowa automatycznie; `--force` wymusza
pełne przetworzenie zawsze.

O tym, czy dokument trafia do pism, decydują dwa pola katalogu:

- `is_active: false` — dokument nieobowiązujący (np. zastąpiony nowszym). Zostaje w
  bazie, ale nie jest używany jako źródło pisma. `lifecycle_status: "superseded"`
  wymaga tej wartości.
- `retrieval_enabled: false` — dokument jest pomijany przez każde wyszukiwanie.

Po aktualizacji kodu uruchom polecenie z `--file` przynajmniej raz: dodaje ono do
istniejącej bazy kolumny, z których korzysta wyszukiwanie.

---

## 7. Notebooki i uruchomienie pipeline'u

Na obecnym etapie projekt jest rozwijany eksperymentalnie przez Jupyter.

Uruchom:

```cmd
jupyter lab
```

Główny notebook demonstracyjny znajduje się w `src/main.ipynb`. Pokazuje przepływ:

```text
Dopytywator → CaseBrief → EvidencePack → DraftLetter → letter.md
```

Notebooki do ingestion znajdują się w
`src/Zrodlowyszukiwator/rag_ingestion/notebooks/`. Mają na początku zmienną
`PROJECT_ROOT` ze ścieżką z komputera autora — przed uruchomieniem ustaw ją na
katalog swojego klonu. Do zwykłego wczytania dokumentów wystarczy polecenie z
punktu 6.1.

---

## 8. Wyświetlator — interfejs Streamlit

Wyświetlator jest prostym interfejsem POC napisanym w Pythonie i Streamlit.
Nie dodaje osobnego backendu ani własnej logiki domenowej; korzysta z istniejących
modułów Dopytywatora, Źródłowyszukiwatora i Pismogeneratora.

### Uruchomienie

Z katalogu głównego repozytorium:

```cmd
streamlit run src\Wyswietlator\app.py
```

Interfejs powinien być dostępny pod adresem `http://localhost:8501`. Przeglądarka
nie otwiera się sama (`server.headless` w `.streamlit/config.toml`) — wpisz adres
ręcznie.

W UI można:

- wybrać dla każdego etapu (Dopytywator, Źródłowyszukiwator, Pismogenerator)
  dostawcę — lokalną Ollamę albo OpenRouter — i model,
- ustawić maksymalną liczbę pytań,
- przeprowadzić rozmowę w chacie,
- zobaczyć `CaseBrief` i znalezione dowody,
- wygenerować oraz pobrać `letter.md`.

Przepływ UI:

```text
chat → CaseBrief → EvidencePack → DraftLetter → letter.md
```

Wyniki są zapisywane lokalnie w katalogu `outputs/YYYYMMDD_HHMMSS_<id>_ui/`:

```text
conversation.json
brief.json
evidence.json
draft.json
letter.md
```

Logi aplikacji są zapisywane osobno dla każdej sprawy w katalogu `logs/`,
pod nazwami `pismak_<timestamp>.log`. Po wybraniu „Nowa sprawa” powstaje
nowy plik, a logi równoległych sesji nie są mieszane.
Jeśli odpowiedź modelu jest niepoprawna albo ucięta, log zawiera jej treść
(maksymalnie 20 000 znaków, zapisaną jako JSON-escaped tekst). Może ona zawierać
dane sprawy, dlatego logi należy chronić tak samo jak pozostałe dane lokalne.

W razie błędu interfejs pokazuje tylko ogólny komunikat. Typ błędu, miejsce w
kodzie, treść i ślad stosu zapisuje log sprawy (zdarzenia `interview_failed`,
`pipeline_failed`, `model_list_failed`). Błędy, których aplikacja nie obsłużyła,
trafiają do wspólnego pliku `logs/app_errors.log`, a przeglądarka pokazuje wtedy
ogólny komunikat Streamlita. Odpowiada za to `.streamlit/config.toml`, wczytywany
z katalogu, z którego startuje aplikacja — uruchamiaj ją więc z katalogu głównego
repozytorium.

Do `logs/app_errors.log` trafiają też wpisy, których nie da się przypisać do
żadnej sprawy. Aplikacja nie trzyma plików logów otwartych, więc można je usuwać
lub przenosić także w trakcie jej działania.

Wyświetlator wymaga działającej Ollamy, PostgreSQL z tabelami `rag_documents` i
`rag_chunks` oraz zainstalowanego `bge-m3`. Przy pustej bazie RAG najpierw trzeba
wykonać ingestion dokumentów (punkt 6.1).

### OpenRouter (opcjonalnie)

Każdy etap generatywny może zamiast lokalnego modelu użyć darmowego routera
OpenRouter (`openrouter/free`). Wpisz własny klucz jako `OPENROUTER_KEY` w `.env`,
uruchom aplikację ponownie i wybierz dostawcę w interfejsie.

Treść zapytań trafia wtedy do usługi zewnętrznej, dlatego interfejs wymaga
zaznaczenia zgody przed wysłaniem. Embeddingi (`bge-m3`) i baza dokumentów
zostają lokalne. Szczegóły i test połączenia opisuje
`src/Wyswietlator/README.md`.


### Retencja i usuwanie danych

Pismak zapisuje na komputerze, na którym działa Streamlit, pełną rozmowę oraz
brief, znalezione dowody, draft i wyrenderowane pismo. Dane pozostają w katalogu
danego uruchomienia pod `outputs/` jako niezaszyfrowane pliki, bez
automatycznego terminu usunięcia. Logi w `logs/` również nie są szyfrowane;
niepoprawne lub ucięte odpowiedzi modelu mogą być w nich zapisane do 20 000
znaków. Osoby z dostępem do systemu plików serwera, w tym jego administrator,
mogą odczytać te dane. UI wymaga potwierdzenia zapoznania się z tą informacją
przed przetwarzaniem sprawy. Nie ma funkcji kasowania spraw; operator usuwa
katalog uruchomienia i odpowiadający mu log ręcznie po zakończeniu pracy lub
testów. Przy zdalnym udostępnieniu aplikacji dane nadal są przechowywane na
hoście właściciela. Nie umieszczaj w sprawach danych osobowych, które nie są
konieczne, i nie publikuj zawartości `outputs/` ani `logs/`.

### Struktura UI

- `src/Wyswietlator/app.py` — komponenty Streamlit, stan sesji, chat i widoki;
- `src/Wyswietlator/services.py` — cienkie adaptery infrastruktury, konfiguracja
   RAG, renderer i zapis wyników;
- zależności interfejsu są zapisane w głównym `requirements.txt`.

### Lokalny serwer i publikacja POC

PostgreSQL jest publikowany wyłącznie na loopback (`127.0.0.1:5432`). Po zmianie
konfiguracji odtwórz kontener i sprawdź wynik:

```cmd
docker compose up -d
docker compose config
```

Ollama powinna nadal używać `OLLAMA_URL=http://127.0.0.1:11434`. Uruchom lokalnie
Streamlit z ograniczonym adresem nasłuchu:

```cmd
run_pismak.bat
```

Albo bez skryptu:

```cmd
.venv\Scripts\python.exe -m streamlit run src\Wyswietlator\app.py --server.address 127.0.0.1 --server.port 8501
```

Nie zmieniaj `--server.address` na `0.0.0.0` i nie przekierowuj portów routera.
Tunel Cloudflare ma kierować wyłącznie do `http://127.0.0.1:8501`; nie twórz
publicznych tras dla portów Ollamy (`11434`), PostgreSQL (`5432`), Docker API ani
katalogu `outputs/`.

#### Stały Cloudflare Tunnel

Pismak korzysta z tunelu zarządzanego zdalnie: `cloudflared` działa jako usługa
Windows i uwierzytelnia się tokenem, a trasy i zasady dostępu ustawia się w panelu
Cloudflare Zero Trust, nie w pliku na dysku. Wymagana jest domena zarządzana przez
Cloudflare. W przykładach `<NAZWA_HOSTA>` oznacza adres w tej domenie, np.
`pismak.example.com`.

Kolejność ma znaczenie: trasa dodana bez aplikacji Access jest od razu publiczna.

1. Utwórz tunel w panelu Zero Trust i zainstaluj usługę poleceniem, które panel
   wyświetli:

   ```cmd
   cloudflared service install <TOKEN>
   ```

2. Dodaj aplikację Access typu Self-hosted dla `<NAZWA_HOSTA>` z polityką
   dopuszczającą wyłącznie wskazane adresy e-mail.
3. Dopiero teraz dodaj w konfiguracji tunelu trasę publiczną `<NAZWA_HOSTA>` →
   `http://127.0.0.1:8501`.

Sprawdź trasy, które usługa faktycznie otrzymała. `cloudflared` udostępnia je
lokalnie na porcie metryk (domyślnie `20241`):

```cmd
curl http://127.0.0.1:20241/config
```

Lista `ingress` ma zawierać tylko `<NAZWA_HOSTA>` z usługą `http://127.0.0.1:8501`
oraz regułę domyślną. Sama reguła `http_status:503` oznacza, że tunel nie publikuje
żadnej usługi.

Sprawdź ochronę z zewnątrz, bez logowania:

```cmd
curl -I https://<NAZWA_HOSTA>
```

Poprawna odpowiedź to przekierowanie `302` na adres w domenie `cloudflareaccess.com`.
Każda inna odpowiedź oznacza, że Access nie chroni aplikacji — usuń trasę i popraw
konfigurację. Sprawdź też logowanie kontem z listy i odmowę dla konta spoza listy,
zanim przekażesz adres testerom.

Usługa `cloudflared` startuje razem z systemem; aplikację uruchamiasz osobno
poleceniem `run_pismak.bat`. Nie umieszczaj tokenu tunelu ani `.env` w repozytorium.

Quick Tunnel (`cloudflared tunnel --url http://127.0.0.1:8501`) nie zapewnia tu
warstwy Cloudflare Access. Traktuj go jako krótką próbę techniczną bez danych
osobowych i nie udostępniaj jego adresu testerom przed skonfigurowaniem ochrony
Access na własnej domenie.

#### Kontrole przed każdym udostępnieniem

- Uruchom `docker compose config` i sprawdź, że PostgreSQL nasłuchuje wyłącznie na `127.0.0.1`.
- Sprawdź lokalnie UI, Ollamę oraz działanie dwóch niezależnych sesji.
- Sprawdź kolejkę równoczesnych generacji i komunikaty przy niedostępności Ollamy lub bazy.
- Przed udostępnieniem zdalnym sprawdź trasy tunelu (`curl http://127.0.0.1:20241/config`) i ochronę Access (`curl -I https://<NAZWA_HOSTA>` zwraca przekierowanie do logowania), w tym odmowę dostępu dla nieuprawnionego konta.
- Automatyczne `pytest` i Ruff są uruchamiane w GitHub Actions przy pushu i pull requeście.

---

## 9. Przykładowa struktura projektu

```text
Pismak/
├── .env_template
├── README.md
├── requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
├── docker-compose.yaml
├── run_pismak.bat
├── .streamlit/config.toml
├── docker/postgres/init/001_extensions.sql
├── modele/
│   ├── bielik/Modelfile
│   └── pllum/Modelfile
├── scripts/test_openrouter.py
├── src/
│   ├── main.ipynb
│   ├── _shared/
│   ├── Dopytywator/
│   ├── Pismogenerator/
│   ├── Wyswietlator/
│   │   ├── app.py
│   │   └── services.py
│   └── Zrodlowyszukiwator/
│       ├── RAG/
│       │   ├── document_catalog.json
│       │   └── RAW/                (dwa dokumenty PDF)
│       ├── rag_ingestion/
│       ├── rag_retrieval/
│       └── brief_to_evidence/
├── tests/
└── .venv/                          (lokalnie, poza repozytorium)
```

---

## 10. Docelowy przepływ

```text
Użytkownik
   ↓
Dopytywator
   ↓
CaseBrief
   ↓
Źródłowyszukiwacz
   │
   ├── planowanie wyszukiwania
   ├── vector search
   ├── keyword search
   ├── metadata filtering
   └── reranking
   ↓
EvidencePack
   ↓
Pismogenerator
   ↓
DraftLetter
   ↓
Faktosprawdzanie
   ↓
Korekta językowa
   ↓
Analiza możliwej odpowiedzi urzędu
   ↓
finalne pismo
   ↓
wybór adresata
   ↓
wysyłka / zapis
```

---

## 11. Cel końcowy

Docelowo Pismak ma działać jako jeden spójny interfejs chatowy.

Użytkownik powinien móc napisać np.:

```text
Chcę, żeby miasto poprawiło bezpieczeństwo pieszych
na skrzyżowaniu obok szkoły.
```

System powinien następnie:

1. dopytać o istotne informacje,
2. zbudować `CaseBrief`,
3. znaleźć odpowiednie polityki, strategie, standardy i dokumenty miejskie,
4. znaleźć podobne rozwiązania z innych miast,
5. stworzyć argumentację opartą na źródłach,
6. wygenerować projekt pisma,
7. oznaczyć twierdzenia wymagające weryfikacji,
8. sprawdzić język i styl,
9. przewidzieć możliwe kontrargumenty i odpowiedź urzędu,
10. pomóc wybrać właściwego adresata,
11. zapisać lub wysłać pismo.

Docelowo lokalne modele powinny być traktowane jako wymienne komponenty:

```text
lokalny PLLuM / Bielik
        ↕
model chmurowy
```

bez konieczności przebudowy logiki Pismaka.

---

## Status projektu

Aktualnie:

```text
Dopytywator           ✅
Pismogenerator        ✅
lokalne LLM / Ollama  ✅
rag_ingestion         ✅ / rozwijane
PostgreSQL + pgvector 🟡 konfiguracja
rag_retrieval         ✅ / rozwijane
brief_to_evidence     ✅ / rozwijane
reranking             ✅ / rozwijane
faktosprawdzanie      ⬜
wysyłka               ⬜
```

Projekt jest obecnie w fazie prototypowania architektury i pipeline'u RAG.
