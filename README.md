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
3. Podsumowanie: „W imieniu Towarzystwa Upiększania Miasta Wrocławia, wnioskuję o ...”
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

### 3.1. Wymagania

Potrzebne są:

- Git,
- Python 3.11 lub nowszy,
- `uv`,
- Docker Desktop,
- Ollama.

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

Modele GGUF mogą zostać zaimportowane do Ollamy przez `Modelfile`.

Przykład:

```text
FROM ./pllum-4b-chat-2512-q4_k_m.gguf
```

Następnie:

```cmd
ollama create pismak-pllum -f Modelfile
```

Analogicznie można dodać Bielika.

### 4.3. Model embeddingowy

Dla RAG używany jest lokalny model embeddingowy:

```cmd
ollama pull bge-m3
```

Sprawdź:

```cmd
ollama list
```

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

### 5.2. `docker-compose.yaml`

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

### 5.3. `.env`

Utwórz lokalny plik konfiguracyjny na podstawie szablonu `.env_template`
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

## 6. Dodanie dokumentu do RAG

Przy każdym nowym dokumencie:

```text
PDF
 ↓
rag_ingestion
 ↓
PostgreSQL / pgvector
```

Najpierw należy przygotować metadata dokumentu, a następnie uruchomić ingestion.

Przykładowo:

```cmd
python -m src.Zrodlowyszukiwator.rag_ingestion.cli ^
   --file "C:\sciezka\do\dokumentu.pdf" ^
   --catalog sciezka\do\katalogu.json
```

Ten sam plik nie powinien być ponownie dodawany, jeśli jego SHA-256 się nie zmienił.

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
`src/Zrodlowyszukiwator/rag_ingestion/notebooks/`.

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

Interfejs powinien być dostępny pod adresem `http://localhost:8501`.

W UI można:

- wybrać modele Ollama dla Dopytywatora, Źródłowyszukiwatora i Pismogeneratora,
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

Wyświetlator wymaga działającej Ollamy, PostgreSQL z tabelami `rag_documents` i
`rag_chunks` oraz zainstalowanego `bge-m3`. Przy pustej bazie RAG najpierw trzeba
wykonać ingestion dokumentów.


### Retencja i usuwanie danych

Pismak zapisuje na komputerze, na którym działa Streamlit, pełną rozmowę oraz
brief, znalezione dowody, draft i wyrenderowane pismo. Dane pozostają w katalogu
danego uruchomienia pod `outputs/` bez automatycznego terminu usunięcia. UI nie
ma funkcji kasowania spraw; operator usuwa cały katalog uruchomienia ręcznie po
zakończeniu pracy lub testów. Dostęp do plików ma każda osoba mająca dostęp do
systemu plików hosta. Przy zdalnym udostępnieniu aplikacji dane nadal są
przechowywane na hoście właściciela. Nie umieszczaj w sprawach danych osobowych,
które nie są konieczne, i nie publikuj zawartości `outputs/`.

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

Wymagana jest domena zarządzana przez Cloudflare. Na komputerze właściciela:

```cmd
cloudflared tunnel login
cloudflared tunnel create pismak
```

Zapisz konfigurację poza repozytorium, w `%USERPROFILE%\.cloudflared\config.yml`
(UUID i ścieżkę credentials podstaw wartościami wyświetlonymi przez `cloudflared`):

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: C:\Users\<USER>\.cloudflared\<TUNNEL_UUID>.json

ingress:
   - hostname: app.pismak.pl
      service: http://127.0.0.1:8501
   - service: http_status:404
```

Utwórz trasę DNS:

```cmd
cloudflared tunnel route dns pismak app.pismak.pl
```

W Cloudflare Zero Trust dodaj aplikację Access typu Self-hosted dla
`app.pismak.pl` i politykę dopuszczającą wyłącznie wskazane adresy e-mail. Sprawdź
logowanie w zwykłym oknie i odmowę dostępu dla niedopuszczonego konta, zanim
przekażesz adres testerom. Uruchom aplikację i tunel w osobnych terminalach:

```cmd
run_pismak.bat
cloudflared tunnel run pismak
```

Nie umieszczaj pliku credentials, tokenów ani `.env` w repozytorium.

Quick Tunnel (`cloudflared tunnel --url http://127.0.0.1:8501`) nie zapewnia tu
warstwy Cloudflare Access. Traktuj go jako krótką próbę techniczną bez danych
osobowych i nie udostępniaj jego adresu testerom przed skonfigurowaniem ochrony
Access na własnej domenie.

#### Kontrole przed każdym udostępnieniem

- Uruchom `docker compose config` i sprawdź, że PostgreSQL nasłuchuje wyłącznie na `127.0.0.1`.
- Sprawdź lokalnie UI, Ollamę oraz działanie dwóch niezależnych sesji.
- Sprawdź kolejkę równoczesnych generacji i komunikaty przy niedostępności Ollamy lub bazy.
- Przed udostępnieniem zdalnym sprawdź politykę Cloudflare Access, w tym odmowę dostępu dla nieuprawnionego konta.
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
├── docker/postgres/init/001_extensions.sql
├── src
│   ├── main.ipynb
│   ├── _shared/
│   ├── Dopytywator/
│   ├── Pismogenerator/
│   ├── Zrodlowyszukiwator/
│       ├── rag_ingestion/
│       ├── rag_retrieval/
│       └── brief_to_evidence/
│   ├── Wyswietlator/
│       ├── app.py
│       └── services.py
├── tests/
└── .venv/
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
