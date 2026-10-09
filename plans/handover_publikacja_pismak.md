# HANDOVER — publikacja Pismaka w internecie z lokalnymi modelami

## 1. Cel

Celem jest opublikowanie **Pismaka** jako publicznie dostępnego POC-u, przy zachowaniu całej ciężkiej logiki lokalnie na komputerze właściciela projektu.

Docelowo użytkownik ma wejść na adres HTTPS, np.:

```text
https://app.pismak.pl
```

i przejść przez cały pipeline:

```text
Użytkownik
   ↓
Dopytywator
   ↓
CaseBrief
   ↓
Źródłowyszukiwator
   ↓
EvidencePack
   ↓
Pismogenerator
   ↓
DraftLetter
   ↓
pismo.md
```

Modele generatywne, embeddingi i baza mają nadal działać lokalnie.

## 2. Obecna architektura

Projekt działa na Windowsie.

```text
Windows
│
├── Python + uv + .venv
├── Streamlit
├── Ollama
│   ├── PLLuM
│   ├── Bielik
│   └── bge-m3
└── Docker
    └── PostgreSQL + pgvector
```

Główny katalog:

```text
C:\Krzysiek\Pismak
```

UI:

```text
wyswietlator/
├── app.py
├── services.py
├── models.py
├── __init__.py
├── requirements-ui.txt
├── .env.ui.example
└── README.md
```

## 3. Moduły Pismaka

### 3.1. Dopytywator

Dopytywator:
- prowadzi rozmowę,
- zadaje jedno pytanie na turę,
- nie powtarza pytań,
- nie wymyśla faktów, przepisów ani instytucji,
- kończy rozmowę po zebraniu wystarczających informacji,
- ma konfigurowalny limit pytań,
- tworzy `CaseBrief`.

Minimalna gotowość `CaseBrief`:
- `title` niepusty,
- `problem_description` niepusty,
- `desired_outcome` niepusty,
- minimum jeden `requested_actions`.

### 3.2. Źródłowyszukiwator

Przepływ:

```text
CaseBrief
 ↓
Query Planner
 ↓
ResearchPlan
 ↓
Hybrid Retrieval
 ├── vector search
 └── full-text search
 ↓
RRF
 ↓
LLM reranker
 ↓
EvidencePack
```

Embeddings:
- `bge-m3`

Generatywny planner/reranker:
- PLLuM / Bielik / inny model z Ollamy.

Baza:
- PostgreSQL,
- pgvector,
- tabele RAG, m.in. `rag_documents`, `rag_chunks`.

### 3.3. Pismogenerator

Pismogenerator dostaje:
- `CaseBrief`,
- `EvidencePack`.

Generuje:
- `DraftLetter`,
- finalnie `letter.md`.

W POC:
- typ pisma = `WNIOSEK`,
- nie dodaje podstaw prawnych,
- nie wymyśla źródeł,
- może korzystać tylko ze źródeł przekazanych przez RAG.

## 4. Obecny UI

UI jest napisany w Streamlit.

Pozwala:
- wybrać model Dopytywatora,
- wybrać model Źródłowyszukiwatora,
- wybrać model Pismogeneratora,
- ustawić maksymalną liczbę pytań,
- prowadzić chat,
- wyświetlić CaseBrief,
- uruchomić RAG,
- wyświetlić źródła,
- wygenerować draft,
- pobrać `pismo.md`.

Nie należy przepisywać go na React ani tworzyć osobnego backendu na etapie POC.

## 5. Docelowa architektura publikacji

Preferowane rozwiązanie:

**Cloudflare Tunnel**

```text
Użytkownik w internecie
        ↓
https://app.pismak.pl
        ↓
Cloudflare
        ↓
Cloudflare Tunnel
        ↓
cloudflared na lokalnym PC
        ↓
127.0.0.1:8501
        ↓
Streamlit
        ↓
Pismak
        ↓
├── Ollama
└── PostgreSQL + pgvector
```

Główna zasada:

> Publiczny ma być tylko Streamlit przez tunel. Ollama i PostgreSQL nie mogą być wystawione do internetu.

## 6. Porty

### Streamlit

Uruchamiać:

```cmd
streamlit run wyswietlator\app.py --server.address 127.0.0.1 --server.port 8501
```

### PostgreSQL

W `docker-compose.yaml` preferować:

```yaml
ports:
  - "127.0.0.1:5432:5432"
```

zamiast:

```yaml
ports:
  - "5432:5432"
```

### Ollama

Pozostawić:

```env
OLLAMA_URL=http://127.0.0.1:11434
```

Nie wystawiać `11434` publicznie.

## 7. `.env`

Przykładowa konfiguracja:

```env
POSTGRES_USER=pismak
POSTGRES_PASSWORD=pismak_dev
POSTGRES_DB=pismak

DATABASE_URL=postgresql://pismak:pismak_dev@localhost:5432/pismak

OLLAMA_URL=http://127.0.0.1:11434

RAG_EMBEDDING_MODEL=bge-m3
RAG_EMBEDDING_DIM=1024
```

`.env` musi pozostać poza repozytorium i być w `.gitignore`.

## 8. Etap 1 — test lokalny

Przed publikacją sprawdzić:

```cmd
cd C:\Krzysiek\Pismak
.venv\Scripts\activate
ollama list
docker ps
streamlit run wyswietlator\app.py --server.address 127.0.0.1 --server.port 8501
```

Następnie wejść:

```text
http://localhost:8501
```

Przetestować cały przepływ:
1. pierwsza wiadomość,
2. pytania Dopytywatora,
3. CaseBrief,
4. RAG,
5. EvidencePack,
6. Pismogenerator,
7. download `pismo.md`.

## 9. Etap 2 — Quick Tunnel

Do pierwszego testu bez własnej domeny:

```cmd
cloudflared tunnel --url http://localhost:8501
```

Powinien powstać tymczasowy adres:

```text
https://<losowa-nazwa>.trycloudflare.com
```

Na tym etapie testować z:
- telefonu przez LTE/5G,
- innego komputera,
- 1–3 testerów.

## 10. Etap 3 — stały tunnel

Po udanym teście:

```text
app.pismak.pl
```

ma wskazywać przez Cloudflare Tunnel na:

```text
http://localhost:8501
```

Nie tworzyć publicznych hostname'ów dla:
- `11434`,
- `5432`,
- Docker API,
- folderu `outputs`.

## 11. Cloudflare Access

Pierwsza publiczna wersja nie powinna być otwarta dla każdego.

Preferowane:
- Cloudflare Access,
- dostęp tylko dla wskazanych adresów e-mail,
- 5–20 testerów.

Nie budować własnego systemu logowania w POC.

## 12. Krytyczna zmiana: obsługa wielu użytkowników

Streamlit ma `st.session_state`, ale przed publikacją należy sprawdzić pełną separację sesji.

Nie wolno przechowywać per-user danych w globalnych zmiennych.

Per-user:
- `messages`,
- `case_brief`,
- `evidence_pack`,
- `draft`,
- `letter_markdown`,
- `run_dir`,
- wybrane modele,
- liczba pytań.

Globalnie mogą być:
- klient Ollama,
- konfiguracja,
- lock,
- cache listy modeli,
- read-only repozytorium.

## 13. Unikalne `session_id`

Dodać:

```python
import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
```

Nie zapisywać runów wyłącznie po timestampie.

Preferowany `run_id`:

```python
from datetime import datetime
import uuid

run_id = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + uuid.uuid4().hex[:8]
)
```

Folder:

```text
outputs/
└── 20260928_203000_a84f9c21/
```

Każda sesja musi mieć własny katalog.

## 14. Krytyczna zmiana: kolejka / lock dla lokalnego LLM

Sprzęt nie jest serwerem wieloużytkownikowym.

Na POC przyjąć:

```text
1 generacja LLM naraz
```

Najprostsza implementacja:

```python
from threading import Lock

LLM_LOCK = Lock()
```

Centralnie:

```python
with LLM_LOCK:
    response = client.chat.completions.create(...)
```

Najlepiej umieścić lock w jednej wspólnej funkcji, np. `generate_structured()`.

Dotyczy:
- Dopytywatora,
- Query Plannera,
- rerankera,
- Pismogeneratora.

Na start można też serializować embeddingi BGE-M3.

## 15. Komunikat podczas oczekiwania

Jeśli lock jest zajęty, UI powinien pokazywać komunikat:

```text
Lokalny model jest zajęty. Oczekiwanie na swoją kolej...
```

Nie trzeba implementować Redis/Celery.

## 16. Limit długości wiadomości

Dodać np.:

```python
MAX_USER_MESSAGE_CHARS = 5000
```

Walidacja:

```python
if len(prompt) > MAX_USER_MESSAGE_CHARS:
    st.error("Wiadomość jest zbyt długa.")
    st.stop()
```

## 17. Limit pytań

Publiczny POC:
- minimalnie 1,
- maksymalnie 10 lub 12.

Nie pozwalać użytkownikowi ustawić np. 100.

## 18. Publiczne błędy

Usunąć z widoku użytkownika:

```python
st.exception(exc)
```

Zamiast tego:

```python
st.error(
    "Wystąpił błąd podczas przetwarzania sprawy. "
    "Spróbuj ponownie za chwilę."
)
```

Pełny traceback logować lokalnie.

## 19. Logging

Dodać katalog:

```text
logs/
```

i log np.:

```text
logs/pismak.log
```

Logować:
- start sesji,
- `session_id`,
- wybrany model,
- start/koniec etapu,
- czas trwania,
- liczbę znalezionych chunków,
- liczbę evidence,
- błędy.

Nie logować domyślnie:
- pełnej treści rozmów,
- danych osobowych,
- pełnych pism.

## 20. Timing

Mierzyć czasy:

```python
import time

t0 = time.perf_counter()
...
elapsed = time.perf_counter() - t0
```

Minimum:
- Dopytywator,
- build CaseBrief,
- Query Planner,
- retrieval,
- reranking,
- Pismogenerator.

To pozwoli ocenić wąskie gardła.

## 21. Bezpieczeństwo danych

Użytkownicy mogą podawać:
- imiona,
- adresy,
- miejsca,
- opis konfliktu,
- dane osób trzecich.

Dlatego:
- `outputs/` nie może być publiczny,
- nie serwować plików bezpośrednio z filesystemu,
- plik do pobrania przekazywać przez `st.download_button`,
- `.env` nie może być publiczny,
- token Cloudflare nie może trafić do repozytorium.

## 22. Folder `outputs`

Pozostaje lokalnie:

```text
C:\Krzysiek\Pismak\outputs
```

Przykład:

```text
outputs/
└── 20260928_203000_a84f9c21/
    ├── conversation.json
    ├── brief.json
    ├── evidence.json
    ├── draft.json
    └── letter.md
```

## 23. Automatyczny start

Docelowo można przygotować:

```text
run_pismak.bat
```

Przykład:

```bat
@echo off
cd /d C:\Krzysiek\Pismak
call .venv\Scripts\activate
streamlit run wyswietlator\app.py --server.address 127.0.0.1 --server.port 8501
```

Uruchamiać przez Windows Task Scheduler.

Cloudflared również powinien docelowo działać jako usługa Windows.

## 24. Co musi być uruchomione

Publiczna aplikacja działa tylko, gdy działa cały stack:

```text
Windows
│
├── internet
├── Ollama
├── Docker Desktop
│   └── PostgreSQL + pgvector
├── Streamlit
└── cloudflared
```

Jeśli komputer się wyłączy lub uśnie — aplikacja przestaje działać. Dla POC jest to akceptowalne.

## 25. Test wielosesyjny

Przed publikacją:

1. otworzyć zwykłe okno Chrome,
2. otworzyć incognito,
3. uruchomić dwie różne sprawy,
4. sprawdzić, czy:
   - chaty się nie mieszają,
   - CaseBriefy się nie mieszają,
   - EvidencePacki się nie mieszają,
   - drafty się nie mieszają,
   - pliki zapisują się w różnych folderach.

## 26. Test dwóch użytkowników równocześnie

Użytkownik A:
- uruchamia generację.

Użytkownik B:
- uruchamia generację w tym samym czasie.

Oczekiwane:
- brak crasha,
- B czeka,
- A kończy,
- potem B dostaje wynik,
- odpowiedzi nie mieszają się.

## 27. Test niedostępnej Ollamy

Jeśli Ollama nie działa:
- UI nie może pokazać tracebacka,
- powinien pojawić się komunikat typu:

```text
Model lokalny jest obecnie niedostępny.
```

## 28. Test niedostępnej bazy

Jeśli PostgreSQL nie działa:

```text
Źródłowyszukiwator jest obecnie niedostępny.
```

Opcjonalnie później można dodać fallback do generowania bez RAG.

## 29. Model degradacji — opcjonalny

Jeżeli baza nie działa:

```text
Dopytywator
 ↓
CaseBrief
 ↓
Pismogenerator bez EvidencePack
```

UI musi wtedy jawnie poinformować:

```text
Pismo wygenerowano bez źródeł RAG.
```

Nie jest to wymagane do pierwszego POC.

## 30. Rate limiting — później

Na pierwszym etapie wystarczy Cloudflare Access + lock.

Jeżeli pojawią się problemy:
- limit spraw na użytkownika,
- limit spraw na godzinę,
- limit czasu sesji.

Nie wdrażać rozbudowanego rate limitera bez potrzeby.

## 31. Czego nie robić

Na tym etapie NIE:
- React,
- FastAPI,
- Kubernetes,
- Redis,
- Celery,
- publiczne Ollama,
- publiczny PostgreSQL,
- port forwarding na routerze,
- osobny VPS,
- migracja modeli do chmury,
- własny auth backend,
- system użytkowników i haseł.

## 32. Sugerowana kolejność implementacji

1. Przeczytać `wyswietlator/app.py`.
2. Przeczytać `wyswietlator/services.py`.
3. Przeczytać `wyswietlator/models.py`.
4. Dodać `session_id`.
5. Dodać unikalny `run_id`.
6. Dodać globalny lock.
7. Dodać limit wiadomości.
8. Ograniczyć max pytań.
9. Usunąć publiczne tracebacki.
10. Dodać logging.
11. Dodać timingi.
12. Zmienić bind PostgreSQL na `127.0.0.1`.
13. Uruchamiać Streamlit na `127.0.0.1`.
14. Przetestować lokalnie.
15. Uruchomić Quick Tunnel.
16. Przetestować z telefonu.
17. Przetestować dwie sesje.
18. Przetestować dwóch jednoczesnych użytkowników.
19. Skonfigurować Named Tunnel.
20. Skonfigurować własną domenę.
21. Skonfigurować Cloudflare Access.
22. Zaktualizować README.

## 33. Oczekiwane pliki po wdrożeniu

Kolejny model powinien przygotować co najmniej:

```text
C:\Krzysiek\Pismak\
├── docker-compose.yaml
├── run_pismak.bat
├── logs/
├── outputs/
└── wyswietlator/
    ├── app.py
    ├── services.py
    ├── models.py
    ├── runtime.py      # opcjonalnie
    ├── __init__.py
    ├── requirements-ui.txt
    └── README.md
```

`runtime.py` może zawierać:
- globalny lock,
- logging,
- timing helpers.

Nie jest wymagany, jeśli zmiany pozostaną małe.

## 34. Kryteria akceptacyjne

### Funkcjonalność
- [ ] publiczny adres HTTPS działa,
- [ ] chat działa,
- [ ] Dopytywator działa,
- [ ] CaseBrief działa,
- [ ] Źródłowyszukiwator działa,
- [ ] Pismogenerator działa,
- [ ] `pismo.md` można pobrać.

### Bezpieczeństwo
- [ ] Ollama nie jest publiczna,
- [ ] PostgreSQL nie jest publiczny,
- [ ] `outputs/` nie jest publiczny,
- [ ] `.env` nie jest publiczny,
- [ ] Cloudflare Access ogranicza dostęp,
- [ ] tracebacki nie są widoczne dla użytkownika.

### Multi-user
- [ ] każda sesja ma własny `session_id`,
- [ ] każda sesja ma własny output,
- [ ] dane użytkowników nie mieszają się,
- [ ] globalny LLM lock działa.

### Stabilność
- [ ] brak crasha przy dwóch użytkownikach,
- [ ] brak crasha przy chwilowej niedostępności modelu,
- [ ] brak crasha przy błędzie DB,
- [ ] błędy są logowane.

## 35. Docelowy UX

Przykładowe statusy:

```text
Dopytywator analizuje odpowiedź...
```

```text
Przygotowuję podsumowanie sprawy...
```

```text
Źródłowyszukiwator przeszukuje dokumenty...
```

```text
Lokalny model jest zajęty. Oczekiwanie na swoją kolej...
```

```text
Pismogenerator przygotowuje draft...
```

## 36. Finalny efekt

Właściciel projektu powinien móc:

1. włączyć komputer,
2. uruchomić potrzebne usługi,
3. wejść na `https://app.pismak.pl`,
4. zalogować się przez Cloudflare Access,
5. rozpocząć sprawę,
6. przejść przez Dopytywator,
7. zobaczyć CaseBrief,
8. uruchomić Źródłowyszukiwator,
9. uruchomić Pismogenerator,
10. pobrać `pismo.md`.

Drugi użytkownik powinien móc zrobić to samo bez mieszania danych.

## 37. Priorytety

### Priorytet 1
- bezpieczeństwo portów,
- sesje,
- lock,
- brak tracebacków.

### Priorytet 2
- Cloudflare Tunnel,
- test z zewnątrz.

### Priorytet 3
- Cloudflare Access,
- logging,
- timingi.

### Priorytet 4
- automatyczny start usług.

## 38. Zakres POC

Zakładać:
- kilku–kilkunastu testerów,
- pojedyncze generacje,
- umiarkowane użycie,
- komputer lokalny pozostaje głównym serwerem.

Nie projektować na tym etapie rozwiązania dla setek równoczesnych użytkowników.

## 39. Handover dla kolejnego modelu — zadanie

Na podstawie tego dokumentu kolejny model ma:

1. zmodyfikować istniejący `wyswietlator`,
2. nie przepisywać projektu od zera,
3. zachować Streamlit,
4. zachować lokalne modele,
5. zachować lokalną bazę,
6. dodać izolację sesji,
7. dodać lock,
8. dodać bezpieczną obsługę błędów,
9. dodać logowanie,
10. przygotować publikację przez Cloudflare Tunnel,
11. przygotować dokładne instrukcje uruchomienia,
12. przygotować checklistę testów,
13. zaktualizować README.

Najważniejsze kryterium: **minimalna liczba nowych technologii i możliwie prosty kod Python**.
