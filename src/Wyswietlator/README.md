# Interfejs Pismaka

## Cel

Pismak pomaga przejść od opisu problemu, przez doprecyzowującą rozmowę i wyszukanie źródeł, do projektu pisma do instytucji publicznej.

## Struktura

- `app.py` — widoki, konfiguracja modeli, stan sesji i obsługa interakcji użytkownika.
- `services.py` — tworzenie klientów i pipeline'ów, renderowanie oraz zapis rezultatów.
- `__init__.py` — oznaczenie katalogu jako pakietu Python.

## Dostawcy modeli

Każdy etap generatywny (Dopytywator, planner/reranker Źródłowyszukiwacza
i Pismogenerator) może korzystać z lokalnej Ollama albo z darmowego
routera OpenRouter (`openrouter/free`). Ustaw `OPENROUTER_KEY` w lokalnym
pliku `.env`; nie dodawaj prawdziwego klucza do repozytorium.

Przy wyborze OpenRouter UI wymaga potwierdzenia przed wysłaniem danych
poza aplikację. OpenRouter wybiera dostępny model, którego rzeczywista
nazwa jest pokazywana po odpowiedzi i zapisywana w `models.json` danego
uruchomienia. Żądania mają ustawione `require_parameters`, więc router
wybiera tylko modele obsługujące odpowiedź zgodną ze schematem JSON.
SDK nie ponawia żądań; aplikacja wysyła jedno dodatkowe żądanie tylko
wtedy, gdy odpowiedź nie spełnia schematu albo została ucięta. Planner
i reranker na OpenRouter nie mają planu zapasowego — błąd jest pokazywany
w interfejsie.

Przed przetwarzaniem sprawy UI wymaga również potwierdzenia informacji o
zapisie danych. Rozmowa i artefakty sprawy są przechowywane na serwerze
aplikacji jako niezaszyfrowane pliki, dostępne dla administratora serwera,
bez automatycznego terminu usunięcia. Logi mogą zawierać niepoprawne lub
ucięte odpowiedzi modelu (do 20 000 znaków), a więc także dane sprawy.

Ręczny test po skonfigurowaniu `.env`:

```powershell
.\.venv\Scripts\python.exe .\scripts\test_openrouter.py
```

Skrypt wykonuje jedno krótkie zapytanie, wypisuje odpowiedź i identyfikator
modelu, ale nigdy nie wypisuje klucza. Nie uruchamia się w `pytest`.
Embeddingi BGE-M3 i lokalny pipeline RAG pozostają niezależne od OpenRouter.

## Przepływ danych

Użytkownik opisuje sprawę w rozmowie; interfejs przekazuje ją do Dopytywatora i przechowuje w stanie sesji. Po utworzeniu `CaseBrief` uruchamia Źródłowyszukiwacz, który zwraca `EvidencePack`, a następnie Pismogenerator, który tworzy i waliduje projekt. Rozmowa jest wyświetlana jako pierwsza, a poniżej znajdują się podsumowanie sprawy, źródła i pismo do pobrania. Rozmowa i kolejne artefakty są zapisywane w katalogu uruchomienia w `outputs/`.

Przepływ: rozmowa → `CaseBrief` → `EvidencePack` → projekt pisma → podgląd, pobranie i zapis wyników.