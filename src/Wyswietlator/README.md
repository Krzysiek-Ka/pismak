# Wyświetlator

## Cel

Wyświetlator jest interfejsem użytkownika aplikacji Pismak. Łączy wywiad, wyszukiwanie źródeł i generowanie pisma w jeden przepływ Streamlit, pokazuje wyniki i pozwala pobrać gotowy projekt.

## Struktura

- `app.py` — widoki, konfiguracja modeli, stan sesji i obsługa interakcji użytkownika.
- `services.py` — tworzenie klientów i pipeline'ów, renderowanie oraz zapis rezultatów.
- `__init__.py` — oznaczenie katalogu jako pakietu Python.

## Przepływ danych

Użytkownik opisuje sprawę w czacie; Wyświetlator przekazuje rozmowę do Dopytywatora i przechowuje ją w stanie sesji. Po utworzeniu `CaseBrief` uruchamia Źródłowyszukiwacz, który zwraca `EvidencePack`, a następnie Pismogenerator, który tworzy i waliduje projekt. Interfejs wyświetla brief, źródła i pismo oraz udostępnia pobranie pliku Markdown. Rozmowa i kolejne artefakty są zapisywane w katalogu uruchomienia w `outputs/`.

Przepływ: rozmowa → `CaseBrief` → `EvidencePack` → projekt pisma → podgląd, pobranie i zapis wyników.