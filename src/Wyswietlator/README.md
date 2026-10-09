# Interfejs Pismaka

## Cel

Pismak pomaga przejść od opisu problemu, przez doprecyzowującą rozmowę i wyszukanie źródeł, do projektu pisma do instytucji publicznej.

## Struktura

- `app.py` — widoki, konfiguracja modeli, stan sesji i obsługa interakcji użytkownika.
- `services.py` — tworzenie klientów i pipeline'ów, renderowanie oraz zapis rezultatów.
- `__init__.py` — oznaczenie katalogu jako pakietu Python.

## Przepływ danych

Użytkownik opisuje sprawę w rozmowie; interfejs przekazuje ją do Dopytywatora i przechowuje w stanie sesji. Po utworzeniu `CaseBrief` uruchamia Źródłowyszukiwacz, który zwraca `EvidencePack`, a następnie Pismogenerator, który tworzy i waliduje projekt. Rozmowa jest wyświetlana jako pierwsza, a poniżej znajdują się podsumowanie sprawy, źródła i pismo do pobrania. Rozmowa i kolejne artefakty są zapisywane w katalogu uruchomienia w `outputs/`.

Przepływ: rozmowa → `CaseBrief` → `EvidencePack` → projekt pisma → podgląd, pobranie i zapis wyników.