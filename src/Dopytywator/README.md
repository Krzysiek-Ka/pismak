# Dopytywator

## Cel

Dopytywator porządkuje opis sprawy przed przygotowaniem pisma. Prowadzi krótki wywiad, a następnie zapisuje ustalenia w postaci `CaseBrief`. Nie wyszukuje źródeł, nie weryfikuje faktów i nie udziela porad prawnych.

## Struktura

- `dopytywator.py` — zbieranie opisu, prowadzenie wywiadu i budowanie briefu.
- `structures.py` — modele rozmowy, twierdzeń, wyniku tury i `CaseBrief`.
- `prompts.py` — instrukcje i formatowanie promptów wywiadu oraz briefu.
- `dopytywator_helper_functions.py` — walidacja gotowości briefu i prezentacja rozmowy oraz briefu.
- `constants.py` — stałe modułu.

## Przepływ danych

Opis użytkownika trafia do rozmowy. Model zadaje kolejne pytania, aż uzna informacje za wystarczające albo zostanie osiągnięty limit. Historia rozmowy jest następnie przekształcana w `CaseBrief`; moduł sprawdza obecność podstawowych pól i jawnie oznacza braki oraz fakty wymagające weryfikacji. `CaseBrief` jest wejściem dla Źródłowyszukiwacza i Pismogeneratora.