# Pismogenerator

## Cel

Pismogenerator przygotowuje projekt pisma na podstawie `CaseBrief`, źródeł przekazanych przez użytkownika i `EvidencePack` ze Źródłowyszukiwacza. Nie wyszukuje samodzielnie dokumentów ani nie uzupełnia treści wiedzą spoza tych danych.

## Struktura

- `pismogenerator.py` — budowanie i walidacja projektu oraz przekazanie go do renderowania.
- `structures.py` — model danych `DraftLetter`.
- `prompts.py` — instrukcje generowania projektu i przygotowanie kontekstu dla modelu.
- `pismogenerator_helper_functions.py` — walidacja projektu, składanie treści listu i renderowanie źródeł.

## Przepływ danych

`CaseBrief`, `EvidencePack` i materiały użytkownika są przekazywane do modelu, który zwraca `DraftLetter`. Projekt jest sprawdzany m.in. pod kątem dozwolonych identyfikatorów źródeł, brakujących treści i ograniczeń Phase 0. Poprawny projekt jest składany do tekstu Markdown; odwołania `E1`, `E2` wskazują źródła z `EvidencePack`, a `U1`, `U2` materiały użytkownika.

Przepływ: `CaseBrief` + `EvidencePack` + materiały użytkownika → `DraftLetter` → walidacja → treść pisma Markdown.