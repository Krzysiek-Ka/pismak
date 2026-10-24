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

Uwagi dla użytkownika zawsze stoją w nawiasach kwadratowych: `[DO UZUPEŁNIENIA: …]` i `[wymaga weryfikacji]`. Prompt prosi model o oznaczanie niepotwierdzonych twierdzeń znacznikiem `[wymaga weryfikacji]`. Renderer zamienia też zapis „(wymaga weryfikacji)” i samo „wymaga weryfikacji” poza nawiasami na `[wymaga weryfikacji]`. Nie dotyka zwrotu z przeczeniem („nie wymaga weryfikacji”).

Renderer porządkuje też to, co model dopisuje wokół treści:
- usuwa „wnioskuję o” z początku podsumowania wniosku, bo formuła już je zawiera;
- zmniejsza pierwszą literę rzeczownika na początku podsumowania („Zwołanie” → „zwołanie”); skrótowce i nazwy zostają;
- zamienia dosłowne `\n` na podziały wiersza;
- usuwa formułę pożegnalną i podpis z końca argumentacji, bo renderer dodaje własne.

Stałe elementy układu pisma (adresat, nadawca, formuła, podpis, nagłówek źródeł) są stałymi w `pismogenerator_helper_functions.py`. Korzysta z nich Pismopoprawiator, który na ich podstawie rozpoznaje akapity tylko do odczytu.

Przepływ: `CaseBrief` + `EvidencePack` + materiały użytkownika → `DraftLetter` → walidacja → treść pisma Markdown.