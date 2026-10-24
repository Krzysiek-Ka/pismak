# Złote zbiory

Każdy moduł ma tu zbiór najważniejszych przypadków brzegowych, sprawdzanych na prawdziwych modelach i bazie, bez atrap. Zasady opisuje [.github/copilot-instructions.md](../../.github/copilot-instructions.md).

```text
tests/golden_set/
  golden_support.py                dostawcy, przypadki, limity, raport przebiegu (wspólne)
  <moduł>/
    test_golden_<moduł>.py         kontrola zbioru (bez modeli) i przebiegi na modelach
    cases/<id_przypadku>.json      jeden plik na przypadek
    results/                       raporty przebiegów, poza Gitem
```

Moduły: `dopytywator`, `zrodlowyszukiwator`, `pismogenerator`, `pismopoprawiator`, `rag_ingestion` (wczytywanie PDF-ów) i `rag_eval` (metryki wyszukiwania).

## Uruchamianie tylko ręczne

Każdy moduł testów ma `pytestmark = pytest.mark.golden`; zwykłe `python -m pytest` i CI pomijają ten marker.

```powershell
.venv\Scripts\python.exe -m pytest tests/golden_set -m golden -v -s                       # całość
.venv\Scripts\python.exe -m pytest tests/golden_set/dopytywator -m golden -k TestGoldenCases  # tylko kontrola zbioru
$env:PISMAK_GOLDEN_PROVIDERS = "bielik,pllum"                                               # opcjonalnie: wybrani dostawcy
```

- **Dostawcy** pochodzą z konfiguracji projektu: każdy `modele/<nazwa>/Modelfile` to model Ollamy `pismak-<nazwa>`, a OpenRouter to `openrouter/free`. Identyfikator testu: `<dostawca>-<przypadek>`.
- **Brak usługi** (model, baza, embeddingi) przed startem kończy się pominięciem z powodem.
- **Limity:** limit tokenów modułu, limit czasu żądania (`limits.timeout_seconds` albo ten z aplikacji) i `limits.max_questions` dla Dopytywatora.
- **Raport:** każdy przebieg zapisuje `results/<czas>_<dostawca>_<przypadek>.json` (model, status `passed` / `failed_check` / `infra_error` / `skipped`, wynik każdej kontroli, czas, ponowienia, błędy, `manual_review`) i log obok. Sekrety z `.env` są zastępowane znacznikiem.
- **OpenRouter:** darmowy router ma dzienny limit żądań (50 bez doładowania konta); po jego wyczerpaniu przebiegi kończą się statusem `infra_error`.

## Przypadek

```json
{
  "id": "unknown_addressee",
  "description": "Co przypadek sprawdza i dlaczego (pełni rolę docstringu).",
  "input": "Opis sprawy albo dane wejściowe modułu.",
  "user_answers": ["Odpowiedzi użytkownika po kolei (Dopytywator)."],
  "limits": { "max_questions": 3 },
  "expected": { "null_fields": ["addressee"] },
  "manual_review": ["Czego nie da się sprawdzić automatycznie; nie wpływa na wynik."]
}
```

`expected` zawiera sprawdzalne właściwości, a nie dokładny tekst. Obsługiwane pola wymienia docstring pliku testu; nieznane pole kończy zbieranie testów błędem.

## Moduły

- **Dopytywator:** wywiad jak w czacie aplikacji; limit pytań, schemat i gotowość briefu, brak liczb spoza rozmowy, puste pola, których użytkownik nie podał (adresat, `available_evidence`).
- **Źródłowyszukiwacz:** po trzy przypadki na dokument korpusu (odpowiedź wprost, z połączenia fragmentów, spoza korpusu); właściwy dokument w pakiecie albo abstencja, fragmenty identyczne z bazą, pominięcie nieobowiązującej Strategii 2030.
- **Pismogenerator:** CaseBrief i EvidencePack z prawdziwymi fragmentami; sekcje pisma, dozwolone źródła i przypisy, brak liczb, cytatów i podstaw prawnych spoza danych.
- **Pismopoprawiator:** dziesięć pism × tryby „całość” i „po akapitach”. Test nie przechodzi tylko przy naruszeniu treści; naprawione błędy i odrzucenia trafiają do raportu. Rozwinięcie skrótu („ul.” → „ulicy”) nie jest naruszeniem.
- **Wczytywanie PDF-ów (`rag_ingestion`):** prawdziwe PDF-y z `RAG/RAW/` i żywa baza; czysty tekst, limity fragmentów, frazy, które nie mogą zginąć, i zgodność indeksu z bieżącym kodem. Brak pliku na dysku kończy się pominięciem.
- **Ewaluacja RAG (`rag_eval`):** pytania do dokumentów korpusu na żywym indeksie; `precision@6` i `recall@6` na konfigurację. Szczegóły: [rag_eval/README.md](rag_eval/README.md).
