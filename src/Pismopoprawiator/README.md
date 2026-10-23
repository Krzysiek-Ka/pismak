# Pismopoprawiator

## Cel

Pismopoprawiator poprawia język gotowego pisma po Pismogeneratorze i nie zmienia jego treści. Wolno mu poprawić ortografię, interpunkcję, gramatykę i fleksję, literówki oraz spacje. Nie może zmienić sensu ani tonu, liczb, dat, nazw własnych, numerów uchwał i artykułów, cytatów w „…”, oznaczeń źródeł `[n]`, adresów URL, frazy „wnioskuję o” ani znaczników niepewności („według obserwacji”, „[wymaga weryfikacji]”). Nie dodaje, nie usuwa i nie przeredagowuje zdań. Uwagi stylistyczne pokazuje jako sugestie i ich nie stosuje.

Etap nigdy nie przerywa pipeline'u. Przy każdym błędzie zwraca pismo bez korekty. Nie dostaje `CaseBrief` ani źródeł, tylko tekst pisma.

## Struktura

- `pismopoprawiator.py` — wywołania modelu, wybór trybu, `correct_letter` i `make_llm`.
- `pismopoprawiator_helper_functions.py` — podział pisma na akapity, strażnicy, stosowanie poprawek, strażnik całości i logi.
- `prompts.py` — instrukcja dla modelu i prompt z ponumerowanymi akapitami.
- `structures.py` — schemat odpowiedzi modelu (`Correction`, `CorrectionResponse`) i wynik etapu (`CorrectionResult`).
- `constants.py` — limity etapu i listy elementów chronionych.

## Przepływ danych

1. Pismo jest dzielone na część do korekty i niezmienny koniec, czyli listę źródeł i sekcję kontrolną. Koniec nie trafia do modelu.
2. Część do korekty jest dzielona na akapity `[A1]`, `[A2]`… Adresat, nadawca, typ pisma, formuła „Z poważaniem,” i podpis są oznaczone jako tylko do odczytu.
3. Model dostaje całe pismo w jednym wywołaniu. Jeśli pismo jest dłuższe niż `WHOLE_LETTER_MAX_CHARS` albo nie mieści się w oknie modelu, dostaje je po akapitach: grupy do `CHUNK_MAX_CHARS` znaków, z poprzednim akapitem jako kontekstem, najwyżej `MAX_CALLS` wywołań.
4. Model nie zwraca tekstu, tylko JSON `{"poprawki": [{"akapit", "oryginal", "poprawka", "kategoria", "uzasadnienie"}]}`.
5. Kod sprawdza każdą poprawkę, a przyjęte stosuje deterministycznie, od końca tekstu.
6. Strażnik całości porównuje wynik z oryginałem. Przy porażce etap zwraca oryginalne pismo.

Wywołanie korzysta z `LLMClient.generate_structured` (`src/_shared/serwer.py`): najwyżej 2 próby, limit odpowiedzi 4096 tokenów, temperatura 0. Limit 300 s na żądanie ustawia `make_llm` przez `with_options` klienta OpenAI; czas czekania w kolejce lokalnego modelu się do niego nie wlicza. Limity etapu są wyższe niż w innych modułach: modele rozumujące z OpenRouter potrzebują miejsca na rozumowanie, a Bielik na CPU nie mieścił się z całym pismem w 90 s. Przy wyborze trybu etap rezerwuje na odpowiedź 2048 tokenów (`POPRAWIATOR_RESERVED_OUTPUT_TOKENS`). Rzeczywisty limit odpowiedzi jest obniżany do miejsca, które zostaje w oknie modelu. Okno kontekstu Ollamy ustala `PARAMETER num_ctx` w `Modelfile` (README, punkt 3.3). Etap odczytuje je przez `detect_context_tokens` i według niego wybiera tryb.

## Strażnicy

Poprawka jest odrzucana, z podanym powodem, gdy:

- dotyczy akapitu nieistniejącego, tylko do odczytu albo widocznego w danym wywołaniu tylko jako kontekst;
- ma nieznaną kategorię, pusty fragment, fragment dłuższy niż 200 znaków albo nic nie zmienia;
- fragmentu nie ma we wskazanym akapicie albo występuje w nim więcej niż raz;
- przecina element chroniony: `[...]`, URL, cytat, frazę chronioną albo formułę otwierającą. Cały element może się znaleźć we fragmencie, ale musi zostać bez zmian;
- zmienia w akapicie:
  - liczby i daty, numery uchwał (`XLVIII/1169/13`), oznaczenia przepisów (`art.`, `§`, `ust.`, `pkt`),
  - oznaczenia w nawiasach kwadratowych, URL-e, cytaty, frazy chronione,
  - nazwy własne, czyli wyrazy wielką literą poza początkiem zdania,
  - liczbę przeczeń, wyrazy modalne albo formatowanie (podziały wierszy, `*`, `#`, `` ` ``);
- jest zbyt odległa od oryginału (`difflib`: podobieństwo < 0,5);
- nakłada się na inną przyjętą poprawkę.

Poprawki z kategorią `styl` są sugestiami i nigdy nie są stosowane.

Strażnik całości sprawdza, że:

- lista źródeł i sekcja kontrolna są identyczne,
- liczba akapitów jest ta sama, a akapity tylko do odczytu się nie zmieniły,
- długość treści zmieniła się najwyżej o 5%,
- elementy chronione w całym piśmie się nie zmieniły.

Strażnicy są ostrożni. Odrzucą też część poprawnych zmian, np. odmianę nazwy własnej albo wielką literę w środku zdania; takie odrzucenia widać w logu i w interfejsie.

## Wynik i logi

`CorrectionResult` zawiera:

- tekst oryginalny i poprawiony,
- poprawki zastosowane i odrzucone (z powodem) oraz sugestie,
- status: `corrected`, `unchanged`, `reverted` albo `failed`,
- tryb, dostawcę, model, okno kontekstu, liczbę wywołań, czas i błędy.

W logu sprawy każda poprawka ma osobną linię: `correction_applied`, `correction_rejected` (z polem `powod`) albo `correction_suggestion`, z akapitem, kategorią, oryginałem, poprawką i uzasadnieniem. Na końcu jest linia podsumowania `correction_finished`. Pełna treść pisma trafia do logu tylko na poziomie DEBUG (`correction_text`), a domyślny poziom logu sprawy to INFO.

## Testy

- `tests/unit_test/pismopoprawiator/` — strażnicy, stosowanie poprawek, tryby, błędy i logi, bez sieci.
- `tests/golden_set/pismopoprawiator/` — złoty zbiór na prawdziwych modelach (tylko ręcznie, `-m golden_set`), porównanie trybu „całość” i „po akapitach”. Rozwinięcie skrótu („ul.” → „ulicy”) nie jest w nim naruszeniem treści.
- `tests/e2e/` — etap „korekta” po Pismogeneratorze; niezmiennik H8 i ocena S10.
