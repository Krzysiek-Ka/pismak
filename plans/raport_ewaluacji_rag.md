# Raport: ewaluacja retrievalu RAG i naprawa ingestion (Pismak, 2026-10-14)

## 1. Podsumowanie

- **Wyszukiwanie odpowiedzi działa po naprawie ingestion.** Wszystkie 4 pytania z odpowiedzią
  przechodzą test (przed naprawą 3 z 4). Właściwy dokument jest wśród 6 fragmentów
  EvidencePack w 4/4 przypadkach. MRR dowodu w EvidencePack wzrósł z 0,42 do 0,69, a pokrycie
  faktów wśród 20 kandydatów z 0,88 do 1,00.
- **Główna luka: system nie rozpoznaje braku podstaw.** Abstencja w 0 z 2 pytań bez odpowiedzi,
  przed naprawą i po niej. Wdrożony jawny `evidence_status` (sekcja 11) działa mechanicznie, ale
  z lokalnym rerankerem `pismak-bielik` ocena trafności zawodzi: abstencja nadal 0/2, do tego
  1 fałszywa abstencja na 4 pytania z odpowiedzią. EvidencePack przedstawia 4–6 fragmentów, które nie odpowiadają na
  pytanie, jako dowody. Kontrakt RAG nie ma progu trafności. Prosty próg podobieństwa cosinusowego
  też nie rozwiąże problemu: po naprawie zakresy dla pytań z odpowiedzią (0,589–0,717) i bez
  odpowiedzi (0,590–0,604) nakładają się.
- **Hybryda nie poprawia wyników względem samego wyszukiwania wektorowego.** MRR dowodu po
  naprawie: wektorowe 0,71, hybrydowe 0,69, leksykalne 0,55. Przed naprawą RRF zepchnął dowód
  bike-01 z 1. na 7. miejsce, czyli poza EvidencePack. Domyślnego rerankera `pismak-pllum`
  nie udało się zmierzyć z powodu ograniczeń sprzętowych (sekcja 8). Reranker `pismak-bielik`
  pogarsza wynik: dowód w EvidencePack w 2/4 przypadkach wobec 4/4 bez rerankera (sekcja 11.5).
- **Metadane źródeł są poprawne:** 0 niezgodności z katalogiem w 627 sprawdzonych fragmentach
  w każdym przebiegu (tytuł, typ, forma prawna, numer i data aktu, status, URL, plik,
  identyfikator, zakres stron). Wszystkie dowody mają poprawnie przypisaną stronę.
- **Parafrazy, synonimy, daty i polskie znaki:** wyszukiwanie wektorowe radzi sobie z
  parafrazami („bez auta” → „transport niesamochodowy”, „zniżki” → „preferencje cenowe”) i z
  pytaniami bez polskich znaków. Wyszukiwanie leksykalne nie radzi sobie z parafrazami, a dla
  pytania o rok 2020 Strategia 2050 zajmuje 3 pierwsze miejsca EvidencePack.
- Naprawa defektów przywróciła 5–16% tekstu każdego dokumentu. Jej efekt uboczny, czyli
  powtarzane nagłówki stron w treści chunków, usuwa wersja 0.3, razem ze znakami sterującymi
  (sekcja 11). Na tym zbiorze pytań nie zmieniło to wyników w mierzalny sposób.

## 2. Co zostało zrobione

1. **Naprawa trzech defektów przygotowania tekstu** (krok 1) oraz mechanizmu, który
   przenosi poprawki do istniejących baz. Zmiany są w `src/Zrodlowyszukiwator/rag_ingestion/`
   (szczegóły w sekcji 3).
2. **Zbiór testowy** `tests/rag_eval/data/rag_eval_questions.jsonl`: po 3 pytania dla każdego
   z 2 zakomitowanych dokumentów (2 z odpowiedzią i 1 bez), zweryfikowane w tekście PDF.
3. **Test ewaluacyjny** `tests/rag_eval/`: testy zbioru bez bazy (w CI) oraz ewaluacja na
   żywym indeksie, uruchamiana na żądanie (`PISMAK_RAG_EVAL=1`), z raportem MD i JSON.
4. **Pomiar przed i po**: baseline na indeksie v0.1, reindeksacja wszystkich 6 dokumentów
   do v0.2, ponowny pomiar.

Kod retrievalu (`rag_retrieval`, `brief_to_evidence`) i schemat bazy pozostały bez zmian.
Dokumenty PDF i katalog metadanych nie były modyfikowane.

## 3. Naprawione defekty ingestion

| # | Defekt | Przyczyna | Naprawa |
|---|---|---|---|
| D1 | Linie uznane za nagłówki znikały z treści chunków | `build_text_units` zapisywał nagłówek tylko w `section_path`, nadpisywanym przez kolejny nagłówek. Regex nagłówka numerowanego łapał każdą linię zaczynającą się liczbą („2015 co najmniej…”, „8 marca 1990 r.…”). | Nagłówek zostaje także w treści. Numer rozdziału ma 1–2 cyfry na poziom, a tytuł zaczyna się wielką literą. Linie zakończone `:` lub `-` nie są nagłówkami. Linie spisu treści są jawnie pomijane. |
| D2 | Złe mapowanie fontu w Polityce Rowerowej: `Ŝ` zamiast `ż`, `ś` zamiast `Ż` w wersalikach | warstwa tekstowa PDF (render strony 5 pokazuje „BIEŻĄCE”, „podróży”) | `repair_legacy_polish_glyphs`, stosowana tylko w dokumencie z sygnaturą `Ŝ` (litera spoza polszczyzny) |
| D3 | Wyrazy przeniesione do następnej linii zostawały rozcięte („trans- portu”) | łączenie linii akapitu spacją | sklejanie `litera-` + mała litera; wyjątek dla łącznika zawieszonego („jedno- i dwu…”) |
| D4 | Poprawki ekstrakcji nie trafiały do istniejących baz | ingestion pomijał plik o tym samym SHA-256 i nie porównywał zapisanej `ingestion_version` | `INGESTION_VERSION = "0.2"`; dokument zapisany starszą wersją jest przetwarzany od nowa |

**Weryfikacja na wszystkich 6 PDF-ach** (stary i nowy kod, bez bazy; linie bez spisu treści):

| dokument | linie utracone: v0.1 → v0.2 | znaki utracone | `Ŝ` w chunkach | rozcięcia `x- y` |
|---|---|---|---|---|
| Strategia Wrocław 2050 | 1073 → 1* | 11,24% → 0,03% | 0 → 0 | 9 → 8 (same zawieszone łączniki „jedno- i”, „zero- i”) |
| Strategia Wrocław 2030 | 461 → 1* | 12,61% → 0,08% | 0 → 0 | 0 → 0 |
| Polityka Rowerowa | 41 → 0 | 8,41% → 0,00% | 94 → 0 | 0 → 0 |
| Wrocławska polityka mobilności | 134 → 0 | 5,67% → 0,00% | 0 → 0 | 208 → 1 („oto- czeniem” na granicy stron) |
| Program Tramwajowy 2.0 | 166 → 1* | 5,74% → 0,08% | 0 → 0 | 3 → 1 (złożenie „tramwajowo- autobusową”) |
| Standardy dostępności | 847 → 0 | 16,14% → 0,00% | 0 → 0 | 11 → 3 (same zawieszone łączniki) |

\* Ta pojedyncza linia nie zginęła. Przecina ją istniejący twardy podział akapitu dłuższego
niż 3500 znaków (np. „zmi|eni”), więc jej treść jest rozdzielona między dwa sąsiednie chunki.

Fakty kontrolne obecne w chunkach dopiero po naprawie: „2015 co najmniej 10%” (Rowerowa),
„Pozytywny > 65%” (tabela monitoringu Mobilności), „1. Kształtowanie miasta kompaktowego”
i „30 proc. powierzchni miasta” (Strategia 2050).

Testy jednostkowe: 10 nowych i 1 zaktualizowany (kontrakt: nagłówek zostaje w treści) w
`test_chunker.py`, `test_pdf_extractor.py` i `test_ingestion_pipeline.py`, w tym regresje na
zakomitowanych PDF-ach. Cały zestaw po zmianach: 238 testów zielonych i 13 pominiętych
(ewaluacja na żywym indeksie bez `PISMAK_RAG_EVAL=1`); `ruff check .` czysty.

Reindeksacja przez istniejące CLI **bez `--force`**: dokumenty o niezmienionym SHA-256 zostały
przetworzone ponownie wyłącznie z powodu wersji (potwierdza D4). Wynik dla Polityki Rowerowej:
`repaired_glyph_pages: [1, 3, 4, 5, 6, 7, 8, 9]`. Backup indeksu v0.1:
`outputs/rag_eval/index_backup_v0.1.dump` (przywrócenie: `docker cp` do kontenera i
`pg_restore -U pismak -d pismak --clean /tmp/index_backup_v0.1.dump`).

## 4. Zbiór testowy

| id | pytanie | odpowiedź w korpusie | źródło (strona PDF) |
|---|---|---|---|
| bike-01 | Jaki udział rowerów w ruchu miasto chciało osiągnąć w 2015, a jaki w 2020 roku? Chcę porównać z tym, co jest teraz. | tak: ≥10% podróży niepieszych w 2015, ≥15% wszystkich podróży w 2020 | Polityka Rowerowa s. 4 i 5 |
| bike-02 | Działam w stowarzyszeniu rowerowym. Czy organizacje społeczne mają jakiś formalny wpływ na to, jak miasto projektuje drogi? | tak: Rada ds. rozwoju ruchu rowerowego z udziałem NGO opiniuje koncepcje i projekty drogowe | Polityka Rowerowa s. 8 |
| bike-03 | Ile kosztuje wypożyczenie roweru miejskiego we Wrocławiu i czy pierwsze minuty są za darmo? | **nie** (brak cennika w 6 dokumentach) | — |
| mob-01 | Ile procent przejazdów po Wrocławiu miało się odbywać bez auta do 2020 roku? Chodzi mi o oficjalny cel miasta. | tak: ≥60% transportu niesamochodowego do 2020 | Polityka mobilności s. 4 |
| mob-02 | Mieszkam w centrum, w strefie płatnego parkowania. Czy miasto przewiduje dla mieszkańców jakieś zniżki za parkowanie? | tak: „preferencje cenowe dla stałych mieszkańców strefy płatnego parkowania” | Polityka mobilności s. 10 |
| mob-03 | Od kiedy we Wrocławiu ma obowiązywać strefa czystego transportu i które samochody nie będą mogły wjechać do centrum? | **nie** (brak SCT, terminu i kryteriów; jest tylko ogólny kierunek i „norma EURO 5” dla autobusów) | — |

Brak odpowiedzi na bike-03 i mob-03 sprawdziłem w tekście **wszystkich 6** dokumentów
indeksu, w tym 4 niezakomitowanych. Przeszukane hasła i powiązane, ale niewystarczające
fragmenty są zapisane w `absence_check`. Test przy każdym przebiegu sprawdza, że wzorce
odpowiedzi nie występują w żadnym chunku indeksu. Każdy cytat i każda fraza dowodu są
automatycznie weryfikowane na deklarowanej stronie PDF.

## 5. Metodyka (skrót)

- **Konfiguracje**: `vector`, `lexical`, `hybrid` (RRF 20+20, jak w produkcji) i `pipeline`
  (`BriefToEvidencePipeline`: 6 fragmentów, najwyżej 3 z dokumentu, bez rerankera).
  Zapytanie powstaje deterministycznie, jak w produkcyjnym fallbacku planera. Filtry są domyślne,
  więc nieaktywna Strategia 2030 jest wykluczona.
- **Dowód**: fragment z właściwego dokumentu zawierający wszystkie frazy jednego faktu z
  `evidence`. Ocena ignoruje artefakty ekstrakcji, żeby stary indeks był oceniany za treść.
- **Miary**: `source_hit@k` (właściwe źródło w top-k), `evidence_hit@k`, `evidence_recall@k`
  (udział pokrytych faktów), `MRR@20` dla źródła i dowodu oraz `evidence_precision@6` (jaka część
  przedstawionych fragmentów faktycznie wspiera odpowiedź). k=6 i k=20 to rozmiar EvidencePack
  i liczba kandydatów dla rerankera w produkcji. MRR pokazuje, jak wysoko jest pierwszy
  trafny dowód, a tylko czoło listy trafia do pisma.
- **Abstencja**: zgodnie z kontraktem RAG. Pusty EvidencePack albo intencja w `unsupported_topics`.
  Test nie wymaga pustej listy wyników wyszukiwania.
- **Kryterium zaliczenia**: pytanie z odpowiedzią wymaga, żeby właściwe źródło i dowód były w
  EvidencePack, a wszystkie fakty wśród 20 kandydatów. Pytanie bez odpowiedzi wymaga abstencji.
  Bez `xfail` i bez progów dobranych pod wynik.
- **Ograniczenie**: n=6 (4 + 2). Jedno pytanie to 25% metryk dla pytań z odpowiedzią,
  dlatego wnioski opierają się na wynikach per pytanie.

## 6. Wyniki zmierzone

Przebiegi (raporty generowane automatycznie, katalog `outputs/` jest w `.gitignore`):

| przebieg | indeks | raport | wynik testów |
|---|---|---|---|
| baseline | v0.1 (przed naprawą) | `outputs/rag_eval/20261014_122633_baseline/` | 3 FAILED, 9 PASSED, 1 SKIPPED |
| after | v0.2 (po naprawie) | `outputs/rag_eval/20261014_163938_after/` | 2 FAILED, 10 PASSED, 1 SKIPPED |
| after-reranker | v0.2 | `outputs/rag_eval/20261014_164220_after-reranker/` | jak „after” + FAILED test rerankera (OOM) |

Oba przebiegi po naprawie dały identyczne rankingi we wszystkich 36 listach wyników, więc
pomiar jest deterministyczny.

### 6.1. Wyniki zbiorcze, pytania z odpowiedzią (n=4), przed → po naprawie

| konfiguracja | source_hit@1 | source_hit@6 | evidence_hit@6 | evidence_recall@20 | MRR źródło | MRR dowód | precyzja dowodów@6 |
|---|---|---|---|---|---|---|---|
| `vector` | 0,75 → 0,75 | 1,00 → 1,00 | 1,00 → 1,00 | 0,88 → 1,00 | 0,83 → 0,83 | 0,67 → 0,71 | 0,21 → 0,25 |
| `lexical` | 0,25 → 0,50 | 0,50 → 0,75 | 0,25 → 0,75 | 0,50 → 0,75 | 0,40 → 0,55 | 0,28 → 0,55 | 0,08 → 0,17 |
| `hybrid` | 0,50 → 0,50 | 1,00 → 1,00 | 0,75 → 1,00 | 0,88 → 1,00 | 0,67 → 0,69 | 0,45 → 0,69 | 0,17 → 0,25 |
| `pipeline` (EvidencePack) | 0,50 → 0,50 | 1,00 → 1,00 | 0,75 → 1,00 | — | 0,67 → 0,69 | 0,42 → 0,69 | 0,29 → 0,38 |

Odsetek pytań z właściwym źródłem w wynikach: 100% w top-6 dla `vector`, `hybrid` i
`pipeline` (przed i po naprawie); dla `lexical` 50% → 75%.

### 6.2. Wyniki per pytanie

Pozycja pierwszego fragmentu z dowodem, przed → po naprawie („—” oznacza brak w zwróconej liście):

| pytanie | test: przed → po | `vector` | `lexical` | `hybrid` | EvidencePack | fragmentów w EvidencePack (po) |
|---|---|---|---|---|---|---|
| bike-01 | FAIL → **PASS** | 1 → 1 (fakty 1/2 → 2/2) | — → 1 | 7 → 1 | — → 1 | 5 |
| bike-02 | PASS → PASS | 3 → 2 | 1 → 1 | 1 → 1 | 1 → 1 | 3 |
| mob-01 | PASS → PASS | 1 → 1 | — → — (dokument poza top-20) | 3 → 4 | 3 → 4 | 4 |
| mob-02 | PASS → PASS | 3 → 3 | 10 → 5 | 3 → 2 | 3 → 2 | 5 |
| bike-03 (bez odp.) | FAIL → FAIL | — | — | — | brak abstencji: 5 → 4 fragmentów jako dowody | 4 |
| mob-03 (bez odp.) | FAIL → FAIL | — | — | — | brak abstencji: 5 → 6 fragmentów jako dowody | 6 |

### 6.3. Właściwe dokumenty kontra podobne słowa (pkt 1 i 4)

- W pytaniach o politykę mobilności Strategia Wrocław 2050 zajmuje 4–5 z 6 pierwszych miejsc
  hybrydy. Dla mob-01 („bez auta do 2020”) pozycje 1–3 EvidencePack to Strategia 2050: wyniki
  ankiety (s. 184) i wykaz dokumentów wykonawczych (s. 218). Właściwy cel „do roku 2020 … nie
  mniej niż 60%” jest dopiero na 4. miejscu. Mieści się w pakiecie tylko dzięki limitowi
  3 fragmentów na dokument.
- Wyszukiwanie leksykalne w ogóle nie znajduje Polityki mobilności dla mob-01 (54. miejsce w
  pełnym rankingu). Pytanie i dokument nie mają wspólnych słów kluczowych: „procent”, „auta”,
  „przejazdów” w pytaniu wobec „%”, „niesamochodowego”, „podróży” w dokumencie.
- Po naprawie pytania rowerowe trafiają we właściwy dokument na 1. miejscu we wszystkich
  konfiguracjach.

### 6.4. Jakość dowodów (pkt 2)

- Precyzja EvidencePack wzrosła z 0,29 do 0,38: około 6 na 10 przedstawionych fragmentów nie
  zawiera faktu potrzebnego do odpowiedzi (kontekst z tego samego dokumentu albo dystraktory).
  Przy 1–2 faktach na pytanie i 6 miejscach ten wskaźnik ma naturalny sufit.
- **EvidencePack bywa niepełny:** po naprawie zawiera 3–6 fragmentów zamiast 6 (bike-02: 3).
  Bez rerankera pipeline najpierw bierze 6 najlepszych kandydatów, a potem stosuje limit 3 na
  dokument ([pipeline.py:71-83](../src/Zrodlowyszukiwator/brief_to_evidence/pipeline.py#L71-L83)).
  Kandydaci z pozycji 7–20 nie uzupełniają odrzuconych miejsc. Naprawione: sekcja 11.6.

### 6.5. Parafrazy, synonimy, polskie znaki, daty (pkt 3)

| zjawisko | obserwacja (po naprawie) |
|---|---|
| parafraza i synonim | Wyszukiwanie wektorowe i hybrydowe: dowód w top-4 dla wszystkich pytań z parafrazą lub synonimem (bike-02, mob-01, mob-02). Leksykalne nie łączy „bez auta” z „niesamochodowy” (mob-01). |
| daty | bike-01: oba cele (2015 i 2020) na pozycjach 1–2. Przed naprawą cel na 2015 r. nie istniał w indeksie. mob-01: właściwy rok w dokumencie, ale Strategia 2050 z innymi latami (2050, 2024) wyżej w rankingu. |
| brak polskich znaków (warianty pochodne) | `vector-ascii`: pozycje dowodu identyczne lub lepsze (mob-02: 3 → 2). `hybrid-ascii`: identyczne lub lepsze (mob-01: 4 → 2). `lexical-ascii`: gorzej dla mob-02 (5 → 7). Konfiguracja `simple` nie usuwa diakrytyków. |
| zepsute kodowanie PDF | `Ŝ` w chunkach: 94 → 0. Dowód bike-01 w pełnym rankingu leksykalnym: 42. miejsce → 1. Łączny efekt D1 (przywrócona linia z celem na 2015 r.) i D2 („podróży” zamiast „podróŜy”). |

### 6.6. Metadane źródła (pkt 5)

W każdym przebiegu 627 zwróconych fragmentów (wszystkie konfiguracje) porównano z
`document_catalog.json`: 0 niezgodności. Zgodność stron: każdy fragment zawierający dowód
obejmuje stronę PDF, na której ten dowód rzeczywiście jest (0 niezgodności).

### 6.7. Rozpoznawanie braku podstaw (pkt 6)

- Abstencja: 0/2 (przed i po). Fałszywe abstencje na pytaniach z odpowiedzią: 0/4.
- `unsupported_topics` jest zawsze puste. Intencja trafia tam tylko wtedy, gdy żaden fragment
  jej nie dopasował, a wyszukiwanie wektorowe zawsze zwraca 20 kandydatów.
- EvidencePack dla bike-03 zawiera ogólne fragmenty Polityki Rowerowej (wprowadzenie, cele,
  dokumenty planistyczne) i fragment Strategii 2050 o tym, jak powstała strategia. Dla mob-03
  zawiera fragmenty Strategii 2050 o strefie centralnej i o wykazie dokumentów oraz obszary
  Polityki mobilności: transport samochodowy, zbiorowy i ochronę środowiska. Ten ostatni
  fragment ma ogólny zapis o strefach zależnych od wpływu pojazdu na środowisko. Żaden fragment
  nie odpowiada na pytanie, a wszystkie trafiłyby do pisma jako dowody. Zapis o strefach
  szczególnie łatwo wziąć za odpowiedź.
- Najwyższe podobieństwo cosinusowe w top-20: przed naprawą zakresy dało się rozdzielić
  (0,607–0,718 wobec 0,582–0,596), po naprawie już nie (0,589–0,717 wobec 0,590–0,604). Przy
  6 pytaniach to tylko obserwacja, ale wskazuje, że sam próg cosinusowy nie wystarczy.
- Kontrola korpusu: wzorce odpowiedzi z `absence_check` nie występują w żadnym chunku indeksu
  (oba indeksy), więc etykiety „bez odpowiedzi” są aktualne.

### 6.8. Hybryda i reranking (pkt 7)

- Hybryda (RRF) nie poprawia rankingu względem samego wyszukiwania wektorowego: MRR dowodu
  0,69 wobec 0,71 po naprawie, 0,45 wobec 0,67 przed. Pomaga dla mob-02 (3 → 2) i bike-02
  (2 → 1), szkodzi dla mob-01 (1 → 4) i, przed naprawą, dla bike-01 (1 → 7). Przyczyna: przy
  zapytaniu leksykalnym równym całemu pytaniu wyszukiwanie OR po wielu słowach zwraca dużo
  szumu, a RRF waży je na równi z wektorowym.
- Zastrzeżenie: w produkcji planer LLM tworzy krótkie `lexical_query` ze słów kluczowych.
  Ewaluacja mierzy tryb fallback, więc wpływ hybrydy z planerem LLM nie jest zmierzony.
- Reranker LLM: **niezmierzony** (sekcja 8).

## 7. Nieudane pytania i prawdopodobne przyczyny

| pytanie | stan | przyczyna |
|---|---|---|
| bike-03 | FAIL (abstencja) | Kontrakt RAG nie ma sygnału braku podstaw opartego na trafności. `unsupported_topics` zależy tylko od tego, czy intencja dopasowała jakikolwiek fragment, a wyszukiwanie wektorowe zawsze coś dopasowuje. Reranker z `allow_fallback=True` przy pustym wyborze też wraca do rankingu hybrydowego. |
| mob-03 | FAIL (abstencja) | Jak wyżej. Dodatkowo pułapka leksykalna: „norma EURO 5” (autobusy) i ogólny kierunek „stref o ograniczonej dostępności” są tematycznie bliskie, ale nie odpowiadają na pytanie o termin i kryteria. |
| bike-01 (przed naprawą) | FAIL → PASS | D1: linia „2015 co najmniej 10%…” była brana za nagłówek i znikała z indeksu. D2 obniżał trafienia leksykalne („podróŜy”). RRF zepchnął jedyny pozostały dowód na 7. miejsce. |
| mob-01 (przechodzi z małym zapasem) | PASS | Dowód na 4. miejscu z 6. Leksykalnie dokument poza top-20 z powodu słownictwa, a Strategia 2050 dominuje. Inaczej sformułowane pytanie łatwo wypchnęłoby dowód z EvidencePack. |

## 8. Nieuruchomione i ograniczenia

- **Reranker z domyślnym modelem `pismak-pllum` niezmierzony.** Pomiar z `pismak-bielik`
  wykonałem później, na indeksie v0.3 (sekcja 11.5).
  - Baseline: przerwany po pierwszym pytaniu. Jedno wywołanie `pismak-pllum` trwało 7 min 51 s
    na CPU, a załadowany model wyczerpał pamięć wirtualną (0,4 GB wolne z 25,4 GB), co
    blokowało uruchamianie innych procesów.
  - Po naprawie: wszystkie 6 wywołań zakończyło się błędem Ollamy `llama-server reported
    out-of-memory during startup`. Harness zapisał błędy jawnie, bez cichego fallbacku, a test
    `test_reranker_configuration_completed` jest FAILED.
  - Pomiar wymaga maszyny z większą ilością pamięci lub GPU albo mniejszego okna kontekstu
    modelu. Celowo nie zmieniałem konfiguracji modelu.
- **Planer LLM niemierzony:** zapytania budowane deterministycznie, tak jak w produkcyjnym
  fallbacku. To założenie zadania (bez wywołań LLM).
- **n = 6** (4 + 2): metryki zbiorcze są gruboziarniste, a różnica 0,02–0,04 to szum.
- **Wyniki dotyczą lokalnego indeksu z 6 dokumentami.** Na świeżym klonie z 2 dokumentami
  dystraktorów jest mniej i ranking będzie inny. W CI uruchamiają się tylko testy zbioru (19 testów, bez bazy).
- Reindeksacja na tej maszynie trwała kilka godzin (embeddingi bge-m3 na CPU przy wysokim
  zużyciu pamięci). To nie wpływa na wyniki, ale ogranicza częstotliwość pomiarów.

## 9. Rekomendacje

**Wdrożone 14.10 w tej kolejności (szczegóły i pomiar w sekcji 11):**

1. **Sygnał braku podstaw w kontrakcie RAG**: `EvidencePack.evidence_status` oparty na ocenie
   rerankera z progiem, bez fallbacku przy poprawnej pustej odpowiedzi modelu.
2. **Usuwanie znaków sterujących** (punktory `\x01`) w `normalize_text`.
3. **Usuwanie powtarzanych nagłówków i stopek stron** w ingestion (efekt uboczny D1).
4. **Pełny EvidencePack:** limit na dokument stosowany podczas wybierania z 20 kandydatów
   (sekcja 11.6).

**Pozostałe (niewdrożone):**

5. **Ranking wobec dokumentów ogólnych:** RRF z równą wagą części leksykalnej pogarsza wynik
   względem samych wektorów w trybie fallback. Warto zmierzyć wagi RRF i tryb z planerem LLM,
   zanim zmieni się ranking.
6. **Leksykalnie:** rozważyć `unaccent` lub polski słownik w `tsvector` (pytania bez znaków
   diakrytycznych, odmiana).
7. **Większy zbiór:** dodać pytania dla 4 lokalnych dokumentów i więcej pytań na dokument, żeby
   metryki zbiorcze miały znaczenie statystyczne. Zasada 3/2/1 dla zakomitowanych PDF-ów jest
   pilnowana automatycznie.

## 10. Jak powtórzyć

```powershell
.venv\Scripts\python.exe -m pytest -q          # cały zestaw (ewaluacja na żywym indeksie pominięta)
$env:PISMAK_RAG_EVAL = "1"; $env:PISMAK_RAG_EVAL_LABEL = "nazwa"
.venv\Scripts\python.exe -m pytest tests/rag_eval -v -rA
# ocena wsparcia i abstencja wymagają rerankera, np.:
# $env:PISMAK_RAG_EVAL_RERANKER_MODEL = "pismak-bielik:latest"   (około 40 min na CPU)
```

Szczegóły formatu zbioru i definicje metryk: `tests/rag_eval/README.md`.

## 11. Wdrożenie rekomendacji 1–3 (14.10)

### 11.1. Sygnał braku podstaw: `evidence_status`

- **Kontrakt:** `EvidencePack.evidence_status` przyjmuje wartości `supported` (reranker ocenił
  kandydatów i wybrał wspierające), `unsupported` (ocenił i nic nie odpowiada na sprawę, albo nic
  nie znaleziono; EvidencePack jest wtedy pusty) lub `unverified` (trafności nikt nie ocenił: brak
  rerankera albo fallback po błędzie modelu). Wartość domyślna `unverified` zachowuje zgodność z
  zapisanymi wcześniej plikami `evidence.json`.
- **Reranker** (`src/Zrodlowyszukiwator/rag_retrieval/reranker.py`):
  - Prompt definiuje skalę ocen: 70–100 bezpośrednie poparcie, 40–69 pośrednie, 0–39 tylko podobny temat.
  - Prompt mówi wprost, że pusta lista `selected` jest poprawną odpowiedzią.
  - Próg `min_score=40`. Oceny niższe niż próg nie trafiają do pakietu.
  - Poprawna pusta odpowiedź modelu jest teraz oceną „brak podstaw”. Wcześniej uruchamiała
    fallback do rankingu hybrydowego, a przy `allow_fallback=False` rzucała wyjątek.
  - Odpowiedź wskazująca wyłącznie nieistniejące `chunk_id` to błąd modelu, a nie ocena.
  - `rerank()` zwraca `RerankOutcome(selected, assessed)`.
- **Pipeline** (`brief_to_evidence/pipeline.py`) ustawia status. **Konsumenci:** kontekst dla
  Pismogeneratora mówi wprost, że fragmenty są nieocenione albo że żaden nie odpowiada na sprawę.
  UI pokazuje odpowiedni komunikat.
- **Dlaczego nie pokrycie słów kluczowych:** pomiar z sekcji 6 pokazuje, że dopasowanie słów nie
  łączy parafraz (mob-01: dokument źródłowy poza top-20 leksykalnie), a progi cosinusowe nie
  rozdzielają pytań. Reguła oparta na słowach odrzucałaby pytania z odpowiedzią, a progu
  dobranego pod 6 pytań nie dałoby się uczciwie uzasadnić.
- **Testy jednostkowe:** 8 nowych i 2 zaktualizowane (reranker: pusta odpowiedź, próg, nieznane
  ID; pipeline: 4 statusy; kontekst dla Pismogeneratora).
- **Ewaluacja:** abstencja to `evidence_status == "unsupported"`. Testy abstencji i fałszywej
  abstencji wymagają rerankera. Nowy test sprawdza, że pipeline bez rerankera nie deklaruje
  `supported`.

### 11.2. Znaki sterujące

`normalize_text` usuwa znaki sterujące C0 (poza `\n` i `\t`). Dotyczyło to 80 punktorów `\x01`
(66 w Polityce Rowerowej, 14 w Polityce mobilności). Po zmianie w żadnym z 6 dokumentów nie ma
znaków sterujących i nie brakuje żadnej linii tekstu.

### 11.3. Nagłówki i stopki stron

Linia jest usuwana od drugiego wystąpienia, gdy jednocześnie:
- zawiera litery;
- dokument ma co najmniej 4 strony;
- linia występuje wyłącznie wśród 3 pierwszych lub 3 ostatnich linii stron;
- pojawia się na co najmniej 25% stron.

Pierwsze wystąpienie zostaje, bo może być tytułem rozdziału. Reguła usunęła dokładnie:

| dokument | usunięte powtórzenia |
|---|---|
| Strategia Wrocław 2050 | „Otwarci na przyszłość – nastawieni na Wrocław” (228), „Kierunki polityki przestrzennej” (92) |
| Standardy dostępności | „Wrocławskie standardy dostępności przestrzeni miejskich” (55) |
| pozostałe 4 dokumenty | nic |

Wiersze tabel powtarzane na kilku stronach i pojedyncze myślniki punktorów zostają. Wynik
ingestion wypisuje usunięte linie w `removed_running_lines`.

### 11.4. Reindeksacja do v0.3 i pomiar

- `INGESTION_VERSION = "0.3"`. Backup indeksu v0.2 jest w `outputs/rag_eval/index_backup_v0.2/`
  (format COPY, instrukcja w `RESTORE.txt`). Backup zrobiłem przez psycopg, bo Docker CLI
  przestał odpowiadać.
- Reindeksacja 6 dokumentów bez `--force` trwała około 25 minut. Liczby chunków po zmianie:
  Rowerowa 28 → 22, Mobilność 46 → 45, Strategia 2050 582 → 439, Standardy 549 → 527,
  Strategia 2030 i Program Tramwajowy bez zmian (239, 178). Kontrola linii w dokumentach
  zakomitowanych: 0 brakujących.
- **Ewaluacja v0.3 bez rerankera** (`outputs/rag_eval/20261014_174045_v0.3/`): 11 PASSED,
  7 SKIPPED (testy wymagające rerankera). Wszystkie 4 pytania z odpowiedzią przechodzą, a
  metadane mają 0 niezgodności w 629 fragmentach.

| konfiguracja | MRR dowodu v0.2 → v0.3 | evidence_hit@6 | precyzja dowodów@6 |
|---|---|---|---|
| `vector` | 0,71 → 0,67 | 1,00 → 1,00 | 0,25 → 0,21 |
| `hybrid` | 0,69 → 0,65 | 1,00 → 1,00 | 0,25 → 0,21 |
| `pipeline` | 0,69 → 0,65 | 1,00 → 1,00 | 0,38 → 0,25 |

  Różnice to przesunięcia o jedną pozycję w dwóch pytaniach (bike-02 wektorowo 2 → 3, mob-02
  hybrydowo 2 → 3). Przy 4 pytaniach to szum. Niższa precyzja `pipeline` wynika z większych
  pakietów (bike-02: 3 → 5 fragmentów, bike-01: 5 → 6) przy tych samych fragmentach z dowodem.
  Zmiany 2 i 3 porządkują tekst indeksu, ale na tym zbiorze pytań nie poprawiają rankingu
  w mierzalny sposób.

### 11.5. Pomiar abstencji z rerankerem (`pismak-bielik`, indeks v0.3)

Domyślny model RAG `pismak-pllum` nie mieści się w pamięci tej maszyny (sekcja 8), więc użyłem
`pismak-bielik:latest`. To model 4,8B z oknem 8192 tokenów, jeden z modeli do wyboru w
aplikacji. Istniejący mechanizm dopasował prompt do okna. Każde z 6 wywołań trwało 6–7 minut
na CPU, cały przebieg 39 minut. Raport: `outputs/rag_eval/20261014_182054_v0.3-reranker-bielik/`.
Wynik testów: 4 FAILED, 14 PASSED.

| pytanie | odpowiedź w korpusie | evidence_status | wybór rerankera | ocena |
|---|---|---|---|---|
| bike-01 | tak | supported | 6 fragmentów, w tym oba cele (2015 i 2020) | poprawnie |
| bike-02 | tak | supported | 5 fragmentów, w tym fragment o Radzie z udziałem NGO | poprawnie |
| mob-01 | tak | supported | 3 fragmenty, wszystkie ze Strategii 2050 (s. 177–183); dowodu z Polityki mobilności brak | dowód utracony |
| mob-02 | tak | **unsupported** | nic | fałszywa abstencja |
| bike-03 | nie | supported | 5 fragmentów (Polityka Rowerowa: cele, Rada; Strategia 2050) | brak abstencji |
| mob-03 | nie | supported | 6 fragmentów (Strategia 2050: strefa centralna; Polityka mobilności: obszary 3, 9, 14) | brak abstencji |

- **Mechanizm działa:** poprawna pusta odpowiedź modelu dała status `unsupported` i pusty
  EvidencePack (mob-02), a status jest widoczny w kontekście Pismogeneratora i w UI.
- **Ocena trafności z tym modelem zawodzi:** abstencja 0/2, fałszywa abstencja 1/4, dowód w
  EvidencePack po rerankingu 2/4. Bez rerankera dowód był w pakiecie w 4/4 przypadkach.
  Model nie odróżnia wsparcia od podobieństwa tematycznego. W obu pytaniach bez odpowiedzi
  dał fragmentom co najmniej 40 punktów (inaczej nie przeszłyby progu), a w mob-02 odrzucił
  fragment z odpowiedzią.
- **Skutek zmiany kontraktu w produkcji:** wcześniej pusty wybór rerankera uruchamiał fallback do
  rankingu hybrydowego, więc w mob-02 do pisma trafiłby fragment z odpowiedzią (3. miejsce
  hybrydy). Teraz pakiet jest pusty ze statusem `unsupported`. Dla pytań spoza korpusu to
  zachowanie właściwe, ale przy słabym modelu oceniającym pojawia się ryzyko fałszywej abstencji.
- Progu i promptu celowo nie stroiłem pod te 6 pytań. Następny krok to mocniejszy model
  oceniający (np. przez OpenRouter albo `pismak-pllum` na GPU) i ten sam pomiar.
- W tym przebiegu harness nie zapisywał ocen ani uzasadnień rerankera. Od teraz zapisuje je
  w `results.json` i w raporcie per pytanie („Wybór `pipeline+reranker`”).

### 11.6. Pełny EvidencePack

- **Zmiana** (`brief_to_evidence/pipeline.py`): limit 3 fragmentów na dokument działa podczas
  wybierania ze wszystkich 20 kandydatów, a nie po obcięciu listy do 6. Dotyczy wyboru bez
  rerankera i wyboru po fallbacku rerankera (status `unverified`). Gdy reranker ocenił
  trafność (status `supported`), pakiet zawiera wyłącznie fragmenty wybrane przez model, nawet
  jeśli jest ich mniej niż 6. Dopisanie nieocenionych fragmentów do pakietu `supported`
  naruszałoby kontrakt z sekcji 11.1.
- **Testy jednostkowe:** 3 nowe (uzupełnianie z dalszych kandydatów, uzupełnianie po fallbacku,
  brak dopisywania do wyboru rerankera). Test uzupełniania nie przechodził na poprzednim kodzie
  (3 fragmenty zamiast 6).
- **Pomiar** na indeksie v0.3 bez rerankera (`outputs/rag_eval/20261014_205227_v0.3-full-pack/`,
  11 PASSED, 7 SKIPPED):

| pytanie | fragmentów w EvidencePack: przed → po | pozycja pierwszego dowodu | fragmentów z dowodem |
|---|---|---|---|
| bike-01 | 6 → 6 | 1 → 1 | 2 → 2 |
| bike-02 | 5 → 6 | 1 → 1 | 1 → 1 |
| bike-03 | 3 → 6 | — | 0 → 0 |
| mob-01 | 4 → 6 | 4 → 4 | 1 → 1 |
| mob-02 | 5 → 6 | 3 → 3 | 1 → 1 |
| mob-03 | 6 → 6 | — | 0 → 0 |

  Pakiet ma teraz zawsze 6 fragmentów. Ranking hybrydowy, pozycje dowodów i liczba fragmentów
  z dowodem są bez zmian. Precyzja dowodów spadła z 0,25 do 0,21, bo te same dowody dzielą
  pakiet z większą liczbą fragmentów kontekstowych. Dla mob-01 dodatkowe miejsca zajęły
  2 fragmenty Polityki Rowerowej, mało związane z pytaniem. Na tym zbiorze zmiana nie dodała
  dowodu. Pomaga wtedy, gdy dowód z innego dokumentu leży na 7. lub dalszej pozycji, a czołówkę
  zajmuje jeden dokument.
