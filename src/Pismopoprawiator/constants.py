POPRAWIATOR_STAGE = "pismopoprawiator"
# Limity tego etapu celowo różnią się od pozostałych. Przy 1024 tokenach modele rozumujące
# z openrouter/free zużywały cały limit na rozumowanie i zwracały pustą odpowiedź (złoty
# zbiór, 2026-10-23); sama lista poprawek zajmuje zwykle 100-600 tokenów.
POPRAWIATOR_MAX_OUTPUT_TOKENS = 4096
# Miejsce zostawiane na odpowiedź przy wyborze trybu: całe pismo czy po akapitach. Mniejsze
# niż limit odpowiedzi, żeby pismo mieściło się w całości w oknie 8192 tokenów Bielika.
POPRAWIATOR_RESERVED_OUTPUT_TOKENS = 2048
# Limit jednego żądania. Bielik na CPU nie poprawił całego pisma w 90 s, a pojedynczy akapit
# zajmował mu 20-80 s. Czas oczekiwania w kolejce lokalnego modelu się do limitu nie wlicza.
POPRAWIATOR_TIMEOUT_SECONDS = 300.0
# Pierwsza próba i jedno ponowienie: tyle wykonuje LLMClient.generate_structured (naprawa
# schematu albo prośba o krótszą odpowiedź). Przekroczenie czasu nie jest ponawiane.
POPRAWIATOR_MAX_ATTEMPTS = 2
POPRAWIATOR_TEMPERATURE = 0.0

# Strażnicy pojedynczej poprawki.
MAX_ORIGINAL_CHARS = 200
# difflib.SequenceMatcher.ratio() między fragmentem a jego poprawką.
MIN_SIMILARITY = 0.5
# Strażnik całości: dopuszczalna względna zmiana długości poprawianej części pisma.
MAX_LENGTH_CHANGE_RATIO = 0.05

# Liczba poprawek, o którą prosi prompt w jednym wywołaniu; więcej nie zmieści się w limicie.
MAX_CORRECTIONS_PER_CALL = 15
# Dłuższe pismo, albo takie, które nie mieści się w oknie modelu, idzie po akapitach.
WHOLE_LETTER_MAX_CHARS = 12_000
# Tryb po akapitach: najwięcej tekstu edytowalnego w jednym wywołaniu i najwięcej wywołań.
CHUNK_MAX_CHARS = 2_500
MAX_CALLS = 6

CATEGORIES = ("ortografia", "interpunkcja", "gramatyka", "fleksja", "literówka", "spacje")
# Uwagi stylistyczne są tylko sugestiami; kod nigdy ich nie stosuje.
STYLE_CATEGORY = "styl"

# Frazy, które muszą zostać dosłownie, niezależnie od tego, czy stoją w nawiasie.
PROTECTED_PHRASES = ("wnioskuję o", "według obserwacji", "wymaga weryfikacji")
# Słowa przeczące. Liczone są też wszystkie wyrazy zaczynające się od „nie”, więc
# poprawka „nie zgodny” → „niezgodny” nie zmienia liczby przeczeń.
NEGATION_WORDS = (
    "ani",
    "bez",
    "brak",
    "braku",
    "nigdy",
    "nic",
    "niczego",
    "żaden",
    "żadna",
    "żadne",
    "żadnego",
    "żadnej",
    "żadnych",
    "żadnym",
)
# Początki wyrazów wyrażających możliwość, powinność lub prośbę. Liczba wyrazów z każdym
# z nich musi zostać taka sama.
MODAL_STEMS = (
    "moż",
    "mog",
    "mógł",
    "powin",
    "należ",
    "mus",
    "trzeba",
    "wolno",
    "konieczn",
    "niezbędn",
    "potrzeb",
    "warto",
    "prawdopodob",
    "ewentualn",
    "chyba",
    "raczej",
    "wydaje",
    "proszę",
    "postul",
    "wnios",
    "żąda",
    "domaga",
)
# Skróty, po których kropka nie kończy zdania, więc następny wyraz wielką literą to nazwa.
ABBREVIATIONS = (
    "al",
    "art",
    "dr",
    "godz",
    "im",
    "inż",
    "ks",
    "lit",
    "mgr",
    "nr",
    "np",
    "ok",
    "os",
    "pkt",
    "pl",
    "por",
    "prof",
    "r",
    "rozdz",
    "św",
    "tel",
    "tj",
    "tzn",
    "tzw",
    "ul",
    "ust",
    "zob",
)
