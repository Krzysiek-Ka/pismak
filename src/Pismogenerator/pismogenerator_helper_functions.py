import re

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack

# Identyfikator źródła w nawiasach kwadratowych zamiast numeru przypisu: [E1], [U2].
_BRACKETED_SOURCE_ID = re.compile(r"\[([EU]\d+)\]")
# Dosłowny „\” na końcu wiersza albo tekstu (twarde łamanie wiersza w Markdownie).
_LINE_END_BACKSLASH = re.compile(r"[ \t]*\\[ \t]*(\n|\Z)")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_user_evidence(values: list[str] | None) -> list[str]:
    """Return the user's materials as numbered for U1, U2…: trimmed, without blank entries."""
    return [str(value).strip() for value in values or [] if str(value).strip()]


def validate_draft(
    draft: DraftLetter,
    case_brief: CaseBrief,
    evidence_pack: EvidencePack,
    user_evidence: list[str],
) -> list[str]:
    """Validate a draft letter against the Phase 0 constraints for the app."""
    errors: list[str] = []

    # Pusty adresat nie jest błędem: renderer wpisze w jego miejsce znacznik do uzupełnienia.

    # Nadawcę, formułę otwierającą i podpis składa deterministycznie renderer w Phase 0.
    if draft.sender.strip():
        errors.append("Pole nadawcy powinno pozostać puste w Phase 0")

    summary = draft.request_summary.strip()

    # Podsumowanie jest używane w stałej formule wnioskującej, dlatego nie może być puste.
    if not summary:
        errors.append("Brak krótkiego podsumowania wniosku")

    # Renderer dokleja przed podsumowaniem „wnioskuję o”; początkowe „o” dawałoby „o o…”.
    if summary.lower().startswith("o "):
        errors.append(
            "request_summary nie powinno zaczynać się od 'o' — renderer dodaje 'wnioskuję o'"
        )

    # Phase 0 nie ustala podstaw prawnych; zakazuje tego prompt Pismogeneratora. Kontrola
    # zwrotów jest wyłączona decyzją właściciela (2026-10-18): szukanie podciągów („art.”)
    # odrzucało poprawne pisma („na start.”). Zwrot wpisany na listę znów odrzuci projekt.
    forbidden_legal_phrases: list[str] = []
    combined_main_text = (draft.request_summary + "\n" + draft.argumentation_markdown).lower()

    for phrase in forbidden_legal_phrases:
        if phrase in combined_main_text:
            errors.append(f"Phase 0 nie powinno dodawać podstawy prawnej: {phrase!r}")

    # Bardzo krótki tekst zwykle oznacza, że model pominął uzasadnienie wniosku.
    if len(draft.argumentation_markdown.strip()) < 100:
        errors.append("Argumentacja projektu jest podejrzanie krótka")

    # Dozwolone są wyłącznie identyfikatory ze znalezionych dowodów albo materiałów użytkownika.
    allowed_source_ids = {evidence.evidence_id for evidence in evidence_pack.evidence}
    allowed_source_ids.update(f"U{index}" for index, _ in enumerate(user_evidence, start=1))
    illegal_sources = [source for source in draft.sources if source not in allowed_source_ids]
    if illegal_sources:
        errors.append("Model dodał niedozwolone źródła: " + "; ".join(illegal_sources))

    # Kontrola odwołań [1], [2], ... w podsumowaniu wniosku i w argumentacji.
    body = draft.request_summary + "\n" + draft.argumentation_markdown
    refs = [int(x) for x in re.findall(r"\[(\d+)\]", body)]
    # Przypis bez listy źródeł nie ma celu, do którego czytelnik mógłby go odnieść.
    if refs and not draft.sources:
        errors.append("Argumentacja zawiera przypisy, ale lista sources jest pusta")

    # Numery przypisów są pozycjami na liście sources liczonymi od 1; inny numer byłby
    # nierozwiązywalny.
    if any(not 1 <= ref <= len(draft.sources) for ref in refs):
        errors.append(
            "Argumentacja odwołuje się do numeru źródła, którego nie ma na liście sources"
        )

    # [E1] albo [U1] zostaje w tekście po normalize_draft tylko wtedy, gdy nie jest dozwolony.
    raw_ids = sorted({f"[{source_id}]" for source_id in _BRACKETED_SOURCE_ID.findall(body)})
    if raw_ids:
        errors.append("Przypisy wskazują źródła spoza dozwolonych: " + ", ".join(raw_ids))

    return errors


# ________________________________
# render listu  letter.md

PHASE0_OPENING_FORMULA = "Jako wieloletni mieszkaniec naszego kochanego miasta, wnioskuję o"
PHASE0_DOCUMENT_TYPE = "WNIOSEK"
# Stałe elementy układu pisma. Pismopoprawiator rozpoznaje po nich akapity tylko do odczytu
# i część pisma, której nie przekazuje modelowi.
RECIPIENT_LABEL = "**Adresat:**"
RECIPIENT_PLACEHOLDER = "[DO UZUPEŁNIENIA: adresat]"
SENDER_LABEL = "**Nadawca:**"
PHASE0_CLOSING = "Z poważaniem,"
PHASE0_SIGNATURE = "[imię i nazwisko]"
SOURCES_HEADING = "## Źródła"
CONTROL_SECTION_HEADING = "## Sekcja kontrolna"

# Uwaga dla użytkownika, tak jak [DO UZUPEŁNIENIA: …], zawsze stoi w nawiasach kwadratowych.
VERIFICATION_PHRASE = "wymaga weryfikacji"
VERIFICATION_MARKER = f"[{VERIFICATION_PHRASE}]"
_VERIFICATION_NOTE = re.compile(
    r"\(\s*(wymaga\s+weryfikacji)\s*\)|(?<!\w)(wymaga\s+weryfikacji)(?!\w)", re.IGNORECASE
)
_SQUARE_BRACKETS = re.compile(r"\[[^\[\]\n]*\]")
_NEGATION_BEFORE = re.compile(r"(?<!\w)nie\s+$", re.IGNORECASE)


def bracket_verification_markers(text: str) -> str:
    """Put every "wymaga weryfikacji" note that is not already bracketed into [ ]."""
    bracketed = [match.span() for match in _SQUARE_BRACKETS.finditer(text)]

    def bracket(match: re.Match[str]) -> str:
        start, end = match.span()
        if any(left <= start and end <= right for left, right in bracketed):
            return match.group(0)
        # „nie wymaga weryfikacji” w nawiasie czytałoby się jak znacznik do sprawdzenia.
        if match.group(2) and _NEGATION_BEFORE.search(text[:start]):
            return match.group(0)
        return f"[{match.group(1) or match.group(2)}]"

    return _VERIFICATION_NOTE.sub(bracket, text)


# Model zaczyna czasem podsumowanie od zwrotu, który renderer już dokleja („wnioskuję o
# wnioskuję o…”), albo od wielkiej litery („wnioskuję o Zwołanie…”).
_REQUEST_PREFIX = re.compile(r"^(?:wnioskuj[eę]|wnosz[eę])\s+o\s+", re.IGNORECASE)
# Końcówki rzeczowników, które zwykle dopełniają „wnioskuję o”: „zwołanie”, „podjęcie”,
# „zgodę”. Nazwa własna na początku podsumowania zostaje z wielkiej litery.
_REQUEST_NOUN_ENDINGS = ("nie", "cie", "ę")
_CLOSING_LINE = re.compile(
    r"^(?:z poważaniem|z wyrazami szacunku|łączę wyrazy szacunku|z pozdrowieniami|pozdrawiam)"
    r"[,.!]?$",
    re.IGNORECASE,
)
_PLACEHOLDER_LINE = re.compile(r"^\[[^\[\]\n]*\]$")


def normalize_request_summary(summary: str) -> str:
    """Make the summary a lowercase complement of the fixed "wnioskuję o" formula."""
    text = summary.strip()
    while match := _REQUEST_PREFIX.match(text):
        text = text[match.end() :]
    first_word = text.split(maxsplit=1)[0] if text else ""
    letters = first_word.rstrip(",.;:")
    if (
        letters[:1].isupper()
        and letters[1:] == letters[1:].lower()
        and letters.lower().endswith(_REQUEST_NOUN_ENDINGS)
    ):
        text = text[0].lower() + text[1:]
    return text


def unescape_newlines(text: str) -> str:
    """Turn the literal "\\n" a model sometimes writes instead of a line break into one."""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n")


def normalize_draft(draft: DraftLetter, allowed_source_ids: list[str]) -> DraftLetter:
    """
    Turn [E1]-style citations into footnote numbers and drop literal backslashes.

    Bielik pisał przypisy jako [E1] / [U1] zamiast [1], a nowe linie jako dosłowne "\\n" albo
    twarde łamanie wiersza ("\\" na końcu wiersza) — renderer numeruje tylko listę sources,
    więc czytelnik nie dopasowałby [E1] do pozycji listy (złoty zbiór, 2026-10-24).
    """
    sources = list(draft.sources)

    def footnote(match: re.Match[str]) -> str:
        source_id = match.group(1)
        if source_id not in sources:
            # Niedozwolony identyfikator zostaje w tekście i zgłasza go validate_draft.
            if source_id not in allowed_source_ids:
                return match.group(0)
            sources.append(source_id)
        return f"[{sources.index(source_id) + 1}]"

    def clean(text: str) -> str:
        text = unescape_newlines(text)
        # Twarde łamanie staje się nowym akapitem: bez niego Markdown sklejałby wiersze.
        text = _LINE_END_BACKSLASH.sub(lambda m: "\n\n" if m.group(1) else "", text)
        text = _EXTRA_BLANK_LINES.sub("\n\n", text)
        return _BRACKETED_SOURCE_ID.sub(footnote, text)

    request_summary = clean(draft.request_summary)
    argumentation = clean(draft.argumentation_markdown)
    return draft.model_copy(
        update={
            "request_summary": request_summary,
            "argumentation_markdown": argumentation,
            "sources": sources,
        }
    )


def strip_closing(argumentation: str) -> str:
    """Drop a closing formula and signature placeholder the model added at the end."""
    # Renderer sam dodaje „Z poważaniem,” i podpis, więc pożegnanie modelu dubluje się.
    lines = argumentation.rstrip().split("\n")

    def cleaned(index: int) -> str:
        return lines[index].strip().rstrip("\\").strip()

    def last_filled(stop: int) -> int:
        index = stop - 1
        while index >= 0 and not cleaned(index):
            index -= 1
        return index

    last = last_filled(len(lines))
    if last >= 0 and _PLACEHOLDER_LINE.match(cleaned(last)):
        last = last_filled(last)
    if last < 0 or not _CLOSING_LINE.match(cleaned(last)):
        return argumentation
    kept = lines[: last_filled(last) + 1]
    # Twarde łamanie wiersza („\” na końcu) przed usuniętym pożegnaniem nie ma już czego łamać.
    if kept:
        kept[-1] = kept[-1].rstrip().rstrip("\\").rstrip()
    return "\n".join(kept)


def render_list(items: list[str], empty_text: str = "Brak.") -> str:
    """Render a list as a markdown-style bullet list."""
    if not items:
        return f"- {empty_text}"
    return "\n".join(f"- {item}" for item in items)


def render_sources(
    sources: list[str],
    evidence_pack: EvidencePack,
    user_evidence: list[str],
) -> str:
    """Render source references with numbered citations."""
    if not sources:
        return "Brak źródeł wskazanych przez użytkownika."
    evidence_by_id = {e.evidence_id: e for e in evidence_pack.evidence}
    rendered = []
    for index, source_id in enumerate(sources, start=1):
        if source_id in evidence_by_id:
            evidence = evidence_by_id[source_id]
            pages = (
                str(evidence.page_start)
                if evidence.page_start == evidence.page_end
                else f"{evidence.page_start}-{evidence.page_end}"
            )
            suffix = f", s. {pages}"
            if evidence.source_url:
                suffix += f", {evidence.source_url}"
            rendered.append(f"[{index}] {source_id}: {evidence.document_title}{suffix}")
        elif source_id.startswith("U") and source_id[1:].isdigit():
            user_index = int(source_id[1:]) - 1
            if 0 <= user_index < len(user_evidence):
                rendered.append(f"[{index}] {source_id}: {user_evidence[user_index]}")
            else:
                rendered.append(f"[{index}] {source_id}")
        else:
            rendered.append(f"[{index}] {source_id}")
    # Lista punktowana: w Markdownie wiersze rozdzielone samym \n zlewają się w jeden akapit.
    return "\n".join(f"- {line}" for line in rendered)


def render_letter(
    draft: DraftLetter,
    evidence_pack: EvidencePack,
    user_evidence: list[str],
) -> str:
    """Turn a DraftLetter model into a final markdown letter body."""
    recipient = draft.recipient.strip() or RECIPIENT_PLACEHOLDER
    request_summary = bracket_verification_markers(
        normalize_request_summary(unescape_newlines(draft.request_summary))
    )
    argumentation = bracket_verification_markers(
        strip_closing(unescape_newlines(draft.argumentation_markdown).strip())
    )

    # Pusta linia między polami nagłówka: bez niej Markdown pokazuje je w jednej linii.
    letter = f"""{RECIPIENT_LABEL} {recipient}

{SENDER_LABEL}

{PHASE0_OPENING_FORMULA} {request_summary}

# {PHASE0_DOCUMENT_TYPE}

{argumentation}

{PHASE0_CLOSING}

{PHASE0_SIGNATURE}

---

{SOURCES_HEADING}

{render_sources(draft.sources, evidence_pack, user_evidence)}

---

{CONTROL_SECTION_HEADING}

### Informacje wymagające weryfikacji przed wysłaniem

{render_list(draft.claims_requiring_verification)}

### Brakujące pola

{render_list(draft.missing_fields)}

### Założenia redakcyjne modelu

{render_list(draft.assumptions_used)}
"""
    return letter.strip()
