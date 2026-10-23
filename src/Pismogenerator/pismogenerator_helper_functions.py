import re

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack


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

    # Kontrola odwołań [1], [2], ...
    refs = [int(x) for x in re.findall(r"\[(\d+)\]", draft.argumentation_markdown)]
    # Przypis bez listy źródeł nie ma celu, do którego czytelnik mógłby go odnieść.
    if refs and not draft.sources:
        errors.append("Argumentacja zawiera przypisy, ale lista sources jest pusta")

    # Numery przypisów są pozycjami na liście sources; większy numer byłby nierozwiązywalny.
    if refs and max(refs) > len(draft.sources):
        errors.append(
            "Argumentacja odwołuje się do numeru źródła, którego nie ma na liście sources"
        )

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
    request_summary = bracket_verification_markers(draft.request_summary.strip())
    argumentation = bracket_verification_markers(draft.argumentation_markdown.strip())

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
