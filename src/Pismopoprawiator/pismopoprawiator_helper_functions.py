from __future__ import annotations

import difflib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass

from src.Pismogenerator.pismogenerator_helper_functions import (
    PHASE0_CLOSING,
    PHASE0_DOCUMENT_TYPE,
    PHASE0_OPENING_FORMULA,
    PHASE0_SIGNATURE,
    RECIPIENT_LABEL,
    SENDER_LABEL,
    SOURCES_HEADING,
)
from src.Pismopoprawiator.constants import (
    ABBREVIATIONS,
    CATEGORIES,
    MAX_LENGTH_CHANGE_RATIO,
    MAX_ORIGINAL_CHARS,
    MIN_SIMILARITY,
    MODAL_STEMS,
    NEGATION_WORDS,
    PROTECTED_PHRASES,
    STYLE_CATEGORY,
)
from src.Pismopoprawiator.structures import (
    Correction,
    CorrectionResult,
    Paragraph,
    RejectedCorrection,
)

LOGGER = logging.getLogger("pismak")

# ________________________________
# podział pisma

# Lista źródeł i sekcja kontrolna zaczynają się od tej linii; renderer dodaje ją na końcu.
_TAIL_MARKER = "\n---\n\n" + SOURCES_HEADING
# Akapity rozdziela co najmniej jedna pusta linia; separator zostaje zachowany co do znaku.
_PARAGRAPH_SEPARATOR = re.compile(r"(\n(?:[ \t]*\n)+)")
_READ_ONLY_BLOCKS = {f"# {PHASE0_DOCUMENT_TYPE}", SENDER_LABEL, PHASE0_CLOSING, PHASE0_SIGNATURE}


def split_letter(letter: str) -> tuple[str, str]:
    """Split a letter into the part sent for correction and the sources and control tail."""
    index = letter.rfind(_TAIL_MARKER)
    if index == -1:
        return letter, ""
    # Białe znaki przed kreską należą do niezmiennego końca, więc treść kończy się podpisem.
    cut = len(letter[:index].rstrip())
    return letter[:cut], letter[cut:]


def is_read_only(text: str) -> bool:
    """Return whether a block was written by the renderer rather than the model."""
    block = text.strip()
    return block in _READ_ONLY_BLOCKS or block.startswith((RECIPIENT_LABEL, SENDER_LABEL))


def split_paragraphs(body: str) -> tuple[list[str], list[Paragraph]]:
    """Return the body split into text parts and separators, and its numbered paragraphs."""
    parts = _PARAGRAPH_SEPARATOR.split(body)
    paragraphs: list[Paragraph] = []
    # Parzyste pozycje to tekst, nieparzyste to separatory.
    for part_index in range(0, len(parts), 2):
        text = parts[part_index]
        if not text.strip():
            continue
        paragraphs.append(
            Paragraph(
                id=f"A{len(paragraphs) + 1}",
                text=text,
                editable=not is_read_only(text),
                part_index=part_index,
            )
        )
    return parts, paragraphs


def join_paragraphs(parts: list[str], paragraphs: list[Paragraph]) -> str:
    """Put the paragraphs back between their original separators."""
    joined = list(parts)
    for paragraph in paragraphs:
        joined[paragraph.part_index] = paragraph.text
    return "".join(joined)


# ________________________________
# elementy chronione

_BRACKETED = re.compile(r"\[[^\[\]\n]*\]")
# Kropka albo przecinek na końcu należą do zdania, a nie do adresu.
_URL = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]]*[^\s<>()\[\].,;:!?]", re.IGNORECASE)
_QUOTE = re.compile(r"„[^„”\"\n]*[”\"]|“[^“”\n]*”|\"[^\"\n]*\"|«[^«»\n]*»")
_PHRASE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(r"\s+".join(map(re.escape, phrase.split())) for phrase in PROTECTED_PHRASES)
    + r")(?!\w)",
    re.IGNORECASE,
)
_FORMULA = re.compile(re.escape(PHASE0_OPENING_FORMULA))
_RESOLUTION = re.compile(r"(?<![\w/])[IVXLCDM]+/\d+(?:/\d+)*")
_NUMBER = re.compile(r"\d+(?:[.,:/\-–]\d+)*")
_LEGAL = re.compile(r"(?:art\.|§|ust\.|pkt\.?|lit\.|rozdz\.)\s*\d+[a-z]?", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+")
_ABBREVIATION_BEFORE = re.compile(r"([^\W\d_]+)\.$")
# Znaki, które mogą stać między końcem zdania a pierwszym wyrazem następnego.
_SENTENCE_OPENERS = " \t*_#>„\"«“'([-–—"
_STRUCTURE_CHARS = "\n*#_`>|\\"

# Rodzaje elementów w kolejności, w jakiej strażnik podaje pierwszą różnicę.
SIGNATURE_LABELS = {
    "numbers": "liczby i daty",
    "resolutions": "numery uchwał",
    "legal": "oznaczenia przepisów",
    "brackets": "oznaczenia w nawiasach kwadratowych",
    "urls": "adresy URL",
    "quotes": "cytaty",
    "phrases": "frazy chronione",
    "formula": "formułę „wnioskuję o”",
    "names": "nazwy własne",
    "negations": "przeczenia",
    "modality": "modalność",
    "structure": "formatowanie",
}
_SPAN_PATTERNS = (_BRACKETED, _URL, _QUOTE, _PHRASE, _FORMULA, _RESOLUTION)
_TOKEN_PATTERNS = (
    ("numbers", _NUMBER, lambda token: token),
    ("resolutions", _RESOLUTION, lambda token: token),
    ("legal", _LEGAL, lambda token: re.sub(r"\s+", "", token.lower())),
    ("brackets", _BRACKETED, lambda token: token),
    ("urls", _URL, lambda token: token),
    ("quotes", _QUOTE, lambda token: token),
    ("phrases", _PHRASE, lambda token: " ".join(token.lower().split())),
    ("formula", _FORMULA, lambda token: token),
)

Signature = dict[str, Counter[str]]


def protected_spans(text: str) -> list[tuple[int, int, str]]:
    """Return the ranges a correction may contain whole but must never cut through."""
    return [
        (match.start(), match.end(), match.group(0))
        for pattern in _SPAN_PATTERNS
        for match in pattern.finditer(text)
    ]


def starts_sentence(text: str, index: int) -> bool:
    """Return whether the word at index opens a sentence, line or list item."""
    before = text[:index].rstrip(_SENTENCE_OPENERS)
    if not before or before[-1] in "\n\\!?…:;":
        return True
    if before[-1] != ".":
        return False
    word = _ABBREVIATION_BEFORE.search(before)
    if word is None:
        # Kropka po liczbie albo nawiasie, np. „1.” na liście lub „[1].” na końcu zdania.
        return True
    token = word.group(1)
    # „ul. Kazimierza” i „J. Kowalski”: kropka skrótu nie kończy zdania.
    return not (token.lower() in ABBREVIATIONS or (len(token) == 1 and token.isupper()))


def content_signature(text: str) -> Signature:
    """Count every element of a paragraph that a language correction must leave as it is."""
    signature: Signature = {kind: Counter() for kind in SIGNATURE_LABELS}
    for kind, pattern, normalize in _TOKEN_PATTERNS:
        for match in pattern.finditer(text):
            signature[kind][normalize(match.group(0))] += 1
    for match in _WORD.finditer(text):
        word = match.group(0)
        lower = word.lower()
        if lower.startswith("nie") or lower in NEGATION_WORDS:
            signature["negations"]["przeczenia"] += 1
        stem = next((stem for stem in MODAL_STEMS if lower.startswith(stem)), None)
        if stem is not None:
            signature["modality"][stem] += 1
        # Wielka litera poza początkiem zdania oznacza nazwę własną albo skrótowiec.
        if word[0].isupper() and not starts_sentence(text, match.start()):
            signature["names"][word] += 1
    for char in _STRUCTURE_CHARS:
        if count := text.count(char):
            signature["structure"]["nowa linia" if char == "\n" else char] = count
    return signature


def merge_signatures(signatures: list[Signature]) -> Signature:
    """Add up paragraph signatures; sentence starts are decided inside each paragraph."""
    merged: Signature = {kind: Counter() for kind in SIGNATURE_LABELS}
    for signature in signatures:
        for kind, counter in signature.items():
            merged[kind].update(counter)
    return merged


def _quoted_list(values: list[str]) -> str:
    shown = ", ".join(f"„{value}”" for value in values[:3])
    return shown + (f" i {len(values) - 3} więcej" if len(values) > 3 else "")


def signature_difference(before: Signature, after: Signature) -> str | None:
    """Describe the first protected element that differs, or return None when none does."""
    for kind, label in SIGNATURE_LABELS.items():
        if before[kind] == after[kind]:
            continue
        if kind == "negations":
            # Przeczenia są tylko liczone, więc różnica to zmiana ich liczby.
            return f"zmienia {label}: {sum(before[kind].values())} → {sum(after[kind].values())}"
        removed = list((before[kind] - after[kind]).elements())
        added = list((after[kind] - before[kind]).elements())
        changes = []
        if removed:
            changes.append("usuwa " + _quoted_list(removed))
        if added:
            changes.append("dodaje " + _quoted_list(added))
        return f"zmienia {label}: " + "; ".join(changes)
    return None


# ________________________________
# strażnicy i stosowanie poprawek


@dataclass(frozen=True)
class AcceptedCorrection:
    """A correction that passed every guard, with its place in the original paragraph."""

    correction: Correction
    position: int
    start: int
    end: int


def count_occurrences(text: str, fragment: str) -> int:
    """Count occurrences of a fragment, overlapping ones included."""
    count, index = 0, text.find(fragment)
    while index != -1:
        count += 1
        index = text.find(fragment, index + 1)
    return count


def _widen(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend a range by one word on each side, staying inside the text."""
    while start > 0 and text[start - 1].isspace():
        start -= 1
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and text[end].isspace():
        end += 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def narrow_correction(correction: Correction, text: str) -> Correction:
    """Cut a correction down to the changed words, with just enough context to be unique."""
    # Małe modele kopiują całe zdania albo akapity. Odcięcie wspólnego początku i końca nie
    # zmienia wyniku w akapicie, a limit długości i podobieństwo dotyczą wtedy samej zmiany.
    original, fixed = correction.oryginal, correction.poprawka
    if not original or original == fixed or count_occurrences(text, original) != 1:
        return correction
    base = text.find(original)
    shorter = min(len(original), len(fixed))
    prefix = 0
    while prefix < shorter and original[prefix] == fixed[prefix]:
        prefix += 1
    suffix = 0
    while suffix < shorter - prefix and original[-1 - suffix] == fixed[-1 - suffix]:
        suffix += 1
    start, end = prefix, len(original) - suffix
    # Zakres powiększa się tylko o wspólne części, więc w poprawce odpowiada mu ten sam kawałek.
    while start > 0 and original[start - 1].isalnum():
        start -= 1
    while end < len(original) and original[end].isalnum():
        end += 1
    # Zmiana samych spacji jest czytelna tylko z sąsiednim wyrazem („trwa  ponad”).
    if start < end and original[start].isspace():
        start = _widen(original, start, end)[0]
    if start < end and original[end - 1].isspace():
        end = _widen(original, start, end)[1]

    spans = [(left - base, right - base) for left, right, _ in protected_spans(text)]

    def cover_spans(start: int, end: int) -> tuple[int, int] | None:
        """Grow the range until no protected element is cut; None when that leaves the fragment."""
        changed = True
        while changed:
            changed = False
            for left, right in spans:
                if start < right and left < end and not (start <= left and right <= end):
                    if left < 0 or right > len(original):
                        return None
                    start, end, changed = min(start, left), max(end, right), True
        return start, end

    while True:
        # Element chroniony musi być we fragmencie w całości albo wcale. Gdy już fragment
        # modelu przecinał taki element, zostaje bez zmian i ocenią go strażnicy.
        covered = cover_spans(start, end)
        if covered is None:
            return correction
        start, end = covered
        fragment = original[start:end].strip()
        if fragment and count_occurrences(text, fragment) == 1:
            break
        if start == 0 and end == len(original):
            return correction
        start, end = _widen(original, start, end)

    # Spacje na brzegach zostają poza fragmentem, jeżeli nie należą do zmiany.
    while start < prefix and original[start].isspace():
        start += 1
    while end > len(original) - suffix and original[end - 1].isspace():
        end -= 1
    delta = len(fixed) - len(original)
    return correction.model_copy(
        update={"oryginal": original[start:end], "poprawka": fixed[start : end + delta]}
    )


def check_correction(
    correction: Correction,
    paragraph: Paragraph | None,
    editable_ids: set[str],
) -> tuple[int | None, str | None]:
    """Return where the correction applies, or why it is rejected."""
    if correction.kategoria not in CATEGORIES:
        return None, f"nieznana kategoria „{correction.kategoria}”"
    if paragraph is None:
        return None, f"nie ma akapitu {correction.akapit}"
    if not paragraph.editable:
        return None, f"akapit {paragraph.id} jest tylko do odczytu"
    if paragraph.id not in editable_ids:
        return None, f"akapit {paragraph.id} nie był edytowalny w tym wywołaniu"

    original, fixed = correction.oryginal, correction.poprawka
    if not original:
        return None, "pusty fragment oryginalny"
    if len(original) > MAX_ORIGINAL_CHARS:
        return None, f"fragment ma {len(original)} znaków (limit {MAX_ORIGINAL_CHARS})"
    if original == fixed:
        return None, "poprawka nie różni się od oryginału"
    occurrences = count_occurrences(paragraph.text, original)
    if occurrences == 0:
        return None, f"fragmentu nie ma w akapicie {paragraph.id}"
    if occurrences > 1:
        return None, f"fragment występuje w akapicie {paragraph.id} {occurrences} razy"

    start = paragraph.text.find(original)
    end = start + len(original)
    for left, right, value in protected_spans(paragraph.text):
        if start < right and left < end and not (start <= left and right <= end):
            return None, f"fragment przecina chroniony element „{value}”"

    similarity = difflib.SequenceMatcher(None, original, fixed, autojunk=False).ratio()
    if similarity < MIN_SIMILARITY:
        return None, f"poprawka zbyt odległa od oryginału (podobieństwo {similarity:.2f})"

    # Porównanie całego akapitu przed i po zmianie: o początku zdania decyduje kontekst.
    corrected = paragraph.text[:start] + fixed + paragraph.text[end:]
    difference = signature_difference(
        content_signature(paragraph.text), content_signature(corrected)
    )
    if difference:
        return None, difference
    return start, None


def review_corrections(
    paragraphs: list[Paragraph],
    corrections: list[Correction],
    editable_ids: set[str],
) -> tuple[list[AcceptedCorrection], list[RejectedCorrection], list[Correction]]:
    """Sort the model's corrections into accepted, rejected and style-only suggestions."""
    positions = {paragraph.id: index for index, paragraph in enumerate(paragraphs)}
    accepted: list[AcceptedCorrection] = []
    rejected: list[RejectedCorrection] = []
    suggestions: list[Correction] = []
    for correction in corrections:
        position = positions.get(correction.akapit)
        paragraph = paragraphs[position] if position is not None else None
        if paragraph is not None:
            correction = narrow_correction(correction, paragraph.text)
        if correction.kategoria == STYLE_CATEGORY:
            suggestions.append(correction)
            continue
        start, reason = check_correction(correction, paragraph, editable_ids)
        if reason is not None or start is None or position is None:
            rejected.append(RejectedCorrection(correction=correction, reason=reason or "-"))
            continue
        accepted.append(
            AcceptedCorrection(correction, position, start, start + len(correction.oryginal))
        )
    return accepted, rejected, suggestions


def apply_corrections(
    paragraphs: list[Paragraph],
    accepted: list[AcceptedCorrection],
) -> tuple[list[Paragraph], list[Correction], list[RejectedCorrection]]:
    """Apply non-overlapping corrections from the end of the text and return the new paragraphs."""
    kept: list[AcceptedCorrection] = []
    rejected: list[RejectedCorrection] = []
    for item in sorted(accepted, key=lambda item: (item.position, item.start, item.end)):
        previous = kept[-1] if kept and kept[-1].position == item.position else None
        if previous is not None and item.start < previous.end:
            repeated = previous.correction.model_dump() == item.correction.model_dump()
            reason = (
                "powtórzona poprawka"
                if repeated
                else f"nakłada się na inną poprawkę w akapicie {item.correction.akapit}"
            )
            rejected.append(RejectedCorrection(correction=item.correction, reason=reason))
            continue
        kept.append(item)

    texts = [paragraph.text for paragraph in paragraphs]
    # Od końca tekstu: zmiana długości nie przesuwa miejsc poprawek, które czekają w kolejce.
    for item in reversed(kept):
        text = texts[item.position]
        texts[item.position] = text[: item.start] + item.correction.poprawka + text[item.end :]
    corrected = [
        paragraph.model_copy(update={"text": text})
        for paragraph, text in zip(paragraphs, texts, strict=True)
    ]
    return corrected, [item.correction for item in kept], rejected


def content_violations(original_letter: str, corrected_letter: str) -> list[str]:
    """List every way the corrected letter differs from the original beyond language fixes."""
    original_body, original_tail = split_letter(original_letter)
    corrected_body, corrected_tail = split_letter(corrected_letter)
    violations: list[str] = []
    if corrected_tail != original_tail:
        violations.append("zmieniono listę źródeł albo sekcję kontrolną")

    _, before = split_paragraphs(original_body)
    _, after = split_paragraphs(corrected_body)
    if len(before) != len(after):
        violations.append(f"liczba akapitów zmieniła się z {len(before)} na {len(after)}")
    else:
        for old, new in zip(before, after, strict=True):
            if not old.editable and old.text != new.text:
                violations.append(f"zmieniono akapit tylko do odczytu {old.id}")

    change = abs(len(corrected_body) - len(original_body)) / max(1, len(original_body))
    if change > MAX_LENGTH_CHANGE_RATIO:
        violations.append(
            f"długość treści zmieniła się o {change:.1%} (limit {MAX_LENGTH_CHANGE_RATIO:.0%})"
        )

    difference = signature_difference(
        merge_signatures([content_signature(paragraph.text) for paragraph in before]),
        merge_signatures([content_signature(paragraph.text) for paragraph in after]),
    )
    if difference:
        violations.append(difference)
    return violations


# ________________________________
# logi


def quoted(text: str | None) -> str:
    """Quote a value for a single log line, keeping Polish characters readable."""
    return json.dumps(text or "", ensure_ascii=False)


def log_result(result: CorrectionResult, session_id: str | None) -> None:
    """Log every correction of a run, applied or not, and a summary line."""
    extra = {"session_id": session_id}
    session = session_id or "-"
    for correction in result.applied:
        LOGGER.info(
            "correction_applied session_id=%s akapit=%s kategoria=%s oryginal=%s poprawka=%s "
            "uzasadnienie=%s",
            session,
            correction.akapit,
            correction.kategoria,
            quoted(correction.oryginal),
            quoted(correction.poprawka),
            quoted(correction.uzasadnienie),
            extra=extra,
        )
    for item in result.rejected:
        correction = item.correction
        LOGGER.info(
            "correction_rejected session_id=%s akapit=%s kategoria=%s oryginal=%s poprawka=%s "
            "uzasadnienie=%s powod=%s",
            session,
            correction.akapit,
            correction.kategoria,
            quoted(correction.oryginal),
            quoted(correction.poprawka),
            quoted(correction.uzasadnienie),
            quoted(item.reason),
            extra=extra,
        )
    for correction in result.suggestions:
        LOGGER.info(
            "correction_suggestion session_id=%s akapit=%s kategoria=%s oryginal=%s "
            "poprawka=%s uzasadnienie=%s",
            session,
            correction.akapit,
            correction.kategoria,
            quoted(correction.oryginal),
            quoted(correction.poprawka),
            quoted(correction.uzasadnienie),
            extra=extra,
        )
    LOGGER.info(
        "correction_finished session_id=%s status=%s mode=%s calls=%d applied=%d rejected=%d "
        "suggestions=%d provider=%s model=%s context_tokens=%s elapsed_seconds=%.3f errors=%s",
        session,
        result.status,
        result.mode or "-",
        result.calls,
        len(result.applied),
        len(result.rejected),
        len(result.suggestions),
        result.provider or "-",
        result.model or "-",
        result.context_tokens or "-",
        result.elapsed_seconds,
        quoted("; ".join(result.errors)),
        extra=extra,
    )
    # Pełna treść pisma tylko na poziomie DEBUG; domyślny poziom logu sprawy to INFO.
    LOGGER.debug(
        "correction_text session_id=%s original=%s corrected=%s",
        session,
        quoted(result.original_text),
        quoted(result.corrected_text),
        extra=extra,
    )
