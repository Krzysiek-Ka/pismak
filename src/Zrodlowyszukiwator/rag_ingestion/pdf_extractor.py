from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pymupdf

from .models import ExtractedPage

_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*(?:\|\s*S\s*t\s*r\s*o\s*n\s*a)?\s*$", re.I)
# Numer strony stoi wśród pierwszych lub ostatnich linii strony. Linia z samą liczbą w środku
# strony to treść — komórka tabeli albo wartość przeniesiona do nowej linii („110” stopni).
_PAGE_NUMBER_EDGE_LINES = 2
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# Znaki sterujące nie są tekstem. Punktory z fontu symbolicznego trafiają do warstwy
# tekstowej np. jako U+0001 i zostawały w treści fragmentów.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Zepsute mapowanie fontu w części starszych PDF-ów: „ż” trafia do warstwy tekstowej
# jako „Ŝ” (U+015C), a „Ż” jako małe „ś”. „Ŝ” nie występuje w polszczyźnie, więc jego
# obecność w dokumencie jednoznacznie wskazuje ten wzorzec.
_LEGACY_SMALL_Z_DOT = "Ŝ"
_WORD_WITH_SMALL_S_ACUTE_RE = re.compile(r"\w*ś\w*")

# Nagłówki i stopki stron (np. tytuł dokumentu u góry każdej strony) powtarzałyby się
# w treści setek fragmentów. Rozpoznajemy je jako linie, które występują wyłącznie na
# brzegach stron i to na znacznej części stron dokumentu. Próg 25% obejmuje też nagłówek
# drukowany naprzemiennie z tytułem rozdziału (strony parzyste i nieparzyste).
_RUNNING_LINE_EDGE = 3
_RUNNING_LINE_MIN_SHARE = 0.25
_RUNNING_LINE_MIN_PAGES = 3
_RUNNING_LINE_MIN_DOCUMENT_PAGES = 4


def repair_legacy_polish_glyphs(text: str) -> str:
    """Restore 'ż' and 'Ż' that a broken PDF font mapping turned into 'Ŝ' and 'ś'."""
    text = text.replace(_LEGACY_SMALL_Z_DOT, "ż")

    def uppercase_z_dot(match: re.Match[str]) -> str:
        """Replace 'ś' with 'Ż' only inside a word that is otherwise written in capitals."""
        word = match.group(0)
        letters = [c for c in word if c.isalpha() and c != "ś"]
        # Małe „ś” w wyrazie pisanym wersalikami jest w tym wzorcu literą „Ż”
        # („BIEśĄCE”); w zwykłych wyrazach („środowisko”) zostaje bez zmian.
        if len(letters) >= 2 and all(c.isupper() for c in letters):
            return word.replace("ś", "Ż")
        return word

    return _WORD_WITH_SMALL_S_ACUTE_RE.sub(uppercase_z_dot, text)


def find_running_lines(page_texts: list[str]) -> set[str]:
    """Return lines that only ever appear at page edges, on many pages: headers and footers."""
    if len(page_texts) < _RUNNING_LINE_MIN_DOCUMENT_PAGES:
        return set()

    edge_pages: defaultdict[str, set[int]] = defaultdict(set)
    body_lines: set[str] = set()
    for idx, text in enumerate(page_texts):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        edge = set(lines[:_RUNNING_LINE_EDGE] + lines[-_RUNNING_LINE_EDGE:])
        for line in lines:
            if line in edge:
                edge_pages[line].add(idx)
            else:
                body_lines.add(line)

    min_pages = max(_RUNNING_LINE_MIN_PAGES, math.ceil(_RUNNING_LINE_MIN_SHARE * len(page_texts)))
    return {
        line
        for line, pages in edge_pages.items()
        # Linia obecna także w środku strony jest treścią, a sam znak („-”, „•”) punktorem.
        if len(pages) >= min_pages
        and line not in body_lines
        and sum(c.isalpha() for c in line) >= 3
    }


def remove_running_lines(page_texts: list[str]) -> list[tuple[str, list[str]]]:
    """Drop repeated page headers and footers, keeping each line's first occurrence."""
    running = find_running_lines(page_texts)
    seen: set[str] = set()
    result: list[tuple[str, list[str]]] = []
    for text in page_texts:
        kept: list[str] = []
        removed: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped in running:
                # Pierwsze wystąpienie bywa prawdziwym tytułem rozdziału, więc zostaje.
                if stripped in seen:
                    removed.append(stripped)
                    continue
                seen.add(stripped)
            kept.append(line)
        result.append((_MULTI_NEWLINE_RE.sub("\n\n", "\n".join(kept)).strip(), removed))
    return result


def normalize_text(text: str) -> str:
    """
    Konserwatywne czyszczenie tekstu jednej strony:
    - normalizacja Unicode,
    - usuwanie soft-hyphen i znaków sterujących,
    - usuwanie numeru strony z brzegu strony,
    - porządkowanie białych znaków.
    Kodowania polskich znaków nie poprawiamy tutaj. Jedyny wyjątek to jednoznaczny
    wzorzec zepsutego fontu, naprawiany w extract_pdf_pages dla całego dokumentu.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub(" ", text)

    lines = [_MULTI_SPACE_RE.sub(" ", raw).strip() for raw in text.splitlines()]
    content = [index for index, line in enumerate(lines) if line]
    edge = set(content[:_PAGE_NUMBER_EDGE_LINES] + content[-_PAGE_NUMBER_EDGE_LINES:])
    lines = [
        line
        for index, line in enumerate(lines)
        if not (index in edge and _PAGE_NUMBER_RE.match(line))
    ]

    text = "\n".join(lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str | Path) -> list[ExtractedPage]:
    """Extract normalized text and page quality metadata from a PDF file."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    # Plik zamykamy od razu po odczycie: na Windows otwarty PDF jest zablokowany do zapisu.
    with pymupdf.open(pdf_path) as doc:
        raw_pages = [page.get_text("text") for page in doc]
    # Wzorzec fontu dotyczy całego dokumentu: „ś” w wersalikach naprawiamy także na
    # stronach, na których akurat nie ma „Ŝ”.
    legacy_glyphs = any(_LEGACY_SMALL_Z_DOT in raw for raw in raw_pages)
    texts: list[str] = []
    repaired_pages: list[bool] = []

    for raw in raw_pages:
        text = normalize_text(raw)
        repaired = False
        if legacy_glyphs:
            fixed = repair_legacy_polish_glyphs(text)
            repaired = fixed != text
            text = fixed
        texts.append(text)
        repaired_pages.append(repaired)

    pages: list[ExtractedPage] = []
    for idx, (text, removed) in enumerate(remove_running_lines(texts)):
        char_count = len(text)

        pages.append(
            ExtractedPage(
                page_number=idx + 1,
                text=text,
                char_count=char_count,
                low_text_quality=char_count < 80,
                legacy_glyphs_repaired=repaired_pages[idx],
                removed_running_lines=removed,
            )
        )

    return pages
