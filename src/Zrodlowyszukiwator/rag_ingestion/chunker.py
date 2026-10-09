from __future__ import annotations

import hashlib
import re

from .models import ChunkRecord, DocumentMetadata, ExtractedPage, TextUnit

_NUMBERED_HEADING = re.compile(r"^\s*((?:\d+\.)*\d+)\.?\s+(.{2,160})$")
_ROMAN_HEADING = re.compile(r"^\s*([IVXLCDM]+)\.\s+(.{2,160})$", re.I)


def _looks_like_heading(line: str) -> tuple[bool, int]:
    """
    Heurystyka bez OCR i bez zależności od layout modelu.
    Zwraca (czy_nagłówek, poziom).
    """
    s = line.strip()
    if not s or len(s) > 180:
        return False, 0

    m = _NUMBERED_HEADING.match(s)
    if m:
        prefix = m.group(1)
        return True, prefix.count(".") + 1

    if _ROMAN_HEADING.match(s):
        return True, 1

    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 5:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio >= 0.85 and len(s.split()) <= 14:
            return True, 1

    # Typowe krótkie nagłówki mieszane, np. "Wstęp", "Spis treści"
    if len(s.split()) <= 8 and s[:1].isupper() and not s.endswith((".", ";", ",")):
        if len(s) <= 70:
            # tylko jeśli linia nie wygląda jak normalne zdanie
            if not re.search(r"\b(jest|są|oraz|który|która|które|dla|przez)\b", s, re.I):
                return True, 2

    return False, 0


def _split_page_to_units(page: ExtractedPage) -> list[tuple[str, int | None]]:
    """
    Zwraca (tekst, heading_level lub None).
    Paragrafy są budowane z kolejnych linii.
    """
    out: list[tuple[str, int | None]] = []
    paragraph: list[str] = []

    def flush() -> None:
        """Append the current paragraph to the page units when it contains text."""
        nonlocal paragraph
        if paragraph:
            text = " ".join(paragraph).strip()
            if text:
                out.append((text, None))
            paragraph = []

    for line in page.text.splitlines():
        line = line.strip()
        if not line:
            flush()
            continue

        is_heading, level = _looks_like_heading(line)
        if is_heading:
            flush()
            out.append((line, level))
        else:
            paragraph.append(line)

    flush()
    return out


def build_text_units(pages: list[ExtractedPage]) -> list[TextUnit]:
    """Convert extracted pages into text units with inherited section paths."""
    section_stack: list[str] = []
    units: list[TextUnit] = []

    for page in pages:
        for text, heading_level in _split_page_to_units(page):
            if heading_level is not None:
                level = max(1, heading_level)
                section_stack = section_stack[: level - 1]
                section_stack.append(text)
                continue

            if text:
                units.append(
                    TextUnit(
                        page_number=page.page_number,
                        section_path=list(section_stack),
                        text=text,
                    )
                )

    return units


def _sha256(text: str) -> str:
    """Return the SHA-256 hexadecimal digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_units(
    units: list[TextUnit],
    metadata: DocumentMetadata,
    *,
    max_chars: int = 3500,
    min_chars: int = 250,
    overlap_units: int = 1,
) -> list[ChunkRecord]:
    """
    Chunking semantyczny:
    - nie rozcina pojedynczego paragrafu, jeśli nie musi,
    - preferuje granice sekcji,
    - zachowuje stronę i ścieżkę sekcji,
    - stosuje niewielki overlap na poziomie paragrafów.
    """
    chunks: list[ChunkRecord] = []
    buffer: list[TextUnit] = []

    def buffer_len(items: list[TextUnit]) -> int:
        """Calculate the approximate rendered length of buffered text units."""
        return sum(len(x.text) + 2 for x in items)

    def flush(force: bool = False) -> None:
        """Materialize the current chunk buffer when its size or force flag allows it."""
        nonlocal buffer
        if not buffer:
            return
        if not force and buffer_len(buffer) < min_chars:
            return

        text = "\n\n".join(x.text for x in buffer).strip()
        if not text:
            buffer = []
            return

        page_start = min(x.page_number for x in buffer)
        page_end = max(x.page_number for x in buffer)
        section_path = buffer[-1].section_path or buffer[0].section_path

        idx = len(chunks)
        text_hash = _sha256(text)
        chunk_id = f"{metadata.document_id}:{idx:05d}:{text_hash[:12]}"

        section = " > ".join(section_path) if section_path else "(bez nagłówka)"
        embedding_text = (
            f"Dokument: {metadata.title}\n"
            f"Typ: {metadata.document_kind.value}\n"
            f"Miasto: {metadata.city}\n"
            f"Sekcja: {section}\n"
            f"Strony PDF: {page_start}-{page_end}\n\n"
            f"{text}"
        )

        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                document_id=metadata.document_id,
                chunk_index=idx,
                page_start=page_start,
                page_end=page_end,
                section_path=section_path,
                text=text,
                embedding_text=embedding_text,
                text_sha256=text_hash,
                metadata={
                    "title": metadata.title,
                    "document_kind": metadata.document_kind.value,
                    "hierarchy_rank": metadata.hierarchy_rank,
                    "city": metadata.city,
                    "issuer": metadata.issuer,
                    "legal_form": metadata.legal_form.value,
                    "adoption_act_number": metadata.adoption_act_number,
                },
            )
        )

        keep = max(0, overlap_units)
        buffer = buffer[-keep:] if keep else []

    for unit in units:
        if buffer:
            current_section = buffer[-1].section_path
            section_changed = unit.section_path != current_section
            would_overflow = buffer_len(buffer) + len(unit.text) + 2 > max_chars

            if section_changed and buffer_len(buffer) >= min_chars:
                flush(force=True)
            elif would_overflow:
                flush(force=True)

        # bardzo długi pojedynczy paragraf: konserwatywny split znakowy
        if len(unit.text) > max_chars:
            if buffer:
                flush(force=True)
            start = 0
            while start < len(unit.text):
                piece = unit.text[start : start + max_chars].strip()
                if piece:
                    temp = TextUnit(
                        page_number=unit.page_number,
                        section_path=unit.section_path,
                        text=piece,
                    )
                    buffer = [temp]
                    flush(force=True)
                start += max_chars
            continue

        buffer.append(unit)

    flush(force=True)
    return chunks
