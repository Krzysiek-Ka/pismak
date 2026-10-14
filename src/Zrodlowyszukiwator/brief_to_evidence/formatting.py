from __future__ import annotations

from .models import EvidencePack


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """Shorten evidence text while preferring a nearby paragraph or sentence boundary."""
    if len(text) <= max_chars:
        return text
    shortened = text[:max_chars]
    # Preferencja: koniec akapitu > koniec zdania > twarde cięcie.
    for marker in ("\n\n", ". ", "; "):
        idx = shortened.rfind(marker)
        if idx >= int(max_chars * 0.65):
            return shortened[: idx + len(marker)].strip()
    return shortened.rstrip() + "…"


def _validity_label(is_active: bool | None) -> str:
    """Describe for the letter prompt whether a source document is in force."""
    if is_active is None:
        return "nieustalony"
    return "obowiązujący" if is_active else "nieobowiązujący (dokument historyczny)"


def evidence_pack_to_markdown(
    pack: EvidencePack,
    *,
    max_chars_per_evidence: int = 2400,
) -> str:
    """Kontekst gotowy do przekazania Pismogeneratorowi."""
    parts = [f"# EvidencePack\n\nSprawa: {pack.query_summary}"]

    if not pack.evidence:
        if pack.evidence_status == "unsupported":
            parts.append(
                "\nBrak znalezionych źródeł. Żaden znaleziony fragment nie odpowiada na "
                "sprawę; dokumenty o podobnej tematyce nie są jej poparciem."
            )
        else:
            parts.append("\nBrak znalezionych źródeł.")
        return "\n".join(parts)

    if pack.evidence_status == "unverified":
        parts.append(
            "Uwaga: trafność fragmentów nie została oceniona. Są to najbardziej podobne "
            "fragmenty z wyszukiwania; cytuj fragment tylko wtedy, gdy jego treść rzeczywiście "
            "dotyczy sprawy."
        )

    for e in pack.evidence:
        section = " > ".join(e.section_path) if e.section_path else "brak"
        pages = str(e.page_start) if e.page_start == e.page_end else f"{e.page_start}-{e.page_end}"
        excerpt = _truncate_at_boundary(e.excerpt, max_chars_per_evidence)

        parts.append(
            f"""
## [{e.evidence_id}] {e.document_title}
- document_kind: {e.document_kind}
- legal_form: {e.legal_form}
- status obowiązywania: {_validity_label(e.is_active)}
- strony PDF: {pages}
- sekcja: {section}
- powód trafności: {e.relevance_reason}
- source_url: {e.source_url or "brak"}

### Fragment źródłowy
{excerpt}
""".strip()
        )

    if pack.unsupported_topics:
        parts.append(
            "## Tematy bez wystarczającego wsparcia\n"
            + "\n".join(f"- {x}" for x in pack.unsupported_topics)
        )

    return "\n\n".join(parts)
