import re

from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator.structures import DraftLetter
from src.Zrodlowyszukiwator.brief_to_evidence.models import EvidencePack


def validate_draft(
    draft: DraftLetter,
    case_brief: CaseBrief,
    evidence_pack: EvidencePack,
    user_evidence: list[str],
) -> list[str]:
    """Validate a draft letter against the Phase 0 constraints for the app."""
    errors: list[str] = []

    if not draft.recipient.strip():
        errors.append("Projekt nie ma adresata ani oznaczenia braku")

    if draft.sender.strip():
        errors.append("Pole nadawcy powinno pozostać puste w Phase 0")

    summary = draft.request_summary.strip()

    if not summary:
        errors.append("Brak krótkiego podsumowania wniosku")

    if summary.lower().startswith("o "):
        errors.append(
            "request_summary nie powinno zaczynać się od 'o' — renderer dodaje 'wnioskuję o'"
        )

    forbidden_legal_phrases = [
        "art.",
        " kpa",
        "kodeks postępowania administracyjnego",
        "dostęp do informacji publicznej",
        "ustawa o dostępie do informacji publicznej",
        "działając w trybie",
    ]

    combined_main_text = (draft.request_summary + "\n" + draft.argumentation_markdown).lower()

    for phrase in forbidden_legal_phrases:
        if phrase in combined_main_text:
            errors.append(f"Phase 0 nie powinno dodawać podstawy prawnej: {phrase!r}")

    if len(draft.argumentation_markdown.strip()) < 100:
        errors.append("Argumentacja projektu jest podejrzanie krótka")

    allowed_source_ids = {evidence.evidence_id for evidence in evidence_pack.evidence}
    allowed_source_ids.update(f"U{index}" for index, _ in enumerate(user_evidence, start=1))
    illegal_sources = [source for source in draft.sources if source not in allowed_source_ids]
    if illegal_sources:
        errors.append("Model dodał niedozwolone źródła: " + "; ".join(illegal_sources))

    # Kontrola odwołań [1], [2], ...
    refs = [int(x) for x in re.findall(r"\[(\d+)\]", draft.argumentation_markdown)]
    if refs and not draft.sources:
        errors.append("Argumentacja zawiera przypisy, ale lista sources jest pusta")

    if refs and max(refs) > len(draft.sources):
        errors.append(
            "Argumentacja odwołuje się do numeru źródła, którego nie ma na liście sources"
        )

    # facts_to_verify powinny pozostać jawne.
    missing_verification_flags = [
        fact
        for fact in case_brief.facts_to_verify
        if fact not in draft.claims_requiring_verification
    ]
    if missing_verification_flags:
        print(
            "Uwaga: poniższe elementy facts_to_verify nie zostały "
            "przepisane literalnie do sekcji kontrolnej:"
        )
        for fact in missing_verification_flags:
            print(" -", fact)

    return errors


# ________________________________
# render listu  letter.md

PHASE0_OPENING_FORMULA = "W imieniu Towarzystwa Upiększania Miasta Wrocławia, wnioskuję o"
PHASE0_DOCUMENT_TYPE = "WNIOSEK"


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
    return "\n".join(rendered)


def render_letter(
    draft: DraftLetter,
    evidence_pack: EvidencePack,
    user_evidence: list[str],
) -> str:
    """Turn a DraftLetter model into a final markdown letter body."""
    recipient = draft.recipient.strip() or "[DO UZUPEŁNIENIA: adresat]"

    letter = f"""**Adresat:** {recipient}
**Nadawca:**

{PHASE0_OPENING_FORMULA} {draft.request_summary.strip()}

# {PHASE0_DOCUMENT_TYPE}

{draft.argumentation_markdown.strip()}

Z poważaniem,

[imię i nazwisko]
Towarzystwo Upiększania Miasta Wrocławia

---

## Źródła

{render_sources(draft.sources, evidence_pack, user_evidence)}

---

## Sekcja kontrolna

### Informacje wymagające weryfikacji przed wysłaniem

{render_list(draft.claims_requiring_verification)}

### Brakujące pola

{render_list(draft.missing_fields)}

### Założenia redakcyjne modelu

{render_list(draft.assumptions_used)}
"""
    return letter.strip()
