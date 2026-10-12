from __future__ import annotations

from src._shared.context_budget import input_char_budget
from src.Dopytywator.structures import CaseBrief
from src.Pismogenerator import pismogenerator_helper_functions, prompts, structures
from src.Pismogenerator.constants import (
    EVIDENCE_MAX_CHARS,
    EVIDENCE_MIN_CHARS,
    PISMOGENERATOR_MAX_OUTPUT_TOKENS,
    PISMOGENERATOR_RESERVED_OUTPUT_TOKENS,
)
from src.Zrodlowyszukiwator.brief_to_evidence import EvidencePack


class DraftValidationError(ValueError):
    """Raised when generated draft content violates application constraints."""

    def __init__(
        self,
        errors: list[str],
        *,
        draft: structures.DraftLetter | None = None,
    ) -> None:
        self.draft = draft
        self.validation_errors = errors
        details = "\n".join(f" - {error}" for error in errors)
        super().__init__(f"DraftLetter nie przeszedł walidacji:\n{details}")


def extract_sources_from_brief(case_brief: CaseBrief) -> list[str]:
    """Extract the allowed source candidates from a brief, preferring explicit sources when present."""
    if hasattr(case_brief, "model_dump"):
        data = case_brief.model_dump()
    elif isinstance(case_brief, dict):
        data = case_brief
    else:
        data = vars(case_brief)

    raw_sources = data.get("sources", [])

    if raw_sources:
        source_candidates = [str(x).strip() for x in raw_sources if str(x).strip()]
    else:
        source_candidates = [
            str(x).strip() for x in case_brief.available_evidence if str(x).strip()
        ]

    # Usuń duplikaty z zachowaniem kolejności.
    source_candidates = list(dict.fromkeys(source_candidates))

    print("Dozwolone źródła / materiały:")
    if source_candidates:
        for i, source in enumerate(source_candidates, start=1):
            print(f" [{i}] {source}")
    else:
        print(" - brak")

    return source_candidates


def fit_draft_user_prompt(
    case_brief: CaseBrief,
    evidence_pack: EvidencePack,
    user_evidence: list[str],
    provider: str,
) -> str:
    """Build the draft prompt, shortening evidence excerpts until it fits the context window."""
    budget = input_char_budget(
        provider,
        reserved_output_tokens=PISMOGENERATOR_RESERVED_OUTPUT_TOKENS,
    ) - len(prompts.PISMOGENERATOR_SYSTEM_PROMPT)
    excerpt_chars = EVIDENCE_MAX_CHARS
    while True:
        user_prompt = prompts.drafr_user_prompt(
            case_brief=case_brief,
            evidence_pack=evidence_pack,
            user_evidence=user_evidence,
            max_chars_per_evidence=excerpt_chars,
        )
        if len(user_prompt) <= budget or excerpt_chars <= EVIDENCE_MIN_CHARS:
            return user_prompt
        excerpt_chars = max(EVIDENCE_MIN_CHARS, int(excerpt_chars * 0.8))


def build_draft(
    case_brief: CaseBrief,
    evidence_pack: EvidencePack,
    llm: object,
    llm_temperature: float = 0.2,
) -> structures.DraftLetter:
    """Generate and validate a draft letter from a brief and EvidencePack."""
    user_evidence = [
        str(value).strip() for value in case_brief.available_evidence if str(value).strip()
    ]
    user_prompt = fit_draft_user_prompt(
        case_brief=case_brief,
        evidence_pack=evidence_pack,
        user_evidence=user_evidence,
        provider=getattr(llm, "provider", "ollama"),
    )
    draft = llm.generate_structured(
        system_prompt=prompts.PISMOGENERATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=structures.DraftLetter,
        temperature=llm_temperature,
        max_output_tokens=PISMOGENERATOR_MAX_OUTPUT_TOKENS,
    )
    # Renderer sam dodaje formułę nadawcy i podpis w Phase 0.
    draft = draft.model_copy(update={"sender": ""})

    draft_errors = pismogenerator_helper_functions.validate_draft(
        draft=draft, case_brief=case_brief, evidence_pack=evidence_pack, user_evidence=user_evidence
    )

    if draft_errors:
        raise DraftValidationError(draft_errors, draft=draft)

    return draft


def build_letter(
    draft: structures.DraftLetter,
    evidence_pack: EvidencePack,
    user_evidence: list[str] | None = None,
) -> str:
    """Render a final letter text from the validated draft structure."""
    return pismogenerator_helper_functions.render_letter(
        draft=draft,
        evidence_pack=evidence_pack,
        user_evidence=user_evidence or [],
    )
