from __future__ import annotations

from typing import Literal

from pydantic import Field, create_model

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


def fit_draft_user_prompt(
    case_brief: CaseBrief,
    evidence_pack: EvidencePack,
    user_evidence: list[str],
    provider: str,
    context_tokens: int | None = None,
) -> str:
    """Build the draft prompt, shortening evidence excerpts until it fits the context window."""
    budget = input_char_budget(
        provider,
        reserved_output_tokens=PISMOGENERATOR_RESERVED_OUTPUT_TOKENS,
        context_tokens=context_tokens,
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


def draft_letter_model(allowed_source_ids: list[str]) -> type[structures.DraftLetter]:
    """Return DraftLetter whose sources field accepts only the given IDs in its JSON schema."""
    # Schemat trafia do gramatyki modelu (Ollama) albo trybu strict (OpenRouter), więc model
    # nie może wpisać "1" zamiast "E1" ani wymyślić źródła. Nazwa zostaje DraftLetter.
    if allowed_source_ids:
        sources = (list[Literal[tuple(allowed_source_ids)]], Field(default_factory=list))
    else:
        sources = (list[str], Field(default_factory=list, max_length=0))
    return create_model("DraftLetter", __base__=structures.DraftLetter, sources=sources)


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
    allowed_source_ids = [evidence.evidence_id for evidence in evidence_pack.evidence] + [
        f"U{index}" for index in range(1, len(user_evidence) + 1)
    ]
    user_prompt = fit_draft_user_prompt(
        case_brief=case_brief,
        evidence_pack=evidence_pack,
        user_evidence=user_evidence,
        provider=getattr(llm, "provider", "ollama"),
        context_tokens=getattr(llm, "context_tokens", None),
    )
    generated = llm.generate_structured(
        system_prompt=prompts.PISMOGENERATOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=draft_letter_model(allowed_source_ids),
        temperature=llm_temperature,
        max_output_tokens=PISMOGENERATOR_MAX_OUTPUT_TOKENS,
    )
    # Renderer sam dodaje formułę nadawcy i podpis w Phase 0.
    draft = structures.DraftLetter.model_validate(generated.model_dump() | {"sender": ""})

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
