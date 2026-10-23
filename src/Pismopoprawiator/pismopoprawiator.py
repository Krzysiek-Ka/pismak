from __future__ import annotations

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

from openai import OpenAI
from pydantic import Field, create_model

from src._shared.context_budget import input_char_budget
from src._shared.serwer import LLMClient, ProviderName, error_log_detail
from src.Pismopoprawiator import pismopoprawiator_helper_functions as helpers
from src.Pismopoprawiator import prompts
from src.Pismopoprawiator.constants import (
    CATEGORIES,
    CHUNK_MAX_CHARS,
    MAX_CALLS,
    POPRAWIATOR_MAX_OUTPUT_TOKENS,
    POPRAWIATOR_RESERVED_OUTPUT_TOKENS,
    POPRAWIATOR_STAGE,
    POPRAWIATOR_TEMPERATURE,
    POPRAWIATOR_TIMEOUT_SECONDS,
    STYLE_CATEGORY,
    WHOLE_LETTER_MAX_CHARS,
)
from src.Pismopoprawiator.structures import (
    Correction,
    CorrectionResponse,
    CorrectionResult,
    Paragraph,
    RejectedCorrection,
)

LOGGER = logging.getLogger("pismak")
Mode = Literal["whole", "paragraphs"]
# Jedno wywołanie modelu: akapity widoczne w prompcie i te z nich, które wolno poprawić.
CallPlan = tuple[list[Paragraph], list[str]]


def make_llm(
    client: OpenAI,
    model: str,
    provider: ProviderName,
    *,
    session_id: str | None = None,
    on_wait: Callable[[], None] | None = None,
    context_tokens: int | None = None,
) -> LLMClient:
    """Wrap the stage client so each request has the Pismopoprawiator timeout and no SDK retry."""
    return LLMClient(
        client=client.with_options(timeout=POPRAWIATOR_TIMEOUT_SECONDS, max_retries=0),
        model_name=model,
        on_wait=on_wait,
        stage=POPRAWIATOR_STAGE,
        session_id=session_id,
        provider=provider,
        context_tokens=context_tokens,
    )


def correction_model(paragraph_ids: list[str]) -> type[CorrectionResponse]:
    """Return the response schema that offers only the given paragraphs and known categories."""
    # Wyliczenia trafiają do gramatyki modelu (Ollama) albo trybu strict (OpenRouter), ale
    # walidacja w Pythonie zostaje łagodna: nieznany akapit lub kategorię odrzuca kod dla
    # jednej poprawki, a nie cała odpowiedź. Nazwy zostają nazwami modeli bazowych.
    correction = create_model(
        "Correction",
        __base__=Correction,
        akapit=(str, Field(json_schema_extra={"enum": paragraph_ids})),
        kategoria=(str, Field(json_schema_extra={"enum": [*CATEGORIES, STYLE_CATEGORY]})),
    )
    return create_model(
        "CorrectionResponse",
        __base__=CorrectionResponse,
        poprawki=(list[correction], Field(default_factory=list)),
    )


def choose_mode(paragraphs: list[Paragraph], provider: str, context_tokens: int | None) -> Mode:
    """Send the whole letter in one call unless it is long or does not fit the model window."""
    editable_ids = [paragraph.id for paragraph in paragraphs if paragraph.editable]
    prompt_chars = len(prompts.POPRAWIATOR_SYSTEM_PROMPT) + len(
        prompts.correction_user_prompt(paragraphs, editable_ids)
    )
    budget = input_char_budget(
        provider,
        reserved_output_tokens=POPRAWIATOR_RESERVED_OUTPUT_TOKENS,
        context_tokens=context_tokens,
    )
    body_chars = sum(len(paragraph.text) for paragraph in paragraphs)
    if body_chars <= WHOLE_LETTER_MAX_CHARS and prompt_chars <= budget:
        return "whole"
    return "paragraphs"


def plan_calls(
    paragraphs: list[Paragraph], mode: Mode, chunk_chars: int = CHUNK_MAX_CHARS
) -> list[CallPlan]:
    """Split the letter into calls: one for the whole letter, or ranges of whole paragraphs."""
    if mode == "whole":
        return [(paragraphs, [paragraph.id for paragraph in paragraphs if paragraph.editable])]

    ranges: list[tuple[int, int]] = []
    first: int | None = None
    last = 0
    size = 0
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.editable:
            continue
        if first is not None and size + len(paragraph.text) > chunk_chars:
            ranges.append((first, last + 1))
            first, size = None, 0
        if first is None:
            first = index
        size += len(paragraph.text)
        last = index
    if first is not None:
        ranges.append((first, last + 1))

    calls: list[CallPlan] = []
    for start, stop in ranges:
        # Poprzedni akapit zostaje w prompcie jako kontekst, którego nie wolno poprawiać.
        context = paragraphs[start - 1 : start] if start > 0 else []
        editable = [paragraph.id for paragraph in paragraphs[start:stop] if paragraph.editable]
        calls.append((context + paragraphs[start:stop], editable))
    return calls


def _call_model(llm: Any, call: CallPlan) -> list[Correction]:
    """Ask the model for corrections of one call's paragraphs."""
    in_call, editable_ids = call
    response = llm.generate_structured(
        system_prompt=prompts.POPRAWIATOR_SYSTEM_PROMPT,
        user_prompt=prompts.correction_user_prompt(in_call, editable_ids),
        response_model=correction_model([paragraph.id for paragraph in in_call]),
        temperature=POPRAWIATOR_TEMPERATURE,
        max_output_tokens=POPRAWIATOR_MAX_OUTPUT_TOKENS,
    )
    return [Correction.model_validate(item.model_dump()) for item in response.poprawki]


def _correct(
    letter: str,
    llm: Any,
    mode: Mode | Literal["auto"],
    session_id: str | None,
    result: CorrectionResult,
    chunk_chars: int,
) -> None:
    """Fill the result: ask the model, check every correction and apply the accepted ones."""
    body, tail = helpers.split_letter(letter)
    parts, paragraphs = helpers.split_paragraphs(body)
    if not any(paragraph.editable for paragraph in paragraphs):
        return

    result.mode = (
        choose_mode(paragraphs, result.provider or "ollama", result.context_tokens)
        if mode == "auto"
        else mode
    )
    calls = plan_calls(paragraphs, result.mode, chunk_chars)
    if len(calls) > MAX_CALLS:
        skipped = [paragraph_id for _, editable in calls[MAX_CALLS:] for paragraph_id in editable]
        result.errors.append(
            f"pominięto akapity {', '.join(skipped)}: limit {MAX_CALLS} wywołań modelu"
        )
        calls = calls[:MAX_CALLS]

    accepted: list[helpers.AcceptedCorrection] = []
    failed_calls = 0
    for call in calls:
        result.calls += 1
        try:
            corrections = _call_model(llm, call)
        except Exception as exc:
            # Nieudane wywołanie nie przekreśla poprawek z pozostałych części pisma.
            failed_calls += 1
            result.errors.append(f"{', '.join(call[1])}: {type(exc).__name__}")
            LOGGER.warning(
                "correction_call_failed session_id=%s paragraphs=%s error_type=%s error_detail=%s",
                session_id or "-",
                ",".join(call[1]),
                type(exc).__name__,
                error_log_detail(exc),
                extra={"session_id": session_id},
            )
            continue
        call_accepted, rejected, suggestions = helpers.review_corrections(
            paragraphs, corrections, set(call[1])
        )
        accepted.extend(call_accepted)
        result.rejected.extend(rejected)
        result.suggestions.extend(suggestions)

    if failed_calls == len(calls):
        result.status = "failed"
        return

    corrected_paragraphs, applied, overlapping = helpers.apply_corrections(paragraphs, accepted)
    result.rejected.extend(overlapping)
    corrected = helpers.join_paragraphs(parts, corrected_paragraphs) + tail

    violations = helpers.content_violations(letter, corrected)
    if violations:
        # Strażnik całości: pismo wraca do wersji sprzed korekty.
        result.rejected.extend(
            RejectedCorrection(correction=correction, reason="wycofano: " + violations[0])
            for correction in applied
        )
        result.errors.append("strażnik całości: " + "; ".join(violations))
        result.status = "reverted"
        return

    result.applied = applied
    result.corrected_text = corrected
    result.status = "corrected" if applied else "unchanged"


def correct_letter(
    letter: str,
    llm: Any,
    *,
    mode: Mode | Literal["auto"] = "auto",
    session_id: str | None = None,
    chunk_chars: int = CHUNK_MAX_CHARS,
) -> CorrectionResult:
    """Correct the language of a rendered letter; never raises and always returns a usable text.

    In the paragraph mode each call holds whole paragraphs up to chunk_chars characters of
    editable text; a value of 1 sends every editable paragraph in its own call.
    """
    started = perf_counter()
    result = CorrectionResult(
        original_text=letter,
        corrected_text=letter,
        status="unchanged",
        provider=getattr(llm, "provider", None),
        model=getattr(llm, "model_name", None),
        context_tokens=getattr(llm, "context_tokens", None),
    )
    try:
        _correct(letter, llm, mode, session_id, result, chunk_chars)
    except Exception as exc:
        # Korekta nigdy nie przerywa pipeline'u: użytkownik dostaje pismo bez korekty.
        result.corrected_text = letter
        result.applied = []
        result.status = "failed"
        result.errors.append(type(exc).__name__)
        LOGGER.error(
            "correction_failed session_id=%s error_type=%s error_detail=%s",
            session_id or "-",
            type(exc).__name__,
            error_log_detail(exc),
            exc_info=exc,
            extra={"session_id": session_id},
        )
    result.model = getattr(llm, "last_model", None) or result.model
    result.elapsed_seconds = round(perf_counter() - started, 3)
    helpers.log_result(result, session_id)
    return result


def failed_result(
    letter: str,
    error: Exception,
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
) -> CorrectionResult:
    """Return the uncorrected letter when the stage could not even prepare its model client."""
    result = CorrectionResult(
        original_text=letter,
        corrected_text=letter,
        status="failed",
        provider=provider,
        model=model,
        errors=[type(error).__name__],
    )
    helpers.log_result(result, session_id)
    return result
