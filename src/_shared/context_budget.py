from __future__ import annotations

import logging
import math
import os

LOGGER = logging.getLogger("pismak")

# Okno kontekstu, w którym muszą się zmieścić prompt i odpowiedź, pochodzi z wybranego
# modelu (src/_shared/model_context.py). Tu są tylko: nazwa zmiennej z opcjonalnym górnym
# limitem oraz wartość zapasowa na wypadek, gdy okna modelu nie udało się ustalić.
CONTEXT_TOKENS_SETTINGS = {
    "ollama": ("PISMAK_CONTEXT_TOKENS", 8192),
    "openrouter": ("PISMAK_OPENROUTER_CONTEXT_TOKENS", 32768),
}
# Przelicznik znaków na tokeny, celowo zaniżony. Pomiar na promptach Pismaka: Bielik 4.5B
# 2,81-4,11 znaku na token, PLLuM 4B 3,29-4,15; najniżej wypada JSON wysyłany do rerankera.
CHARS_PER_TOKEN = 2.5
SAFETY_MARGIN_TOKENS = 256
MIN_OUTPUT_TOKENS = 1024


class LLMInputLimitError(RuntimeError):
    """Raised when a prompt cannot fit in the model's context window."""

    def __init__(self, message: str, *, context_tokens: int | None = None) -> None:
        super().__init__(message)
        self.context_tokens = context_tokens


def _configured_limit(name: str) -> int | None:
    """Return the optional upper limit of the context window set in the environment."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} musi być liczbą całkowitą tokenów.") from exc
    if value <= SAFETY_MARGIN_TOKENS + MIN_OUTPUT_TOKENS:
        raise ValueError(
            f"{name} musi być większe niż {SAFETY_MARGIN_TOKENS + MIN_OUTPUT_TOKENS} tokenów."
        )
    return value


def context_window_tokens(provider: str, context_tokens: int | None = None) -> int:
    """Return the window to plan for: the model's own one, capped by the optional setting."""
    name, fallback = CONTEXT_TOKENS_SETTINGS.get(provider, CONTEXT_TOKENS_SETTINGS["ollama"])
    limit = _configured_limit(name)
    if context_tokens is None:
        return fallback if limit is None else limit
    return context_tokens if limit is None else min(context_tokens, limit)


def estimate_tokens(chars: int) -> int:
    """Estimate the token count of a text of the given length, erring on the high side."""
    return math.ceil(chars / CHARS_PER_TOKEN)


def input_char_budget(
    provider: str,
    *,
    reserved_output_tokens: int,
    context_tokens: int | None = None,
) -> int:
    """Return how many prompt characters fit while leaving room for the reserved output."""
    window = context_window_tokens(provider, context_tokens)
    return max(0, int((window - SAFETY_MARGIN_TOKENS - reserved_output_tokens) * CHARS_PER_TOKEN))


def fit_output_tokens(
    *,
    provider: str,
    prompt_chars: int,
    max_output_tokens: int,
    stage: str,
    session_id: str | None = None,
    context_tokens: int | None = None,
) -> int:
    """Lower the output limit so the prompt and the answer fit in the context window together."""
    context_tokens = context_window_tokens(provider, context_tokens)
    prompt_tokens = estimate_tokens(prompt_chars)
    output_tokens = min(max_output_tokens, context_tokens - SAFETY_MARGIN_TOKENS - prompt_tokens)
    LOGGER.info(
        "llm_context_budget stage=%s session_id=%s provider=%s context_tokens=%d "
        "prompt_chars=%d estimated_prompt_tokens=%d max_output_tokens=%d",
        stage,
        session_id or "-",
        provider,
        context_tokens,
        prompt_chars,
        prompt_tokens,
        output_tokens,
        extra={"session_id": session_id},
    )
    if output_tokens < min(max_output_tokens, MIN_OUTPUT_TOKENS):
        raise LLMInputLimitError(
            f"Dane wejściowe ({prompt_chars} znaków) nie mieszczą się w oknie kontekstu "
            f"modelu ({context_tokens} tokenów).",
            context_tokens=context_tokens,
        )
    return output_tokens


def usage_summary(completion: object) -> str:
    """Format the token usage reported by the provider for a log line."""
    usage = getattr(completion, "usage", None)
    return (
        f"prompt_tokens={getattr(usage, 'prompt_tokens', None) or '-'} "
        f"completion_tokens={getattr(usage, 'completion_tokens', None) or '-'}"
    )
