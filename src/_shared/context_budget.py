from __future__ import annotations

import logging
import math
import os

LOGGER = logging.getLogger("pismak")

# Okno kontekstu, w którym muszą się zmieścić prompt i odpowiedź. Wartość dla Ollamy
# nie może przekraczać num_ctx ustawionego po stronie serwera: Ollama obcina dłuższy
# prompt bez zgłaszania błędu.
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
    """Raised when a prompt cannot fit in the configured context window."""


def context_window_tokens(provider: str) -> int:
    """Return the configured context window for the selected provider."""
    name, default = CONTEXT_TOKENS_SETTINGS.get(provider, CONTEXT_TOKENS_SETTINGS["ollama"])
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} musi być liczbą całkowitą tokenów.") from exc
    if value <= SAFETY_MARGIN_TOKENS + MIN_OUTPUT_TOKENS:
        raise ValueError(
            f"{name} musi być większe niż {SAFETY_MARGIN_TOKENS + MIN_OUTPUT_TOKENS} tokenów."
        )
    return value


def estimate_tokens(chars: int) -> int:
    """Estimate the token count of a text of the given length, erring on the high side."""
    return math.ceil(chars / CHARS_PER_TOKEN)


def input_char_budget(provider: str, *, reserved_output_tokens: int) -> int:
    """Return how many prompt characters fit while leaving room for the reserved output."""
    tokens = context_window_tokens(provider) - SAFETY_MARGIN_TOKENS - reserved_output_tokens
    return max(0, int(tokens * CHARS_PER_TOKEN))


def fit_output_tokens(
    *,
    provider: str,
    prompt_chars: int,
    max_output_tokens: int,
    stage: str,
    session_id: str | None = None,
) -> int:
    """Lower the output limit so the prompt and the answer fit in the context window together."""
    context_tokens = context_window_tokens(provider)
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
            f"modelu ({context_tokens} tokenów)."
        )
    return output_tokens


def usage_summary(completion: object) -> str:
    """Format the token usage reported by the provider for a log line."""
    usage = getattr(completion, "usage", None)
    return (
        f"prompt_tokens={getattr(usage, 'prompt_tokens', None) or '-'} "
        f"completion_tokens={getattr(usage, 'completion_tokens', None) or '-'}"
    )
