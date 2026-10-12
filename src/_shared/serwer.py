import json
import logging
from collections.abc import Callable
from time import perf_counter
from typing import Literal, TypeVar

from openai import OpenAI
from openai.lib._parsing import type_to_response_format_param
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json

from src._shared.context_budget import fit_output_tokens, usage_summary
from src._shared.llm_runtime import llm_generation_lock

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1/"
# Lokalny model na CPU przetwarza pełny prompt nawet kilkanaście minut. Ponowienia SDK
# są wyłączone, bo każde wysłałoby to samo wielominutowe żądanie od początku.
OLLAMA_REQUEST_TIMEOUT_SECONDS = 900.0
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_KEY_PLACEHOLDER = "sk-or-v1-WSTAW_TUTAJ_SWÓJ_KLUCZ"
ProviderName = Literal["ollama", "openrouter"]
MAX_INTERVIEW_QUESTIONS = 3
ANALYST_TEMPERATURE = 0.1
DEBUG = False
LOGGER = logging.getLogger("pismak")
MAX_LOGGED_MODEL_RESPONSE_CHARS = 20_000
COMPACT_OUTPUT_INSTRUCTION = (
    "Poprzednia odpowiedź przekroczyła limit. Zwróć krótszą, zwięzłą odpowiedź "
    "z kompletem wymaganych pól i poprawną strukturą JSON. Skróć opisy i listy; "
    "nie dodawaj komentarzy ani tekstu poza JSON."
)


def log_model_response(
    *,
    outcome: Literal["invalid", "truncated"],
    stage: str,
    session_id: str | None,
    response: str | None,
    provider: str,
    model: str,
    attempt: int,
) -> None:
    """Log a safely encoded model response when it could not be used as returned."""
    if response is None:
        return
    logged_response = response[:MAX_LOGGED_MODEL_RESPONSE_CHARS]
    was_truncated = len(response) > MAX_LOGGED_MODEL_RESPONSE_CHARS
    LOGGER.warning(
        "llm_model_response outcome=%s stage=%s session_id=%s provider=%s model=%s "
        "attempt=%d response_chars=%d response_log_truncated=%s response_json=%s",
        outcome,
        stage,
        session_id or "-",
        provider,
        model,
        attempt,
        len(response),
        str(was_truncated).lower(),
        json.dumps(logged_response, ensure_ascii=False),
        extra={"session_id": session_id},
    )


def check_available_model() -> None:
    """Print the list of models currently exposed by a local Ollama instance."""
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",  # wymagane przez SDK, ignorowane przez Ollamę
        timeout=180.0,
    )

    try:
        available_models = [m.id for m in client.models.list().data]
    except Exception as exc:
        raise RuntimeError(
            f"Nie udało się połączyć z Ollamą pod {OLLAMA_BASE_URL}. Sprawdź, czy Ollama działa."
        ) from exc

    if not available_models:
        raise RuntimeError("Ollama działa, ale nie zwróciła żadnych modeli. Sprawdź `ollama list`.")

    print("Modele widoczne w Ollama:")
    for i, model_name in enumerate(available_models, start=1):
        print(f"  [{i}] {model_name}")


# ____________________________________________________________________________________________-
# rapair
REPAIR_PROMPT = """
Poprzednia odpowiedź nie spełniła wymaganego schematu danych.
Błędy walidacji:
{validation_errors}
Zwróć ponownie pełny obiekt zgodny ze schematem. Bez komentarza i bez Markdown poza wartościami pól.
""".strip()

# _________________________________________________________________________________________
# client


class LLMConfigurationError(RuntimeError):
    """Raised when the selected LLM provider is not configured."""


def set_up_client(type: Literal["local"] | ProviderName = "local") -> OpenAI:
    """Create an OpenAI-compatible client for Ollama or OpenRouter."""
    if type in ("local", "ollama"):
        return OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    if type == "openrouter":
        import os

        api_key = os.getenv("OPENROUTER_KEY", "").strip()
        if not api_key or api_key == OPENROUTER_KEY_PLACEHOLDER:
            raise LLMConfigurationError(
                "Brak OPENROUTER_KEY. Dodaj klucz do pliku .env i uruchom aplikację ponownie."
            )
        return OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=60.0,
            max_retries=0,
        )
    raise ValueError(f"Nieznany dostawca modelu: {type}")


T = TypeVar("T", bound=BaseModel)


class LLMResponseError(RuntimeError):
    """Raised when a model fails to return valid structured output."""


class LLMOutputLimitError(LLMResponseError):
    """Raised when the output token limit prevents a complete model response."""

    def __init__(self, message: str, *, partial_response: str | None = None) -> None:
        super().__init__(message)
        self.partial_response = partial_response


def structured_output_error_detail(error: Exception) -> str:
    """Summarize structured-output failures without logging rejected input values."""
    if isinstance(error, ValidationError):
        details = []
        for item in error.errors(include_input=False, include_url=False, include_context=False):
            location = ".".join(str(part) for part in item["loc"]) or "odpowiedź"
            details.append(f"{location}: {item['msg']} ({item['type']})")
        return "; ".join(details) or "odpowiedź nie spełnia schematu"
    if isinstance(error, LLMResponseError):
        return str(error)
    return type(error).__name__


def recover_partial_structured_response(
    content: str | None,
    response_model: type[T],
) -> T | None:
    """Return a truncated JSON prefix only when it still validates as the requested model."""
    if not content:
        return None
    try:
        partial_data = from_json(content, allow_partial=True)
        return response_model.model_validate(partial_data)
    except (ValueError, ValidationError):
        return None


class LLMClient:
    """Thin wrapper around the OpenAI-compatible client for structured responses."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        on_wait: Callable[[], None] | None = None,
        stage: str = "llm_generation",
        session_id: str | None = None,
        provider: ProviderName = "ollama",
        context_tokens: int | None = None,
    ) -> None:
        """Initialize the client wrapper with an OpenAI-compatible client and model name."""
        self.client = client
        self.model_name = model_name
        self.on_wait = on_wait
        self.stage = stage
        self.session_id = session_id
        self.provider = provider
        self.context_tokens = context_tokens
        self.last_model: str | None = None

    def _call_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
        max_output_tokens: int,
        attempt: int,
    ) -> T:
        """Call the model with a structured response schema and return the parsed object."""
        started = perf_counter()
        response_content: str | None = None
        try:
            # Limit wyjścia jest zmniejszany, gdy prompt zajmuje większą część okna kontekstu.
            max_output_tokens = fit_output_tokens(
                provider=self.provider,
                prompt_chars=len(system_prompt) + len(user_prompt),
                max_output_tokens=max_output_tokens,
                stage=self.stage,
                session_id=self.session_id,
                context_tokens=self.context_tokens,
            )
            with llm_generation_lock(
                self.on_wait,
                stage=self.stage,
                session_id=self.session_id,
            ):
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=type_to_response_format_param(response_model),
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
            self.last_model = getattr(completion, "model", None)
            choice = completion.choices[0]
            response_content = choice.message.content
            if getattr(choice, "finish_reason", None) == "length":
                raise LLMOutputLimitError(
                    "Model przekroczył limit tokenów odpowiedzi.",
                    partial_response=response_content,
                )
            if response_content:
                result = response_model.model_validate_json(response_content)
            else:
                raise LLMResponseError("Model nie zwrócił danych.")
        except LLMOutputLimitError as exc:
            response_content = exc.partial_response
            retry_scheduled = attempt == 1
            LOGGER.warning(
                "llm_output_truncated stage=%s session_id=%s provider=%s model=%s "
                "max_output_tokens=%d attempt=%d retry_scheduled=%s "
                "elapsed_seconds=%.3f",
                self.stage,
                self.session_id or "-",
                self.provider,
                self.last_model or self.model_name,
                max_output_tokens,
                attempt,
                str(retry_scheduled).lower(),
                perf_counter() - started,
                extra={"session_id": self.session_id},
            )
            log_model_response(
                outcome="truncated",
                stage=self.stage,
                session_id=self.session_id,
                response=response_content,
                provider=self.provider,
                model=self.last_model or self.model_name,
                attempt=attempt,
            )
            raise
        except Exception as exc:
            retry_scheduled = attempt == 1 and isinstance(exc, (ValidationError, LLMResponseError))
            LOGGER.error(
                "llm_request_failed stage=%s session_id=%s provider=%s model=%s "
                "response_model=%s max_output_tokens=%d attempt=%d retry_scheduled=%s "
                "error_type=%s error_detail=%s "
                "elapsed_seconds=%.3f",
                self.stage,
                self.session_id or "-",
                self.provider,
                self.last_model or self.model_name,
                response_model.__name__,
                max_output_tokens,
                attempt,
                str(retry_scheduled).lower(),
                type(exc).__name__,
                structured_output_error_detail(exc),
                perf_counter() - started,
                extra={"session_id": self.session_id},
            )
            log_model_response(
                outcome="invalid",
                stage=self.stage,
                session_id=self.session_id,
                response=response_content,
                provider=self.provider,
                model=self.last_model or self.model_name,
                attempt=attempt,
            )
            raise

        LOGGER.info(
            "llm_request_completed stage=%s session_id=%s provider=%s model=%s "
            "max_output_tokens=%d attempt=%d elapsed_seconds=%.3f %s",
            self.stage,
            self.session_id or "-",
            self.provider,
            self.last_model or self.model_name,
            max_output_tokens,
            attempt,
            perf_counter() - started,
            usage_summary(completion),
            extra={"session_id": self.session_id},
        )
        return result

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        repair_prompt: str = REPAIR_PROMPT,
        response_model: type[T],
        temperature: float = 0.0,
        max_output_tokens: int,
    ) -> T:
        """Request structured output with an output-token cap and at most one repair."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens musi być dodatnie.")
        try:
            return self._call_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                attempt=1,
            )
        except LLMOutputLimitError as first_error:
            compact_prompt = user_prompt + "\n\n" + COMPACT_OUTPUT_INSTRUCTION
            try:
                return self._call_structured(
                    system_prompt=system_prompt,
                    user_prompt=compact_prompt,
                    response_model=response_model,
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    attempt=2,
                )
            except LLMOutputLimitError as retry_error:
                partial_response = retry_error.partial_response or first_error.partial_response
                recovered = recover_partial_structured_response(
                    partial_response,
                    response_model,
                )
                if recovered is not None:
                    LOGGER.warning(
                        "llm_partial_response_recovered stage=%s session_id=%s",
                        self.stage,
                        self.session_id or "-",
                        extra={"session_id": self.session_id},
                    )
                    return recovered
                raise LLMOutputLimitError(
                    "Odpowiedź nadal przekracza limit. Jej części nie dało się "
                    "zwalidować jako kompletnej struktury.",
                    partial_response=partial_response,
                ) from retry_error
            except (ValidationError, LLMResponseError) as retry_error:
                raise LLMOutputLimitError(
                    "Skrócona odpowiedź nie spełniła wymaganego schematu.",
                    partial_response=first_error.partial_response,
                ) from retry_error
        except (ValidationError, LLMResponseError) as first_error:
            repair = repair_prompt.format(validation_errors=str(first_error))
            try:
                return self._call_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt + "\n\n" + repair,
                    response_model=response_model,
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    attempt=2,
                )
            except LLMOutputLimitError:
                raise
            except (ValidationError, LLMResponseError) as second_error:
                raise LLMResponseError(
                    f"Model dwukrotnie zwrócił niepoprawny structured output dla "
                    f"{response_model.__name__}. Ostatni błąd: "
                    f"{structured_output_error_detail(second_error)}"
                ) from second_error
