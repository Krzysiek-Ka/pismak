import json
import logging
import os
import re
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal, TypeVar

from openai import APIConnectionError, APIStatusError, InternalServerError, OpenAI
from openai.lib._parsing import type_to_response_format_param
from pydantic import BaseModel, ValidationError
from pydantic_core import from_json

from src._shared.context_budget import fit_output_tokens, usage_summary
from src._shared.llm_runtime import llm_generation_lock

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# Lokalny model na CPU przetwarza pełny prompt nawet kilkanaście minut. Ponowienia SDK
# są wyłączone, bo każde wysłałoby to samo wielominutowe żądanie od początku.
OLLAMA_REQUEST_TIMEOUT_SECONDS = 900.0
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Darmowe modele rozumujące z openrouter/free przekraczały 60 s (złoty zbiór, 2026-10-24).
OPENROUTER_REQUEST_TIMEOUT_SECONDS = 120.0
OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_KEY_PLACEHOLDER = "sk-or-v1-WSTAW_TUTAJ_SWÓJ_KLUCZ"
# Router openrouter/free wybierał też modele, które ignorują response_format. Z tym ustawieniem
# OpenRouter kieruje żądanie tylko do dostawców obsługujących wszystkie jego parametry.
OPENROUTER_ROUTING = {"provider": {"require_parameters": True}}
ProviderName = Literal["ollama", "openrouter"]
LOGGER = logging.getLogger("pismak")
MAX_LOGGED_MODEL_RESPONSE_CHARS = 20_000
REDACTED = "<ukryte>"
# 'user_id': 'user_…' (repr słownika błędu) albo "user_id":"user_…" (JSON).
_USER_ID = re.compile(r"""(['"]user_id['"]\s*:\s*['"])[^'"]+""")
# „Not found for account 'sxpm…'” od dostawcy, także w escapowanym JSON-ie (\').
_ACCOUNT_ID = re.compile(r"""(for account \\?['"])[^'"\\]+""")
COMPACT_OUTPUT_INSTRUCTION = (
    "Poprzednia odpowiedź przekroczyła limit. Zwróć krótszą, zwięzłą odpowiedź "
    "z kompletem wymaganych pól i poprawną strukturą JSON. Skróć opisy i listy; "
    "nie dodawaj komentarzy ani tekstu poza JSON."
)
# Temperatura ponownego żądania. Przy 0.0 model zapętlony w pierwszej próbie powtarzał
# w drugiej dokładnie tę samą pętlę (e2e bielik-buspas, 2026-10-18).
RETRY_TEMPERATURE = 0.1


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


def ollama_base_url() -> str:
    """Return the OpenAI-compatible endpoint of the Ollama server named in OLLAMA_URL."""
    return os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/") + "/v1/"


def check_available_model() -> None:
    """Print the list of models currently exposed by a local Ollama instance."""
    base_url = ollama_base_url()
    client = OpenAI(
        base_url=base_url,
        api_key="ollama",  # wymagane przez SDK, ignorowane przez Ollamę
        timeout=180.0,
    )

    try:
        available_models = [m.id for m in client.models.list().data]
    except Exception as exc:
        raise RuntimeError(
            f"Nie udało się połączyć z Ollamą pod {base_url}. Sprawdź, czy Ollama działa."
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


def provider_request_options(provider: str) -> dict[str, Any]:
    """Return the extra arguments a provider's chat request needs for structured output."""
    if provider == "openrouter":
        return {"extra_body": OPENROUTER_ROUTING}
    return {}


def set_up_client(type: Literal["local"] | ProviderName = "local") -> OpenAI:
    """Create an OpenAI-compatible client for Ollama or OpenRouter."""
    if type in ("local", "ollama"):
        return OpenAI(
            base_url=ollama_base_url(),
            api_key="ollama",
            timeout=OLLAMA_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    if type == "openrouter":
        api_key = os.getenv("OPENROUTER_KEY", "").strip()
        if not api_key or api_key == OPENROUTER_KEY_PLACEHOLDER:
            raise LLMConfigurationError(
                "Brak OPENROUTER_KEY. Dodaj klucz do pliku .env i uruchom aplikację ponownie."
            )
        return OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=OPENROUTER_REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    raise ValueError(f"Nieznany dostawca modelu: {type}")


T = TypeVar("T", bound=BaseModel)


class LLMResponseError(RuntimeError):
    """Raised when a model fails to return valid structured output."""


class LLMOutputLimitError(LLMResponseError):
    """Raised when the output token limit prevents a complete model response."""

    def __init__(
        self,
        message: str,
        *,
        partial_response: str | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_response = partial_response
        # Limit faktycznie wysłany w żądaniu, po dopasowaniu do okna kontekstu.
        self.max_output_tokens = max_output_tokens


class LLMServiceError(RuntimeError):
    """Raised when the model server ended the request without any answer."""


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


def redact_account_ids(text: str) -> str:
    """Hide the OpenRouter user id and the upstream provider's account id in an error text."""
    text = _USER_ID.sub(r"\g<1>" + REDACTED, text)
    return _ACCOUNT_ID.sub(r"\g<1>" + REDACTED, text)


def error_log_detail(error: Exception) -> str:
    """Describe any failure on one log line; schema errors still omit rejected values."""
    if isinstance(error, (ValidationError, LLMResponseError)):
        return structured_output_error_detail(error)
    # Błąd OpenRouter niesie user_id konta i identyfikator konta dostawcy (np. Nvidia).
    return redact_account_ids(" ".join(str(error).split())) or type(error).__name__


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


def is_transient_provider_error(error: Exception) -> bool:
    """Return whether an OpenRouter failure is worth one more request with the same prompt."""
    # Timeout dziedziczy po APIConnectionError. „Provider returned error” (np. 404) oznacza
    # błąd dostawcy wybranego przez router, który przy następnym żądaniu wybiera zwykle inny.
    if isinstance(error, (APIConnectionError, InternalServerError)):
        return True
    return isinstance(error, APIStatusError) and "Provider returned error" in str(error)


def provider_error_message(completion: Any) -> str:
    """Describe a provider error sent in a 200 response that has no choices."""
    extra = getattr(completion, "model_extra", None) or {}
    error = extra.get("error") if isinstance(extra, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    detail = f": {redact_account_ids(message)}" if message else ""
    return f"Dostawca modelu zwrócił błąd zamiast odpowiedzi{detail}."


RetryKind = Literal["compact", "repair", "transient"]


class LLMClient:
    """
    Jedyne miejsce wywołania modelu z odpowiedzią strukturalną w Pismaku.

    Dopasowuje limit odpowiedzi do okna kontekstu, czeka w kolejce lokalnego modelu, waliduje
    odpowiedź i wykonuje najwyżej jedno ponowienie: prośbę o krótszą odpowiedź po ucięciu,
    naprawę schematu po niepoprawnej odpowiedzi albo to samo żądanie po chwilowym błędzie
    OpenRouter. Planner, reranker i pozostałe etapy korzystają wyłącznie z tej klasy.
    """

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

    def _retry_kind(self, error: Exception) -> RetryKind | None:
        """Return how a failed first attempt is retried, or None when it is not retried."""
        if isinstance(error, LLMOutputLimitError):
            return "compact"
        if isinstance(error, (ValidationError, LLMResponseError)):
            return "repair"
        if self.provider == "openrouter" and is_transient_provider_error(error):
            return "transient"
        return None

    def _call_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
        max_output_tokens: int,
        seed: int | None,
        attempt: int,
    ) -> T:
        """Call the model with a structured response schema and return the parsed object."""
        started = perf_counter()
        response_content: str | None = None
        finish_reason: str | None = None
        completion: Any = None
        # Seed tylko dla Ollamy: na OpenRouter require_parameters zawęziłby routing do dostawców
        # obsługujących seed.
        seed_option = {"seed": seed} if seed is not None and self.provider == "ollama" else {}
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
                provider=self.provider,
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
                    **seed_option,
                    **provider_request_options(self.provider),
                )
            self.last_model = getattr(completion, "model", None)
            choices = getattr(completion, "choices", None)
            if not choices:
                # OpenRouter przekazuje błąd dostawcy także w odpowiedzi 200, bez choices.
                raise LLMResponseError(provider_error_message(completion))
            choice = choices[0]
            response_content = choice.message.content
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                raise LLMOutputLimitError(
                    "Model przekroczył limit tokenów odpowiedzi.",
                    partial_response=response_content,
                    max_output_tokens=max_output_tokens,
                )
            if response_content:
                result = response_model.model_validate_json(response_content)
            elif self.provider == "ollama":
                # Gramatyka Ollamy nie pozwala na pustą odpowiedź; pusta oznacza, że proces
                # modelu się zatrzymał (np. z braku pamięci), a ponowienie trafi w to samo.
                raise LLMServiceError("Serwer modelu zakończył żądanie bez odpowiedzi.")
            else:
                raise LLMResponseError("Model nie zwrócił danych.")
        except LLMOutputLimitError as exc:
            response_content = exc.partial_response
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
                str(attempt == 1).lower(),
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
            retry_scheduled = attempt == 1 and self._retry_kind(exc) is not None
            # finish_reason=stop przy uciętym JSON-ie wskazuje na przerwanie po stronie serwera
            # modelu (np. brak pamięci), a nie na limit odpowiedzi.
            LOGGER.error(
                "llm_request_failed stage=%s session_id=%s provider=%s model=%s "
                "response_model=%s max_output_tokens=%d attempt=%d retry_scheduled=%s "
                "finish_reason=%s error_type=%s error_detail=%s "
                "elapsed_seconds=%.3f %s",
                self.stage,
                self.session_id or "-",
                self.provider,
                self.last_model or self.model_name,
                response_model.__name__,
                max_output_tokens,
                attempt,
                str(retry_scheduled).lower(),
                finish_reason or "-",
                type(exc).__name__,
                error_log_detail(exc),
                perf_counter() - started,
                usage_summary(completion),
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

    def _output_limit_error(
        self, message: str, error: LLMOutputLimitError, started: float
    ) -> LLMOutputLimitError:
        """Log the final output-limit failure and return the error to raise."""
        LOGGER.warning(
            "llm_output_limit stage=%s session_id=%s provider=%s model=%s "
            "max_output_tokens=%d elapsed_seconds=%.3f",
            self.stage,
            self.session_id or "-",
            self.provider,
            self.last_model or self.model_name,
            error.max_output_tokens or 0,
            perf_counter() - started,
            extra={"session_id": self.session_id},
        )
        return LLMOutputLimitError(
            message,
            partial_response=error.partial_response,
            max_output_tokens=error.max_output_tokens,
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        repair_prompt: str = REPAIR_PROMPT,
        response_model: type[T],
        temperature: float = 0.0,
        max_output_tokens: int,
        seed: int | None = None,
    ) -> T:
        """Request structured output with an output-token cap and at most one retry."""
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens musi być dodatnie.")
        started = perf_counter()
        call = {
            "system_prompt": system_prompt,
            "response_model": response_model,
            "max_output_tokens": max_output_tokens,
            "seed": seed,
        }
        try:
            return self._call_structured(
                **call, user_prompt=user_prompt, temperature=temperature, attempt=1
            )
        except Exception as exc:
            first_error = exc
            kind = self._retry_kind(exc)
            if kind is None:
                raise

        if kind == "compact":
            retry_prompt = user_prompt + "\n\n" + COMPACT_OUTPUT_INSTRUCTION
        elif kind == "repair":
            detail = structured_output_error_detail(first_error)
            retry_prompt = user_prompt + "\n\n" + repair_prompt.format(validation_errors=detail)
        else:
            retry_prompt = user_prompt
        # Chwilowy błąd dostawcy to nie wina odpowiedzi, więc żądanie zostaje takie samo.
        retry_temperature = temperature if kind == "transient" else RETRY_TEMPERATURE
        try:
            return self._call_structured(
                **call, user_prompt=retry_prompt, temperature=retry_temperature, attempt=2
            )
        except LLMOutputLimitError as retry_error:
            if retry_error.partial_response is None and isinstance(
                first_error, LLMOutputLimitError
            ):
                retry_error.partial_response = first_error.partial_response
            recovered = recover_partial_structured_response(
                retry_error.partial_response, response_model
            )
            if recovered is not None:
                LOGGER.warning(
                    "llm_partial_response_recovered stage=%s session_id=%s",
                    self.stage,
                    self.session_id or "-",
                    extra={"session_id": self.session_id},
                )
                return recovered
            raise self._output_limit_error(
                "Wynik modelu został ucięty i nie dało się go zwalidować jako kompletnej "
                "struktury.",
                retry_error,
                started,
            ) from retry_error
        except (ValidationError, LLMResponseError) as retry_error:
            detail = structured_output_error_detail(retry_error)
            if isinstance(first_error, LLMOutputLimitError):
                raise self._output_limit_error(
                    "Skrócona odpowiedź nie spełniła wymaganego schematu.", first_error, started
                ) from retry_error
            if kind == "transient":
                raise LLMResponseError(
                    f"Model nie zwrócił poprawnego structured output dla "
                    f"{response_model.__name__} po ponowieniu. Ostatni błąd: {detail}"
                ) from retry_error
            raise LLMResponseError(
                f"Model dwukrotnie zwrócił niepoprawny structured output dla "
                f"{response_model.__name__}. Ostatni błąd: {detail}"
            ) from retry_error
