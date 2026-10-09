from collections.abc import Callable
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from src._shared.llm_runtime import llm_generation_lock

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1/"
MAX_INTERVIEW_QUESTIONS = 3
ANALYST_TEMPERATURE = 0.1
DEBUG = False


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


def set_up_client(type: str = "local") -> OpenAI:
    """Create an OpenAI-compatible client configured for the local Ollama backend."""
    if type == "local":
        return OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=180.0,
        )
    raise ValueError("only local implemented")


T = TypeVar("T", bound=BaseModel)


class LLMResponseError(RuntimeError):
    """Raised when a model fails to return valid structured output."""


class LLMClient:
    """Thin wrapper around the OpenAI-compatible client for structured responses."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        on_wait: Callable[[], None] | None = None,
        stage: str = "llm_generation",
        session_id: str | None = None,
    ) -> None:
        """Initialize the client wrapper with an OpenAI-compatible client and model name."""
        self.client = client
        self.model_name = model_name
        self.on_wait = on_wait
        self.stage = stage
        self.session_id = session_id

    def _call_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        """Call the model with a structured response schema and return the parsed object."""
        with llm_generation_lock(
            self.on_wait,
            stage=self.stage,
            session_id=self.session_id,
        ):
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                temperature=temperature,
            )
        message = completion.choices[0].message
        if getattr(message, "parsed", None) is not None:
            return message.parsed
        if message.content:
            return response_model.model_validate_json(message.content)
        raise LLMResponseError("Model nie zwrócił danych.")

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        repair_prompt: str = REPAIR_PROMPT,
        response_model: type[T],
        temperature: float = 0.0,
    ) -> T:
        """Request structured output and retry once with a repair prompt on validation failure."""
        try:
            return self._call_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                temperature=temperature,
            )
        except Exception as first_error:
            repair = repair_prompt.format(validation_errors=str(first_error))
            try:
                return self._call_structured(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt + "\n\n" + repair,
                    response_model=response_model,
                    temperature=0.0,
                )
            except Exception as second_error:
                raise LLMResponseError(
                    "Model dwukrotnie zwrócił niepoprawny structured output. "
                    f"Ostatni błąd: {second_error}"
                ) from second_error
