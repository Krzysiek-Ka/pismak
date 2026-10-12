from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from src._shared.context_budget import fit_output_tokens, usage_summary
from src._shared.llm_runtime import llm_generation_lock
from src._shared.serwer import (
    COMPACT_OUTPUT_INSTRUCTION,
    LLMOutputLimitError,
    log_model_response,
    recover_partial_structured_response,
)

from ..rag_retrieval.models import SearchIntent
from .models import ResearchPlan

ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS = 4096
LOGGER = logging.getLogger("pismak")


class LLMQueryPlanner:
    """CaseBrief -> 3-5 celowanych zapytań do miejskiego RAG."""

    SYSTEM_PROMPT = """
Jesteś plannerem wyszukiwania źródeł dla Pismaka.

Na podstawie CaseBrief przygotuj 3-5 zapytań do bazy oficjalnych dokumentów
miejskich (strategie, polityki, programy, plany, standardy, zarządzenia).

Dla każdego intentu przygotuj:
- semantic_query: pełne pytanie/zdanie dobre dla wyszukiwania wektorowego,
- lexical_query: krótka kombinacja najważniejszych terminów/nazw/numerów,
- purpose: po co to zapytanie jest potrzebne,
- priority: 1-3.

Zasady:
1. Nie odpowiadaj na sprawę i nie twórz argumentacji.
2. Nie dodawaj faktów nieobecnych w CaseBrief.
3. Szukaj osobno źródeł dla problemu, celu i konkretnych requested_actions.
4. Jeśli CaseBrief zawiera dokładną nazwę ulicy, dokumentu, numer lub termin
   techniczny, zachowaj go w lexical_query.
5. Nie filtruj się wyłącznie do strategii wysokiego poziomu; szukaj też
   konkretnych standardów, programów i zasad wykonawczych.
""".strip()

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        on_wait: Any = None,
        session_id: str | None = None,
        allow_fallback: bool = True,
        provider: str = "ollama",
        context_tokens: int | None = None,
    ) -> None:
        """Initialize the query planner with an LLM client and model identifier."""
        self.client = client
        self.model = model
        self.on_wait = on_wait
        self.session_id = session_id
        self.allow_fallback = allow_fallback
        self.provider = provider
        self.context_tokens = context_tokens
        self.last_model: str | None = None

    @staticmethod
    def _brief_dict(case_brief: Any) -> dict[str, Any]:
        """Convert a CaseBrief-like value into a dictionary for planning."""
        if isinstance(case_brief, dict):
            return case_brief
        if isinstance(case_brief, BaseModel):
            return case_brief.model_dump(mode="json")
        raise TypeError("case_brief musi być dict albo Pydantic BaseModel")

    def _llm_plan(self, brief: dict[str, Any]) -> ResearchPlan:
        """Request a structured research plan from the configured language model."""
        schema = ResearchPlan.model_json_schema()
        started = perf_counter()
        user_prompt = json.dumps(brief, ensure_ascii=False, indent=2)
        content: str | None = None
        attempt = 1
        try:
            for attempt in (1, 2):
                max_tokens = fit_output_tokens(
                    provider=self.provider,
                    prompt_chars=len(self.SYSTEM_PROMPT) + len(user_prompt),
                    max_output_tokens=ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
                    stage="query_planner",
                    session_id=self.session_id,
                    context_tokens=self.context_tokens,
                )
                with llm_generation_lock(
                    self.on_wait,
                    stage="query_planner",
                    session_id=self.session_id,
                ):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=max_tokens,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ResearchPlan",
                                "strict": True,
                                "schema": schema,
                            },
                        },
                    )
                self.last_model = getattr(response, "model", None)
                choice = response.choices[0]
                content = choice.message.content
                if getattr(choice, "finish_reason", None) == "length":
                    LOGGER.warning(
                        "llm_output_truncated stage=query_planner session_id=%s provider=%s "
                        "model=%s max_output_tokens=%d attempt=%d retry_scheduled=%s",
                        self.session_id or "-",
                        self.provider,
                        self.last_model or self.model,
                        ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
                        attempt,
                        str(attempt == 1).lower(),
                        extra={"session_id": self.session_id},
                    )
                    log_model_response(
                        outcome="truncated",
                        stage="query_planner",
                        session_id=self.session_id,
                        response=content,
                        provider=self.provider,
                        model=self.last_model or self.model,
                        attempt=attempt,
                    )
                    if attempt == 1:
                        user_prompt += "\n\n" + COMPACT_OUTPUT_INSTRUCTION
                        continue
                    partial = recover_partial_structured_response(content, ResearchPlan)
                    if partial is not None:
                        LOGGER.warning(
                            "llm_partial_response_recovered stage=query_planner session_id=%s",
                            self.session_id or "-",
                            extra={"session_id": self.session_id},
                        )
                        return partial
                    raise LLMOutputLimitError(
                        "Plan wyszukiwania został ucięty i nie dało się go zwalidować.",
                        partial_response=content,
                    )
                if not content:
                    raise RuntimeError("Planner zwrócił pustą odpowiedź")
                plan = ResearchPlan.model_validate_json(content)
                break
            else:
                raise RuntimeError("Planner nie zwrócił odpowiedzi.")
        except LLMOutputLimitError:
            LOGGER.warning(
                "llm_output_limit stage=query_planner session_id=%s provider=%s model=%s "
                "max_output_tokens=%d elapsed_seconds=%.3f",
                self.session_id or "-",
                self.provider,
                self.last_model or self.model,
                ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
                perf_counter() - started,
                extra={"session_id": self.session_id},
            )
            log_model_response(
                outcome="invalid",
                stage="query_planner",
                session_id=self.session_id,
                response=content,
                provider=self.provider,
                model=self.last_model or self.model,
                attempt=attempt,
            )
            raise
        except Exception as exc:
            LOGGER.error(
                "llm_request_failed stage=query_planner session_id=%s provider=%s model=%s "
                "max_output_tokens=%d attempt=1 retry_scheduled=false error_type=%s "
                "elapsed_seconds=%.3f",
                self.session_id or "-",
                self.provider,
                self.last_model or self.model,
                ZRODLOWYSZUKIWACZ_PLANNER_MAX_OUTPUT_TOKENS,
                type(exc).__name__,
                perf_counter() - started,
                extra={"session_id": self.session_id},
            )
            raise
        self.last_model = getattr(response, "model", None)
        LOGGER.info(
            "llm_request_completed stage=query_planner session_id=%s provider=%s model=%s "
            "max_output_tokens=%d attempt=1 elapsed_seconds=%.3f %s",
            self.session_id or "-",
            self.provider,
            self.last_model or self.model,
            max_tokens,
            perf_counter() - started,
            usage_summary(response),
            extra={"session_id": self.session_id},
        )
        return plan

    @staticmethod
    def _fallback_plan(brief: dict[str, Any]) -> ResearchPlan:
        """Build a deterministic research plan from the brief when model planning is unavailable."""
        problem = str(brief.get("problem_description") or "").strip()
        goal = str(brief.get("desired_outcome") or "").strip()
        actions = [str(x).strip() for x in (brief.get("requested_actions") or []) if str(x).strip()]

        intents: list[SearchIntent] = []
        if problem:
            intents.append(
                SearchIntent(
                    semantic_query=problem,
                    lexical_query=problem[:220],
                    purpose="Źródła opisujące problem i właściwy kierunek polityki miasta.",
                    priority=3,
                )
            )
        if goal:
            intents.append(
                SearchIntent(
                    semantic_query=goal,
                    lexical_query=goal[:220],
                    purpose="Źródła wspierające oczekiwany rezultat.",
                    priority=2,
                )
            )
        for action in actions[:3]:
            intents.append(
                SearchIntent(
                    semantic_query=action,
                    lexical_query=action[:220],
                    purpose="Źródła odnoszące się do konkretnego postulatu.",
                    priority=3,
                )
            )

        return ResearchPlan(
            case_summary=goal or problem or "Sprawa z CaseBrief",
            intents=intents[:5],
        )

    def plan(self, case_brief: Any) -> ResearchPlan:
        """Return an LLM research plan or a deterministic fallback plan."""
        brief = self._brief_dict(case_brief)
        try:
            plan = self._llm_plan(brief)
            if plan.intents:
                return plan
            if not self.allow_fallback:
                raise RuntimeError("Planner zwrócił plan bez zapytań.")
        except Exception:
            if not self.allow_fallback:
                raise
        return self._fallback_plan(brief)
