from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from src._shared.llm_runtime import llm_generation_lock

from ..rag_retrieval.models import SearchIntent
from .models import ResearchPlan


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
    ) -> None:
        """Initialize the query planner with an LLM client and model identifier."""
        self.client = client
        self.model = model
        self.on_wait = on_wait
        self.session_id = session_id

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
        with llm_generation_lock(
            self.on_wait,
            stage="query_planner",
            session_id=self.session_id,
        ):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(brief, ensure_ascii=False, indent=2),
                    },
                ],
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ResearchPlan",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Planner zwrócił pustą odpowiedź")
        return ResearchPlan.model_validate_json(content)

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
        except Exception:
            pass
        return self._fallback_plan(brief)
