from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Streamlit uruchamia skrypt z katalogiem pliku na sys.path, a niekoniecznie
# z katalogiem repozytorium; dodaj oba poziomy przed importami src.* i services.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("PISMAK_PROJECT_ROOT", str(HERE.parent.parent))).resolve()

for import_root in (PROJECT_ROOT, HERE):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from services import (
    MAX_INTERVIEW_QUESTIONS,
    build_zrodlowyszukiwator,
    close_case_logging,
    configure_logging,
    error_location,
    get_llm_client,
    get_openai_client,
    list_generation_models,
    llm_error_message,
    render_letter,
    save_run,
    validate_user_message,
)

from src._shared.serwer import OPENROUTER_MODEL, LLMClient, LLMOutputLimitError
from src.Dopytywator import prompts as dopytywator_prompts
from src.Dopytywator.constants import DOPYTYWATOR_MAX_OUTPUT_TOKENS
from src.Dopytywator.dopytywator_helper_functions import validate_brief_readiness
from src.Dopytywator.structures import CaseBrief, ConversationMessage, InterviewTurnResult
from src.Pismogenerator import pismogenerator
from src.Pismogenerator.pismogenerator import DraftValidationError
from src.Zrodlowyszukiwator.brief_to_evidence import EvidencePack

LOGGER = logging.getLogger("pismak")

st.set_page_config(
    page_title="Pismak",
    page_icon="📝",
    layout="wide",
)


def init_state() -> None:
    """Initialize missing Streamlit session-state values for a new case."""
    # Streamlit wykonuje cały skrypt ponownie po każdej interakcji.
    # session_state przechowuje dane, które mają przetrwać takie ponowne wykonanie.
    defaults = {
        "session_id": uuid.uuid4().hex,
        "stage": "INTERVIEW",
        "messages": [],
        "question_count": 0,
        "case_brief": None,
        "evidence_pack": None,
        "draft": None,
        "letter_markdown": None,
        "run_dir": None,
        "model_usage": {},
    }
    new_session = "session_id" not in st.session_state
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    configure_logging(PROJECT_ROOT, st.session_state.session_id)
    if new_session:
        LOGGER.info(
            "session_started session_id=%s",
            st.session_state.session_id,
            extra={"session_id": st.session_state.session_id},
        )


def notify_model_wait() -> None:
    """Show a non-sensitive notice while a model request waits for its turn."""
    st.info("Model jest zajęty. Oczekiwanie na swoją kolej...")


def show_thinking_overlay(message: str):
    """Display a large blocking overlay while a model request is running."""
    overlay = st.empty()
    overlay.markdown(
        f"""
        <style>
        .pismak-thinking-overlay {{
            position: fixed;
            inset: 0;
            z-index: 1000000;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(15, 23, 42, 0.72);
            backdrop-filter: blur(5px);
        }}
        .pismak-thinking-card {{
            display: flex;
            align-items: center;
            gap: 1.25rem;
            width: min(90vw, 560px);
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 1rem;
            background: var(--secondary-background-color, #fff);
            color: var(--text-color, #172033);
            box-shadow: 0 1.5rem 4rem rgba(0, 0, 0, 0.35);
        }}
        .pismak-thinking-spinner {{
            flex: 0 0 2.5rem;
            width: 2.5rem;
            height: 2.5rem;
            border: 4px solid rgba(128, 128, 128, 0.25);
            border-top-color: #2563eb;
            border-radius: 50%;
            animation: pismak-spin 0.8s linear infinite;
        }}
        .pismak-thinking-card strong {{ font-size: 1.35rem; }}
        .pismak-thinking-card p {{ margin: 0.35rem 0 0; opacity: 0.8; }}
        @keyframes pismak-spin {{ to {{ transform: rotate(360deg); }} }}
        </style>
        <div class="pismak-thinking-overlay" role="status" aria-live="polite">
            <div class="pismak-thinking-card">
                <span class="pismak-thinking-spinner"></span>
                <div><strong>Model myśli</strong><p>{message}</p></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return overlay


def reset_case() -> None:
    """Clear the current case state and initialize a fresh UI session."""
    # Usuwamy tylko dane tej sprawy; ustawienia modeli pozostają w widgetach.
    close_case_logging(st.session_state.session_id)
    for key in [
        "session_id",
        "stage",
        "messages",
        "question_count",
        "case_brief",
        "evidence_pack",
        "draft",
        "letter_markdown",
        "run_dir",
        "model_usage",
    ]:
        if key in st.session_state:
            del st.session_state[key]


# Klient Ollama jest zasobem współdzielonym, więc nie twórz go przy każdym rerun.
@st.cache_resource(show_spinner=False)
def load_client() -> OpenAI:
    """Load and cache the OpenAI-compatible Ollama client."""
    return get_openai_client()


# Lista modeli może być przechowywana przez 30 sekund zamiast pobierania jej stale.
@st.cache_data(ttl=30, show_spinner=False)
def load_models() -> list[str]:
    """Load and cache the available generation model names."""
    client = get_openai_client()
    return list_generation_models(client)


def record_model_usage(stage: str, provider: str, configured_model: str, llm: LLMClient) -> None:
    """Keep provider and returned model identifiers in the local run metadata."""
    model = llm.last_model or configured_model
    usage = st.session_state.model_usage
    usage.setdefault(stage, []).append({"provider": provider, "model": model})
    LOGGER.info(
        "llm_model_used stage=%s provider=%s model=%s session_id=%s",
        stage,
        provider,
        model,
        st.session_state.session_id,
        extra={"session_id": st.session_state.session_id},
    )


def require_external_processing_consent(providers: tuple[str, ...]) -> bool:
    """Prevent prompts from leaving the app until the user acknowledges the privacy notice."""
    if "openrouter" in providers and not st.session_state.get(
        "cfg_external_processing_consent", False
    ):
        st.error("Potwierdź zgodę na przetwarzanie zewnętrzne przed wysłaniem zapytania.")
        return False
    return True


def show_brief(brief: CaseBrief) -> None:
    """Render the current CaseBrief summary and its missing-information warnings."""
    with st.container(border=True):
        st.subheader("Podsumowanie sprawy")
        st.markdown(f"**{brief.title}**")
        st.write(brief.problem_description)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Oczekiwany rezultat**")
            st.write(brief.desired_outcome)
        with col2:
            st.markdown("**Lokalizacja**")
            st.write(brief.location or "—")

        st.markdown("**Wnioskowane działania**")
        for action in brief.requested_actions:
            st.markdown(f"- {action}")

        if brief.missing_information:
            with st.expander("Brakujące informacje"):
                for item in brief.missing_information:
                    st.markdown(f"- {item}")

        if brief.facts_to_verify:
            with st.expander("Fakty do weryfikacji"):
                for item in brief.facts_to_verify:
                    st.markdown(f"- {item}")


def show_evidence(evidence_pack: EvidencePack) -> None:
    """Render retrieved evidence excerpts and their source metadata."""
    evidence = getattr(evidence_pack, "evidence", [])
    with st.expander(
        f"Źródła znalezione przez Źródłowyszukiwator ({len(evidence)})",
        expanded=False,
    ):
        if not evidence:
            st.info("Nie znaleziono źródeł.")
            return

        for idx, item in enumerate(evidence, start=1):
            pages = (
                str(item.page_start)
                if item.page_start == item.page_end
                else f"{item.page_start}–{item.page_end}"
            )
            st.markdown(f"**[{idx}] {item.document_title} — s. {pages}**")
            if item.section_path:
                st.caption(" > ".join(item.section_path))
            st.write(item.relevance_reason)
            st.code(item.excerpt[:1200], language=None)


# Ten kod wykonuje się przy każdym rerun, dlatego inicjalizacja musi być idempotentna.
init_state()

st.title("Pismak - twój asystent w pisaniu wniosków")
st.caption("Od opisu sprawy, przez sprawdzone źródła, do projektu pisma.")

# Panel konfiguracji zapisuje wartości widgetów w session_state dzięki ich kluczom.
with st.container(border=True):
    st.subheader("Konfiguracja")

    provider_columns = st.columns([2, 2, 2, 1])
    provider_options = ["ollama", "openrouter"]
    provider_labels = {
        "ollama": "Lokalny model (Ollama)",
        "openrouter": "Losowy darmowy model (OpenRouter)",
    }

    with provider_columns[0]:
        dopytywator_provider = st.selectbox(
            "Dostawca — Dopytywator",
            provider_options,
            format_func=provider_labels.get,
            key="cfg_dopytywator_provider",
            disabled=st.session_state.question_count > 0,
        )

    with provider_columns[1]:
        zrodla_provider = st.selectbox(
            "Dostawca — Źródłowyszukiwacz",
            provider_options,
            format_func=provider_labels.get,
            key="cfg_zrodla_provider",
            disabled=st.session_state.case_brief is not None,
        )

    with provider_columns[2]:
        pismo_provider = st.selectbox(
            "Dostawca — Pismogenerator",
            provider_options,
            format_func=provider_labels.get,
            key="cfg_pismo_provider",
            disabled=st.session_state.draft is not None,
        )

    with provider_columns[3]:
        max_questions = int(
            st.number_input(
                "Max pytań",
                min_value=1,
                max_value=MAX_INTERVIEW_QUESTIONS,
                value=1,
                step=1,
                disabled=st.session_state.question_count > 0,
            )
        )

    models: list[str] = []
    model_list_error: Exception | None = None
    if "ollama" in (dopytywator_provider, zrodla_provider, pismo_provider):
        try:
            models = load_models()
        except Exception as exc:
            model_list_error = exc
            LOGGER.exception("model_list_failed")

    model_columns = st.columns([2, 2, 2, 1])

    def choose_model(label: str, provider: str, key: str, disabled: bool) -> str | None:
        if provider == "openrouter":
            st.caption(f"OpenRouter dobiera darmowy model ({OPENROUTER_MODEL}).")
            return OPENROUTER_MODEL
        if models:
            return st.selectbox(label, models, key=key, disabled=disabled)
        if model_list_error:
            st.error("Nie udało się pobrać modeli Ollama. Sprawdź, czy Ollama działa.")
        else:
            st.error("Brak modeli generatywnych w Ollama. Sprawdź `ollama list`.")
        return None

    with model_columns[0]:
        dopytywator_model = choose_model(
            "Model — Dopytywator",
            dopytywator_provider,
            "cfg_dopytywator_model",
            st.session_state.question_count > 0,
        )

    with model_columns[1]:
        zrodla_model = choose_model(
            "Model — Źródłowyszukiwacz",
            zrodla_provider,
            "cfg_zrodla_model",
            st.session_state.case_brief is not None,
        )
        st.caption("Planner i reranker. Embeddings: " + os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"))

    with model_columns[2]:
        pismo_model = choose_model(
            "Model — Pismogenerator",
            pismo_provider,
            "cfg_pismo_model",
            st.session_state.draft is not None,
        )

    selected_providers = (dopytywator_provider, zrodla_provider, pismo_provider)
    if "openrouter" in selected_providers:
        st.warning(
            "OpenRouter wysyła treść zapytań do usługi zewnętrznej i dalej do dostawcy "
            "wybranego przez router. Dostawcy mogą stosować różne zasady retencji i "
            "wykorzystania danych. Nie wysyłaj poufnych danych bez świadomej decyzji."
        )
        st.checkbox(
            "Rozumiem i zgadzam się na wysyłanie treści zapytań do OpenRouter.",
            key="cfg_external_processing_consent",
        )

    model_usage = st.session_state.model_usage
    if model_usage:
        with st.expander("Modele użyte w tej sprawie", expanded=False):
            for stage, calls in model_usage.items():
                if calls:
                    latest = calls[-1]
                    st.caption(f"{stage}: {latest['provider']} — {latest['model']}")

    if st.button("Nowa sprawa", type="secondary"):
        reset_case()
        st.rerun()


st.divider()
st.subheader("Rozmowa")

# Historia rozmowy jest renderowana ze stanu, a nie przechowywana w lokalnej zmiennej.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if st.session_state.case_brief is None:
    # Dopóki nie ma briefu, chat_input prowadzi wywiad z Dopytywatorem.
    placeholder = (
        "Opisz sprawę własnymi słowami..."
        if not st.session_state.messages
        else "Odpowiedz Dopytywatorowi..."
    )

    prompt = st.chat_input(placeholder)

    if prompt:
        if not require_external_processing_consent((dopytywator_provider,)):
            st.stop()
        if dopytywator_model is None:
            st.error("Wybierz dostawcę Ollama z dostępnym modelem albo OpenRouter.")
            st.stop()
        try:
            validate_user_message(prompt)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        # chat_input zwraca tekst tylko po zatwierdzeniu wiadomości przez użytkownika.
        st.session_state.messages.append({"role": "user", "content": prompt.strip()})

        thinking_overlay = show_thinking_overlay(
            "Analizuję opis i przygotowuję odpowiedź lub podsumowanie sprawy."
        )
        try:
            client = (
                load_client()
                if dopytywator_provider == "ollama"
                else get_llm_client(dopytywator_provider)
            )
            interview_started = time.perf_counter()
            conversation = [
                ConversationMessage.model_validate(message) for message in st.session_state.messages
            ]
            dopytywator_llm = LLMClient(
                client=client,
                model_name=dopytywator_model,
                on_wait=notify_model_wait,
                stage="dopytywator",
                session_id=st.session_state.session_id,
                provider=dopytywator_provider,
            )

            # Po osiągnięciu limitu pytań pomijamy kolejne pytanie i budujemy brief.
            if st.session_state.question_count >= max_questions:
                brief = dopytywator_llm.generate_structured(
                    system_prompt=dopytywator_prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT,
                    user_prompt=dopytywator_prompts.brief_user_prompt(conversation),
                    response_model=CaseBrief,
                    temperature=0.0,
                    max_output_tokens=DOPYTYWATOR_MAX_OUTPUT_TOKENS,
                )
                record_model_usage(
                    "Dopytywator",
                    dopytywator_provider,
                    dopytywator_model,
                    dopytywator_llm,
                )
                st.session_state.case_brief = brief
                st.session_state.stage = "BRIEF_READY"
            else:
                # W przeciwnym razie model decyduje, czy zadać następne pytanie.
                result = dopytywator_llm.generate_structured(
                    system_prompt=dopytywator_prompts.DOPYTYWATOR_INTERVIEW_SYSTEM_PROMPT,
                    user_prompt=dopytywator_prompts.interviewer_user_prompt(
                        conversation,
                        question_number=st.session_state.question_count + 1,
                        max_interview_questions=max_questions,
                    ),
                    response_model=InterviewTurnResult,
                    temperature=0.1,
                    max_output_tokens=DOPYTYWATOR_MAX_OUTPUT_TOKENS,
                )
                record_model_usage(
                    "Dopytywator",
                    dopytywator_provider,
                    dopytywator_model,
                    dopytywator_llm,
                )

                if result.ready_to_build_brief:
                    # Model uznał, że informacje są wystarczające do zbudowania briefu.
                    brief = dopytywator_llm.generate_structured(
                        system_prompt=dopytywator_prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT,
                        user_prompt=dopytywator_prompts.brief_user_prompt(conversation),
                        response_model=CaseBrief,
                        temperature=0.0,
                        max_output_tokens=DOPYTYWATOR_MAX_OUTPUT_TOKENS,
                    )
                    record_model_usage(
                        "Dopytywator",
                        dopytywator_provider,
                        dopytywator_model,
                        dopytywator_llm,
                    )
                    st.session_state.case_brief = brief
                    st.session_state.stage = "BRIEF_READY"

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "Mam wystarczające informacje. Przygotowałem podsumowanie sprawy."
                            ),
                        }
                    )
                else:
                    # Odpowiedź modelu jest pytaniem; zapisujemy je przed rerunem.
                    question = result.next_question.strip()
                    st.session_state.question_count += 1
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": question,
                        }
                    )

                    if st.session_state.question_count >= max_questions:
                        # Brief zostanie zbudowany po bieżącej odpowiedzi
                        # użytkownika w następnej turze.
                        pass

            if st.session_state.case_brief is not None:
                # Walidacja nie blokuje draftu, ale pokazuje użytkownikowi braki briefu.
                errors = validate_brief_readiness(st.session_state.case_brief)
                if errors:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "Brief został utworzony, ale ma braki: " + "; ".join(errors)
                            ),
                        }
                    )

                run_dir = save_run(
                    project_root=PROJECT_ROOT,
                    messages=st.session_state.messages,
                    case_brief=st.session_state.case_brief,
                    model_usage=st.session_state.model_usage,
                    run_dir=st.session_state.run_dir,
                )
                st.session_state.run_dir = run_dir

            LOGGER.info(
                "interview_turn_completed session_id=%s model=%s elapsed_seconds=%.3f",
                st.session_state.session_id,
                dopytywator_model,
                time.perf_counter() - interview_started,
                extra={"session_id": st.session_state.session_id},
            )
            # Wszystkie zmiany stanu są już zapisane, więc odświeżamy interfejs.
            st.rerun()

        except Exception as exc:
            if isinstance(exc, LLMOutputLimitError):
                LOGGER.warning(
                    "interview_output_truncated session_id=%s",
                    st.session_state.session_id,
                    extra={"session_id": st.session_state.session_id},
                )
                message = llm_error_message(exc)
                if exc.partial_response:
                    message += f"\n\nCzęściowa odpowiedź:\n\n{exc.partial_response[:1000]}"
                st.warning(message)
            else:
                message = llm_error_message(exc)
                location = error_location(exc)
                if location:
                    message += f"\n\nMiejsce błędu: {location}"
                LOGGER.error(
                    "interview_failed session_id=%s stage=dopytywator provider=%s "
                    "model=%s error_type=%s error_location=%s error_message=%s",
                    st.session_state.session_id,
                    dopytywator_provider,
                    dopytywator_model,
                    type(exc).__name__,
                    location or "-",
                    " ".join(message.splitlines()),
                    extra={"session_id": st.session_state.session_id},
                )
                st.error(message)
        finally:
            thinking_overlay.empty()

else:
    st.chat_input(
        "Wywiad zakończony — podsumowanie sprawy znajduje się poniżej.",
        disabled=True,
    )


if st.session_state.case_brief is not None:
    # Wyniki są celowo umieszczone pod historią pytań i odpowiedzi.
    show_brief(st.session_state.case_brief)

    if st.session_state.evidence_pack is not None:
        show_evidence(st.session_state.evidence_pack)

    if st.session_state.letter_markdown is None:
        if (
            st.button(
                "Znajdź źródła i wygeneruj pismo",
                type="primary",
                use_container_width=True,
            )
            and require_external_processing_consent((zrodla_provider, pismo_provider))
            and zrodla_model is not None
            and pismo_model is not None
        ):
            thinking_overlay = show_thinking_overlay(
                "Wyszukuję źródła i przygotowuję projekt pisma."
            )
            try:
                zrodla_client = (
                    load_client()
                    if zrodla_provider == "ollama"
                    else get_llm_client(zrodla_provider)
                )
                pipeline_started = time.perf_counter()

                with st.status("Uruchamiam pipeline...", expanded=True) as status:
                    st.write("1/2 Źródłowyszukiwator")
                    LOGGER.info(
                        "rag_started session_id=%s provider=%s model=%s",
                        st.session_state.session_id,
                        zrodla_provider,
                        zrodla_model,
                        extra={"session_id": st.session_state.session_id},
                    )
                    zrodla, docs, chunks = build_zrodlowyszukiwator(
                        client=zrodla_client,
                        model=zrodla_model,
                        on_wait=notify_model_wait,
                        session_id=st.session_state.session_id,
                        provider=zrodla_provider,
                    )
                    st.caption(f"Baza RAG: {docs} dokumentów, {chunks} chunków.")
                    LOGGER.info(
                        "rag_ready session_id=%s documents=%d chunks=%d",
                        st.session_state.session_id,
                        docs,
                        chunks,
                        extra={"session_id": st.session_state.session_id},
                    )

                    try:
                        evidence_pack = zrodla.run(st.session_state.case_brief)
                    finally:
                        for stage_name, component in (
                            ("Planner", zrodla.planner),
                            ("Reranker", zrodla.reranker),
                        ):
                            if component.last_model:
                                model_usage = st.session_state.model_usage
                                model_usage.setdefault(stage_name, []).append(
                                    {
                                        "provider": zrodla_provider,
                                        "model": component.last_model,
                                    }
                                )
                                LOGGER.info(
                                    "llm_model_used stage=%s provider=%s model=%s session_id=%s",
                                    stage_name,
                                    zrodla_provider,
                                    component.last_model,
                                    st.session_state.session_id,
                                    extra={"session_id": st.session_state.session_id},
                                )
                        st.session_state.run_dir = save_run(
                            project_root=PROJECT_ROOT,
                            messages=st.session_state.messages,
                            case_brief=st.session_state.case_brief,
                            model_usage=st.session_state.model_usage,
                            run_dir=st.session_state.run_dir,
                        )
                    st.session_state.evidence_pack = evidence_pack
                    st.write(f"Znaleziono {len(evidence_pack.evidence)} źródeł.")

                    st.write("2/2 Pismogenerator")
                    pismo_client = (
                        load_client()
                        if pismo_provider == "ollama"
                        else get_llm_client(pismo_provider)
                    )
                    pismo_llm = LLMClient(
                        client=pismo_client,
                        model_name=pismo_model,
                        on_wait=notify_model_wait,
                        stage="pismogenerator",
                        session_id=st.session_state.session_id,
                        provider=pismo_provider,
                    )
                    draft = pismogenerator.build_draft(
                        case_brief=st.session_state.case_brief,
                        evidence_pack=evidence_pack,
                        llm=pismo_llm,
                        llm_temperature=0.2,
                    )
                    record_model_usage(
                        "Pismogenerator",
                        pismo_provider,
                        pismo_model,
                        pismo_llm,
                    )
                    letter = render_letter(
                        draft,
                        evidence_pack,
                        st.session_state.case_brief.available_evidence,
                    )

                    st.session_state.draft = draft
                    st.session_state.letter_markdown = letter
                    st.session_state.stage = "DONE"

                    run_dir = save_run(
                        project_root=PROJECT_ROOT,
                        messages=st.session_state.messages,
                        case_brief=st.session_state.case_brief,
                        evidence_pack=evidence_pack,
                        draft=draft,
                        letter_markdown=letter,
                        model_usage=st.session_state.model_usage,
                        run_dir=st.session_state.run_dir,
                    )
                    st.session_state.run_dir = run_dir
                    LOGGER.info(
                        "pipeline_completed session_id=%s model=%s evidence=%d elapsed_seconds=%.3f",
                        st.session_state.session_id,
                        pismo_model,
                        len(evidence_pack.evidence),
                        time.perf_counter() - pipeline_started,
                        extra={"session_id": st.session_state.session_id},
                    )

                    status.update(label="Pismo gotowe", state="complete")

                st.rerun()

            except Exception as exc:
                if isinstance(exc, LLMOutputLimitError):
                    LOGGER.warning(
                        "pipeline_output_truncated session_id=%s",
                        st.session_state.session_id,
                        extra={"session_id": st.session_state.session_id},
                    )
                    message = llm_error_message(exc)
                    if exc.partial_response:
                        message += f"\n\nCzęściowa odpowiedź:\n\n{exc.partial_response[:1000]}"
                    st.warning(message)
                else:
                    if isinstance(exc, DraftValidationError) and exc.draft is not None:
                        st.session_state.run_dir = save_run(
                            project_root=PROJECT_ROOT,
                            messages=st.session_state.messages,
                            case_brief=st.session_state.case_brief,
                            evidence_pack=st.session_state.evidence_pack,
                            draft_with_error=exc.draft,
                            draft_validation_errors=exc.validation_errors,
                            model_usage=st.session_state.model_usage,
                            run_dir=st.session_state.run_dir,
                        )
                    message = llm_error_message(exc)
                    location = error_location(exc)
                    if location:
                        message += f"\n\nMiejsce błędu: {location}"
                    LOGGER.error(
                        "pipeline_failed session_id=%s error_type=%s error_location=%s "
                        "error_message=%s",
                        st.session_state.session_id,
                        type(exc).__name__,
                        location or "-",
                        " ".join(message.splitlines()),
                        extra={"session_id": st.session_state.session_id},
                    )
                    st.error(message)
            finally:
                thinking_overlay.empty()


# Gotowy draft jest wyświetlany niezależnie od tego, czy użytkownik odświeżył stronę.
if st.session_state.letter_markdown is not None:
    with st.container(border=True):
        st.subheader("Projekt pisma")
        st.markdown(st.session_state.letter_markdown)

        st.download_button(
            "Pobierz pismo.md",
            data=st.session_state.letter_markdown.encode("utf-8"),
            file_name="pismo.md",
            mime="text/markdown",
            type="primary",
            use_container_width=True,
        )
