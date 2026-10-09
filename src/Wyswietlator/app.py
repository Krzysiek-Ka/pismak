from __future__ import annotations

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
    configure_logging,
    get_openai_client,
    list_generation_models,
    render_letter,
    save_run,
    validate_user_message,
)

from src._shared.serwer import LLMClient
from src.Dopytywator import prompts as dopytywator_prompts
from src.Dopytywator.dopytywator_helper_functions import validate_brief_readiness
from src.Dopytywator.structures import CaseBrief, ConversationMessage, InterviewTurnResult
from src.Pismogenerator import pismogenerator
from src.Zrodlowyszukiwator.brief_to_evidence import EvidencePack

LOGGER = configure_logging(PROJECT_ROOT)

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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            if key == "session_id":
                LOGGER.info("session_started session_id=%s", value)


def notify_model_wait() -> None:
    """Show a non-sensitive notice when another session is using the local model."""
    st.info("Lokalny model jest zajęty. Oczekiwanie na swoją kolej...")


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
    for key in [
        "stage",
        "messages",
        "question_count",
        "case_brief",
        "evidence_pack",
        "draft",
        "letter_markdown",
        "run_dir",
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

st.title("Pismak")
st.caption("Od opisu sprawy, przez sprawdzone źródła, do projektu pisma.")

try:
    models = load_models()
except Exception:
    LOGGER.exception("model_list_failed")
    st.error("Model lokalny jest obecnie niedostępny. Spróbuj ponownie za chwilę.")
    st.stop()

if not models:
    st.error("Ollama nie zwróciła żadnego modelu generatywnego. Sprawdź `ollama list`.")
    st.stop()


# Panel konfiguracji zapisuje wartości widgetów w session_state dzięki ich kluczom.
with st.container(border=True):
    st.subheader("Konfiguracja")

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])

    with c1:
        dopytywator_model = st.selectbox(
            "Dopytywator",
            models,
            key="cfg_dopytywator_model",
            disabled=st.session_state.question_count > 0,
        )

    with c2:
        zrodla_model = st.selectbox(
            "Źródłowyszukiwator",
            models,
            key="cfg_zrodla_model",
            disabled=st.session_state.case_brief is not None,
        )
        st.caption("Planner i reranker. Embeddings: " + os.getenv("RAG_EMBEDDING_MODEL", "bge-m3"))

    with c3:
        pismo_model = st.selectbox(
            "Pismogenerator",
            models,
            key="cfg_pismo_model",
            disabled=st.session_state.draft is not None,
        )

    with c4:
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
            client = load_client()
            interview_started = time.perf_counter()
            conversation = [
                ConversationMessage.model_validate(message) for message in st.session_state.messages
            ]

            # Po osiągnięciu limitu pytań pomijamy kolejne pytanie i budujemy brief.
            if st.session_state.question_count >= max_questions:
                brief = LLMClient(
                    client=client,
                    model_name=dopytywator_model,
                    on_wait=notify_model_wait,
                    stage="case_brief",
                    session_id=st.session_state.session_id,
                ).generate_structured(
                    system_prompt=dopytywator_prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT,
                    user_prompt=dopytywator_prompts.brief_user_prompt(conversation),
                    response_model=CaseBrief,
                    temperature=0.0,
                )
                st.session_state.case_brief = brief
                st.session_state.stage = "BRIEF_READY"
            else:
                # W przeciwnym razie model decyduje, czy zadać następne pytanie.
                result = LLMClient(
                    client=client,
                    model_name=dopytywator_model,
                    on_wait=notify_model_wait,
                    stage="dopytywator",
                    session_id=st.session_state.session_id,
                ).generate_structured(
                    system_prompt=dopytywator_prompts.DOPYTYWATOR_INTERVIEW_SYSTEM_PROMPT,
                    user_prompt=dopytywator_prompts.interviewer_user_prompt(
                        conversation,
                        question_number=st.session_state.question_count + 1,
                        max_interview_questions=max_questions,
                    ),
                    response_model=InterviewTurnResult,
                    temperature=0.1,
                )

                if result.ready_to_build_brief:
                    # Model uznał, że informacje są wystarczające do zbudowania briefu.
                    brief = LLMClient(
                        client=client,
                        model_name=dopytywator_model,
                        on_wait=notify_model_wait,
                        stage="case_brief",
                        session_id=st.session_state.session_id,
                    ).generate_structured(
                        system_prompt=dopytywator_prompts.DOPYTYWATOR_BUILD_BRIEF_SYSTEM_PROMPT,
                        user_prompt=dopytywator_prompts.brief_user_prompt(conversation),
                        response_model=CaseBrief,
                        temperature=0.0,
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
                    run_dir=st.session_state.run_dir,
                )
                st.session_state.run_dir = run_dir

            LOGGER.info(
                "interview_turn_completed session_id=%s model=%s elapsed_seconds=%.3f",
                st.session_state.session_id,
                dopytywator_model,
                time.perf_counter() - interview_started,
            )
            # Wszystkie zmiany stanu są już zapisane, więc odświeżamy interfejs.
            st.rerun()

        except Exception:
            LOGGER.exception("interview_failed session_id=%s", st.session_state.session_id)
            st.error("Wystąpił błąd podczas rozmowy. Spróbuj ponownie za chwilę.")
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
        if st.button(
            "Znajdź źródła i wygeneruj pismo",
            type="primary",
            use_container_width=True,
        ):
            thinking_overlay = show_thinking_overlay(
                "Wyszukuję źródła i przygotowuję projekt pisma."
            )
            try:
                client = load_client()
                pipeline_started = time.perf_counter()

                with st.status("Uruchamiam pipeline...", expanded=True) as status:
                    st.write("1/2 Źródłowyszukiwator")
                    LOGGER.info(
                        "rag_started session_id=%s model=%s",
                        st.session_state.session_id,
                        zrodla_model,
                    )
                    zrodla, docs, chunks = build_zrodlowyszukiwator(
                        client=client,
                        model=zrodla_model,
                        on_wait=notify_model_wait,
                        session_id=st.session_state.session_id,
                    )
                    st.caption(f"Baza RAG: {docs} dokumentów, {chunks} chunków.")
                    LOGGER.info(
                        "rag_ready session_id=%s documents=%d chunks=%d",
                        st.session_state.session_id,
                        docs,
                        chunks,
                    )

                    evidence_pack = zrodla.run(st.session_state.case_brief)
                    st.session_state.evidence_pack = evidence_pack
                    st.write(f"Znaleziono {len(evidence_pack.evidence)} źródeł.")

                    st.write("2/2 Pismogenerator")
                    draft = pismogenerator.build_draft(
                        case_brief=st.session_state.case_brief,
                        evidence_pack=evidence_pack,
                        llm=LLMClient(
                            client=client,
                            model_name=pismo_model,
                            on_wait=notify_model_wait,
                            stage="pismogenerator",
                            session_id=st.session_state.session_id,
                        ),
                        llm_temperature=0.2,
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
                        run_dir=st.session_state.run_dir,
                    )
                    st.session_state.run_dir = run_dir
                    LOGGER.info(
                        "pipeline_completed session_id=%s model=%s evidence=%d elapsed_seconds=%.3f",
                        st.session_state.session_id,
                        pismo_model,
                        len(evidence_pack.evidence),
                        time.perf_counter() - pipeline_started,
                    )

                    status.update(label="Pismo gotowe", state="complete")

                st.rerun()

            except Exception:
                LOGGER.exception("pipeline_failed session_id=%s", st.session_state.session_id)
                st.error("Wystąpił błąd podczas przetwarzania sprawy. Spróbuj ponownie za chwilę.")
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
