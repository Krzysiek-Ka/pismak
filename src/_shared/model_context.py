from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from src._shared.llm_runtime import llm_generation_lock

LOGGER = logging.getLogger("pismak")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_FREE_ROUTER = "openrouter/free"
OPENROUTER_CACHE_SECONDS = 3600.0
LOOKUP_TIMEOUT_SECONDS = 10.0
MODEL_LOAD_TIMEOUT_SECONDS = 300.0

_STATE_LOCK = threading.Lock()
_LAST_KNOWN: dict[tuple[str, str], int] = {}
_OPENROUTER_CACHE: dict[str, Any] = {}


def _request_json(url: str, payload: dict[str, Any] | None = None, *, timeout: float) -> Any:
    """Send a GET, or a JSON POST when a payload is given, and decode the JSON response."""
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "pismak"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _ollama_url() -> str:
    """Return the base URL of the local Ollama server."""
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def _tagged(model: str) -> str:
    """Return an Ollama model name with its tag, treating a missing tag as `latest`."""
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


def _ollama_loaded_context(model: str) -> int | None:
    """Return the context window Ollama is using for a currently loaded model."""
    loaded = _request_json(f"{_ollama_url()}/api/ps", timeout=LOOKUP_TIMEOUT_SECONDS)
    for entry in loaded.get("models", []):
        if _tagged(str(entry.get("name") or "")) == _tagged(model) and entry.get("context_length"):
            return int(entry["context_length"])
    return None


def _ollama_context(
    model: str,
    on_wait: Callable[[], None] | None,
    session_id: str | None,
) -> int | None:
    """Return the window Ollama serves the model with, loading the model when necessary."""
    context = _ollama_loaded_context(model)
    if context is None:
        # Puste żądanie ładuje model z tym samym oknem, z jakim obsłuży on właściwe zapytanie,
        # a blokada pilnuje, by ładowanie nie nakładało się na trwającą generację.
        with llm_generation_lock(on_wait, stage="model_load", session_id=session_id):
            _request_json(
                f"{_ollama_url()}/api/generate",
                {"model": model, "stream": False},
                timeout=MODEL_LOAD_TIMEOUT_SECONDS,
            )
        context = _ollama_loaded_context(model)
    return context


def _openrouter_models() -> list[dict[str, Any]]:
    """Return the public OpenRouter model list, refreshed at most once per cache period."""
    with _STATE_LOCK:
        fetched_at = _OPENROUTER_CACHE.get("fetched_at")
        if fetched_at is not None and time.monotonic() - fetched_at < OPENROUTER_CACHE_SECONDS:
            return _OPENROUTER_CACHE["models"]
    models = _request_json(OPENROUTER_MODELS_URL, timeout=LOOKUP_TIMEOUT_SECONDS)["data"]
    with _STATE_LOCK:
        _OPENROUTER_CACHE.update(fetched_at=time.monotonic(), models=models)
    return models


def _openrouter_context(model: str) -> int | None:
    """Return the context window of an OpenRouter model or of the free router's candidates."""
    models = _openrouter_models()
    if model != OPENROUTER_FREE_ROUTER:
        windows = [m.get("context_length") for m in models if m.get("id") == model]
        return int(windows[0]) if windows and windows[0] else None

    # Router sam wybiera darmowy model, więc bezpieczne jest najmniejsze okno wśród tych,
    # które obsługują odpowiedź strukturalną wymaganą przez Pismaka.
    free = [m for m in models if str(m.get("id", "")).endswith(":free")]
    structured = [
        m
        for m in free
        if {"response_format", "structured_outputs"} & set(m.get("supported_parameters") or [])
    ]
    windows = [int(m["context_length"]) for m in structured or free if m.get("context_length")]
    return min(windows) if windows else None


def detect_context_tokens(
    provider: str,
    model: str,
    *,
    on_wait: Callable[[], None] | None = None,
    session_id: str | None = None,
) -> int | None:
    """Return the real context window of the selected model, or None when it is unknown."""
    try:
        if provider == "ollama":
            context = _ollama_context(model, on_wait, session_id)
        elif provider == "openrouter":
            context = _openrouter_context(model)
        else:
            context = None
    except Exception as exc:
        # Wykrywanie jest pomocnicze: właściwy błąd połączenia pokaże samo żądanie do modelu.
        LOGGER.warning(
            "context_window_detection_failed provider=%s model=%s session_id=%s error_type=%s",
            provider,
            model,
            session_id or "-",
            type(exc).__name__,
            extra={"session_id": session_id},
        )
        context = None

    with _STATE_LOCK:
        if context is None:
            context = _LAST_KNOWN.get((provider, model))
        else:
            _LAST_KNOWN[(provider, model)] = context
    LOGGER.info(
        "context_window_detected provider=%s model=%s session_id=%s context_tokens=%s",
        provider,
        model,
        session_id or "-",
        context or "-",
        extra={"session_id": session_id},
    )
    return context
