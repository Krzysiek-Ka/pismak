"""Fixtures shared by the unit tests of every module."""

from __future__ import annotations

import httpx
import openai
import pytest

OPENROUTER_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _status_error(error_class: type[openai.APIStatusError], status_code: int, message: str):
    return error_class(
        message,
        response=httpx.Response(status_code, request=OPENROUTER_REQUEST),
        body=None,
    )


# Błędy, po których LLMClient ponawia żądanie do OpenRouter raz, z tym samym promptem.
TRANSIENT_ERRORS = [
    pytest.param(lambda: openai.APITimeoutError(request=OPENROUTER_REQUEST), id="timeout"),
    pytest.param(lambda: openai.APIConnectionError(request=OPENROUTER_REQUEST), id="connection"),
    pytest.param(
        lambda: _status_error(openai.InternalServerError, 502, "upstream failure"),
        id="server_error",
    ),
    pytest.param(
        lambda: _status_error(
            openai.NotFoundError, 404, "Error code: 404 - Provider returned error"
        ),
        id="provider_returned_error_404",
    ),
]
# Błędy, których ponowienie nic nie zmieni: zła konfiguracja, odrzucone żądanie, dzienny limit.
PERMANENT_ERRORS = [
    pytest.param(
        lambda: _status_error(openai.RateLimitError, 429, "free-models-per-day"), id="rate_limit"
    ),
    pytest.param(
        lambda: _status_error(openai.BadRequestError, 400, "invalid schema"), id="bad_request"
    ),
    pytest.param(
        lambda: _status_error(openai.AuthenticationError, 401, "invalid key"), id="authentication"
    ),
]


@pytest.fixture(params=TRANSIENT_ERRORS + PERMANENT_ERRORS)
def provider_error(request) -> Exception:
    """A timeout or an HTTP error as the OpenAI SDK raises it for a chat request."""
    return request.param()


@pytest.fixture(params=TRANSIENT_ERRORS)
def transient_provider_error(request) -> Exception:
    """A provider failure that LLMClient retries once on OpenRouter."""
    return request.param()


@pytest.fixture(params=PERMANENT_ERRORS)
def permanent_provider_error(request) -> Exception:
    """A provider failure that LLMClient never retries."""
    return request.param()
