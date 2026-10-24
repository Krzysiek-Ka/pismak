"""Fixtures shared by the unit tests of every module."""

from __future__ import annotations

import httpx
import openai
import pytest

OPENROUTER_REQUEST = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _status_error(error_class: type[openai.APIStatusError], status_code: int):
    return error_class(
        "upstream failure",
        response=httpx.Response(status_code, request=OPENROUTER_REQUEST),
        body=None,
    )


@pytest.fixture(
    params=[
        pytest.param(lambda: openai.APITimeoutError(request=OPENROUTER_REQUEST), id="timeout"),
        pytest.param(
            lambda: openai.APIConnectionError(request=OPENROUTER_REQUEST), id="connection"
        ),
        pytest.param(lambda: _status_error(openai.RateLimitError, 429), id="rate_limit"),
        pytest.param(lambda: _status_error(openai.InternalServerError, 502), id="server_error"),
        pytest.param(lambda: _status_error(openai.BadRequestError, 400), id="bad_request"),
    ]
)
def provider_error(request) -> Exception:
    """A timeout or an HTTP error as the OpenAI SDK raises it for a chat request."""
    return request.param()
