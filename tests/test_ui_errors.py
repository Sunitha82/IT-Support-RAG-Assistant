
"""Tests for user-friendly API error messages."""

import pytest

from app.ui.error_messages import get_api_error_message
import requests

from app.ui.error_messages import get_request_error_message


@pytest.mark.parametrize(
    "status_code",
    [429, 500, 502, 503, 504],
)
def test_http_error_message(status_code):
    """HTTP errors should produce the correct user message."""

    response = requests.Response()
    response.status_code = status_code

    error = requests.exceptions.HTTPError(
        "Simulated API failure",
        response=response,
    )

    assert get_request_error_message(error) == (
        get_api_error_message(status_code)
    )


def test_connection_error_message():
    error = requests.exceptions.ConnectionError(
        "Connection refused"
    )

    message = get_request_error_message(error)

    assert "Unable to connect" in message
    assert "Connection refused" not in message


def test_timeout_error_message():
    error = requests.exceptions.Timeout(
        "Request timed out"
    )

    message = get_request_error_message(error)

    assert "timed out" in message.lower()


def test_generic_network_error_message():
    error = requests.exceptions.RequestException(
        "Unexpected network failure"
    )

    message = get_request_error_message(error)

    assert "network error" in message.lower()


@pytest.mark.parametrize(
    "status_code, expected_message",
    [
        (429, "The AI service is busy."),
        (502, "The AI service returned an invalid response."),
        (503, "The IT Support service is temporarily unavailable."),
        (504, "The AI service took too long to respond."),
        (500, "Something went wrong while processing your question."),
        (400, "Unable to process your request."),
    ],
)
def test_api_error_messages(status_code, expected_message):
    """Verify safe messages for different HTTP statuses."""

    message = get_api_error_message(status_code)

    assert expected_message in message

def test_authentication_error_message():
    """HTTP 401 should display a user-friendly authentication error."""

    from app.ui.error_messages import get_api_error_message

    message = get_api_error_message(401)

    assert "Authentication failed" in message
    assert "administrator" in message


def test_guardrail_error_is_meaningful_without_reflecting_server_text():
    response = requests.Response()
    response.status_code = 422
    response._content = b'{"detail":{"code":"input_guardrail_rejected","message":"private server details"}}'
    message = get_request_error_message(requests.HTTPError(response=response))
    assert "IT support" in message
    assert "internal instructions" in message
    assert "private server details" not in message
