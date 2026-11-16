
"""User-friendly error messages for the Streamlit frontend."""


def get_api_error_message(status_code: int) -> str:
    """Return a safe message for an API HTTP error."""

    error_messages = {
        401: (
        "Authentication failed. Please contact your "
        "administrator to check the API configuration."
        ),
        429: (
            "The AI service is busy. "
            "Please try again shortly."
        ),
        502: (
            "The AI service returned an invalid response. "
            "Please try again."
        ),
        503: (
            "The IT Support service is temporarily "
            "unavailable. Please try again later."
        ),
        504: (
            "The AI service took too long to respond. "
            "Please try again."
        ),
        500: (
            "Something went wrong while processing "
            "your question."
        ),
    }

    return error_messages.get(
        status_code,
        "Unable to process your request.",
    )


def get_request_error_message(error: Exception) -> str:
    """Convert a requests exception into a safe user message."""

    import requests

    if isinstance(error, requests.exceptions.HTTPError):
        response = error.response

        if response is not None:
            if response.status_code == 422:
                try:
                    detail = response.json().get("detail", {})
                except (ValueError, AttributeError):
                    detail = {}
                # Display only our known error code, never arbitrary server text.
                if isinstance(detail, dict) and detail.get("code") == "input_guardrail_rejected":
                    return (
                        "I can help with IT support questions, but I can't reveal internal "
                        "instructions or follow requests to override them. "
                        "Please ask about VPN, passwords, or software access."
                    )
                return "Please enter an IT support question of 2,000 characters or fewer."
            return get_api_error_message(response.status_code)

        return "Unable to process your request."

    if isinstance(error, requests.exceptions.Timeout):
        return (
            "The request timed out while contacting "
            "the IT Support API. Please try again."
        )

    if isinstance(error, requests.exceptions.ConnectionError):
        return (
            "Unable to connect to the IT Support API. "
            "Please check that the backend service is running."
        )

    if isinstance(error, requests.exceptions.RequestException):
        return (
            "A network error occurred while contacting "
            "the IT Support API."
        )

    return "An unexpected error occurred."
