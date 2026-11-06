
"""Reusable Azure OpenAI client for the IT Support RAG application."""

import logging
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    AzureOpenAI,
    RateLimitError,
)

from app.core.config import get_settings
from app.core.exceptions import (
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


class AzureLLMClient:
    """Handle communication with the configured Azure OpenAI deployment."""

    def __init__(self, client: AzureOpenAI | None = None) -> None:
        settings = get_settings()

        self.deployment = settings.azure_openai_deployment

        if not self.deployment:
            raise ValueError(
                "AZURE_OPENAI_DEPLOYMENT is not configured"
            )

        if client is not None:
            # Dependency injection supports mocked API calls in tests.
            self.client = client
            return

        if not settings.azure_openai_endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT is not configured"
            )

        if not settings.azure_openai_api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is not configured"
            )

        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            timeout=30.0,
            max_retries=2,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a text response using Azure OpenAI."""

        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("Prompts must not be empty")

        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            if not response.choices:
                raise LLMResponseError(
                    "Azure OpenAI returned no choices"
                )

            content = response.choices[0].message.content

            if not content or not content.strip():
                raise LLMResponseError(
                    "Azure OpenAI returned an empty response"
                )

            duration_ms = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            logger.info(
                "Azure OpenAI response generated successfully",
                extra={"duration_ms": duration_ms},
            )

            return content

        except APITimeoutError as exc:
            logger.warning(
                "Azure OpenAI request timed out"
            )
            raise LLMTimeoutError(
                "The AI service timed out"
            ) from exc

        except RateLimitError as exc:
            logger.warning(
                "Azure OpenAI rate limit exceeded"
            )
            raise LLMRateLimitError(
                "The AI service is receiving too many requests"
            ) from exc

        except AuthenticationError as exc:
            logger.error(
                "Azure OpenAI authentication failed"
            )
            raise LLMAuthenticationError(
                "The AI service authentication failed"
            ) from exc

        except APIConnectionError as exc:
            logger.error(
                "Azure OpenAI connection failed"
            )
            raise LLMUnavailableError(
                "The AI service is unavailable"
            ) from exc

        except APIStatusError as exc:
            logger.error(
                "Azure OpenAI returned an API error",
                extra={"status_code": exc.status_code},
            )
            raise LLMServiceError(
                "The AI service returned an error"
            ) from exc

        except LLMResponseError:
            logger.error(
                "Azure OpenAI returned an invalid response"
            )
            raise

        except Exception:
            logger.exception(
                "Unexpected Azure OpenAI client failure"
            )
            raise
