
"""Application-specific exceptions."""


class LLMServiceError(Exception):
    """Base exception for LLM service failures."""


class LLMTimeoutError(LLMServiceError):
    """The LLM request exceeded its timeout."""


class LLMRateLimitError(LLMServiceError):
    """The LLM provider rejected the request due to rate limits."""


class LLMAuthenticationError(LLMServiceError):
    """The LLM provider rejected the configured credentials."""


class LLMUnavailableError(LLMServiceError):
    """The LLM provider is temporarily unavailable."""


class LLMResponseError(LLMServiceError):
    """The LLM returned an invalid or empty response."""


class RetrievalServiceError(Exception):
    """Base exception for retrieval failures."""


class VectorStoreError(RetrievalServiceError):
    """ChromaDB storage or search operation failed."""


class EmbeddingServiceError(RetrievalServiceError):
    """Failed to generate an embedding."""
