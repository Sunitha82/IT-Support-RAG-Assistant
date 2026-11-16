"""Optional metadata-only tracing; telemetry must not break answering."""
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from dotenv import load_dotenv
from app.core.keyvault import get_secret

load_dotenv()
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_langfuse_client():
    if os.getenv("LANGFUSE_ENABLED", "false").lower() != "true":
        return None
    try:
        from langfuse import Langfuse
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        if not secret_key and os.getenv("USE_AZURE_KEY_VAULT", "false").lower() == "true":
            secret_key = get_secret("langfuse-secret-key")
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
        if not all([public_key, secret_key, base_url]):
            logger.warning("Langfuse configuration incomplete; tracing unavailable")
            return None
        return Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url,
                        timeout=3, flush_interval=1, tracing_enabled=True)
    except Exception:
        logger.warning("Langfuse initialization failed; answering continues without tracing")
        return None

class SafeObservation:
    def __init__(self, span=None):
        self.span = span

    @property
    def trace_id(self):
        try:
            return self.span.trace_id if self.span is not None else None
        except Exception:
            return None

    def update(self, **kwargs):
        if self.span is not None:
            try:
                self.span.update(**kwargs)
            except Exception:
                logger.warning("Langfuse observation update failed")

@contextmanager
def observation(name: str, as_type: str = "span", **kwargs):
    """Only pass counts/identifiers, never raw question/document/answer text."""
    manager = None
    span = SafeObservation()
    try:
        client = get_langfuse_client()
        if client is not None:
            candidate = client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
            entered = candidate.__enter__()
            manager = candidate
            span = SafeObservation(entered)
    except Exception:
        logger.warning("Langfuse observation could not start")
    try:
        yield span
    except BaseException as exc:
        # Avoid SDK auto-capture of sensitive exception messages.
        span.update(level="ERROR", status_message="Stage failed",
                    metadata={"error_type": type(exc).__name__})
        raise
    finally:
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                logger.warning("Langfuse observation could not finish")

def flush_traces():
    try:
        client = get_langfuse_client()
        if client is not None:
            client.flush()
    except Exception:
        logger.warning("Langfuse flush failed")
