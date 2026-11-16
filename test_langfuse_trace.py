"""Optional manual connectivity check. Importing this file sends nothing."""
import argparse
from app.observability.langfuse_tracing import get_langfuse_client, observation, flush_traces

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send-test", action="store_true",
                        help="Authenticate and send a metadata-only test trace to configured Langfuse")
    args = parser.parse_args()
    if not args.send_test:
        print("No trace sent. Configure Langfuse, then use --send-test when ready.")
        return
    client = get_langfuse_client()
    if client is None:
        raise SystemExit("Tracing is disabled or unavailable. Check LANGFUSE_ENABLED, keys and base URL.")
    try:
        authenticated = client.auth_check()
    except Exception:
        raise SystemExit("Langfuse authentication check failed. Check connectivity and project configuration.")
    if not authenticated:
        raise SystemExit("Langfuse did not accept the configured project credentials.")
    with observation("rag-connectivity-test", input={"test_type": "connectivity"}) as span:
        span.update(output={"status": "completed"})
        trace_id = span.trace_id
    flush_traces()
    print("Trace queued; flush completed. Confirm delivery in the Langfuse project.")
    print("Trace ID:", trace_id)

if __name__ == "__main__":
    main()
