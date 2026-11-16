"""Offline real-model regression checks. No LLM calls or trace exports."""
import os
os.environ["USE_AZURE_KEY_VAULT"] = "false"
os.environ["LANGFUSE_ENABLED"] = "false"

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from app.ingestion.document_loader import load_documents
from app.ingestion.parent_child_chunking import chunk_documents_parent_child
from app.retrieval.parent_child_store import ParentChildStore
from app.retrieval.enhanced_retriever import EnhancedRetriever

CASES = [
    ("My VPN is connected but I cannot access an internal application.", "vpn_troubleshooting.txt"),
    ("My secure work connection says connected, but company websites won't open. What should I try?", "vpn_troubleshooting.txt"),
    ("How do I request access to company software?", "software_access.txt"),
    ("How often should I change my network password?", "password_policy.txt"),
    ("What are the IT helpdesk working hours?", "helpdesk_support.txt"),
    ("What is the company's maternity leave entitlement?", None),
    ("What is the company's annual bonus policy?", None),
]

def main():
    with TemporaryDirectory() as directory:
        parents, children = chunk_documents_parent_child(load_documents())
        store = ParentChildStore(Path(directory))
        store.add_chunks(parents, children)
        retriever = EnhancedRetriever(store=store)
        failures = []
        for question, expected in CASES:
            results = retriever.retrieve(question, top_k=5)
            passed = (bool(results) and results[0].source == expected) if expected else not results
            print(json.dumps({"question": question, "passed": passed,
                              "results": [{"source": r.source, "distance": round(r.distance, 4),
                                           "rerank_score": round(r.rerank_score, 4)} for r in results]}), flush=True)
            if not passed:
                failures.append(question)
        if failures:
            raise SystemExit(f"Retrieval regressions: {failures}")

if __name__ == "__main__":
    main()
