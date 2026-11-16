"""User-facing topic descriptions, filtered against the actual indexed sources."""
from pathlib import Path

TOPICS = [
    {"source": "vpn_troubleshooting.txt", "title": "VPN & connectivity",
     "description": "Troubleshoot access to internal applications when your VPN is connected.",
     "question": "My VPN is connected but I cannot access an internal application."},
    {"source": "password_policy.txt", "title": "Network passwords",
     "description": "Employee password rules, password changes and locked accounts.",
     "question": "How often should I change my network password?"},
    {"source": "software_access.txt", "title": "Software access",
     "description": "Find out how to request access and which approvals you need.",
     "question": "How do I request access to company software?"},
    {"source": "remote_work_policy.txt", "title": "Remote working",
     "description": "Understand remote-working approval and access requirements.",
     "question": "What are the requirements for working remotely?"},
    {"source": "helpdesk_support.txt", "title": "IT helpdesk",
     "description": "Check support hours and the kinds of issues the helpdesk handles.",
     "question": "What are the IT helpdesk working hours?"},
]


def build_topic_catalog(source_titles: dict[str, str]) -> list[dict[str, str]]:
    """Never advertise a known topic whose source is absent from the index."""
    topics = [dict(topic) for topic in TOPICS if topic["source"] in source_titles]
    known = {topic["source"] for topic in TOPICS}
    for source, title in sorted(source_titles.items()):
        if source not in known:
            title = title or Path(source).stem.replace("_", " ").replace("-", " ").capitalize()
            topics.append({"source": source, "title": title,
                           "description": "Ask about the guidance in this indexed document.",
                           "question": f"What guidance is available in the {title} document?"})
    return topics


def coverage_guidance(answer: str, topics: list[dict], outcome: str = "") -> str | None:
    """Offer useful next steps when retrieval or generation reports no guidance."""
    unsupported = outcome == "no_evidence" or answer.strip().lower().startswith(
        ("i don't have enough information", "i do not have enough information"))
    if not unsupported:
        return None
    titles = ", ".join(topic["title"].lower() for topic in topics)
    if titles:
        return (f"You can ask me about {titles}. If your question isn't covered, "
                "contact your IT helpdesk for the right guidance.")
    return "Try another IT support question, or contact your IT helpdesk for guidance."
