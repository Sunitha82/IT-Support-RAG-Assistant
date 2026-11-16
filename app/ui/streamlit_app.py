
"""Streamlit frontend for the IT Support RAG Assistant."""

import os

import requests
import streamlit as st
from app.ui.error_messages import get_request_error_message
from app.ui.topic_catalog import coverage_guidance
from app.ui.styles import CSS


def display_protection_details(data):
    """Show measured detections without exposing the original personal details."""
    privacy = data.get("privacy", {})
    if privacy:
        with st.expander("Privacy checks and tracing"):
            st.write(f"Personal details masked in your question: {privacy.get('input_detections', 0)}")
            st.write(f"Personal details masked in retrieved evidence: {privacy.get('context_detections', 0)}")
            if privacy.get("output_checked"):
                st.write(f"Personal details masked in the generated answer: {privacy.get('output_detections', 0)}")
            else:
                st.caption("No model answer was generated; output masking was not needed.")
            st.caption("Detection is best effort and does not guarantee that every personal detail is recognised.")
            if data.get("trace_id"):
                st.code(data["trace_id"], language=None)
                st.caption("Langfuse trace queued. Confirm delivery in your Langfuse project. Traces contain counts and timings, not question or answer text.")
            else:
                st.caption("Langfuse tracing is disabled or unavailable for this request.")



# ------------------------------
# Page configuration
# ------------------------------

st.set_page_config(
    page_title="IT Support Assistant",
    page_icon="💻",
    layout="wide",
)

API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
)
API_KEY = os.getenv("RAG_API_KEY", "")
#st.sidebar.caption(f"Connected API: {API_URL}")


# ------------------------------
# Styling
# ------------------------------

st.markdown(CSS, unsafe_allow_html=True)

# ------------------------------
# RAG Insights
# ------------------------------

def display_rag_insights(retrieval_details, similarity_threshold=0.50):
    """Display retrieval scores and visually flag chunks below the threshold.

    This threshold is diagnostic only; it does not filter LLM context.
    """

    if not retrieval_details:
        return

    # Sort retrieved chunks by rank.
    details = sorted(
        retrieval_details,
        key=lambda item: item["rank"],
    )

    with st.expander("RAG Insights — Document Similarity"):

        st.caption(
            "These scores show how closely each retrieved "
            "document chunk matches your question. "
            "They do not represent the probability "
            "that the answer is correct."
        )
        st.caption(
            "Cosine scores belong to matched child chunks. The text below is the expanded parent section. "
            "The cross-encoder reranks candidates; its score is not a percentage or a confidence value."
        )

        # Use chunk IDs to distinguish chunks
        # from the same document.
        labels = [
            f"{item['source']} — {item['chunk_id']}"
            for item in details
        ]

        similarities = [
            round(float(item["similarity"]) * 100, 2)
            for item in details
        ]

        # Display horizontal similarity chart.
        chart_data = {
            "Document chunk": labels,
            "Cosine similarity (%)": similarities,
        }

        st.bar_chart(
            chart_data,
            x="Document chunk",
            y="Cosine similarity (%)",
            horizontal=True,
            sort=False,
        )

        # Diagnostic highlighting only: the API decides which chunks reach the LLM.
        passing = sum(float(item["similarity"]) >= similarity_threshold for item in details)
        st.caption(
            f"Threshold: {similarity_threshold:.0%} · "
            f"{passing} pass · {len(details) - passing} below threshold. "
            "This setting does not change retrieval or LLM context."
        )

        # Display exact retrieval scores.
        st.subheader("Retrieval details")

        table_data = [
            {
                "Threshold status": (
                    "PASS" if float(item["similarity"]) >= similarity_threshold
                    else "BELOW"
                ),
                "Rank": item["rank"],
                "Matched child": item.get("child_id"),
                "Reranker score": item.get("rerank_score"),
                "Fusion score (RRF)": item.get("rrf_score"),
                "Source": item["source"],
                "Chunk ID": item["chunk_id"],
                "Distance": item["distance"],
                "Similarity (%)": round(
                    float(item["similarity"]) * 100, 2
                ),
            }
            for item in details
        ]

        st.dataframe(
            table_data,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Threshold status": st.column_config.TextColumn("Threshold status"),
            },
        )

        # Show clear per-chunk pass/fail highlighting independently of chart colours.
        for item in details:
            similarity = float(item["similarity"])
            label = f"Rank {item['rank']} · {item['source']} · {item['chunk_id']}"
            if similarity >= similarity_threshold:
                st.success(f"PASS · {label} · {similarity:.1%}")
            else:
                st.warning(f"BELOW THRESHOLD · {label} · {similarity:.1%}")

        st.subheader("Retrieved document text")
        st.caption("These are the document chunks returned by retrieval. They may contain internal information; only show this panel to authorised users.")
        for item in details:
            chunk_label = f"Chunk {item['rank']} — {item['source']} ({item['chunk_id']})"
            with st.expander(chunk_label):
                st.caption(f"Similarity: {float(item['similarity']):.1%} | Distance: {item['distance']}")
                st.text(item.get("content") or "Chunk text not provided by API. Rebuild the API with the updated response schema.")



# ------------------------------
# Navigation, topic discovery and welcome screen
# ------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_topics(base_url, api_key):
    try:
        response = requests.get(f"{base_url.rstrip('/')}/topics",
                                headers={"X-API-Key": api_key}, timeout=8)
        response.raise_for_status()
        payload = response.json()
        return payload.get("topics", [])
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return []


if "messages" not in st.session_state:
    st.session_state.messages = []

topics = load_topics(API_URL, API_KEY)
selected_question = None

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">IT</div><div>'
                '<div class="brand-name">IT Support</div>'
                '<div class="brand-caption">Your workplace knowledge guide</div></div></div>',
                unsafe_allow_html=True)
    if st.button("New conversation", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
    with st.expander("About this assistant"):
        st.write("Ask an IT question in your own words. I look for relevant guidance "
                 "in your company's indexed documents and show the sources behind the answer.")
        st.markdown("**What I can help with**")
        if topics:
            for topic in topics:
                st.markdown(f"**{topic['title']}**")
                st.caption(topic["description"])
        else:
            st.caption("The current topic list could not be loaded. Refresh it below.")
        st.markdown("**A few things to know**")
        st.write("I can explain guidance, but I can't reset passwords, change settings, "
                 "grant access or raise a ticket for you.")
        st.write("If I can't find enough information, I'll say so. Your IT helpdesk "
                 "can guide you on questions outside these documents.")
        st.caption("Employee network password rules are not automatically Wi-Fi or router password rules.")
        st.caption("Recognised personal details are masked before search and AI generation. "
                   "Detection is best effort; avoid sharing sensitive information you don't need to include.")
    if st.button("Refresh available topics", use_container_width=True):
        load_topics.clear()
        st.rerun()
    with st.expander("Demo & search settings"):
        top_k = st.slider("Maximum results", min_value=1, max_value=10, value=5, step=1,
                          help="Maximum number of retrieved sections supplied as context.")
        similarity_threshold = st.slider(
            "Highlight similarity above", min_value=0.0, max_value=1.0, value=0.50, step=0.05,
            help="Changes visual highlighting only. It does not filter evidence or change answers.")
        st.caption("Open RAG Insights under an answer to inspect the search. "
                   "Similarity is not answer confidence.")
    st.divider()
    st.caption("Company guidance, with sources.")
    st.caption("Check the source for policy details. For urgent support, contact your IT helpdesk.")

st.markdown('<div class="eyebrow">YOUR IT KNOWLEDGE, MADE SIMPLE</div>', unsafe_allow_html=True)
if not st.session_state.messages:
    st.title("Welcome.")
    st.markdown('<p class="welcome-copy">A little clarity for your working day. '
                'Ask about the IT guidance in your company documents, '
                'and I\'ll help you find an answer with sources you can check.</p>', unsafe_allow_html=True)
    # Inline on the welcome screen so the page starts at the greeting rather
    # than auto-scrolling past it to a bottom-pinned chat widget.
    with st.container():
        typed_question = st.chat_input("Ask about your workplace IT guidance…", key="welcome_question")
    if topics:
        st.caption(f"{len(topics)} topics available in the current knowledge base")
        st.subheader("What can I help you with?")
        for start in range(0, len(topics), 3):
            columns = st.columns(3, gap="medium")
            for offset, topic in enumerate(topics[start:start + 3]):
                with columns[offset]:
                    with st.container(border=True, key=f"topic_card_{start + offset}"):
                        st.subheader(topic["title"])
                        st.write(topic["description"])
                        if st.button(topic["question"], key=f"example_{topic['source']}",
                                     use_container_width=True):
                            selected_question = topic["question"]
            # Keep the final row balanced with a useful scope reminder.
            remainder = len(topics[start:start + 3])
            if start + 3 >= len(topics) and remainder < 3:
                with columns[remainder]:
                    with st.container(border=True, key="topic_card_scope"):
                        st.subheader("Looking for something else?")
                        st.write("I only answer from the documents available to me. "
                                 "If your topic isn't covered, your IT helpdesk can point you in the right direction.")
                        st.caption("See About this assistant for the current scope.")
    else:
        st.info("The available topic list is temporarily unavailable. "
                "You can still ask an IT question, or use Refresh available topics in the sidebar.")
    st.caption("Choose a question above, or type your own below.")
else:
    st.title("IT Support Assistant")
    st.caption("Clear answers, with the sources behind them.")


def display_answer(data, similarity_threshold):
    st.markdown(data["answer"])
    guidance = coverage_guidance(data["answer"], topics, data.get("outcome", ""))
    if guidance:
        st.info(guidance)
    if data.get("sources"):
        with st.expander("Document sources"):
            labels = {topic["source"]: topic["title"] for topic in topics}
            for source in data["sources"]:
                st.write(labels.get(source, source))
                if source in labels:
                    st.caption(source)
    display_protection_details(data)
    display_rag_insights(data.get("retrieval_details", []), similarity_threshold)


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/support_agent:"):
        if message["role"] == "user":
            st.text(message["content"])
        else:
            display_answer({**message, "answer": message["content"]}, similarity_threshold)

if st.session_state.messages:
    typed_question = st.chat_input("Ask about your workplace IT guidance…", key="conversation_question")
question = selected_question or typed_question
if question:
    user_message_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": "Question submitted."})
    with st.chat_message("user", avatar=":material/person:"):
        user_question_placeholder = st.empty()
        user_question_placeholder.markdown("Processing your question…")
    with st.chat_message("assistant", avatar=":material/support_agent:"):
        with st.spinner("Finding the right guidance…"):
            try:
                response = requests.post(
                    f"{API_URL.rstrip('/')}/ask",
                    json={"question": question, "top_k": top_k},
                    headers={"X-API-Key": API_KEY}, timeout=90)
                response.raise_for_status()
                data = response.json()
                safe_question = data["question"]
                st.session_state.messages[user_message_index]["content"] = safe_question
                user_question_placeholder.text(safe_question)
                display_answer(data, similarity_threshold)
                st.session_state.messages.append({
                    "role": "assistant", "content": data["answer"],
                    "sources": data.get("sources", []),
                    "retrieval_details": data.get("retrieval_details", []),
                    "privacy": data.get("privacy", {}), "trace_id": data.get("trace_id"),
                    "tracing_status": data.get("tracing_status"),
                    "outcome": data.get("outcome", ""),
                })
                st.rerun()
            except requests.exceptions.RequestException as exc:
                user_question_placeholder.markdown("Question submitted.")
                if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None and exc.response.status_code == 422:
                    st.warning(get_request_error_message(exc))
                else:
                    st.error(get_request_error_message(exc))
                if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                    request_id = exc.response.headers.get("X-Request-ID")
                    if request_id and exc.response.status_code != 422:
                        st.caption(f"Support reference: {request_id}")
            except (KeyError, ValueError, TypeError):
                user_question_placeholder.markdown("Question submitted.")
                st.error("The answer couldn't be displayed. Please try again.")
