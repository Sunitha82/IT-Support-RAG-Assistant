
"""Streamlit frontend for the IT Support RAG Assistant."""

import os

import requests
import streamlit as st
from app.ui.error_messages import get_request_error_message



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

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
    }
    .app-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------
# RAG Insights
# ------------------------------

def display_rag_insights(retrieval_details):
    """Display similarity scores for retrieved ChromaDB chunks."""

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

        # Display exact retrieval scores.
        st.subheader("Retrieval details")

        table_data = [
            {
                "Rank": item["rank"],
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
        )


# ------------------------------
# Sidebar
# ------------------------------

with st.sidebar:

    st.title("IT Support")
    st.caption("AI-powered knowledge assistant")

    if st.button(
        "New conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("Example questions")

    examples = [
        "My VPN is connected but I cannot access an internal application.",
        "How often should I change my network password?",
        "How do I request access to company software?",
        "What are the IT helpdesk working hours?",
    ]

    selected_question = None

    for example in examples:
        if st.button(
            example,
            use_container_width=True,
        ):
            selected_question = example

    st.divider()

    st.caption(
        "Answers are based on indexed IT support documents."
    )


# ------------------------------
# Main page
# ------------------------------

st.title("IT Support Assistant")

st.markdown(
    '<p class="app-subtitle">'
    'Get answers to your IT questions using the company knowledge base.'
    '</p>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------------------
# Display chat history
# ------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Document sources"):
                for source in message["sources"]:
                    st.write(source)

        if message["role"] == "assistant":
            display_rag_insights(
                message.get("retrieval_details", [])
            )


# ------------------------------
# Chat input
# ------------------------------

typed_question = st.chat_input(
    "Ask a question about VPN, passwords, software access..."
)

question = selected_question or typed_question

if question:

    # Save the user's question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the IT knowledge base..."
        ):

            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    headers={"X-API-Key": API_KEY},
                    timeout=90,
                )

                response.raise_for_status()

                data = response.json()

                answer = data["answer"]
                sources = data.get("sources", [])
                retrieval_details = data.get(
                    "retrieval_details", []
                )

                # Display generated answer.
                st.markdown(answer)

                # Display document sources.
                if sources:
                    with st.expander("Document sources"):
                        for source in sources:
                            st.write(source)

                # Display similarity chart separately.
                display_rag_insights(
                    retrieval_details
                )

                # Save assistant response in chat history.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieval_details": retrieval_details,
                    }
                )

            
            
            except requests.exceptions.RequestException as exc:
                message = get_request_error_message(exc)
                st.error(message)

                # Display request ID when the API provides one.
                if (
                    isinstance(exc, requests.exceptions.HTTPError)
                    and exc.response is not None
                ):
                    response = exc.response

                    request_id = response.headers.get("X-Request-ID")

                    if request_id:
                        st.caption(
                            f"Support reference: {request_id}"
                        )

            except (KeyError, ValueError, TypeError):
                st.error(
                    "The API returned an unexpected response."
                )


            except requests.exceptions.Timeout:
                st.error(
                    "The request timed out while contacting "
                    "the IT Support API. Please try again."
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Unable to connect to the IT Support API. "
                    "Please check that the backend service is running."
                )

            except requests.exceptions.RequestException:
                st.error(
                    "A network error occurred while contacting "
                    "the IT Support API."
                )

            except (KeyError, ValueError, TypeError):
                st.error(
                    "The API returned an unexpected response."
                )

