"""Exercise the Streamlit user flow without a browser or backend request."""
from unittest.mock import Mock
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pytest
from app.ui.topic_catalog import TOPICS


@pytest.fixture(autouse=True)
def mock_topic_metadata(monkeypatch):
    response = Mock()
    response.json.return_value = {"topics": TOPICS, "source_count": 5}
    monkeypatch.setattr("requests.get", Mock(return_value=response))
    # Prevent a prior test's topic cache from masking the mock response.
    import streamlit as st
    st.cache_data.clear()


def test_ui_replaces_user_input_with_masked_question(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "question": "My email is <EMAIL_ADDRESS>. Help with VPN.",
        "answer": "Reconnect to the VPN.", "sources": ["vpn.txt"],
        "retrieval_details": [], "privacy": {"input_detections": 1,
        "context_detections": 0, "output_detections": 0, "output_checked": True},
        "trace_id": "a" * 32,
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    path = Path(__file__).resolve().parents[1] / "app/ui/streamlit_app.py"
    app = AppTest.from_file(str(path), default_timeout=30).run()
    app.chat_input[0].set_value("My email is input@example.com. Help with VPN.").run()
    assert not app.exception
    assert "input@example.com" not in str(app.session_state["messages"])
    assert "<EMAIL_ADDRESS>" in app.session_state["messages"][0]["content"]
    assert app.session_state["messages"][1]["privacy"]["input_detections"] == 1
    assert app.session_state["messages"][1]["trace_id"] == "a" * 32


def test_welcome_topics_and_about_are_visible_without_a_model_call(monkeypatch):
    post = Mock()
    monkeypatch.setattr("requests.post", post)
    path = Path(__file__).resolve().parents[1] / "app/ui/streamlit_app.py"
    app = AppTest.from_file(str(path), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Welcome."
    buttons = {button.label for button in app.button}
    assert {topic["question"] for topic in TOPICS} <= buttons
    assert "About this assistant" in {expander.label for expander in app.expander}
    post.assert_not_called()


def test_example_button_submits_question_and_new_conversation_resets(monkeypatch):
    response = Mock()
    response.json.return_value = {"question": TOPICS[0]["question"],
                                  "answer": "Reconnect to the VPN.", "sources": [TOPICS[0]["source"]]}
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", post)
    path = Path(__file__).resolve().parents[1] / "app/ui/streamlit_app.py"
    app = AppTest.from_file(str(path), default_timeout=30).run()
    next(b for b in app.button if b.label == TOPICS[0]["question"]).click().run()
    assert not app.exception
    assert post.call_args.kwargs["json"]["question"] == TOPICS[0]["question"]
    assert len(app.session_state["messages"]) == 2
    next(b for b in app.button if b.label == "New conversation").click().run()
    assert app.session_state["messages"] == []
    assert app.title[0].value == "Welcome."


def test_uncovered_question_offers_supported_topics(monkeypatch):
    response = Mock()
    response.json.return_value = {"question": "When can I change my Wi-Fi password?",
        "answer": "I don't have enough information in the knowledge base to answer that question.",
        "sources": []}
    monkeypatch.setattr("requests.post", Mock(return_value=response))
    path = Path(__file__).resolve().parents[1] / "app/ui/streamlit_app.py"
    app = AppTest.from_file(str(path), default_timeout=30).run()
    app.chat_input[0].set_value("When can I change my Wi-Fi password?").run()
    assert not app.exception
    assert any("network passwords" in item.value and "IT helpdesk" in item.value for item in app.info)
