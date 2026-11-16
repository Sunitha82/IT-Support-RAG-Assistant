from types import SimpleNamespace
from fastapi.testclient import TestClient
from app.ui.topic_catalog import build_topic_catalog, coverage_guidance
from app.api.main import app
from app.api.routes import rag


def test_catalog_does_not_advertise_unindexed_topics():
    topics = build_topic_catalog({"password_policy.txt": "Password Policy"})
    assert [topic["title"] for topic in topics] == ["Network passwords"]
    assert build_topic_catalog({}) == []


def test_new_document_metadata_is_visible_without_inventing_scope():
    topics = build_topic_catalog({"printer_guide.txt": "Printer Guide"})
    assert topics[0]["title"] == "Printer Guide"
    assert "Printer Guide" in topics[0]["question"]


def test_normal_answer_gets_no_unsupported_warning():
    assert coverage_guidance("Reconnect to VPN", []) is None


def test_topics_endpoint_reads_active_index_and_requires_auth(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "topic-test-key")
    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(
        retrieval_mode="enhanced", parent_child_faiss_dir="active"))
    monkeypatch.setattr(rag, "ParentChildStore", lambda path: SimpleNamespace(parents={
        "p": SimpleNamespace(source="password_policy.txt", title="Password Policy")}))
    with TestClient(app) as client:
        assert client.get("/topics").status_code == 401
        response = client.get("/topics", headers={"X-API-Key": "topic-test-key"})
    assert response.status_code == 200
    assert response.json()["source_count"] == 1
    assert response.json()["topics"][0]["title"] == "Network passwords"
