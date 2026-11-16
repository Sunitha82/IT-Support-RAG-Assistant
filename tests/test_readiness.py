"""Readiness must inspect the index used to answer questions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import rag


@pytest.mark.parametrize("count,status", [(6, 200), (0, 503)])
def test_standard_readiness(monkeypatch, count, status):
    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(retrieval_mode="standard"))
    store = Mock()
    store.count.return_value = count
    monkeypatch.setattr(rag, "create_vector_store", lambda: store)
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == status
    if status == 200:
        assert response.json()["indexed_chunks"] == count


@pytest.mark.parametrize("parents,status", [({"parent-1": object()}, 200), ({}, 503)])
def test_enhanced_readiness_checks_active_index(monkeypatch, parents, status):
    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(
        retrieval_mode="enhanced", parent_child_faiss_dir="active-index"
    ))
    store = SimpleNamespace(
        vector_store=SimpleNamespace(count=lambda: 1, documents=[{"chunk_id": "child-1"}]),
        parents=parents,
        child_to_parent={"child-1": "parent-1"},
    )
    factory = Mock(return_value=store)
    monkeypatch.setattr(rag, "ParentChildStore", factory)
    standard_factory = Mock(side_effect=AssertionError("Wrong index checked"))
    monkeypatch.setattr(rag, "create_vector_store", standard_factory)
    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == status
    factory.assert_called_once_with("active-index")
    standard_factory.assert_not_called()
