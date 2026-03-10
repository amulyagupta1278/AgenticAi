"""API-level tests for the /agent/* endpoints.

Uses FastAPI TestClient with mocked agent internals so we don't need
real model files or a running server.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _build_mock_orchestrator():
    """Build a mock OrchestratorAgent that returns realistic responses."""
    orch = MagicMock()
    orch._ready = True

    orch.process.return_value = {
        "ticket_id": "API-001",
        "status": "auto_resolved",
        "classification": {
            "category": "technical_support",
            "confidence": 0.87,
            "all_scores": {"technical_support": 0.87, "billing_and_payments": 0.13},
            "confidence_tier": "high",
        },
        "resolution": {
            "resolved": True,
            "answer": "Try clearing your browser cache and restarting.",
            "source_ticket_subject": "Browser issues",
            "similarity_score": 0.82,
            "matched_category": "technical_support",
        },
        "routing": None,
        "processing_time_ms": 45.2,
        "agent_trace": [],
    }

    orch.get_insights.return_value = {
        "generated_at": "2025-01-15T10:00:00",
        "total_tickets_analyzed": 5000,
        "insights": [
            {
                "category": "technical_support",
                "trend": "increasing",
                "ticket_count": 1200,
                "percentage": 24.0,
                "recurring_tags": ["login", "crash"],
                "recommendation": "Investigate login-related failures.",
            }
        ],
        "top_recurring_issues": [
            {"category": "technical_support", "tag": "login", "count": 320}
        ],
        "prevention_recommendations": [
            "Add self-service password reset to reduce login tickets.",
        ],
    }

    orch.health.return_value = {
        "status": "ok",
        "agents": {
            "classifier": {"agent": "classifier", "ok": True},
            "resolution": {"agent": "resolution", "ok": True},
            "routing": {"agent": "routing", "ok": True},
            "prevention": {"agent": "prevention", "ok": True},
        },
        "rag_index_loaded": True,
        "model_loaded": True,
    }

    return orch


@pytest.fixture
def client():
    """TestClient with mocked agent system and sklearn models."""
    mock_orch = _build_mock_orchestrator()

    # also mock out the sklearn state for /predict, /health, /classes
    encoder = MagicMock()
    encoder.classes_ = np.array(["billing_and_payments", "technical_support"])
    encoder.inverse_transform.return_value = np.array(["technical_support"])
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.15, 0.85]])
    vectorizer = MagicMock()
    vectorizer.transform.return_value = MagicMock()
    mock_state = {"model": model, "vectorizer": vectorizer, "encoder": encoder}

    with patch("src.api.agent_router._orchestrator", mock_orch):
        with patch("src.api.main._state", mock_state):
            from src.api.main import app
            yield TestClient(app, raise_server_exceptions=True)


# ── POST /agent/resolve ─────────────────────────────────────────────────

class TestResolveEndpoint:

    def test_resolve_returns_200(self, client):
        resp = client.post("/agent/resolve", json={
            "description": "My application crashes when uploading large files",
        })
        assert resp.status_code == 200

    def test_resolve_has_expected_fields(self, client):
        resp = client.post("/agent/resolve", json={
            "description": "Cannot process payment on checkout page",
            "ticket_id": "T-999",
        })
        body = resp.json()
        assert "status" in body
        assert "classification" in body
        assert "resolution" in body
        assert body["classification"]["category"] == "technical_support"

    def test_resolve_auto_resolved_status(self, client):
        resp = client.post("/agent/resolve", json={
            "description": "browser keeps freezing on dashboard",
        })
        assert resp.json()["status"] == "auto_resolved"

    def test_resolve_includes_answer(self, client):
        resp = client.post("/agent/resolve", json={
            "description": "page loads slowly after update",
        })
        body = resp.json()
        assert body["resolution"]["resolved"] is True
        assert body["resolution"]["answer"] is not None

    def test_resolve_short_description_rejected(self, client):
        resp = client.post("/agent/resolve", json={"description": "hi"})
        assert resp.status_code == 422

    def test_resolve_empty_body_rejected(self, client):
        resp = client.post("/agent/resolve", json={})
        assert resp.status_code == 422


# ── GET /agent/insights ─────────────────────────────────────────────────

class TestInsightsEndpoint:

    def test_insights_returns_200(self, client):
        resp = client.get("/agent/insights")
        assert resp.status_code == 200

    def test_insights_has_recommendations(self, client):
        resp = client.get("/agent/insights")
        body = resp.json()
        assert "prevention_recommendations" in body
        assert len(body["prevention_recommendations"]) > 0

    def test_insights_has_category_data(self, client):
        resp = client.get("/agent/insights")
        body = resp.json()
        assert len(body["insights"]) > 0
        assert body["insights"][0]["category"] == "technical_support"


# ── GET /agent/status ───────────────────────────────────────────────────

class TestStatusEndpoint:

    def test_status_returns_200(self, client):
        resp = client.get("/agent/status")
        assert resp.status_code == 200

    def test_status_shows_agents(self, client):
        body = client.get("/agent/status").json()
        assert body["status"] == "ok"
        assert "classifier" in body["agents"]
        assert "resolution" in body["agents"]

    def test_status_shows_rag_loaded(self, client):
        body = client.get("/agent/status").json()
        assert body["rag_index_loaded"] is True
        assert body["model_loaded"] is True


# ── Status endpoint without agents ──────────────────────────────────────

def test_status_when_not_initialized():
    """When the agent system hasn't started, should still return 200."""
    with patch("src.api.agent_router._orchestrator", None):
        with patch("src.api.main._state", {}):
            from src.api.main import app
            c = TestClient(app)
            resp = c.get("/agent/status")
            assert resp.status_code == 200
            assert resp.json()["status"] == "not_initialized"
