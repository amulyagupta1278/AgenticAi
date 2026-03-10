"""Integration tests for the orchestrator pipeline.

Mocks out the sklearn/FAISS internals but exercises the full
classify → resolve → route flow.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mock_classifier():
    from src.agents.classifier_agent import ClassifierAgent
    agent = ClassifierAgent()
    encoder = MagicMock()
    encoder.classes_ = np.array(["billing_and_payments", "technical_support"])
    encoder.inverse_transform.return_value = np.array(["technical_support"])
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.12, 0.88]])
    vectorizer = MagicMock()
    vectorizer.transform.return_value = MagicMock()
    agent._model = model
    agent._vectorizer = vectorizer
    agent._encoder = encoder
    agent._ready = True
    return agent


def _mock_resolver(resolves=True):
    from src.agents.resolution_agent import ResolutionAgent
    agent = ResolutionAgent()
    agent._ready = True
    if resolves:
        agent._fallback = False
        mock_index = MagicMock()
        mock_index.ntotal = 50
        mock_index.search.return_value = (
            np.array([[0.82]], dtype=np.float32),
            np.array([[0]]),
        )
        agent._index = mock_index
        agent._metadata = [
            {"category": "technical_support", "subject": "crash fix",
             "answer": "Please reinstall the app."},
        ]
        mock_embed = MagicMock()
        mock_embed.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        agent._embed_model = mock_embed
    else:
        agent._fallback = True  # forces no resolution
    return agent


@pytest.fixture
def orchestrator_auto_resolve():
    """Orchestrator that will auto-resolve tickets."""
    from src.agents.orchestrator import OrchestratorAgent
    orch = OrchestratorAgent()
    orch.classifier = _mock_classifier()
    orch.resolver = _mock_resolver(resolves=True)
    orch.router.load()
    orch.prevention.load()
    orch._ready = True
    return orch


@pytest.fixture
def orchestrator_route():
    """Orchestrator that will route (not resolve) tickets."""
    from src.agents.orchestrator import OrchestratorAgent
    orch = OrchestratorAgent()
    orch.classifier = _mock_classifier()
    orch.resolver = _mock_resolver(resolves=False)
    orch.router.load()
    orch.prevention.load()
    orch._ready = True
    return orch


class TestOrchestratorAutoResolve:

    @patch("src.agents.orchestrator.start_ticket_trace", return_value=None)
    @patch("src.agents.orchestrator.end_ticket_trace")
    def test_auto_resolves(self, mock_end, mock_start, orchestrator_auto_resolve):
        result = orchestrator_auto_resolve.process(
            text="my app crashes on startup", ticket_id="TEST-001")
        assert result["status"] == "auto_resolved"
        assert result["resolution"]["resolved"] is True
        assert result["resolution"]["answer"] is not None

    @patch("src.agents.orchestrator.start_ticket_trace", return_value=None)
    @patch("src.agents.orchestrator.end_ticket_trace")
    def test_classification_in_response(self, mock_end, mock_start, orchestrator_auto_resolve):
        result = orchestrator_auto_resolve.process(text="upload fails")
        assert result["classification"]["category"] == "technical_support"
        assert result["classification"]["confidence"] > 0

    @patch("src.agents.orchestrator.start_ticket_trace", return_value=None)
    @patch("src.agents.orchestrator.end_ticket_trace")
    def test_no_routing_when_resolved(self, mock_end, mock_start, orchestrator_auto_resolve):
        result = orchestrator_auto_resolve.process(text="printer jam")
        assert result["routing"] is None

    @patch("src.agents.orchestrator.start_ticket_trace", return_value=None)
    @patch("src.agents.orchestrator.end_ticket_trace")
    def test_agent_trace_populated(self, mock_end, mock_start, orchestrator_auto_resolve):
        result = orchestrator_auto_resolve.process(text="slow loading")
        # at minimum classifier and resolver should be in the trace
        assert len(result["agent_trace"]) >= 2

    @patch("src.agents.orchestrator.start_ticket_trace", return_value=None)
    @patch("src.agents.orchestrator.end_ticket_trace")
    def test_processing_time_tracked(self, mock_end, mock_start, orchestrator_auto_resolve):
        result = orchestrator_auto_resolve.process(text="error on checkout")
        assert result["processing_time_ms"] > 0


class TestOrchestratorRouting:

    @patch("src.agents.orchestrator.start_ticket_trace", return_value=None)
    @patch("src.agents.orchestrator.end_ticket_trace")
    def test_routes_when_not_resolved(self, mock_end, mock_start, orchestrator_route):
        result = orchestrator_route.process(
            text="billing issue with my invoice", ticket_id="TEST-002")
        assert result["status"] in ("routed", "escalated")
        assert result["routing"] is not None
        assert "team" in result["routing"]

    @patch("src.agents.orchestrator.start_ticket_trace", return_value=None)
    @patch("src.agents.orchestrator.end_ticket_trace")
    def test_escalation_shows_in_status(self, mock_end, mock_start, orchestrator_route):
        result = orchestrator_route.process(
            text="URGENT production outage everything is down")
        if result["routing"]["escalated"]:
            assert result["status"] == "escalated"


class TestOrchestratorHealth:

    def test_health_when_ready(self, orchestrator_auto_resolve):
        h = orchestrator_auto_resolve.health()
        assert h["status"] == "ok"
        assert "classifier" in h["agents"]
        assert "resolution" in h["agents"]
        assert "routing" in h["agents"]
        assert "prevention" in h["agents"]

    def test_health_shows_rag_loaded(self, orchestrator_auto_resolve):
        h = orchestrator_auto_resolve.health()
        assert h["model_loaded"] is True
        # resolver not in fallback → rag is loaded
        assert h["rag_index_loaded"] is True

    def test_insights_returns_data(self, orchestrator_auto_resolve):
        result = orchestrator_auto_resolve.get_insights()
        # prevention agent returns a dict with these keys
        assert "prevention_recommendations" in result or "total_tickets_analyzed" in result
