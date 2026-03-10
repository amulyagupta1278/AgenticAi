"""Unit tests for individual agents — everything is mocked so we don't
need real model files or FAISS indexes to run these."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── ClassifierAgent ─────────────────────────────────────────────────────

class TestClassifierAgent:

    @pytest.fixture
    def agent(self):
        from src.agents.classifier_agent import ClassifierAgent

        agent = ClassifierAgent()

        # mock out the sklearn artifacts
        encoder = MagicMock()
        encoder.classes_ = np.array([
            "billing_and_payments", "general_inquiry", "technical_support"])
        encoder.inverse_transform.return_value = np.array(["technical_support"])

        model = MagicMock()
        model.predict.return_value = np.array([2])
        model.predict_proba.return_value = np.array([[0.05, 0.10, 0.85]])

        vectorizer = MagicMock()
        vectorizer.transform.return_value = MagicMock()

        agent._model = model
        agent._vectorizer = vectorizer
        agent._encoder = encoder
        agent._ready = True
        return agent

    def test_returns_correct_category(self, agent):
        result = agent.process(text="my app keeps crashing when I upload files")
        assert result["category"] == "technical_support"

    def test_confidence_is_a_float(self, agent):
        result = agent.process(text="cannot reset password")
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1.0

    def test_high_confidence_tier(self, agent):
        # mock returns 0.85 → should be "high" (threshold 0.75)
        result = agent.process(text="internet not working")
        assert result["confidence_tier"] == "high"

    def test_medium_confidence_tier(self, agent):
        agent._model.predict_proba.return_value = np.array([[0.30, 0.20, 0.50]])
        result = agent.process(text="some vague issue")
        assert result["confidence_tier"] == "medium"

    def test_low_confidence_tier(self, agent):
        agent._model.predict_proba.return_value = np.array([[0.35, 0.30, 0.35]])
        result = agent.process(text="umm hello")
        assert result["confidence_tier"] == "low"

    def test_all_scores_present(self, agent):
        result = agent.process(text="refund request for order 123")
        assert result["all_scores"] is not None
        assert "technical_support" in result["all_scores"]

    def test_classes_property(self, agent):
        assert len(agent.classes) == 3

    def test_health_when_ready(self, agent):
        h = agent.health()
        assert h["ok"] is True
        assert h["agent"] == "classifier"


# ── ResolutionAgent ─────────────────────────────────────────────────────

class TestResolutionAgent:

    @pytest.fixture
    def agent(self):
        from src.agents.resolution_agent import ResolutionAgent

        agent = ResolutionAgent()
        # simulate a loaded FAISS setup
        agent._ready = True
        agent._fallback = False

        # mock FAISS index
        mock_index = MagicMock()
        mock_index.ntotal = 100
        # simulate: one good match (score=0.85 at index 3) and some noise
        mock_index.search.return_value = (
            np.array([[0.85, 0.60, 0.30]], dtype=np.float32),
            np.array([[3, 7, 12]]),
        )
        agent._index = mock_index

        # fake metadata
        agent._metadata = [
            {"category": "billing_and_payments", "subject": "charge question", "answer": "ans0"},
            {"category": "technical_support", "subject": "app crash", "answer": "ans1"},
            {"category": "billing_and_payments", "subject": "refund", "answer": "ans2"},
            {"category": "technical_support", "subject": "upload bug", "answer": "Try clearing cache"},
            {"category": "general_inquiry", "subject": "hours", "answer": "ans4"},
            {"category": "billing_and_payments", "subject": "invoice", "answer": "ans5"},
            {"category": "technical_support", "subject": "slow load", "answer": "ans6"},
            {"category": "general_inquiry", "subject": "where office", "answer": "ans7"},
            {"category": "technical_support", "subject": "login error", "answer": "ans8"},
            {"category": "billing_and_payments", "subject": "payment fail", "answer": "ans9"},
            {"category": "technical_support", "subject": "api error", "answer": "ans10"},
            {"category": "general_inquiry", "subject": "misc", "answer": "ans11"},
            {"category": "billing_and_payments", "subject": "tax", "answer": "ans12"},
        ]

        # mock embedding model
        mock_embed = MagicMock()
        mock_embed.encode.return_value = np.random.randn(1, 384).astype(np.float32)
        agent._embed_model = mock_embed

        return agent

    def test_resolves_when_similar(self, agent):
        result = agent.process(
            text="file upload keeps failing",
            category="technical_support",
            confidence=0.80,
        )
        assert result["resolved"] is True
        assert result["answer"] == "Try clearing cache"
        assert result["similarity_score"] >= 0.40

    def test_skips_on_low_confidence(self, agent):
        result = agent.process(
            text="something weird happening",
            category="technical_support",
            confidence=0.30,  # below MIN_CONFIDENCE
        )
        assert result["resolved"] is False

    def test_no_match_wrong_category(self, agent):
        # the best match (idx 3) is "technical_support" but we ask for billing
        result = agent.process(
            text="payment issue",
            category="billing_and_payments",
            confidence=0.80,
        )
        # idx=7 is general_inquiry, idx=12 is billing but score is 0.30 < threshold
        assert result["resolved"] is False

    def test_fallback_mode_returns_empty(self, agent):
        agent._fallback = True
        result = agent.process(
            text="test",
            category="technical_support",
            confidence=0.80,
        )
        assert result["resolved"] is False

    def test_health(self, agent):
        h = agent.health()
        assert h["ok"] is True


# ── RoutingAgent ────────────────────────────────────────────────────────

class TestRoutingAgent:

    @pytest.fixture
    def agent(self):
        from src.agents.routing_agent import RoutingAgent

        agent = RoutingAgent()
        agent.load()  # loads from routing_rules.json or falls back to defaults
        return agent

    def test_routes_to_correct_team(self, agent):
        result = agent.process(
            text="my bill looks wrong",
            category="billing_and_payments",
            confidence=0.85,
            confidence_tier="high",
        )
        assert result["team"] == "Finance Team"

    def test_escalation_on_keyword(self, agent):
        result = agent.process(
            text="PRODUCTION IS DOWN everything is broken",
            category="technical_support",
            confidence=0.90,
            confidence_tier="high",
        )
        assert result["escalated"] is True
        assert result["priority_level"] == "critical"

    def test_escalation_on_low_confidence(self, agent):
        result = agent.process(
            text="something is not working right",
            category="general_inquiry",
            confidence=0.30,
            confidence_tier="low",
        )
        assert result["escalated"] is True
        assert result["priority_level"] == "high"

    def test_incident_low_confidence_critical(self, agent):
        result = agent.process(
            text="system behaving oddly",
            category="it_support",
            confidence=0.35,
            confidence_tier="low",
            ticket_type="Incident",
        )
        assert result["escalated"] is True
        assert result["priority_level"] == "critical"

    def test_medium_confidence_bumps_priority(self, agent):
        result = agent.process(
            text="need help with login",
            category="technical_support",
            confidence=0.55,
            confidence_tier="medium",
        )
        # technical_support default is "medium", bump → "high"
        assert result["priority_level"] == "high"

    def test_high_confidence_keeps_default(self, agent):
        result = agent.process(
            text="general question about hours",
            category="general_inquiry",
            confidence=0.90,
            confidence_tier="high",
        )
        assert result["priority_level"] == "low"  # general_inquiry default

    def test_reason_field_populated(self, agent):
        result = agent.process(
            text="urgent security breach detected",
            category="it_support",
            confidence=0.80,
            confidence_tier="high",
        )
        assert len(result["reason"]) > 0
        assert "category=" in result["reason"]
