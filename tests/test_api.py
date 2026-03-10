import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def mock_state():
    encoder = MagicMock()
    encoder.classes_ = np.array(["billing_and_payments", "general_inquiry", "technical_support"])
    encoder.inverse_transform.return_value = np.array(["technical_support"])

    model = MagicMock()
    model.predict.return_value = np.array([2])
    model.predict_proba.return_value = np.array([[0.05, 0.10, 0.85]])

    vectorizer = MagicMock()
    vectorizer.transform.return_value = MagicMock()

    return {"model": model, "vectorizer": vectorizer, "encoder": encoder}


@pytest.fixture
def client(mock_state):
    with patch("src.api.main._state", mock_state):
        from src.api.main import app
        yield TestClient(app, raise_server_exceptions=True)


def test_health_with_model(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert "classes" in body


def test_health_no_model():
    with patch("src.api.main._state", {}):
        from src.api.main import app
        c = TestClient(app)
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is False


def test_predict_returns_expected_fields(client):
    resp = client.post("/predict", json={"description": "App crashes when I try to upload a file"})
    assert resp.status_code == 200
    body = resp.json()
    assert "ticket_type" in body
    assert "confidence" in body
    assert 0 <= body["confidence"] <= 1


def test_predict_returns_all_scores(client):
    resp = client.post("/predict", json={"description": "Payment failed on checkout"})
    body = resp.json()
    assert body["all_scores"] is not None
    assert len(body["all_scores"]) == 3  # matches mock encoder.classes_


def test_predict_short_description_rejected(client):
    resp = client.post("/predict", json={"description": "hi"})
    assert resp.status_code == 422  # pydantic min_length validation


def test_predict_no_model():
    with patch("src.api.main._state", {}):
        from src.api.main import app
        c = TestClient(app)
        resp = c.post("/predict", json={"description": "cannot log in to my account please help"})
        assert resp.status_code == 503


def test_classes_endpoint(client):
    resp = client.get("/classes")
    assert resp.status_code == 200
    assert "classes" in resp.json()
