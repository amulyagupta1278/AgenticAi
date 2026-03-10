"""
Classifier Agent — wraps the existing sklearn pipeline (TF-IDF + model)
and adds confidence tier logic so downstream agents know how much to
trust the prediction.
"""

import re
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.base import BaseAgent

MODELS_DIR = ROOT / "models"

# thresholds picked from looking at the calibration curve during EDA
HIGH_CONF = 0.75
MED_CONF = 0.45


def _clean(text: str) -> str:
    """Mirror the cleaning in src/api/main.py so predictions are consistent."""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class ClassifierAgent(BaseAgent):

    def __init__(self):
        super().__init__("classifier")
        self._model = None
        self._vectorizer = None
        self._encoder = None

    def load(self):
        needed = {
            "model": MODELS_DIR / "classifier.pkl",
            "vectorizer": MODELS_DIR / "tfidf_vectorizer.pkl",
            "encoder": MODELS_DIR / "label_encoder.pkl",
        }
        missing = [str(p) for p in needed.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing model artifacts: {missing}")

        self._model = joblib.load(needed["model"])
        self._vectorizer = joblib.load(needed["vectorizer"])
        self._encoder = joblib.load(needed["encoder"])
        self._ready = True

    def process(self, text: str, **kw) -> dict:
        cleaned = _clean(text)
        vec = self._vectorizer.transform([cleaned])
        idx = int(self._model.predict(vec)[0])
        category = str(self._encoder.inverse_transform([idx])[0])

        confidence = 0.0
        all_scores = None

        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba(vec)[0]
            confidence = float(np.max(probs))
            all_scores = {
                c: round(float(p), 4)
                for c, p in zip(self._encoder.classes_, probs)
            }

        if confidence >= HIGH_CONF:
            tier = "high"
        elif confidence >= MED_CONF:
            tier = "medium"
        else:
            tier = "low"

        return {
            "category": category,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "confidence_tier": tier,
        }

    @property
    def classes(self) -> list:
        if self._encoder is None:
            return []
        return self._encoder.classes_.tolist()
