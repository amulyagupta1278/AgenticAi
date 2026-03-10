import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.api.schemas import HealthResponse, PredictionResponse, TicketRequest
from src.api.agent_router import router as agent_router, init_agents, shutdown_agents
from src.api.metrics import instrument_app
from src.agentops_config import init_agentops

MODELS_DIR = ROOT / "models"

# module-level state — avoids reloading on every request
_state: dict = {}


def _clean(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_models():
    required = {
        "model": MODELS_DIR / "classifier.pkl",
        "vectorizer": MODELS_DIR / "tfidf_vectorizer.pkl",
        "encoder": MODELS_DIR / "label_encoder.pkl",
    }
    missing = [str(v) for k, v in required.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}. Run training first.")

    for key, path in required.items():
        _state[key] = joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_models()
        classes = _state["encoder"].classes_.tolist()
        print(f"Models loaded — {len(classes)} classes: {classes}")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")

    # boot up the agent system (classifier + RAG resolver + router + prevention)
    try:
        init_agentops()
        init_agents()
        print("Agent system initialized")
    except Exception as e:
        print(f"[WARNING] Agent init failed: {e}")

    yield

    shutdown_agents()
    _state.clear()


app = FastAPI(
    title="Ticket Classifier",
    description="Classify customer support tickets into issue categories.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(agent_router)
instrument_app(app)


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = "model" in _state
    classes = _state["encoder"].classes_.tolist() if loaded else None
    return HealthResponse(status="ok", model_loaded=loaded, classes=classes)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TicketRequest):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")

    cleaned = _clean(req.description)
    vec = _state["vectorizer"].transform([cleaned])

    idx = int(_state["model"].predict(vec)[0])
    ticket_type = str(_state["encoder"].inverse_transform([idx])[0])

    confidence = 0.0
    all_scores = None

    if hasattr(_state["model"], "predict_proba"):
        probs = _state["model"].predict_proba(vec)[0]
        confidence = float(np.max(probs))
        all_scores = {
            cls: round(float(p), 4)
            for cls, p in zip(_state["encoder"].classes_, probs)
        }

    return PredictionResponse(
        ticket_type=ticket_type,
        confidence=round(confidence, 4),
        all_scores=all_scores,
    )


@app.get("/classes")
def list_classes():
    if "encoder" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"classes": _state["encoder"].classes_.tolist()}


# uvicorn src.api.main:app --reload
