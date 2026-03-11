"""
FastAPI router for the multi-agent system.

Three endpoints:
  POST /agent/resolve  — run a ticket through the full pipeline
  GET  /agent/insights — get trend analysis and prevention recommendations
  GET  /agent/status   — check all agent health
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.orchestrator import OrchestratorAgent
from src.agents.schemas import (
    AgentHealthResponse,
    AgentResponse,
    AgentTicket,
    InsightsResponse,
)
from src.api.metrics import (
    AGENT_LATENCY,
    AGENT_RESOLUTION,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])

# module-level instance — loaded during app startup via init_agents()
_orchestrator: OrchestratorAgent | None = None


def init_agents():
    """Called from main.py lifespan to boot up the agent system."""
    global _orchestrator
    _orchestrator = OrchestratorAgent()
    _orchestrator.load()
    logger.info("Agent system initialized")


def shutdown_agents():
    global _orchestrator
    _orchestrator = None


# ------------------------------------------------------------------


@router.post("/resolve", response_model=AgentResponse)
def resolve_ticket(req: AgentTicket):
    """Full agentic pipeline: classify → resolve (RAG) → route if needed."""
    if _orchestrator is None or not _orchestrator._ready:
        raise HTTPException(503, "Agent system not ready — run training pipeline first.")

    result = _orchestrator.process(
        text=req.description,
        ticket_id=req.ticket_id,
    )

    # record prometheus metrics
    cat = result["classification"]["category"]
    conf = result["classification"]["confidence"]
    PREDICTION_COUNT.labels(category=cat).inc()
    PREDICTION_CONFIDENCE.observe(conf)
    AGENT_RESOLUTION.labels(status=result["status"]).inc()
    AGENT_LATENCY.observe(result["processing_time_ms"] / 1000)

    return AgentResponse(**result)


@router.get("/insights", response_model=InsightsResponse)
def get_insights():
    """Prevention insights — trend analysis and proactive recommendations."""
    if _orchestrator is None or not _orchestrator._ready:
        raise HTTPException(503, "Agent system not ready.")

    result = _orchestrator.get_insights()
    return InsightsResponse(**result)


@router.get("/simulation")
def get_simulation():
    """Return pre-computed simulation metrics if available."""
    import json
    metrics_path = ROOT / "data" / "processed" / "simulation_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(404, "Simulation metrics not found. Run: python scripts/run_simulation.py")
    return json.loads(metrics_path.read_text())


@router.get("/status", response_model=AgentHealthResponse)
def agent_status():
    """Health check for the full agent system."""
    if _orchestrator is None:
        return AgentHealthResponse(
            status="not_initialized", agents={},
            rag_index_loaded=False, model_loaded=False,
        )
    return AgentHealthResponse(**_orchestrator.health())
