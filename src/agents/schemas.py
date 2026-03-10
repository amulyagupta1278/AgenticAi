"""Pydantic models for the agent pipeline request/response shapes."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---- input ----

class AgentTicket(BaseModel):
    description: str = Field(..., min_length=5, description="Raw ticket text")
    ticket_id: Optional[str] = None
    priority: Optional[str] = None
    ticket_type: Optional[str] = None  # Incident / Request / Problem / Change


# ---- per-agent results ----

class ClassificationResult(BaseModel):
    category: str
    confidence: float
    all_scores: Optional[Dict[str, float]] = None
    confidence_tier: str  # high / medium / low


class ResolutionResult(BaseModel):
    resolved: bool
    answer: Optional[str] = None
    source_ticket_subject: Optional[str] = None
    similarity_score: float = 0.0
    matched_category: Optional[str] = None


class RoutingResult(BaseModel):
    team: str
    priority_level: str = "medium"
    ticket_type: Optional[str] = None
    escalated: bool = False
    reason: str = ""


# ---- orchestrator response ----

class AgentResponse(BaseModel):
    ticket_id: Optional[str] = None
    status: str  # auto_resolved / routed / escalated
    classification: ClassificationResult
    resolution: ResolutionResult
    routing: Optional[RoutingResult] = None
    processing_time_ms: float
    agent_trace: List[Dict] = []


# ---- prevention / insights ----

class PreventionInsight(BaseModel):
    category: str
    trend: str  # increasing / decreasing / stable
    ticket_count: int
    percentage: float
    recurring_tags: List[str]
    recommendation: str


class InsightsResponse(BaseModel):
    generated_at: str
    total_tickets_analyzed: int
    insights: List[PreventionInsight]
    top_recurring_issues: List[Dict]
    prevention_recommendations: List[str]


# ---- health ----

class AgentHealthResponse(BaseModel):
    status: str
    agents: Dict[str, Dict]
    rag_index_loaded: bool
    model_loaded: bool
