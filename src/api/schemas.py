from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TicketRequest(BaseModel):
    description: str = Field(..., min_length=5, description="Raw ticket text from the customer")


class PredictionResponse(BaseModel):
    ticket_type: str
    confidence: float
    all_scores: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: Optional[List[str]] = None
