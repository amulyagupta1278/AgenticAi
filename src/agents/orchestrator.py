"""
Orchestrator — the top-level coordinator that runs tickets through
the classify → resolve → route pipeline.

Each step is timed independently so we can see where latency comes from
in the agent_trace. AgentOps traces are created per-ticket when configured.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.base import BaseAgent
from src.agents.classifier_agent import ClassifierAgent
from src.agents.resolution_agent import ResolutionAgent
from src.agents.routing_agent import RoutingAgent
from src.agents.prevention_agent import PreventionAgent
from src.agentops_config import start_ticket_trace, end_ticket_trace


class OrchestratorAgent(BaseAgent):

    def __init__(self):
        super().__init__("orchestrator")
        self.classifier = ClassifierAgent()
        self.resolver = ResolutionAgent()
        self.router = RoutingAgent()
        self.prevention = PreventionAgent()

    def load(self):
        self.classifier.load()
        self.resolver.load()
        self.router.load()
        self.prevention.load()
        self._ready = True

    # ------------------------------------------------------------------

    def process(self, text: str, ticket_id: str = None, **kw) -> dict:
        t0 = time.perf_counter()
        trace = []
        agentops_ctx = start_ticket_trace(ticket_id)

        try:
            # step 1 — classify
            clf = self.classifier._timed(text=text)
            trace.append(clf)

            category = clf["category"]
            confidence = clf["confidence"]
            tier = clf["confidence_tier"]

            # step 2 — try to resolve via RAG
            res = self.resolver._timed(
                text=text, category=category, confidence=confidence)
            trace.append(res)

            # step 3 — route if we couldn't resolve
            routing = None
            if not res["resolved"]:
                routing = self.router._timed(
                    text=text,
                    category=category,
                    confidence=confidence,
                    confidence_tier=tier,
                )
                trace.append(routing)

            # figure out the final status
            if res["resolved"]:
                status = "auto_resolved"
            elif routing and routing.get("escalated"):
                status = "escalated"
            else:
                status = "routed"

            elapsed = time.perf_counter() - t0

            end_ticket_trace(agentops_ctx, "success")

            return {
                "ticket_id": ticket_id,
                "status": status,
                "classification": {
                    "category": category,
                    "confidence": confidence,
                    "all_scores": clf.get("all_scores"),
                    "confidence_tier": tier,
                },
                "resolution": {
                    "resolved": res["resolved"],
                    "answer": res.get("answer"),
                    "source_ticket_subject": res.get("source_ticket_subject"),
                    "similarity_score": res.get("similarity_score", 0.0),
                    "matched_category": res.get("matched_category"),
                },
                "routing": routing,
                "processing_time_ms": round(elapsed * 1000, 2),
                "agent_trace": trace,
            }

        except Exception:
            end_ticket_trace(agentops_ctx, "error")
            raise

    # ------------------------------------------------------------------

    def get_insights(self) -> dict:
        return self.prevention._timed()

    def health(self) -> dict:
        return {
            "status": "ok" if self._ready else "not_initialized",
            "agents": {
                a.name: a.health()
                for a in [self.classifier, self.resolver,
                          self.router, self.prevention]
            },
            "rag_index_loaded": (self.resolver._ready
                                 and not self.resolver._fallback),
            "model_loaded": self.classifier._ready,
        }
