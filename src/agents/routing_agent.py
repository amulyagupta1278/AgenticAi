"""
Routing Agent — decides which support team gets the ticket and at what
priority.  Uses a combination of the predicted category, classifier
confidence, ticket type (ITIL), and urgency keywords.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.base import BaseAgent

RULES_PATH = ROOT / "knowledge_base" / "routing_rules.json"

# words that signal something is on fire, regardless of category
DEFAULT_ESCALATION_KW = [
    "outage", "down", "emergency", "security", "breach",
    "fraud", "urgent", "critical", "production down",
    "locked out", "compromised", "data loss", "legal",
]

PRIORITY_BUMP = {"low": "medium", "medium": "high", "high": "critical"}


class RoutingAgent(BaseAgent):

    def __init__(self):
        super().__init__("routing")
        self._rules: dict = {}

    def load(self):
        if RULES_PATH.exists():
            with open(RULES_PATH) as f:
                self._rules = json.load(f)
        else:
            self._rules = self._defaults()
        self._ready = True

    # ------------------------------------------------------------------

    def process(self, text: str, category: str, confidence: float,
                confidence_tier: str, ticket_type: str = None, **kw) -> dict:

        text_lower = text.lower()
        teams = self._rules.get("teams", {})
        esc_keywords = self._rules.get("escalation_keywords", DEFAULT_ESCALATION_KW)

        # check for urgency
        matched_esc = [kw for kw in esc_keywords if kw in text_lower]
        escalated = len(matched_esc) > 0

        team_info = teams.get(category, {"team": "General Support",
                                         "default_priority": "medium"})
        team = team_info["team"]
        base_prio = team_info["default_priority"]

        # priority logic:
        #  - escalation keywords found → critical
        #  - incident type + low confidence → critical (could be a misrouted emergency)
        #  - low confidence → high (needs human eyes)
        #  - medium confidence → bump one level
        #  - high confidence → keep the category default
        if escalated:
            priority = "critical"
        elif ticket_type and ticket_type.lower() == "incident" and confidence_tier == "low":
            priority = "critical"
            escalated = True
        elif confidence_tier == "low":
            priority = "high"
            escalated = True
        elif confidence_tier == "medium":
            priority = PRIORITY_BUMP.get(base_prio, "medium")
        else:
            priority = base_prio

        reasons = []
        if matched_esc:
            reasons.append(f"urgency keywords: {', '.join(matched_esc[:3])}")
        if confidence_tier == "low":
            reasons.append(f"low confidence ({confidence:.2f})")
        reasons.append(f"category={category}")

        return {
            "team": team,
            "priority_level": priority,
            "ticket_type": ticket_type,
            "escalated": escalated,
            "reason": "; ".join(reasons),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _defaults() -> dict:
        """Fallback routing table when the JSON file is missing."""
        return {
            "teams": {
                "technical_support": {"team": "Engineering Support", "default_priority": "medium"},
                "billing_and_payments": {"team": "Finance Team", "default_priority": "medium"},
                "customer_service": {"team": "Customer Success", "default_priority": "low"},
                "product_support": {"team": "Product Team", "default_priority": "medium"},
                "returns_and_exchanges": {"team": "Returns Desk", "default_priority": "medium"},
                "it_support": {"team": "IT Helpdesk", "default_priority": "medium"},
                "sales_and_pre_sales": {"team": "Sales Team", "default_priority": "low"},
                "service_outages_and_maintenance": {"team": "Infrastructure / SRE", "default_priority": "high"},
                "human_resources": {"team": "HR Department", "default_priority": "medium"},
                "general_inquiry": {"team": "General Support", "default_priority": "low"},
            },
            "escalation_keywords": DEFAULT_ESCALATION_KW,
        }
