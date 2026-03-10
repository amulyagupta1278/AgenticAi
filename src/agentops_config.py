"""
AgentOps SDK configuration.

Wraps the agentops init/trace lifecycle so the rest of the codebase
doesn't need to worry about whether an API key is configured or not.
When AGENTOPS_API_KEY isn't set, every function is a silent no-op.
"""

import logging
import os

logger = logging.getLogger(__name__)

_AGENTOPS_KEY = os.environ.get("AGENTOPS_API_KEY", "")
_initialized = False


def init_agentops():
    """Call once during app startup. Safe to call multiple times."""
    global _initialized
    if _initialized or not _AGENTOPS_KEY:
        return

    try:
        import agentops
        agentops.init(
            api_key=_AGENTOPS_KEY,
            default_tags=["ticket-classifier", "agentic-ai"],
            instrument_llm_calls=False,   # we use sklearn, not LLMs
            auto_start_session=False,
        )
        _initialized = True
        logger.info("AgentOps SDK initialized")
    except ImportError:
        logger.warning("agentops package not installed — skipping")
    except Exception as exc:
        logger.warning("AgentOps init failed: %s", exc)


def start_ticket_trace(ticket_id: str = None):
    """Begin an AgentOps trace for one ticket. Returns context or None."""
    if not _initialized:
        return None
    try:
        import agentops
        return agentops.start_trace(
            trace_name=f"ticket-{ticket_id or 'anon'}",
            tags=["ticket-processing"],
        )
    except Exception:
        return None


def end_ticket_trace(ctx, status: str = "success"):
    """Close an AgentOps trace. status should be 'success' or 'error'."""
    if ctx is None:
        return
    try:
        import agentops
        state = "Success" if status == "success" else "Error"
        agentops.end_trace(ctx, end_state=state)
    except Exception:
        pass
