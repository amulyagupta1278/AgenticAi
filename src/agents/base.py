"""
Base class for all agents in the multi-agent pipeline.

Every agent follows the same contract: load resources, process input,
report health. The _timed wrapper gives us per-step latency tracking
without cluttering the actual logic.
"""

import time
from abc import ABC, abstractmethod


class BaseAgent(ABC):

    def __init__(self, name: str):
        self.name = name
        self._ready = False

    @abstractmethod
    def process(self, **kwargs) -> dict:
        """Run the agent's core logic. Returns a flat dict of results."""
        ...

    def health(self) -> dict:
        return {"agent": self.name, "ok": self._ready}

    def _timed(self, **kwargs) -> dict:
        """Wrap process() with timing metadata for the orchestrator trace."""
        t0 = time.perf_counter()
        result = self.process(**kwargs)
        elapsed = time.perf_counter() - t0
        result["_agent"] = self.name
        result["_ms"] = round(elapsed * 1000, 2)
        return result
