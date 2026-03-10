"""
Prometheus metrics for the ticket classifier API.

Exposes counters, histograms, and info gauges that get scraped by
Prometheus at /metrics.  The instrument_app() function wires up a
middleware that auto-records request count and latency for every endpoint.
"""

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    Counter, Histogram, Info, make_asgi_app,
)

# --- counters & histograms ---

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Number of predictions by category",
    ["category"],
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

AGENT_RESOLUTION = Counter(
    "agent_resolutions_total",
    "Agent pipeline outcomes",
    ["status"],  # auto_resolved / routed / escalated
)

AGENT_LATENCY = Histogram(
    "agent_pipeline_duration_seconds",
    "Full agent pipeline latency",
    buckets=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
)

MODEL_INFO = Info("model", "Current model metadata")


def instrument_app(app: FastAPI):
    """Add Prometheus middleware and mount the /metrics endpoint."""

    @app.middleware("http")
    async def _prom_middleware(request: Request, call_next) -> Response:
        # figure out which endpoint this is hitting
        path = request.url.path
        # don't track metrics or docs endpoints
        if path in ("/metrics", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=path,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=path).observe(duration)

        return response

    # mount the prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
