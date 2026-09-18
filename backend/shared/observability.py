from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(service_name: str, level: int = logging.INFO) -> None:
    """Route this process's logging through a single JSON-formatted handler.

    Safe to call more than once per process (a service's main.py can be
    re-imported by multiple test modules) - always replaces the root
    logger's handlers rather than appending, so repeat calls never produce
    duplicate log lines.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    logging.getLogger(service_name).setLevel(level)


def metrics_response() -> Response:
    """Render current process metrics in Prometheus text-exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# market-data-service counters. Defined here (not in app/pipeline.py or
# app/main.py) because those modules are purged from sys.modules and
# re-imported by tests - re-running a `Counter(...)`/`Gauge(...)` call
# against the global default REGISTRY on a second import raises
# "Duplicated timeseries in CollectorRegistry". backend.shared.observability
# is a normal dotted package, never purged, so these are created exactly
# once per process and simply imported (not re-created) elsewhere.
MARKET_DATA_TICKS_PROCESSED = Counter(
    "market_data_ticks_processed_total",
    "Market ticks processed and pushed to the market-events stream",
    ["symbol"],
)
MARKET_DATA_DERIV_RECONNECTS = Counter(
    "market_data_deriv_reconnects_total",
    "Successful Deriv WebSocket reconnects",
    ["symbol"],
)
MARKET_DATA_ACTIVE_SUBSCRIPTIONS = Gauge(
    "market_data_active_subscriptions",
    "Current number of active bot subscriptions to market data",
)
