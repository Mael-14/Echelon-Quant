from __future__ import annotations

import asyncio
import importlib
import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.shared.observability import MARKET_DATA_ACTIVE_SUBSCRIPTIONS, configure_logging

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_configure_logging_emits_json_lines():
    configure_logging("test-observability-service")
    logger = logging.getLogger("test-observability-service")

    stream = io.StringIO()
    logging.getLogger().handlers[0].stream = stream

    logger.info("hello %s", "world")

    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test-observability-service"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def _import_account_service_main():
    service_dir = REPO_ROOT / "backend" / "services" / "account-service"
    sys.path.insert(0, str(service_dir))
    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            del sys.modules[module_name]
    return importlib.import_module("app.main")


def test_metrics_endpoint_returns_prometheus_format():
    main_module = _import_account_service_main()

    with TestClient(main_module.app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "python_gc_objects_collected_total" in response.text


def _import_pipeline_module():
    service_dir = REPO_ROOT / "backend" / "services" / "market-data-service"
    sys.path.insert(0, str(service_dir))
    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            del sys.modules[module_name]
    return importlib.import_module("app.pipeline")


class _FakeDerivClient:
    """Stands in for the real DerivClient so this test never touches the network."""

    def __init__(self) -> None:
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def subscribe(self, request: dict) -> dict:
        return {}

    async def receive(self):
        await asyncio.Event().wait()

    async def reconnect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _no_real_postgres():
    async def fake_create_pool(dsn):
        raise ConnectionError("no Postgres in this test")

    with patch("asyncpg.create_pool", side_effect=fake_create_pool):
        yield


@pytest.mark.asyncio
async def test_active_subscriptions_gauge_tracks_subscribe_and_unsubscribe():
    pipeline_module = _import_pipeline_module()
    monkeypatch_target = pipeline_module.DerivClient
    pipeline_module.DerivClient = _FakeDerivClient
    try:
        from backend.shared.config import Settings

        settings = Settings(postgres_host="fake", redis_host="fake")
        pipeline = pipeline_module.MarketDataPipeline(settings=settings, symbols=[])

        await pipeline.subscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
        await asyncio.sleep(0)
        assert MARKET_DATA_ACTIVE_SUBSCRIPTIONS._value.get() == 1

        await pipeline.subscribe_for_bot(user_id="u1", symbol="V75_2S", bot_id="bot-b")
        await asyncio.sleep(0)
        assert MARKET_DATA_ACTIVE_SUBSCRIPTIONS._value.get() == 2

        await pipeline.unsubscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
        assert MARKET_DATA_ACTIVE_SUBSCRIPTIONS._value.get() == 1

        await pipeline.unsubscribe_for_bot(user_id="u1", symbol="V75_2S", bot_id="bot-b")
        assert MARKET_DATA_ACTIVE_SUBSCRIPTIONS._value.get() == 0
    finally:
        pipeline_module.DerivClient = monkeypatch_target
