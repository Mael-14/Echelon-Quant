from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient


def _import_api_gateway_main(*, purge_cache: bool):
    # The service directory name contains a hyphen and isn't importable as a normal package,
    # so add the service folder to sys.path and import `app.main` from there. Other test
    # modules (bot-service, market-data-service) also expose an `app` package under a
    # different directory; since module imports are cached by name in sys.modules regardless
    # of sys.path order, purge any cached `app`/`app.*` modules once per test to avoid picking
    # up the wrong service - but only on the *first* import of the test, so the module we
    # patch below is the same one the test later runs against.
    import importlib
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    service_dir = repo_root / "backend" / "apps" / "api-gateway"
    sys.path.insert(0, str(service_dir))

    if purge_cache:
        for module_name in list(sys.modules):
            if module_name == "app" or module_name.startswith("app."):
                del sys.modules[module_name]

    return importlib.import_module("app.main")


def _bot_payload(bot_id: str = "bot-1") -> dict:
    return {
        "bot_id": bot_id,
        "name": "Momentum Alpha",
        "account": "paper-account-1",
        "symbols": ["EURUSD", "XAUUSD"],
        "max_positions": 5,
        "max_positions_per_symbol": 2,
        "min_lot": 0.01,
        "max_lot": 2.0,
        "risk_per_trade": 0.01,
        "max_daily_loss": 250.0,
        "max_drawdown": 1000.0,
        "strategy": "momentum_v1",
    }


def _bot_record(bot_id: str = "bot-1", state: str = "CREATED") -> dict:
    return {
        "config": _bot_payload(bot_id),
        "status": {
            "bot_id": bot_id,
            "state": state,
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
            "uptime_seconds": 0,
            "error": None,
        },
    }


@pytest.fixture
def main_module():
    return _import_api_gateway_main(purge_cache=True)


def _install_mock_bot_service(main_module, handler) -> None:
    mock_client = httpx.AsyncClient(
        base_url=main_module.settings.bot_service_url,
        transport=httpx.MockTransport(handler),
    )
    main_module._ensure_bot_service_client = lambda: mock_client


def test_create_bot_proxies_to_bot_service(main_module) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json=_bot_record("bot-1"))

    _install_mock_bot_service(main_module, handler)

    with TestClient(main_module.app) as client:
        resp = client.post("/api/v1/bots", json=_bot_payload("bot-1"))

    assert resp.status_code == 201
    assert resp.json()["config"]["bot_id"] == "bot-1"
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/v1/bots")
    assert captured["body"]["bot_id"] == "bot-1"


def test_get_bot_forwards_not_found_status(main_module) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Bot 'missing' not found"})

    _install_mock_bot_service(main_module, handler)

    with TestClient(main_module.app) as client:
        resp = client.get("/api/v1/bots/missing")

    assert resp.status_code == 404
    assert resp.json()["detail"] == "Bot 'missing' not found"


def test_start_bot_proxies_action_endpoint(main_module) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/bots/bot-1/start"
        assert request.method == "POST"
        return httpx.Response(200, json=_bot_record("bot-1", state="RUNNING")["status"])

    _install_mock_bot_service(main_module, handler)

    with TestClient(main_module.app) as client:
        resp = client.post("/api/v1/bots/bot-1/start")

    assert resp.status_code == 200
    assert resp.json()["state"] == "RUNNING"


def test_bot_service_unreachable_returns_bad_gateway(main_module) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    _install_mock_bot_service(main_module, handler)

    with TestClient(main_module.app) as client:
        resp = client.get("/api/v1/bots")

    assert resp.status_code == 502
