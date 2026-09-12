from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.shared.schemas import BotConfig, BotLifecycleState


BOT_SERVICE_ROOT = Path(__file__).resolve().parents[1] / "backend" / "services" / "bot-service"


def _import_bot_service_modules():
    root = str(BOT_SERVICE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Ensure this test imports the bot-service app package specifically.
    for module_name in ("app.main", "app.service", "app"):
        sys.modules.pop(module_name, None)

    service_module = importlib.import_module("app.service")
    main_module = importlib.import_module("app.main")
    return service_module, main_module


@pytest.fixture
def bot_modules():
    return _import_bot_service_modules()


@pytest.fixture
def client(bot_modules):
    _, main_module = bot_modules
    main_module.bot_manager._bots.clear()
    return TestClient(main_module.app)


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
        "dynamic_lot": True,
        "pyramiding": False,
        "trailing_stop": True,
        "break_even": True,
        "partial_close": True,
    }


def test_bot_manager_start_pause_stop_lifecycle(bot_modules) -> None:
    service_module, _ = bot_modules
    manager = service_module.BotManager()

    manager.create_bot(BotConfig(**_bot_payload("bot-lifecycle")))

    running = manager.start_bot("bot-lifecycle")
    assert running.state == BotLifecycleState.RUNNING

    paused = manager.pause_bot("bot-lifecycle")
    assert paused.state == BotLifecycleState.PAUSED

    stopped = manager.stop_bot("bot-lifecycle")
    assert stopped.state == BotLifecycleState.STOPPED


def test_bot_manager_rejects_invalid_transition(bot_modules) -> None:
    service_module, _ = bot_modules
    manager = service_module.BotManager()

    manager.create_bot(BotConfig(**_bot_payload("bot-invalid")))

    with pytest.raises(RuntimeError, match="Invalid transition"):
        manager.pause_bot("bot-invalid")


def test_create_and_get_bot_endpoints(client: TestClient) -> None:
    create_response = client.post("/api/v1/bots", json=_bot_payload("bot-http"))

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["config"]["bot_id"] == "bot-http"
    assert created["status"]["state"] == "CREATED"

    get_response = client.get("/api/v1/bots/bot-http")
    assert get_response.status_code == 200
    assert get_response.json()["config"]["name"] == "Momentum Alpha"


def test_start_pause_stop_endpoints(client: TestClient) -> None:
    client.post("/api/v1/bots", json=_bot_payload("bot-http-2"))

    start_response = client.post("/api/v1/bots/bot-http-2/start")
    assert start_response.status_code == 200
    assert start_response.json()["state"] == "RUNNING"

    pause_response = client.post("/api/v1/bots/bot-http-2/pause")
    assert pause_response.status_code == 200
    assert pause_response.json()["state"] == "PAUSED"

    stop_response = client.post("/api/v1/bots/bot-http-2/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["state"] == "STOPPED"


def test_pause_from_created_returns_conflict(client: TestClient) -> None:
    client.post("/api/v1/bots", json=_bot_payload("bot-created"))

    pause_response = client.post("/api/v1/bots/bot-created/pause")

    assert pause_response.status_code == 409
    assert "Invalid transition" in pause_response.json()["detail"]


def test_missing_bot_returns_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/bots/does-not-exist")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
