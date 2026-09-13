import asyncio

import pytest
from fastapi.testclient import TestClient


class FakePipeline:
    def __init__(self, *, settings=None, symbols=None):
        self.settings = settings
        self.symbols = list(symbols) if symbols is not None else []
        self._tasks = {}

    async def run(self):
        # run forever until stopped; in tests we won't await it
        await asyncio.Event().wait()

    async def stop(self):
        return None

    async def subscribe_for_bot(self, *, user_id: str, symbol: str, bot_id: str, account_id: str | None = None, app_id: int | None = None):
        key = f"{user_id}:{symbol}:{bot_id}"
        self._tasks[key] = True
        return key

    async def unsubscribe_for_bot(self, *, user_id: str, symbol: str, bot_id: str):
        key = f"{user_id}:{symbol}:{bot_id}"
        self._tasks.pop(key, None)


@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch):
    # Replace the real pipeline with a fake before the app startup runs.
    # The service directory name contains a hyphen and isn't importable as a normal package,
    # so add the service folder to sys.path and import `app.main` from there.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    service_dir = repo_root / "backend" / "services" / "market-data-service"
    sys.path.insert(0, str(service_dir))
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "MarketDataPipeline", FakePipeline)
    yield


def test_subscribe_unsubscribe_endpoints():
    # Import the service `app` by inserting the service directory into sys.path
    import sys
    from pathlib import Path
    import importlib

    repo_root = Path(__file__).resolve().parents[1]
    service_dir = repo_root / "backend" / "services" / "market-data-service"
    sys.path.insert(0, str(service_dir))
    main_mod = importlib.import_module("app.main")
    app = main_mod.app

    with TestClient(app) as client:
        payload = {"user_id": "u1", "symbol": "XAUUSD", "bot_id": "b1"}
        r = client.post("/api/v1/market/subscribe", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert "subscription_key" in data
        assert data["subscription_key"] == "u1:XAUUSD:b1"

        r2 = client.post("/api/v1/market/unsubscribe", json=payload)
        assert r2.status_code == 204
