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

    async def subscribe_for_bot(
        self,
        *,
        user_id: str,
        symbol: str,
        bot_id: str,
        account_id: str | None = None,
        app_id: int | None = None,
    ):
        key = f"{user_id}:{symbol}:{bot_id}"
        self._tasks[key] = True
        return key

    async def unsubscribe_for_bot(self, *, user_id: str, symbol: str, bot_id: str):
        key = f"{user_id}:{symbol}:{bot_id}"
        self._tasks.pop(key, None)


def _import_market_data_service_main(*, purge_cache: bool):
    # The service directory name contains a hyphen and isn't importable as a normal package,
    # so add the service folder to sys.path and import `app.main` from there. Other test
    # modules (e.g. bot-service) also expose an `app` package under a different directory;
    # since module imports are cached by name in sys.modules regardless of sys.path order,
    # purge any cached `app`/`app.*` modules once per test to avoid picking up the wrong
    # service - but only on the *first* import of the test, so the module we patch below
    # is the same one the test later runs against.
    import importlib
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    service_dir = repo_root / "backend" / "services" / "market-data-service"
    sys.path.insert(0, str(service_dir))

    if purge_cache:
        for module_name in list(sys.modules):
            if module_name == "app" or module_name.startswith("app."):
                del sys.modules[module_name]

    return importlib.import_module("app.main")


@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch):
    # Replace the real pipeline with a fake before the app startup runs.
    main_mod = _import_market_data_service_main(purge_cache=True)

    monkeypatch.setattr(main_mod, "MarketDataPipeline", FakePipeline)
    yield


def test_subscribe_unsubscribe_endpoints():
    # Reuse the module the `stub_pipeline` fixture already imported and patched -
    # purging the cache again here would re-run app.main fresh and lose the patch.
    main_mod = _import_market_data_service_main(purge_cache=False)
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
