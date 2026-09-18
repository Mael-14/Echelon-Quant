from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

BOT_SERVICE_ROOT = Path(__file__).resolve().parents[1] / "backend" / "services" / "bot-service"


def _import_bot_service_modules():
    root = str(BOT_SERVICE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Ensure this test imports the bot-service app package specifically, not some
    # other service's `app` package left over in sys.modules from another test file.
    for module_name in list(sys.modules):
        if module_name == "app" or module_name.startswith("app."):
            del sys.modules[module_name]

    import importlib

    service_module = importlib.import_module("app.service")
    store_module = importlib.import_module("app.store")
    return service_module, store_module


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


class _FakeConn:
    def __init__(self, table: dict) -> None:
        self._table = table

    async def execute(self, query: str, *args) -> None:
        if "INSERT INTO bots" in query:
            bot_id, config_json, status_json = args
            self._table[bot_id] = (config_json, status_json)
            return
        raise AssertionError(f"Unexpected query: {query}")

    async def fetch(self, query: str, *args):
        assert "SELECT config, status FROM bots" in query
        return [{"config": c, "status": s} for c, s in self._table.values()]


class _FakeAcquireCtx:
    def __init__(self, table: dict) -> None:
        self._table = table

    async def __aenter__(self) -> _FakeConn:
        return _FakeConn(self._table)

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakePool:
    def __init__(self, table: dict) -> None:
        self._table = table

    def acquire(self) -> _FakeAcquireCtx:
        return _FakeAcquireCtx(self._table)

    async def close(self) -> None:
        pass


@pytest.fixture
def bot_service_modules():
    return _import_bot_service_modules()


@pytest.fixture
def fake_table():
    # Simulates the `bots` Postgres table as {bot_id: (config_json, status_json)},
    # shared across BotStore instances to simulate persistence surviving a restart.
    return {}


def _make_store(store_module, settings_module, fake_table: dict):
    settings = settings_module.Settings(postgres_host="fake")
    store = store_module.BotStore(settings=settings)
    return store


@pytest.mark.asyncio
async def test_save_then_load_all_round_trips_bot(bot_service_modules, fake_table) -> None:
    service_module, store_module = bot_service_modules
    from backend.shared.config import Settings

    settings = Settings(postgres_host="fake")
    store = store_module.BotStore(settings=settings)

    record = service_module.BotRecord(
        config=service_module.BotConfig(**_bot_payload("bot-1")),
        status=service_module.BotStatus(bot_id="bot-1"),
    )

    async def fake_create_pool(dsn):
        return _FakePool(fake_table)

    with patch("asyncpg.create_pool", side_effect=fake_create_pool):
        await store.save(record)
        loaded = await store.load_all()

    assert len(loaded) == 1
    assert loaded[0].config.bot_id == "bot-1"
    assert loaded[0].status.bot_id == "bot-1"


@pytest.mark.asyncio
async def test_save_upserts_rather_than_duplicating(bot_service_modules, fake_table) -> None:
    service_module, store_module = bot_service_modules
    from backend.shared.config import Settings

    settings = Settings(postgres_host="fake")
    store = store_module.BotStore(settings=settings)

    def make_record(state: str) -> "service_module.BotRecord":
        return service_module.BotRecord(
            config=service_module.BotConfig(**_bot_payload("bot-1")),
            status=service_module.BotStatus(bot_id="bot-1", state=state),
        )

    async def fake_create_pool(dsn):
        return _FakePool(fake_table)

    with patch("asyncpg.create_pool", side_effect=fake_create_pool):
        await store.save(make_record("CREATED"))
        await store.save(make_record("RUNNING"))
        loaded = await store.load_all()

    assert len(loaded) == 1
    assert loaded[0].status.state == "RUNNING"


@pytest.mark.asyncio
async def test_restart_hydrates_bot_manager_from_store(bot_service_modules, fake_table) -> None:
    service_module, store_module = bot_service_modules
    from backend.shared.config import Settings

    settings = Settings(postgres_host="fake")

    async def fake_create_pool(dsn):
        return _FakePool(fake_table)

    with patch("asyncpg.create_pool", side_effect=fake_create_pool):
        # "Before restart": a manager creates a bot and persists it.
        store_before = store_module.BotStore(settings=settings)
        manager_before = service_module.BotManager()
        manager_before.create_bot(service_module.BotConfig(**_bot_payload("bot-1")))
        manager_before.start_bot("bot-1")
        await store_before.save(manager_before.get_bot("bot-1"))

        # "After restart": a brand new manager and store, hydrated from the same table.
        store_after = store_module.BotStore(settings=settings)
        manager_after = service_module.BotManager()
        for loaded_record in await store_after.load_all():
            manager_after.load_bot(loaded_record)

    restarted = manager_after.get_bot("bot-1")
    assert restarted is not None
    assert restarted.status.state == "RUNNING"
