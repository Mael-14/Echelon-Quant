from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


def _import_pipeline_module(*, purge_cache: bool = True):
    # The service directory name contains a hyphen and isn't importable as a normal package,
    # so add the service folder to sys.path and import `app.pipeline` from there. Other test
    # modules (bot-service, api-gateway) also expose an `app` package under a different
    # directory, so purge any cached `app`/`app.*` modules first to avoid picking up the
    # wrong service.
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

    return importlib.import_module("app.pipeline")


class FakeDerivClient:
    """Stands in for the real DerivClient so tests never touch the network.

    `receive()` blocks until the test cancels the consumer task, mimicking a
    live tick socket that's simply idle.
    """

    instances: list["FakeDerivClient"] = []

    def __init__(self) -> None:
        FakeDerivClient.instances.append(self)
        self.connected = False
        self.disconnected = False
        self.subscribed_requests: list[dict] = []

    async def connect(self) -> None:
        self.connected = True

    async def subscribe(self, request: dict) -> dict:
        self.subscribed_requests.append(request)
        return {}

    async def receive(self):
        await asyncio.Event().wait()

    async def reconnect(self) -> None:
        pass

    async def disconnect(self) -> None:
        self.disconnected = True


@pytest.fixture
def pipeline_module():
    return _import_pipeline_module()


@pytest.fixture(autouse=True)
def _no_real_postgres():
    async def fake_create_pool(dsn):
        raise ConnectionError("no Postgres in this test")

    with patch("asyncpg.create_pool", side_effect=fake_create_pool):
        yield


@pytest.fixture
def fake_deriv(pipeline_module, monkeypatch):
    FakeDerivClient.instances.clear()
    monkeypatch.setattr(pipeline_module, "DerivClient", FakeDerivClient)
    return FakeDerivClient


def _make_pipeline(pipeline_module, symbols=()):
    from backend.shared.config import Settings

    settings = Settings(postgres_host="fake", redis_host="fake")
    return pipeline_module.MarketDataPipeline(settings=settings, symbols=symbols)


@pytest.mark.asyncio
async def test_two_bots_on_same_symbol_share_one_deriv_connection(pipeline_module, fake_deriv):
    pipeline = _make_pipeline(pipeline_module)

    await pipeline.subscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
    await pipeline.subscribe_for_bot(user_id="u2", symbol="XAUUSD", bot_id="bot-b")
    await asyncio.sleep(0)  # let both consumer tasks reach their first await point

    assert len(fake_deriv.instances) == 1
    assert "XAUUSD" in pipeline._symbol_tasks
    assert pipeline._symbol_subscribers["XAUUSD"] == {"u1:bot-a", "u2:bot-b"}


@pytest.mark.asyncio
async def test_different_symbols_get_separate_connections(pipeline_module, fake_deriv):
    pipeline = _make_pipeline(pipeline_module)

    await pipeline.subscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
    await pipeline.subscribe_for_bot(user_id="u1", symbol="V75_2S", bot_id="bot-a")
    await asyncio.sleep(0)

    assert len(fake_deriv.instances) == 2
    assert set(pipeline._symbol_tasks.keys()) == {"XAUUSD", "V75_2S"}


@pytest.mark.asyncio
async def test_unsubscribe_one_of_two_keeps_shared_stream_alive(pipeline_module, fake_deriv):
    pipeline = _make_pipeline(pipeline_module)

    await pipeline.subscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
    await pipeline.subscribe_for_bot(user_id="u2", symbol="XAUUSD", bot_id="bot-b")
    await asyncio.sleep(0)

    await pipeline.unsubscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")

    assert "XAUUSD" in pipeline._symbol_tasks
    assert fake_deriv.instances[0].disconnected is False
    assert pipeline._symbol_subscribers["XAUUSD"] == {"u2:bot-b"}


@pytest.mark.asyncio
async def test_unsubscribe_last_bot_tears_down_stream(pipeline_module, fake_deriv):
    pipeline = _make_pipeline(pipeline_module)

    await pipeline.subscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
    await asyncio.sleep(0)

    await pipeline.unsubscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")

    assert "XAUUSD" not in pipeline._symbol_tasks
    assert "XAUUSD" not in pipeline._symbol_subscribers
    assert fake_deriv.instances[0].disconnected is True


@pytest.mark.asyncio
async def test_default_symbol_survives_all_bots_unsubscribing(pipeline_module, fake_deriv):
    pipeline = _make_pipeline(pipeline_module, symbols=["XAUUSD"])
    run_task = asyncio.create_task(pipeline.run())
    await asyncio.sleep(0)  # let run() register the default subscriber and start the task

    await pipeline.subscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
    await asyncio.sleep(0)
    assert len(fake_deriv.instances) == 1  # still just the one default-started connection

    await pipeline.unsubscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")

    assert "XAUUSD" in pipeline._symbol_tasks  # default sentinel keeps it alive
    assert fake_deriv.instances[0].disconnected is False

    await pipeline.stop()
    await run_task
    assert fake_deriv.instances[0].disconnected is True


@pytest.mark.asyncio
async def test_subscribe_for_bot_is_idempotent_for_same_key(pipeline_module, fake_deriv):
    pipeline = _make_pipeline(pipeline_module)

    key1 = await pipeline.subscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
    key2 = await pipeline.subscribe_for_bot(user_id="u1", symbol="XAUUSD", bot_id="bot-a")
    await asyncio.sleep(0)

    assert key1 == key2
    assert len(fake_deriv.instances) == 1
    assert pipeline._symbol_subscribers["XAUUSD"] == {"u1:bot-a"}
