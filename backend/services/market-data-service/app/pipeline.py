from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import asyncpg
import redis.asyncio as redis

from backend.shared.config import Settings
from backend.shared.deriv_client import DerivClient, DerivClientError
from backend.shared.observability import (
    MARKET_DATA_ACTIVE_SUBSCRIPTIONS,
    MARKET_DATA_DERIV_RECONNECTS,
    MARKET_DATA_TICKS_PROCESSED,
)

log = logging.getLogger("market-data-pipeline")


@dataclass
class MarketTick:
    ts: datetime
    symbol: str
    bid: float | None
    ask: float | None
    price: float | None
    raw: Mapping[str, Any]


class MarketDataPipeline:
    # Bots/users never get their own Deriv connection: ticks are public data (see
    # docs/deriv-api.md "Practical usage notes" - prefer the public WebSocket for
    # read-only market data, auth is only needed for trading). One connection per
    # unique symbol is shared and fanned out to every interested bot via Redis.
    _DEFAULT_SUBSCRIBER = "__default__"

    def __init__(self, *, settings: Settings, symbols: Iterable[str]):
        self.settings = settings
        self.symbols = list(symbols)
        self._running = False
        self._stop_event = asyncio.Event()

        self._redis: redis.Redis | None = None
        self._pg_pool: asyncpg.Pool | None = None
        self._redis_lock = asyncio.Lock()
        self._pg_lock = asyncio.Lock()

        # One DerivClient/task per unique symbol, shared across every subscriber.
        self._deriv_clients: dict[str, DerivClient] = {}
        self._symbol_tasks: dict[str, asyncio.Task] = {}
        self._symbol_subscribers: dict[str, set[str]] = {}
        self._symbol_lock = asyncio.Lock()

        # Maps a subscribe_for_bot() key -> (symbol, subscriber_id) so
        # unsubscribe_for_bot() can find what to release.
        self._subscriptions: dict[str, tuple[str, str]] = {}

    async def _ensure_redis(self) -> redis.Redis:
        if self._redis is not None:
            return self._redis
        async with self._redis_lock:
            if self._redis is None:
                self._redis = redis.from_url(self.settings.redis_url)
        return self._redis

    async def _ensure_pg(self) -> asyncpg.Pool | None:
        # Schema is managed by Alembic migrations (backend/alembic/versions/0001_initial_schema.py)
        # rather than created here - run `alembic upgrade head` before starting this service.
        if self._pg_pool is not None:
            return self._pg_pool
        async with self._pg_lock:
            if self._pg_pool is None:
                # Settings.database_url is SQLAlchemy-style 'postgresql+asyncpg://...'
                # asyncpg expects 'postgresql://...'
                dsn = self.settings.database_url.replace("+asyncpg", "")
                try:
                    self._pg_pool = await asyncpg.create_pool(dsn)
                except Exception:
                    # Could not connect to Postgres (not available in local/test environments).
                    # Log and continue without a DB connection; persistence and token lookups
                    # will be skipped (and retried by the next caller).
                    log.warning("Could not connect to Postgres at %s; continuing without DB", dsn)
                    self._pg_pool = None
        return self._pg_pool

    async def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        self._running = True
        try:
            # The configured default symbols are always-on: register a permanent
            # sentinel subscriber so their shared task is never torn down.
            for symbol in self.symbols:
                await self._acquire_symbol(symbol, subscriber_id=self._DEFAULT_SUBSCRIBER)

            # Wait until stop is requested
            await self._stop_event.wait()

            # Cancel every symbol task on stop
            tasks = list(self._symbol_tasks.values())
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        finally:
            self._running = False
            self._symbol_tasks.clear()
            self._symbol_subscribers.clear()
            self._subscriptions.clear()
            MARKET_DATA_ACTIVE_SUBSCRIPTIONS.set(0)
            # Disconnect any Deriv clients a cancelled task didn't get to clean up itself
            for client in list(self._deriv_clients.values()):
                try:
                    await client.disconnect()
                except Exception:
                    pass
            if self._redis is not None:
                try:
                    await self._redis.aclose()
                except Exception:
                    pass
            if self._pg_pool is not None:
                try:
                    await self._pg_pool.close()
                except Exception:
                    pass

    async def _acquire_symbol(self, symbol: str, *, subscriber_id: str) -> None:
        """Register `subscriber_id`'s interest in `symbol`, starting its shared
        consumer task if this is the first subscriber."""
        async with self._symbol_lock:
            self._symbol_subscribers.setdefault(symbol, set()).add(subscriber_id)
            if symbol not in self._symbol_tasks:
                self._symbol_tasks[symbol] = asyncio.create_task(self._consume_symbol(symbol))

    async def _release_symbol(self, symbol: str, *, subscriber_id: str) -> None:
        """Remove `subscriber_id`'s interest in `symbol`, tearing down the shared
        consumer task once nothing (including the default sentinel) needs it."""
        async with self._symbol_lock:
            subscribers = self._symbol_subscribers.get(symbol)
            if subscribers is None:
                return
            subscribers.discard(subscriber_id)
            if subscribers:
                return
            del self._symbol_subscribers[symbol]
            task = self._symbol_tasks.pop(symbol, None)

        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _consume_symbol(self, symbol: str) -> None:
        """Maintain a single public Deriv tick stream for `symbol` and fan it out
        to Redis/Postgres. Shared by every bot/user subscribed to this symbol."""
        redis_client = await self._ensure_redis()
        stream_name = self.settings.redis_stream_market_events

        deriv = DerivClient()
        self._deriv_clients[symbol] = deriv

        try:
            try:
                await deriv.connect()
            except DerivClientError as exc:
                log.exception("Failed to open Deriv connection for %s: %s", symbol, exc)
                return

            try:
                await deriv.subscribe({"ticks": symbol})
            except DerivClientError as exc:
                log.exception("Failed to subscribe to %s: %s", symbol, exc)
                return

            while not self._stop_event.is_set():
                try:
                    msg = await deriv.receive()
                except DerivClientError:
                    # Attempt reconnect and continue
                    log.exception("Deriv receive failed for %s, attempting reconnect", symbol)
                    try:
                        await deriv.reconnect()
                        MARKET_DATA_DERIV_RECONNECTS.labels(symbol=symbol).inc()
                    except Exception:
                        await asyncio.sleep(1)
                    continue

                # Normalize to MarketTick if possible
                tick = self._parse_tick(msg, symbol)
                if tick is None:
                    continue

                MARKET_DATA_TICKS_PROCESSED.labels(symbol=symbol).inc()

                payload = {
                    "ts": tick.ts.isoformat(),
                    "symbol": tick.symbol,
                    "price": str(tick.price) if tick.price is not None else "",
                    "bid": str(tick.bid) if tick.bid is not None else "",
                    "ask": str(tick.ask) if tick.ask is not None else "",
                    "tick": json.dumps(tick.raw),
                }

                # Push to Redis stream - this is the fan-out point: every bot/user
                # subscribed to this symbol reads the same stream and filters by
                # the `symbol` field, rather than each getting a dedicated socket.
                try:
                    await redis_client.xadd(stream_name, payload)
                except Exception:
                    log.exception("Failed to push tick to Redis for %s", symbol)

                # Persist to Postgres (if available). Re-resolve the pool each tick so a
                # DB that was down at task startup gets picked up once it recovers.
                pg_pool = await self._ensure_pg()
                if pg_pool is not None:
                    try:
                        async with pg_pool.acquire() as conn:
                            await conn.execute(
                                "INSERT INTO market_ticks(ts, symbol, bid, ask, price, tick) "
                                "VALUES($1,$2,$3,$4,$5,$6::jsonb)",
                                tick.ts,
                                tick.symbol,
                                tick.bid,
                                tick.ask,
                                tick.price,
                                json.dumps(tick.raw),
                            )
                    except Exception:
                        log.exception("Failed to persist tick for %s", symbol)
        finally:
            try:
                await deriv.disconnect()
            except Exception:
                pass
            self._deriv_clients.pop(symbol, None)

    async def subscribe_for_bot(
        self,
        *,
        user_id: str,
        symbol: str,
        bot_id: str,
        account_id: str | None = None,
        app_id: int | None = None,
    ) -> str:
        """Register a bot's interest in a symbol's shared tick stream.

        Ticks are public data, so this does not open a dedicated Deriv connection -
        `account_id`/`app_id` are accepted for API compatibility (and in case an
        account-scoped stream is ever needed) but currently unused. Returns a
        subscription key identifying this bot's registration.
        """
        del account_id, app_id  # unused: ticks don't need per-account auth
        key = f"{user_id}:{symbol}:{bot_id}"
        if key in self._subscriptions:
            return key
        subscriber_id = f"{user_id}:{bot_id}"
        await self._acquire_symbol(symbol, subscriber_id=subscriber_id)
        self._subscriptions[key] = (symbol, subscriber_id)
        MARKET_DATA_ACTIVE_SUBSCRIPTIONS.set(len(self._subscriptions))
        return key

    async def unsubscribe_for_bot(self, *, user_id: str, symbol: str, bot_id: str) -> None:
        key = f"{user_id}:{symbol}:{bot_id}"
        entry = self._subscriptions.pop(key, None)
        if entry is None:
            return
        subscribed_symbol, subscriber_id = entry
        await self._release_symbol(subscribed_symbol, subscriber_id=subscriber_id)
        MARKET_DATA_ACTIVE_SUBSCRIPTIONS.set(len(self._subscriptions))

    def _parse_tick(self, msg: Any, fallback_symbol: str) -> MarketTick | None:
        # Deriv returns messages with 'tick' key for tick subscription
        data = msg if isinstance(msg, dict) else None
        if not data:
            return None

        tick_obj = None
        if "tick" in data and isinstance(data["tick"], dict):
            tick_obj = data["tick"]
        elif "history" in data and isinstance(data["history"], dict):
            # sometimes history responses include last ticks
            tick_obj = data["history"].get("last")
        else:
            # Unknown message type
            return None

        # timestamp
        epoch = tick_obj.get("epoch") or tick_obj.get("time")
        if isinstance(epoch, (int, float)):
            ts = datetime.fromtimestamp(float(epoch), tz=timezone.utc)
        else:
            ts = datetime.now(tz=timezone.utc)

        symbol = tick_obj.get("symbol") or fallback_symbol

        # Deriv 'tick' object often has 'quote' for price
        price = None
        if "quote" in tick_obj:
            try:
                price = float(tick_obj["quote"])
            except Exception:
                price = None
        elif "price" in tick_obj:
            try:
                price = float(tick_obj["price"])
            except Exception:
                price = None

        bid = None
        ask = None
        # Some providers include bid/ask; attempt to extract
        if "bid" in tick_obj:
            try:
                bid = float(tick_obj["bid"]) if tick_obj["bid"] is not None else None
            except Exception:
                bid = None
        if "ask" in tick_obj:
            try:
                ask = float(tick_obj["ask"]) if tick_obj["ask"] is not None else None
            except Exception:
                ask = None

        return MarketTick(ts=ts, symbol=symbol, bid=bid, ask=ask, price=price, raw=tick_obj)
