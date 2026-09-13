from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

import asyncpg
import redis.asyncio as redis

from backend.shared.config import Settings
from backend.shared.deriv_client import DerivClient, DerivClientError
from cryptography.fernet import Fernet, InvalidToken

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
    def __init__(self, *, settings: Settings, symbols: Iterable[str]):
        self.settings = settings
        self.symbols = list(symbols)
        self._running = False
        self._stop_event = asyncio.Event()

        self._redis: redis.Redis | None = None
        self._pg_pool: asyncpg.Pool | None = None
        # We'll create one DerivClient per subscription to support per-user OTPs
        self._deriv_clients: dict[str, DerivClient] = {}
        self._tasks: dict[str, asyncio.Task] = {}

        self._fernet: Fernet | None = None

    async def _ensure_redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self.settings.redis_url)
        return self._redis

    async def _ensure_pg(self) -> asyncpg.Pool:
        if self._pg_pool is None:
            # Settings.database_url is SQLAlchemy-style 'postgresql+asyncpg://...'
            # asyncpg expects 'postgresql://...'
            dsn = self.settings.database_url.replace("+asyncpg", "")
            try:
                self._pg_pool = await asyncpg.create_pool(dsn)
                await self._create_table()
            except Exception:
                # Could not connect to Postgres (not available in local/test environments).
                # Log and continue without a DB connection; persistence and token lookups will be skipped.
                log.warning("Could not connect to Postgres at %s; continuing without DB", dsn)
                self._pg_pool = None

            # prepare Fernet if key is present
            key = getattr(self.settings, "deriv_token_key", None)
            if key:
                try:
                    self._fernet = Fernet(key.encode("utf-8") if isinstance(key, str) else key)
                except Exception:
                    self._fernet = None
        return self._pg_pool

    async def _create_table(self) -> None:
        assert self._pg_pool is not None
        async with self._pg_pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_ticks (
                    id SERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    bid NUMERIC,
                    ask NUMERIC,
                    price NUMERIC,
                    tick JSONB
                );
                """
            )

    async def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        self._running = True
        try:
            # Start one consumer task per configured symbol.
            subscribe_tasks = [asyncio.create_task(self._subscribe_and_consume(sym)) for sym in self.symbols]

            # Wait until stop is requested
            await self._stop_event.wait()

            # Cancel subscription tasks on stop
            for t in subscribe_tasks:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        finally:
            self._running = False
            # Disconnect any per-subscription Deriv clients
            for client in list(self._deriv_clients.values()):
                try:
                    await client.disconnect()
                except Exception:
                    pass
            if self._redis is not None:
                try:
                    await self._redis.close()
                except Exception:
                    pass
            if self._pg_pool is not None:
                try:
                    await self._pg_pool.close()
                except Exception:
                    pass

    async def _subscribe_and_consume(self, symbol: str, *, user_id: str | None = None, account_id: str | None = None, app_id: int | None = None, bot_id: str | None = None) -> None:
        """Subscribe to tick updates for a symbol and forward to Redis/Postgres.

        If `user_id` is provided, this subscription will attempt to fetch the stored token
        for that user and authenticate the Deriv client via OTP for an account-specific socket.
        """
        redis_client = await self._ensure_redis()
        pg_pool = await self._ensure_pg()
        stream_name = self.settings.redis_stream_market_events

        # Create a distinct DerivClient for this subscription so each bot/user can have its own OTP socket
        deriv = DerivClient()

        # If user provided, attempt to fetch stored token from DB (if DB available)
        token: str | None = None
        if user_id is not None and pg_pool is not None:
            try:
                async with pg_pool.acquire() as conn:
                    row = await conn.fetchrow("SELECT token, account_id, app_id FROM deriv_tokens WHERE user_id = $1", user_id)
                if row:
                    enc = row["token"]
                    # decrypt if possible
                    if self._fernet is not None:
                        try:
                            token = self._fernet.decrypt(enc.encode("utf-8")).decode("utf-8")
                        except InvalidToken:
                            token = enc
                    else:
                        token = enc
                    # prefer explicit account_id/app_id passed in to method
                    account_id = account_id or row.get("account_id")
                    app_id = app_id or row.get("app_id")
            except Exception:
                log.exception("Failed to read token for user %s", user_id)

        request = {"ticks": symbol}

        # Authenticate or connect
        try:
            if token and account_id:
                await deriv.authenticate(token=token, account_id=account_id, app_id=app_id)
            else:
                await deriv.connect()
        except DerivClientError as exc:
            log.exception("Failed to open Deriv connection for %s (user=%s): %s", symbol, user_id, exc)
            try:
                await deriv.disconnect()
            except Exception:
                pass
            return

        # register client for cleanup
        key = f"{user_id or 'public'}:{symbol}:{bot_id or ''}"
        self._deriv_clients[key] = deriv

        try:
            await deriv.subscribe(request)
        except DerivClientError as exc:
            log.exception("Failed to subscribe to %s: %s", symbol, exc)
            return

        while not self._stop_event.is_set():
            try:
                msg = await deriv.receive()
            except DerivClientError:
                # Attempt reconnect and continue
                log.exception("Deriv receive failed for %s (user=%s), attempting reconnect", symbol, user_id)
                try:
                    await deriv.reconnect()
                except Exception:
                    await asyncio.sleep(1)
                continue

            # Normalize to MarketTick if possible
            tick = self._parse_tick(msg, symbol)
            if tick is None:
                continue

            payload = {
                "ts": tick.ts.isoformat(),
                "symbol": tick.symbol,
                "price": str(tick.price) if tick.price is not None else "",
                "bid": str(tick.bid) if tick.bid is not None else "",
                "ask": str(tick.ask) if tick.ask is not None else "",
                "tick": json.dumps(tick.raw),
            }

            # Push to Redis stream
            try:
                await redis_client.xadd(stream_name, payload)
            except Exception:
                log.exception("Failed to push tick to Redis for %s", symbol)

            # Persist to Postgres (if available)
            if pg_pool is not None:
                try:
                    async with pg_pool.acquire() as conn:
                        await conn.execute(
                            "INSERT INTO market_ticks(ts, symbol, bid, ask, price, tick) VALUES($1,$2,$3,$4,$5,$6::jsonb)",
                            tick.ts,
                            tick.symbol,
                            tick.bid,
                            tick.ask,
                            tick.price,
                            json.dumps(tick.raw),
                        )
                except Exception:
                    log.exception("Failed to persist tick for %s", symbol)

        # cleanup when loop exits
        try:
            await deriv.disconnect()
        except Exception:
            pass
        self._deriv_clients.pop(key, None)

    async def subscribe_for_bot(self, *, user_id: str, symbol: str, bot_id: str, account_id: str | None = None, app_id: int | None = None) -> str:
        """Start a subscription for a specific bot (user-scoped). Returns subscription key."""
        key = f"{user_id}:{symbol}:{bot_id}"
        if key in self._tasks:
            return key
        task = asyncio.create_task(self._subscribe_and_consume(symbol, user_id=user_id, account_id=account_id, app_id=app_id, bot_id=bot_id))
        self._tasks[key] = task
        return key

    async def unsubscribe_for_bot(self, *, user_id: str, symbol: str, bot_id: str) -> None:
        key = f"{user_id}:{symbol}:{bot_id}"
        task = self._tasks.pop(key, None)
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

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
