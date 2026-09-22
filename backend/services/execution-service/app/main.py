"""execution-service: signals in, managed demo trades out.

Consumes `signal-events`, maps each accepted signal onto a Deriv multiplier
contract, and manages the exit until it closes.

**Nothing reaches Deriv until the guards pass.** `trader.refuse` runs first on
every signal, and its first check is the demo-mode one. That check reads a
single shared property (`settings.is_demo_trading`) rather than comparing the
mode string here, so this service and any other cannot drift apart on what
"demo" means.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

import redis.asyncio as redis
from fastapi import FastAPI
from pydantic import BaseModel

from backend.shared.config import get_settings
from backend.shared.deriv_v3_client import DerivV3Client, DerivV3Error
from backend.shared.observability import (
    EXECUTION_OPEN_POSITIONS,
    EXECUTION_ORDERS_PLACED,
    EXECUTION_ORDERS_REFUSED,
    configure_logging,
    metrics_response,
)

from .trader import TradeRefused, TradingState, close_all, manage, place

settings = get_settings()
configure_logging("execution-service")
log = logging.getLogger("execution-service")

app = FastAPI(title="Execution Service", version="0.1.0", debug=settings.debug)

state = TradingState()
runtime: dict[str, Any] = {"loop": None, "redis": None, "client": None, "balance": 0.0}


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "execution-service"
    environment: str = "development"
    trade_mode: str = "demo"
    open_positions: int = 0
    realised_r_today: float = 0.0
    halted: bool = False
    detail: str | None = None


def _exit_policy() -> Any:
    from forex_agent.execution.exit_policy import ExitPolicy

    return ExitPolicy()


async def _watch(client: DerivV3Client, symbol: str) -> None:
    """Follow one contract to close, applying the exit policy to each update."""
    position = state.open_positions.get(symbol)
    if position is None:
        return
    policy = _exit_policy()
    try:
        stream = await client.proposal_open_contract(position.contract_id)
        async for update in stream:
            contract = update.get("proposal_open_contract") or {}
            quote = contract.get("current_spot")
            if quote is None:
                continue

            outcome = await manage(client, position, quote=float(quote), policy=policy)
            if outcome.startswith("closed") or contract.get("is_sold"):
                profit = float(contract.get("profit") or 0.0)
                risk = position.order.stop_loss_amount or 1.0
                state.realised_r_today += profit / risk
                state.open_positions.pop(symbol, None)
                EXECUTION_OPEN_POSITIONS.set(len(state.open_positions))
                log.info("closed %s: %s (%.2f R today)", symbol, outcome, state.realised_r_today)
                await stream.cancel()
                return
    except asyncio.CancelledError:
        raise
    except DerivV3Error:
        log.exception("Lost the contract stream for %s", symbol)


async def _consume() -> None:
    """Read signal-events and act on each accepted signal."""
    client = DerivV3Client(app_id=settings.deriv_app_id, ws_url=settings.deriv_v3_ws_url)
    runtime["client"] = client
    stream = settings.redis_stream_signal_events
    last_id = "$"

    try:
        await client.connect()
        if settings.deriv_token:
            account = await client.authorize(settings.deriv_token)
            loginid = str(account.get("loginid", ""))
            # A real-money token authorises exactly as happily as a demo one;
            # the loginid is the only thing that tells them apart.
            if settings.is_demo_trading and not loginid.upper().startswith("VRTC"):
                state.halted = True
                state.halt_reason = (
                    f"token authorised as {loginid}, which is not a demo (VRTC) account"
                )
                log.error(state.halt_reason)
            balance = await client.balance()
            runtime["balance"] = float(balance.get("balance") or 0.0)

        while True:
            entries = await runtime["redis"].xread({stream: last_id}, count=10, block=5000)
            for _stream, messages in entries or []:
                for message_id, fields in messages:
                    last_id = message_id
                    try:
                        await _handle(client, fields)
                    except TradeRefused as refused:
                        symbol = _symbol_of(fields)
                        EXECUTION_ORDERS_REFUSED.labels(
                            symbol=symbol, reason=str(refused)[:48]
                        ).inc()
                        log.info("refused %s: %s", symbol, refused)
                    except Exception:
                        log.exception("Failed to act on signal %s", message_id)
    except asyncio.CancelledError:
        raise
    finally:
        await client.close()


def _symbol_of(fields: dict[str, str]) -> str:
    with contextlib.suppress(Exception):
        return str(json.loads(fields["signal"])["symbol"])
    return "unknown"


async def _handle(client: DerivV3Client, fields: dict[str, str]) -> None:
    signal = json.loads(fields["signal"])
    symbol = str(signal["symbol"])
    side = str(signal["side"])
    if side == "hold":
        return

    position = await place(
        client,
        settings=settings,
        state=state,
        symbol=symbol,
        side=side,
        entry_price=float(fields["entry"]),
        stop_distance=float(fields["stop_distance"]),
        reward_r=float(fields["reward_r"]),
        cost_r=float(fields["cost_r"]),
        p_win=float(fields["p_win"]),
        balance=float(runtime["balance"]),
    )
    EXECUTION_ORDERS_PLACED.labels(symbol=symbol, contract_type=position.order.contract_type).inc()
    EXECUTION_OPEN_POSITIONS.set(len(state.open_positions))
    log.info("bought %s %s contract %s", symbol, position.order.contract_type, position.contract_id)
    asyncio.create_task(_watch(client, symbol))


@app.on_event("startup")
async def startup_event() -> None:
    state.roll_day()
    if not settings.is_demo_trading:
        log.warning("DERIV_TRADE_MODE is %r; no orders will be placed", settings.deriv_trade_mode)
    with contextlib.suppress(Exception):
        runtime["redis"] = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
    if runtime["redis"] is not None and settings.deriv_app_id:
        runtime["loop"] = asyncio.create_task(_consume())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    task = runtime.get("loop")
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    client = runtime.get("redis")
    if client is not None:
        with contextlib.suppress(Exception):
            await client.aclose()


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(
        status="degraded" if state.halted else "ok",
        environment=settings.environment,
        trade_mode=settings.deriv_trade_mode,
        open_positions=len(state.open_positions),
        realised_r_today=round(state.realised_r_today, 4),
        halted=state.halted,
        detail=state.halt_reason or None,
    )


@app.get("/metrics")
async def metrics():
    return metrics_response()


@app.post("/api/v1/execution/halt")
async def halt(reason: str = "manual halt") -> dict[str, Any]:
    """Stop trading and close everything open.

    The counterpart to bot-service's emergency stop, at the layer that actually
    holds contracts. Safe to call when nothing is open.
    """
    client = runtime.get("client")
    closed = 0
    if client is not None:
        closed = await close_all(client, state, reason=reason)
    else:
        state.halted = True
        state.halt_reason = reason
    EXECUTION_OPEN_POSITIONS.set(len(state.open_positions))
    return {"halted": True, "reason": reason, "closed": closed}


@app.post("/api/v1/execution/resume")
async def resume() -> dict[str, Any]:
    state.halted = False
    state.halt_reason = ""
    return {"halted": False}
