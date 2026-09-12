from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SharedSchema(BaseModel):
    """Base model for shared schemas across services."""

    model_config = ConfigDict(extra="forbid")


class BotLifecycleState(str, Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class BotConfig(SharedSchema):
    bot_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    account: str = Field(min_length=1)
    symbols: list[str] = Field(default_factory=list)
    max_positions: int = Field(gt=0)
    max_positions_per_symbol: int = Field(gt=0)
    min_lot: float = Field(gt=0)
    max_lot: float = Field(gt=0)
    risk_per_trade: float = Field(gt=0, le=1)
    max_daily_loss: float = Field(ge=0)
    max_drawdown: float = Field(ge=0)
    strategy: str = Field(min_length=1)
    dynamic_lot: bool = False
    pyramiding: bool = False
    trailing_stop: bool = False
    break_even: bool = False
    partial_close: bool = False


class BotStatus(SharedSchema):
    bot_id: str = Field(min_length=1)
    state: BotLifecycleState = BotLifecycleState.CREATED
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_seconds: int = Field(default=0, ge=0)
    error: str | None = None


class Instrument(SharedSchema):
    symbol: str = Field(min_length=1)
    base_asset: str = Field(min_length=1)
    quote_asset: str = Field(min_length=1)
    tick_size: float = Field(gt=0)
    min_qty: float = Field(gt=0)
    is_active: bool = True


class Account(SharedSchema):
    account_id: str = Field(min_length=1)
    broker: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    balance: float
    equity: float
    margin_used: float = Field(default=0, ge=0)
    available_balance: float


class MarketTick(SharedSchema):
    symbol: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    last: float = Field(gt=0)
    volume: float = Field(ge=0)


class MarketCandle(SharedSchema):
    symbol: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    open_time: datetime
    close_time: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class Signal(SharedSchema):
    signal_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    side: Literal["buy", "sell", "hold"]
    strength: float = Field(ge=0, le=1)
    reason: str | None = None


class Prediction(SharedSchema):
    model_name: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    horizon_minutes: int = Field(gt=0)
    predicted_return: float
    confidence: float = Field(ge=0, le=1)


class OrderRequest(SharedSchema):
    client_order_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    quantity: float = Field(gt=0)
    price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)
    time_in_force: Literal["GTC", "IOC", "FOK"] = "GTC"


class OrderResponse(SharedSchema):
    order_id: str = Field(min_length=1)
    client_order_id: str = Field(min_length=1)
    status: Literal["accepted", "rejected", "filled", "partial", "cancelled"]
    filled_qty: float = Field(default=0, ge=0)
    avg_fill_price: float | None = Field(default=None, gt=0)
    message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Position(SharedSchema):
    account_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: Literal["long", "short", "flat"]
    quantity: float = Field(ge=0)
    entry_price: float = Field(ge=0)
    mark_price: float = Field(ge=0)
    unrealized_pnl: float = 0
    realized_pnl: float = 0


class RiskDecision(SharedSchema):
    approved: bool
    reason: str = Field(min_length=1)
    max_allowed_qty: float | None = Field(default=None, gt=0)
    recommended_leverage: float | None = Field(default=None, gt=0)
    risk_score: float = Field(ge=0, le=1)
