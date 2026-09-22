"""Validated domain objects shared by the trading-agent layers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from math import isfinite


def _positive(value: float, label: str) -> float:
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return value


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Regime(str, Enum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    UNKNOWN = "unknown"


class Impact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


#: Candle granularities the Deriv ``ticks_history`` API accepts. Anything else
#: is rejected with ``InputValidationFailed``, so longer structural timeframes
#: must be resampled from the largest supported source granularity.
DERIV_GRANULARITIES = (60, 120, 180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400)


class Timeframe(str, Enum):
    #: Execution frame for the scalp profile. Deriv serves it natively at
    #: ``granularity=60``, so unlike W1 it needs no resampling.
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def seconds(self) -> int:
        return {"1m": 60, "5m": 300, "15m": 900, "4h": 14_400, "1d": 86_400, "1w": 604_800}[self.value]

    @property
    def history_granularity(self) -> int:
        """Largest Deriv-supported granularity that divides into this timeframe.

        ``W1`` has no native Deriv granularity, so weekly structure is rebuilt
        from daily candles rather than requested directly.
        """
        return max(value for value in DERIV_GRANULARITIES if value <= self.seconds)


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    quote: float
    timestamp: datetime
    #: Deriv quotes both sides on every symbol this agent trades, but the
    #: fields are optional so a feed that omits them still produces a valid
    #: tick; the spread gate decides what to do about a missing spread.
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        _positive(self.quote, "quote")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        for label in ("bid", "ask"):
            value = getattr(self, label)
            if value is not None:
                _positive(value, label)
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError("ask must not be below bid")

    @property
    def utc_timestamp(self) -> datetime:
        return self.timestamp.astimezone(timezone.utc)

    @property
    def spread(self) -> float | None:
        """Ask minus bid, or ``None`` when the feed did not quote both."""
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    timeframe: Timeframe
    start: datetime
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must not be empty")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("bar times must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("bar end must be after start")
        values = {"open": self.open, "high": self.high, "low": self.low, "close": self.close}
        for label, value in values.items():
            _positive(value, label)
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("high/low do not contain open and close")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    side: Side
    entry_price: float
    stop_price: float
    take_profit: float
    confidence: float
    timeframe: Timeframe
    reason: str

    def __post_init__(self) -> None:
        entry = _positive(self.entry_price, "entry_price")
        stop = _positive(self.stop_price, "stop_price")
        target = _positive(self.take_profit, "take_profit")
        if self.side is Side.BUY and not stop < entry < target:
            raise ValueError("BUY signal requires stop < entry < take_profit")
        if self.side is Side.SELL and not target < entry < stop:
            raise ValueError("SELL signal requires take_profit < entry < stop")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")


@dataclass(frozen=True, slots=True)
class NewsEvent:
    currency: str
    starts_at: datetime
    ends_at: datetime
    impact: Impact
    title: str = ""

    def __post_init__(self) -> None:
        if not self.currency.strip():
            raise ValueError("currency must not be empty")
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("news times must be timezone-aware")
        if self.ends_at <= self.starts_at:
            raise ValueError("news event end must be after start")

    def is_active(self, at: datetime) -> bool:
        if at.tzinfo is None:
            raise ValueError("comparison time must be timezone-aware")
        instant = at.astimezone(timezone.utc)
        return self.starts_at.astimezone(timezone.utc) <= instant <= self.ends_at.astimezone(timezone.utc)
