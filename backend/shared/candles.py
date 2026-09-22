"""OHLC candles: fetching, rolling windows, and persistence.

Nothing in the platform produced candles before this. `market-data-service`
streams raw ticks to the `market-events` Redis stream, but `encode_features`
needs multi-timeframe OHLC, and the `MarketCandle` schema in
`backend/shared/schemas/trading.py` sat unused.

**Poll on frame close rather than aggregating ticks.** Aggregating means
tracking partial bars, gap-filling weekends and holidays, deduplicating on
reconnect, and deciding when a bar is final. Asking Deriv for the closed bar is
one request with none of that, and it is the same data. `next_close` computes
when to ask.

The paging in :func:`fetch_candles` is the shape already proven in
`fa-ml-toolkit/scripts/download_deriv_training_data.py`: walk backwards on
``end = oldest - 1``, and decide termination by *backward progress* rather than
by row count. Deriv honours ``end`` on intraday granularities but ignores it for
some symbols on the daily frame, answering every request with the same rolling
window shifted a second or two. Each such page adds a few unseen epochs, so a
"did the set grow" test never trips and the loop walks hundreds of pages
without reaching further back.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from backend.shared.schemas.trading import MarketCandle

#: Deriv serves each of these natively, so nothing here is resampled. W1 is
#: absent deliberately: Deriv has no weekly granularity, and rebuilding weekly
#: bars from daily ones is the caller's problem, not this module's.
DERIV_GRANULARITY: dict[str, int] = {
    "1m": 60,
    "2m": 120,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14_400,
    "8h": 28_800,
    "1d": 86_400,
}

#: Deriv's per-request cap on `ticks_history`.
PAGE = 1000


class CandleError(RuntimeError):
    """A timeframe or candle payload cannot be used."""


def granularity_for(timeframe: str) -> int:
    try:
        return DERIV_GRANULARITY[timeframe]
    except KeyError:
        raise CandleError(
            f"Deriv has no native granularity for {timeframe!r}; "
            f"choose one of {', '.join(sorted(DERIV_GRANULARITY))}"
        ) from None


def next_close(timeframe: str, *, after: datetime | None = None) -> datetime:
    """When the bar containing ``after`` closes.

    Frames divide the UTC day evenly, so the boundary is just the next multiple
    of the granularity since the epoch. Poll a little after this, not exactly
    on it: Deriv publishes the closed bar once its own clock has passed the
    boundary.
    """
    seconds = granularity_for(timeframe)
    moment = after or datetime.now(tz=timezone.utc)
    epoch = int(moment.timestamp())
    return datetime.fromtimestamp((epoch // seconds + 1) * seconds, tz=timezone.utc)


def candle_from_payload(payload: dict[str, Any], *, symbol: str, timeframe: str) -> MarketCandle:
    """One Deriv candle as a :class:`MarketCandle`.

    Deriv candle history carries no volume -- the API does not return one, and
    only the live tick stream can supply it -- so volume defaults to 0 rather
    than being invented.
    """
    try:
        epoch = int(float(payload["epoch"]))
        open_time = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return MarketCandle(
            symbol=symbol,
            timeframe=timeframe,
            open_time=open_time,
            close_time=open_time + timedelta(seconds=granularity_for(timeframe)),
            open=float(payload["open"]),
            high=float(payload["high"]),
            low=float(payload["low"]),
            close=float(payload["close"]),
            volume=float(payload.get("volume") or 0.0),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CandleError(f"Unusable candle payload for {symbol} {timeframe}") from exc


async def fetch_candles(
    client: Any,
    symbol: str,
    *,
    timeframe: str,
    count: int = PAGE,
) -> list[MarketCandle]:
    """Up to ``count`` closed candles, oldest first.

    ``client`` is a :class:`~backend.shared.deriv_v3_client.DerivV3Client`,
    taken as a parameter rather than constructed so tests can pass a fake.
    """
    granularity = granularity_for(timeframe)
    collected: dict[int, dict[str, Any]] = {}
    end: int | str = "latest"
    oldest_seen: int | None = None

    while len(collected) < count:
        page = await client.ticks_history(
            symbol, granularity=granularity, count=min(PAGE, count - len(collected)), end=end
        )
        if not page:
            break

        for payload in page:
            try:
                collected[int(float(payload["epoch"]))] = payload
            except (KeyError, TypeError, ValueError):
                continue

        try:
            oldest = int(float(page[0]["epoch"]))
        except (KeyError, IndexError, TypeError, ValueError):
            break
        # Backward progress, not row count: see the module docstring.
        if oldest_seen is not None and oldest >= oldest_seen:
            break
        oldest_seen = oldest
        end = oldest - 1

    ordered = [collected[key] for key in sorted(collected)][-count:]
    return [candle_from_payload(p, symbol=symbol, timeframe=timeframe) for p in ordered]


class CandleWindow:
    """A rolling window of the newest candles for one symbol and timeframe.

    Bounded so a long-running service does not grow without limit, and
    deduplicated by open time so a reconnect that re-delivers the last few bars
    updates them in place instead of appending duplicates.
    """

    def __init__(self, symbol: str, timeframe: str, *, maxlen: int = 1500) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.maxlen = maxlen
        self._by_open: dict[datetime, MarketCandle] = {}
        self._order: deque[datetime] = deque()

    def __len__(self) -> int:
        return len(self._order)

    def extend(self, candles: Iterable[MarketCandle]) -> int:
        """Merge candles in. Returns how many open times were new."""
        added = 0
        for candle in candles:
            if candle.open_time in self._by_open:
                # A re-delivered bar is the authoritative version: the earlier
                # one may have been read while it was still forming.
                self._by_open[candle.open_time] = candle
                continue
            self._by_open[candle.open_time] = candle
            self._order.append(candle.open_time)
            added += 1

        if added:
            self._order = deque(sorted(self._order))
            while len(self._order) > self.maxlen:
                self._by_open.pop(self._order.popleft(), None)
        return added

    @property
    def candles(self) -> list[MarketCandle]:
        return [self._by_open[key] for key in self._order]

    @property
    def newest(self) -> MarketCandle | None:
        return self._by_open[self._order[-1]] if self._order else None

    def bars(self) -> list[Any]:
        """The window as toolkit ``Bar`` objects, for ``encode_features``.

        The toolkit is imported lazily so that services which only move candles
        around -- market-data-service -- do not need `forex_agent` installed.
        """
        from forex_agent.models import Bar, Timeframe

        frame = Timeframe(self.timeframe)
        return [
            Bar(
                candle.symbol,
                frame,
                candle.open_time,
                candle.close_time,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                int(candle.volume),
            )
            for candle in self.candles
        ]


# ------------------------------------------------------------------ persistence

UPSERT_SQL = """
    INSERT INTO market_candles
        (symbol, timeframe, open_time, close_time, open, high, low, close, volume)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    ON CONFLICT (symbol, timeframe, open_time) DO UPDATE SET
        close_time = EXCLUDED.close_time,
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume
"""

SELECT_SQL = """
    SELECT symbol, timeframe, open_time, close_time, open, high, low, close, volume
    FROM market_candles
    WHERE symbol = $1 AND timeframe = $2
    ORDER BY open_time DESC
    LIMIT $3
"""


async def persist_candles(pool: Any, candles: Sequence[MarketCandle]) -> int:
    """Upsert candles. Returns how many rows were sent.

    Upsert rather than insert because the newest bar may be re-fetched: polling
    a moment early returns the bar still forming, and the next poll must be
    allowed to correct it.
    """
    if not candles:
        return 0
    rows = [
        (
            c.symbol,
            c.timeframe,
            c.open_time,
            c.close_time,
            c.open,
            c.high,
            c.low,
            c.close,
            c.volume,
        )
        for c in candles
    ]
    async with pool.acquire() as conn:
        await conn.executemany(UPSERT_SQL, rows)
    return len(rows)


async def load_candles(
    pool: Any, symbol: str, timeframe: str, *, limit: int = 1500
) -> list[MarketCandle]:
    """The newest stored candles, oldest first, so a service warm-starts.

    Without this a restart cannot encode anything until it has re-downloaded a
    full macro window, which at 600 H4 bars is 100 days of history.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(SELECT_SQL, symbol, timeframe, limit)
    candles = [
        MarketCandle(
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            open_time=row["open_time"],
            close_time=row["close_time"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for row in rows
    ]
    candles.reverse()
    return candles
