"""Candles: paging, window bookkeeping, and the frame-close clock.

Hand-rolled fakes, no network, matching tests/test_deriv_client.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.shared.candles import (
    CandleError,
    CandleWindow,
    candle_from_payload,
    fetch_candles,
    granularity_for,
    load_candles,
    next_close,
    persist_candles,
)


def _payload(epoch: int, close: float = 1.10) -> dict:
    return {"epoch": epoch, "open": 1.0, "high": 1.2, "low": 0.9, "close": close}


class FakeClient:
    """Serves candles from a fixed history, honouring `end` like Deriv does."""

    def __init__(self, epochs: list[int], *, honour_end: bool = True) -> None:
        self.history = {e: _payload(e) for e in epochs}
        self.honour_end = honour_end
        self.calls: list[dict] = []

    async def ticks_history(self, symbol, *, granularity, count, end="latest", start=None):
        self.calls.append(
            {"symbol": symbol, "granularity": granularity, "count": count, "end": end}
        )
        keys = sorted(self.history)
        if self.honour_end and end != "latest":
            keys = [k for k in keys if k <= int(end)]
        return [self.history[k] for k in keys[-count:]]


class FakePool:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[tuple[str, list]] = []

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool

            async def __aexit__(self, *exc):
                return False

        return _Ctx()

    async def executemany(self, sql, rows):
        self.executed.append((sql, list(rows)))

    async def fetch(self, sql, *args):
        return self.rows


# ------------------------------------------------------------------ timeframes


def test_granularity_maps_to_deriv_seconds():
    assert granularity_for("1m") == 60
    assert granularity_for("4h") == 14_400


def test_an_unsupported_frame_says_what_is_supported():
    """W1 has no Deriv granularity; failing loudly beats resampling silently."""
    with pytest.raises(CandleError, match="1w"):
        granularity_for("1w")


@pytest.mark.parametrize(
    ("timeframe", "now", "expected"),
    [
        ("1m", "2026-09-22T10:30:15", "2026-09-22T10:31:00"),
        ("5m", "2026-09-22T10:31:00", "2026-09-22T10:35:00"),
        ("4h", "2026-09-22T10:31:00", "2026-09-22T12:00:00"),
        ("1d", "2026-09-22T10:31:00", "2026-09-23T00:00:00"),
    ],
)
def test_next_close_lands_on_the_frame_boundary(timeframe, now, expected):
    moment = datetime.fromisoformat(now).replace(tzinfo=timezone.utc)
    assert next_close(timeframe, after=moment) == datetime.fromisoformat(expected).replace(
        tzinfo=timezone.utc
    )


def test_next_close_on_an_exact_boundary_moves_to_the_next_one():
    """Otherwise a poll firing exactly on close would re-target the bar it just read."""
    boundary = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)
    assert next_close("4h", after=boundary) == datetime(2026, 9, 22, 16, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------- payload


def test_a_candle_carries_its_own_close_time():
    candle = candle_from_payload(_payload(1_600_000_000), symbol="frxEURUSD", timeframe="1m")

    assert candle.close_time == candle.open_time.replace(second=candle.open_time.second) + (
        candle.close_time - candle.open_time
    )
    assert (candle.close_time - candle.open_time).total_seconds() == 60
    # Deriv candle history has no volume field at all.
    assert candle.volume == 0.0


def test_a_malformed_candle_is_rejected_not_defaulted():
    with pytest.raises(CandleError):
        candle_from_payload({"epoch": 1, "open": 1.0}, symbol="frxEURUSD", timeframe="1m")


# ----------------------------------------------------------------------- paging


@pytest.mark.asyncio
async def test_fetch_returns_candles_oldest_first():
    client = FakeClient([60, 120, 180])

    candles = await fetch_candles(client, "frxEURUSD", timeframe="1m", count=3)

    assert [int(c.open_time.timestamp()) for c in candles] == [60, 120, 180]


@pytest.mark.asyncio
async def test_fetch_pages_backwards_until_it_has_enough():
    client = FakeClient([60 * i for i in range(1, 2501)])

    candles = await fetch_candles(client, "frxEURUSD", timeframe="1m", count=2500)

    assert len(candles) == 2500
    assert len(client.calls) > 1
    # Each page after the first walks strictly further back than the last.
    assert client.calls[0]["end"] == "latest"
    ends = [call["end"] for call in client.calls[1:]]
    assert ends == sorted(ends, reverse=True)
    assert len(set(ends)) == len(ends)


@pytest.mark.asyncio
async def test_a_broker_ignoring_end_terminates_instead_of_looping():
    """The reason termination is backward progress and not row count.

    Deriv ignores `end` for some symbols on the daily frame and answers every
    request with the same window. A "did the set grow" test never trips, so the
    loop would walk hundreds of pages without reaching further back.
    """
    client = FakeClient([86_400 * i for i in range(1, 11)], honour_end=False)

    candles = await fetch_candles(client, "frxEURUSD", timeframe="1d", count=5000)

    assert len(candles) == 10
    assert len(client.calls) == 2  # one page, then one that made no progress


@pytest.mark.asyncio
async def test_an_empty_response_stops_the_walk():
    client = FakeClient([])

    assert await fetch_candles(client, "frxEURUSD", timeframe="1m", count=100) == []


# ----------------------------------------------------------------------- window


def _candles(epochs, *, close=1.10):
    return [
        candle_from_payload(_payload(e, close), symbol="frxEURUSD", timeframe="1m") for e in epochs
    ]


def test_window_keeps_candles_in_time_order():
    window = CandleWindow("frxEURUSD", "1m")

    window.extend(_candles([180, 60, 120]))

    assert [int(c.open_time.timestamp()) for c in window.candles] == [60, 120, 180]
    assert int(window.newest.open_time.timestamp()) == 180


def test_a_redelivered_bar_replaces_rather_than_duplicates():
    """A reconnect re-sends the last bars, and the newest may have been partial."""
    window = CandleWindow("frxEURUSD", "1m")
    window.extend(_candles([60, 120], close=1.10))

    added = window.extend(_candles([120], close=1.25))

    assert added == 0
    assert len(window) == 2
    assert window.newest.close == 1.25


def test_window_is_bounded():
    window = CandleWindow("frxEURUSD", "1m", maxlen=3)

    window.extend(_candles([60, 120, 180, 240, 300]))

    assert len(window) == 3
    assert [int(c.open_time.timestamp()) for c in window.candles] == [180, 240, 300]


def test_window_converts_to_toolkit_bars():
    """This is the handoff to encode_features, so the types must line up."""
    from forex_agent.models import Timeframe

    window = CandleWindow("frxEURUSD", "1m")
    window.extend(_candles([60, 120]))

    bars = window.bars()

    assert len(bars) == 2
    assert bars[0].timeframe is Timeframe.M1
    assert bars[0].symbol == "frxEURUSD"
    assert bars[-1].close == 1.10


def test_an_empty_window_has_no_newest():
    assert CandleWindow("frxEURUSD", "1m").newest is None


# ------------------------------------------------------------------ persistence


@pytest.mark.asyncio
async def test_persist_sends_one_row_per_candle():
    pool = FakePool()

    written = await persist_candles(pool, _candles([60, 120]))

    assert written == 2
    sql, rows = pool.executed[0]
    assert "ON CONFLICT" in sql  # the newest bar re-fetches and must correct itself
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_persisting_nothing_touches_the_database():
    pool = FakePool()

    assert await persist_candles(pool, []) == 0
    assert pool.executed == []


@pytest.mark.asyncio
async def test_load_returns_oldest_first_for_warm_start():
    """The query orders newest-first to use the index; callers need the reverse."""
    pool = FakePool(
        [
            {
                "symbol": "frxEURUSD",
                "timeframe": "1m",
                "open_time": datetime.fromtimestamp(e, tz=timezone.utc),
                "close_time": datetime.fromtimestamp(e + 60, tz=timezone.utc),
                "open": 1.0,
                "high": 1.2,
                "low": 0.9,
                "close": 1.1,
                "volume": 0,
            }
            for e in (180, 120, 60)
        ]
    )

    candles = await load_candles(pool, "frxEURUSD", "1m")

    assert [int(c.open_time.timestamp()) for c in candles] == [60, 120, 180]
