"""Download MetaTrader 5 candle history for training and gate replay.

The MT5 counterpart of ``download_deriv_training_data.py``, writing the same
two artefacts in the same shapes so ``backtest_gates.py`` and ``null_test.py``
consume either without knowing which broker produced it:

* a ``(symbol, timeframe, close)`` CSV, and
* with ``--raw``, the full OHLC JSON keyed ``{symbol: {frame: [candle, ...]}}``
  where each candle is ``{"epoch", "open", "high", "low", "close"}`` and
  **epoch is the candle's start**.

Two things differ from the Deriv script, both in our favour.

There is no pagination. Deriv caps a response at roughly 1000 candles and its
history has to be walked backwards page by page, with a guard for the daily
frame where it stops honouring ``end``. ``copy_rates_from_pos`` takes a count
and returns it, so depth is limited only by what the terminal has downloaded.

There is far more of it. That is the point of collecting here rather than from
the Deriv WebSocket: the terminal holds years where the API offered weeks.
**Scroll each chart back in the terminal before running this** -- MT5 fetches
history lazily, and a symbol you have never opened at a given timeframe will
return only what it happens to hold.

Run it with the interpreter that has the ``MetaTrader5`` wheel, on the machine
running the terminal:

    .venv311\\Scripts\\python scripts\\download_mt5_training_data.py \\
        --count 200000 --raw data/mt5_history.json
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forex_agent.config import MT5_FOREX_MARKETS, settings

#: Frame name -> the attribute holding MetaTrader's timeframe constant. Named
#: rather than numeric because the values are an implementation detail of the
#: package: H4 is 16388, not 240.
FRAMES = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
    "1w": "TIMEFRAME_W1",
}

#: Nominal seconds per bar, used to drop the bar still forming and to size the
#: ranged request.
FRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "4h": 14_400, "1d": 86_400, "1w": 604_800,
}


def connect(mt5, config) -> None:
    kwargs: dict[str, object] = {}
    if config.mt5_terminal_path:
        kwargs["path"] = config.mt5_terminal_path
    if config.mt5_login:
        kwargs.update(
            login=config.mt5_login, password=config.mt5_password, server=config.mt5_server
        )
    if not mt5.initialize(**kwargs):
        raise SystemExit(f"could not reach the MT5 terminal: {mt5.last_error()}")
    info = mt5.account_info()
    if info is None:
        raise SystemExit("the terminal returned no account information")
    kind = "demo" if info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "LIVE"
    print(f"connected to account {info.login} ({kind}) on {info.server}", flush=True)


def resolve(mt5, wanted: str) -> str | None:
    """Match a configured name to the broker's own, which may be suffixed."""
    if mt5.symbol_info(wanted) is not None:
        return wanted
    for symbol in mt5.symbols_get() or ():
        if symbol.name.startswith(wanted):
            return symbol.name
    return None


#: Ranged-request windows to try, narrowest first, in days.
#:
#: Too *wide* is the failure mode, and it bites differently per frame. AUDUSD
#: answered a 109-year daily request with nothing and a five-year one with
#: 1301 bars; EURUSD answered the 109-year request happily. Worse, asking for
#: 500k M5 bars computes a 4.75-year window, and **every intraday frame
#: returned nothing at all** because even the narrowest rung was too wide --
#: silently replacing a good history file with empty 1m/5m/15m.
#:
#: So the narrow rungs are unconditional. The caller's computed window is
#: added to this ladder rather than replacing its floor.
RANGE_WINDOW_DAYS = (30, 90, 365, 365 * 5, 365 * 20, 365 * 60)


def prime_by_range(mt5, symbol: str, timeframe: int, count: int, seconds: int):
    """Force the terminal to download a frame it has not cached.

    Walks widening windows and keeps the largest non-empty answer, so deep
    history is not truncated to five years and a symbol that refuses a wide
    request still returns something.
    """
    now = datetime.now(timezone.utc)
    # Clamped: asking for 500k weekly bars computes a 9,580-year window,
    # which overflows datetime before it ever reaches the terminal. No
    # broker holds more than the widest ladder rung anyway.
    wanted = min(int(seconds * count / 86_400) + 1, max(RANGE_WINDOW_DAYS))
    # Union, not replacement: the computed window is a hint about how much is
    # wanted, never a floor on what is tried.
    windows = sorted({wanted, *RANGE_WINDOW_DAYS})
    best = None
    for days in windows:
        found = mt5.copy_rates_range(
            symbol, timeframe, now - timedelta(days=days), now
        )
        found_len = 0 if found is None else len(found)
        if found_len == 0:
            # Too wide for this symbol and frame. Keep walking only if
            # nothing has worked yet -- once something has, wider requests
            # returning empty means the ceiling has been found.
            if best is not None:
                break
            continue
        if found_len > (0 if best is None else len(best)):
            best = found
        if found_len >= count:
            break
    return best


def fetch_frame(mt5, symbol: str, frame: str, count: int) -> list[dict]:
    """Completed candles for one frame, oldest first.

    MetaTrader downloads history lazily, and ``copy_rates_from_pos`` does not
    trigger the download: on a frame the terminal has never held it fails
    outright with "Terminal: Call failed". ``copy_rates_range`` *does* trigger
    it. Measured on a fresh terminal, D1 went from 0 bars to 2843 back to
    2015, and W1 is only ever reachable this way -- it still reports nothing
    by position even after the range request has populated the cache.

    That is not a slow warm-up, it is a silent hole in the training set. The
    strategy needs 1W/1D/4H structural consensus, so a frame that comes back
    empty trains a model on data the live agent will never have.
    """
    timeframe = getattr(mt5, FRAMES[frame])
    seconds = FRAME_SECONDS[frame]

    # ``rates`` is a numpy structured array, so ``rates or ()`` is ambiguous
    # and ``if rates`` raises. Length is the only safe emptiness test.
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)
    if rates is None or len(rates) == 0:
        rates = prime_by_range(mt5, symbol, timeframe, count, seconds)
    if rates is None or len(rates) == 0:
        return []

    # A ranged request includes the bar still forming, which ``start_pos=1``
    # would have excluded. Writing a partial candle into training data as
    # though it had closed is look-ahead.
    cutoff = datetime.now(timezone.utc).timestamp()
    candles = [
        {
            "epoch": int(rate["time"]),
            "open": float(rate["open"]),
            "high": float(rate["high"]),
            "low": float(rate["low"]),
            "close": float(rate["close"]),
            # Tick count, not traded size: MT5 reports `real_volume` as 0 on
            # every retail forex feed, so `tick_volume` is the only weighting
            # available. It is what the VWAP in strategy/features.py reads;
            # without it that column silently degrades to an unweighted mean.
            "volume": int(rate["tick_volume"]),
        }
        for rate in rates
        if int(rate["time"]) + seconds <= cutoff
    ]
    candles.sort(key=lambda candle: candle["epoch"])
    return candles[-count:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/mt5_training.csv"))
    parser.add_argument("--raw", type=Path, default=None, help="also write full OHLC JSON")
    parser.add_argument("--symbols", nargs="+", default=list(MT5_FOREX_MARKETS))
    parser.add_argument("--frames", nargs="+", default=list(FRAMES))
    parser.add_argument(
        "--count", type=int, default=100_000,
        help="candles per frame; the terminal caps this at what it holds",
    )
    args = parser.parse_args()

    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except ImportError:
        raise SystemExit(
            "MetaTrader5 is not installed in this interpreter. It is a "
            "Windows-only wheel and must be run alongside the terminal:\n"
            "    .venv311\\Scripts\\python -m pip install MetaTrader5"
        )

    unknown = [frame for frame in args.frames if frame not in FRAMES]
    if unknown:
        raise SystemExit(f"unknown frame(s) {unknown}; choose from {list(FRAMES)}")

    connect(mt5, settings)
    try:
        history: dict[str, dict[str, list[dict]]] = {}
        rows: list[tuple[str, str, float]] = []
        thin: list[str] = []

        for wanted in args.symbols:
            resolved = resolve(mt5, wanted)
            if resolved is None:
                print(f"  {wanted:<10} not published by this broker; skipped", flush=True)
                continue
            if not mt5.symbol_select(resolved, True):
                print(f"  {wanted:<10} could not be selected; skipped", flush=True)
                continue

            # Keyed by the *configured* name so the artefact matches what the
            # agent trades, whatever this broker calls its instrument.
            history[wanted] = {}
            summary = []
            for frame in args.frames:
                candles = fetch_frame(mt5, resolved, frame, args.count)
                history[wanted][frame] = candles
                rows.extend((wanted, frame, c["close"]) for c in candles)
                if candles:
                    span = (candles[-1]["epoch"] - candles[0]["epoch"]) / 86_400
                    oldest = datetime.fromtimestamp(candles[0]["epoch"], tz=timezone.utc)
                    summary.append(f"{frame}={len(candles)} ({span:.0f}d, from {oldest:%Y-%m-%d})")
                    # Only flag a frame that looks truncated for its depth.
                    # A weekly frame legitimately holds a few hundred bars,
                    # so comparing every frame to --count cries wolf.
                    if len(candles) < 200:
                        thin.append(f"{wanted} {frame} ({len(candles)})")
                else:
                    summary.append(f"{frame}=0")
                    thin.append(f"{wanted} {frame} (0)")
            label = wanted if resolved == wanted else f"{wanted}->{resolved}"
            print(f"  {label:<18} " + "  ".join(summary), flush=True)
    finally:
        mt5.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("symbol", "timeframe", "close"))
        writer.writerows(rows)
    print(f"saved {args.output} ({len(rows)} rows)")

    if args.raw:
        import json

        args.raw.parent.mkdir(parents=True, exist_ok=True)
        args.raw.write_text(json.dumps(history), encoding="utf-8")
        print(f"saved {args.raw}")

    if thin:
        # Not an error: it is how MT5 behaves, and it is silently ruinous for
        # a model trained on what looks like a full set.
        print(
            "\nWARNING: these returned far fewer candles than requested, which "
            "usually means the terminal has not downloaded that history yet.\n"
            "Open each chart at that timeframe, scroll back until it stops "
            "loading, then re-run:\n  " + "\n  ".join(thin)
        )


if __name__ == "__main__":
    main()
