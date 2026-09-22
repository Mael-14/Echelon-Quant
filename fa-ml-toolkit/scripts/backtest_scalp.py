"""Sequential M5 scalp backtest: cascade entry, candle trigger, managed exit.

Everything before this measured *labels* -- fixed triple barriers scored in
bulk, every bar treated as an independent trade. That answers whether a feature
carries information. It does not answer what an account would have done, and
the two disagree for three reasons this script removes:

*One position at a time.* The label studies scored 43,940 overlapping trades in
a five-month window. An account holds one. Entries arriving while a trade is
open never happen, which changes both the trade count and *which* trades are in
the sample.

*A trigger, not a clock.* Entry requires a candlestick rejection in the
direction of the daily tide, at structure, after a retracement -- not the
arrival of the next bar.

*A managed exit.* The fixed 1.5R barrier is replaced by the production
``ExitPolicy``: break-even once the trade has earned its risk, a trailing stop
past that, and a time stop. A winner that keeps running is not handed back at a
fixed target; one that stalls is not given back to the stop. The policy is
imported from ``execution/exit_policy.py`` rather than reimplemented, so what
this measures is what the live agent would do.

**The cascade runs at its own cadence.** Daily state is recomputed when a daily
bar closes and 4-hour state when a 4-hour bar closes -- not once per M5 bar.
That is both what "check D1 daily, H4 every four hours" means and the only way
this finishes: the candle pattern fires on roughly half of all bars, so
recomputing a 400-bar EMA behind it costs an encode on 50,000 bars per symbol.

Intrabar order is pessimistic: within each bar the adverse extreme is visited
before the favourable one, because OHLC cannot say which came first and only
the unfavourable reading cannot flatter the result.

Usage:

    python scripts/backtest_scalp.py --history data/deriv_history.json
"""
from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import sqrt
from pathlib import Path

from forex_agent.execution.exit_policy import ExitPolicy, evaluate_exit
from forex_agent.models import Bar, Side, Timeframe
from forex_agent.strategy.features import pivot_levels
from forex_agent.strategy.indicators import (
    candlestick_confirmation,
    ema,
    swing_points,
    true_range_atr,
)

MACRO_WINDOW = 400
EXEC_WINDOW = 200

#: Spread drag in R against a 1.5xATR stop, per execution frame, charged once
#: per round trip. Cost in R is spread/stop_distance and ATR grows with the
#: square root of time, so the drag falls as 1/sqrt(time) -- ADR-004 calls this
#: the largest lever on expectancy the toolkit found.
#:
#: M5, M15, H4 and D1 are measured. **M1 is extrapolated** from M5 by that same
#: law, and the law reproduces the measured cells only to within about 20%, so
#: re-measure it against a real Deriv spread before trusting an M1 result.
SPREAD_DRAG_R = {
    Timeframe.M1: 0.095 * sqrt(300 / 60),
    Timeframe.M5: 0.095,
    Timeframe.M15: 0.045,
    Timeframe.H4: 0.017,
    Timeframe.D1: 0.007,
}
STOP_ATR_MULTIPLE = 1.5


def default_spread_atr(frame: Timeframe) -> float:
    """Spread in ATR units for an execution frame, or 0 if it has no entry."""
    drag = SPREAD_DRAG_R.get(frame)
    return drag * STOP_ATR_MULTIPLE if drag is not None else 0.0


@dataclass(slots=True)
class Trade:
    symbol: str
    side: str
    opened_at: datetime
    entry: float
    stop: float
    target: float
    risk: float
    pattern: str
    cost_r: float = 0.0
    closed_at: datetime | None = None
    exit_price: float = 0.0
    reason: str = ""
    r: float = 0.0


@dataclass(slots=True)
class MacroState:
    """What a structural frame contributes, recomputed only when it closes."""

    trend_score: float
    levels: list[float] = field(default_factory=list)


def load_bars(symbol: str, candles: list[dict], timeframe: Timeframe) -> list[Bar]:
    span = timedelta(seconds=timeframe.seconds)
    bars: list[Bar] = []
    for candle in candles:
        start = datetime.fromtimestamp(float(candle["epoch"]), tz=timezone.utc)
        try:
            bars.append(
                Bar(
                    symbol, timeframe, start, start + span,
                    float(candle["open"]), float(candle["high"]),
                    float(candle["low"]), float(candle["close"]),
                    int(candle.get("volume", 0) or 0),
                )
            )
        except ValueError:
            continue
    bars.sort(key=lambda bar: bar.start)
    return bars


def macro_state(bars: list[Bar], cut: int, args) -> MacroState | None:
    """Trend score and key levels for the most recently *closed* macro bar."""
    window = bars[max(0, cut - MACRO_WINDOW) : cut]
    if len(window) < args.ema_length:
        return None
    atr = true_range_atr(window, args.atr_length)
    if atr <= 0:
        return None
    baseline = ema([bar.close for bar in window], args.ema_length)
    return MacroState(
        trend_score=(window[-1].close - baseline) / atr,
        levels=pivot_levels(window, strength=2, lookback=args.pivot_lookback),
    )


def precompute_entries(d1, h4, m5, args) -> tuple[dict[int, dict], dict[str, int]]:
    """Every bar's entry candidate, independent of whether a position is open.

    Computed once and shared by both exit policies, because the expensive work
    -- the cascade and the M5 structure read -- does not depend on the exit.
    """
    d1_ends = [bar.end for bar in d1]
    h4_ends = [bar.end for bar in h4] if h4 is not None else []
    signals: dict[int, dict] = {}
    funnel = {"bars": 0, "candle": 0, "trend": 0, "pullback": 0, "structure": 0, "geometry": 0}

    d1_cut = h4_cut = -1
    daily = fourh = None

    for index in range(args.exec_window, len(m5)):
        funnel["bars"] += 1
        bar = m5[index]
        recent = m5[max(0, index - args.candle_within) : index + 1]

        side = pattern = None
        for candidate in (Side.BUY, Side.SELL):
            found = candlestick_confirmation(recent, candidate, within=args.candle_within)
            if found is not None:
                side, pattern = candidate, found
                break
        if side is None:
            continue
        funnel["candle"] += 1

        # The cascade, refreshed only when a structural bar has closed.
        now = bar.end
        new_d1 = bisect.bisect_right(d1_ends, now)
        if new_d1 != d1_cut:
            d1_cut, daily = new_d1, macro_state(d1, new_d1, args)
        if h4 is not None:
            new_h4 = bisect.bisect_right(h4_ends, now)
            if new_h4 != h4_cut:
                h4_cut, fourh = new_h4, macro_state(h4, new_h4, args)
            if fourh is None:
                continue
        if daily is None or not daily.levels:
            continue

        want = 1.0 if side is Side.BUY else -1.0
        if daily.trend_score * want < args.min_trend:
            continue
        if fourh is not None and fourh.trend_score * want < 0:
            continue
        funnel["trend"] += 1

        window = m5[max(0, index + 1 - args.exec_window) : index + 1]
        swing = swing_points(window, strength=2, lookback=args.swing_lookback)
        high, low = swing["high"], swing["low"]
        if high is None or low is None or high <= low:
            continue
        entry = bar.close
        retracement = (entry - low) / (high - low)
        pull = retracement if side is Side.BUY else 1.0 - retracement
        if not (args.pull_low <= pull <= args.pull_high):
            continue
        funnel["pullback"] += 1

        atr = true_range_atr(window, args.atr_length)
        if atr <= 0:
            continue
        # Structure behind the trade, measured in M5 ATRs as the spec defines.
        if side is Side.BUY:
            below = [lv for lv in daily.levels if lv <= entry]
            behind = (entry - (max(below) if below else daily.levels[0])) / atr
        else:
            above = [lv for lv in daily.levels if lv >= entry]
            behind = ((min(above) if above else daily.levels[-1]) - entry) / atr
        if not (0.0 <= behind <= args.max_structure_atr):
            continue
        funnel["structure"] += 1

        level = low if side is Side.BUY else high
        risk = (entry - level) if side is Side.BUY else (level - entry)
        if not (args.min_stop_atr * atr <= risk <= args.max_stop_atr * atr):
            continue
        funnel["geometry"] += 1

        signals[index] = {
            "side": "BUY" if side is Side.BUY else "SELL",
            "pattern": pattern,
            "entry": entry,
            "stop": level,
            "risk": risk,
            "target": entry + args.reward * risk if side is Side.BUY else entry - args.reward * risk,
            "cost_r": args.spread_atr * atr / risk,
        }
    return signals, funnel


def simulate(symbol, m5, signals, policy, args) -> list[Trade]:
    """Walk the bars holding at most one position."""
    trades: list[Trade] = []
    open_trade: Trade | None = None
    stop_now = peak = 0.0

    for index in range(args.exec_window, len(m5)):
        bar = m5[index]

        if open_trade is not None:
            long = open_trade.side == "BUY"
            adverse = bar.low if long else bar.high
            favourable = bar.high if long else bar.low
            for quote in (bar.open, adverse, favourable, bar.close):
                check = evaluate_exit(
                    side=open_trade.side, entry_price=open_trade.entry,
                    stop_price=stop_now, take_profit=open_trade.target,
                    risk_distance=open_trade.risk, peak_price=peak,
                    opened_at=open_trade.opened_at, quote=quote, now=bar.end,
                    policy=policy, cost_r=open_trade.cost_r,
                )
                if check.stop_price is not None:
                    stop_now = check.stop_price
                if check.peak_price is not None:
                    peak = check.peak_price
                if check.close_reason:
                    fill = fill_price(check.close_reason, open_trade, bar, stop_now, quote, long)
                    gain = (fill - open_trade.entry) if long else (open_trade.entry - fill)
                    open_trade.closed_at = bar.end
                    open_trade.exit_price = fill
                    open_trade.reason = check.close_reason
                    open_trade.r = gain / open_trade.risk - open_trade.cost_r
                    trades.append(open_trade)
                    open_trade = None
                    break
            continue

        signal = signals.get(index)
        if signal is None:
            continue
        open_trade = Trade(
            symbol=symbol, side=signal["side"], opened_at=bar.end,
            entry=signal["entry"], stop=signal["stop"], target=signal["target"],
            risk=signal["risk"], pattern=signal["pattern"], cost_r=signal["cost_r"],
        )
        stop_now = signal["stop"]
        peak = signal["entry"]

    return trades


def fill_price(reason: str, trade: "Trade", bar: Bar, stop_now: float,
               quote: float, long: bool) -> float:
    """Where the order actually filled, which is not where the bar reached.

    A resting order fills *at its level*, not at the extreme the bar printed
    on its way through. Booking the extreme credits a take profit with the
    whole overshoot and charges a stop with the whole spike -- which is how a
    -1R stop came out averaging -1.90R with a worst case of -31R.

    The exception is a gap. When the bar *opens* beyond the level there was no
    price at the level to fill against, and the fill is the open -- worse than
    the stop on a loss, better than the target on a win. That is the real
    behaviour of a weekend or news gap and it must not be smoothed away.

    A time stop is a market order, so it fills where it fires.
    """
    if reason == "time stop":
        return quote
    if reason == "take profit":
        level = trade.target
        gapped = bar.open >= level if long else bar.open <= level
    else:
        # "stop loss" and "trailing stop" both exit at the stop in force. On a
        # breach that level equals `stop_now`: the trail only tightens on a new
        # favourable extreme, and a quote cannot both set one and breach it.
        level = stop_now
        gapped = bar.open <= level if long else bar.open >= level
    return bar.open if gapped else level


def report(trades: list[Trade], title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)
    if not trades:
        print("  no trades")
        return
    values = [t.r for t in trades]
    wins = [v for v in values if v > 0]
    total = sum(values)
    equity = peak_eq = drawdown = 0.0
    for value in values:
        equity += value
        peak_eq = max(peak_eq, equity)
        drawdown = min(drawdown, equity - peak_eq)
    gross = [t.r + t.cost_r for t in trades]
    costs = [t.cost_r for t in trades]
    print(f"  trades            {len(trades)}")
    print(f"  win rate          {len(wins) / len(values):.4f}")
    print(f"  total R           {total:+.1f}")
    print(f"  R per trade       {total / len(values):+.4f}")
    # Gross separates "the entry has no edge" from "the edge is smaller than
    # the spread". They are different problems with different answers.
    print(f"  R per trade GROSS {sum(gross) / len(gross):+.4f}")
    print(f"  mean cost         {sum(costs) / len(costs):.4f} R")
    # Is the gross figure distinguishable from zero? That is the whole
    # question: a zero-edge entry cannot be rescued by cutting costs, it can
    # only be brought closer to zero from below.
    n = len(gross)
    mean_g = sum(gross) / n
    var = sum((g - mean_g) ** 2 for g in gross) / max(1, n - 1)
    se = (var / n) ** 0.5
    print(f"  gross t-stat      {mean_g / se:+.2f}  (se {se:.4f}, n {n})")
    print(f"  max drawdown      {drawdown:+.1f} R")
    print(f"  best / worst      {max(values):+.2f} / {min(values):+.2f} R")
    reasons: dict[str, list[float]] = {}
    for trade in trades:
        reasons.setdefault(trade.reason, []).append(trade.r)
    print(f"  {'exit reason':<18}{'n':>7}{'mean R':>10}")
    for reason, group in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"  {reason:<18}{len(group):>7}{sum(group) / len(group):>+10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/deriv_history.json"))
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="restrict to these symbols; default is every symbol in the file")
    parser.add_argument("--reward", type=float, default=1.5)
    parser.add_argument("--atr-length", type=int, default=14)
    parser.add_argument("--ema-length", type=int, default=200)
    parser.add_argument("--pivot-lookback", type=int, default=180)
    parser.add_argument("--swing-lookback", type=int, default=60)
    parser.add_argument("--candle-within", type=int, default=1)
    parser.add_argument("--min-trend", type=float, default=0.0)
    parser.add_argument("--pull-low", type=float, default=0.0)
    parser.add_argument("--pull-high", type=float, default=0.62)
    parser.add_argument("--max-structure-atr", type=float, default=8.0)
    parser.add_argument("--min-stop-atr", type=float, default=0.3)
    parser.add_argument("--max-stop-atr", type=float, default=5.0)
    parser.add_argument("--spread-atr", type=float, default=None,
                        help="spread in ATR units; defaults per --exec-frame from "
                             "the ADR-004 drag table (1m is extrapolated)")
    parser.add_argument("--structure-frame", type=Timeframe, default=Timeframe.H4,
                        choices=list(Timeframe), metavar="{1m,5m,15m,4h,1d,1w}",
                        help="frame supplying key levels and the primary trend "
                             "(default 4h)")
    parser.add_argument("--confirm-frame", type=Timeframe, default=None,
                        choices=list(Timeframe), metavar="{1m,5m,15m,4h,1d,1w}",
                        help="optional second trend filter between structure and "
                             "execution; omit for a single-frame cascade")
    parser.add_argument("--exec-frame", type=Timeframe, default=Timeframe.M1,
                        choices=list(Timeframe), metavar="{1m,5m,15m,4h,1d,1w}",
                        help="frame the trade is triggered and simulated on (default 1m)")
    parser.add_argument("--exec-window", type=int, default=EXEC_WINDOW,
                        help="trailing execution bars for swings and ATR")
    parser.add_argument("--hold-seconds", type=float, default=900.0)
    parser.add_argument("--breakeven-at-r", type=float, default=0.7)
    parser.add_argument("--trail-at-r", type=float, default=1.2)
    parser.add_argument("--trail-distance-r", type=float, default=0.6)
    args = parser.parse_args()

    # Each frame in the cascade must be strictly coarser than the one below it,
    # or the "structure" being read is the execution frame's own noise.
    chain = [args.structure_frame]
    if args.confirm_frame is not None:
        chain.append(args.confirm_frame)
    chain.append(args.exec_frame)
    for coarser, finer in zip(chain, chain[1:]):
        if coarser.seconds <= finer.seconds:
            raise SystemExit(
                f"cascade must run coarse to fine; {coarser.value} does not sit "
                f"above {finer.value}"
            )
    if args.spread_atr is None:
        args.spread_atr = default_spread_atr(args.exec_frame)
        if args.spread_atr <= 0.0:
            raise SystemExit(
                f"no spread drag on record for {args.exec_frame.value}; "
                "pass --spread-atr explicitly"
            )

    managed = ExitPolicy(
        max_hold_seconds=args.hold_seconds, breakeven_at_r=args.breakeven_at_r,
        breakeven_offset_r=0.1, trail_at_r=args.trail_at_r,
        trail_distance_r=args.trail_distance_r,
    )
    # Same entries, exited the old way, to isolate what management is worth.
    fixed = ExitPolicy(max_hold_seconds=args.hold_seconds)

    history = json.loads(args.history.read_text())
    cascade = " -> ".join(f.value for f in chain)
    print(f"Sequential backtest ({cascade}) -- one position at a time, candle-triggered")
    drag = args.spread_atr / STOP_ATR_MULTIPLE
    caveat = " EXTRAPOLATED, not measured" if args.exec_frame is Timeframe.M1 else " measured (ADR-004)"
    print(f"cost: spread {args.spread_atr:.4f} ATR = {drag:.4f}R per round trip;{caveat}")
    print(f"exit: hold {args.hold_seconds:.0f}s, breakeven {args.breakeven_at_r}R, "
          f"trail {args.trail_at_r}R by {args.trail_distance_r}R, reward {args.reward}R\n")

    all_managed: list[Trade] = []
    all_fixed: list[Trade] = []
    totals = {"bars": 0, "candle": 0, "trend": 0, "pullback": 0, "structure": 0, "geometry": 0}
    for symbol, frames in sorted(history.items()):
        if args.symbols and symbol not in args.symbols:
            continue
        d1 = load_bars(symbol, frames.get(args.structure_frame.value) or [], args.structure_frame)
        if args.confirm_frame is None:
            h4 = None
        else:
            h4 = load_bars(symbol, frames.get(args.confirm_frame.value) or [], args.confirm_frame)
        m5 = load_bars(symbol, frames.get(args.exec_frame.value) or [], args.exec_frame)
        if len(d1) < args.ema_length:
            print(f"  {symbol}: skipped ({args.structure_frame.value} has {len(d1)} "
                  f"bars, needs {args.ema_length})")
            continue
        if h4 is not None and len(h4) < args.ema_length:
            print(f"  {symbol}: skipped ({args.confirm_frame.value} has {len(h4)} "
                  f"bars, needs {args.ema_length})")
            continue
        if len(m5) < args.exec_window:
            print(f"  {symbol}: skipped ({args.exec_frame.value} has {len(m5)} "
                  f"bars, needs {args.exec_window})")
            continue
        signals, funnel = precompute_entries(d1, h4, m5, args)
        for key in totals:
            totals[key] += funnel[key]
        a = simulate(symbol, m5, signals, managed, args)
        b = simulate(symbol, m5, signals, fixed, args)
        print(f"  {symbol}: {len(signals)} candidates -> {len(a)} trades, "
              f"{sum(t.r for t in a):+.1f}R managed", flush=True)
        all_managed.extend(a)
        all_fixed.extend(b)

    print(f"\n  entry funnel: {totals['bars']} bars -> candle {totals['candle']}"
          f" -> trend {totals['trend']} -> pullback {totals['pullback']}"
          f" -> structure {totals['structure']} -> sizeable {totals['geometry']}")

    report(all_fixed, "FIXED EXIT (stop, target, clock) -- the old label, traded")
    report(all_managed, "MANAGED EXIT (break-even + trail) -- the production policy")


if __name__ == "__main__":
    main()
