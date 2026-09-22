"""Point-in-time multi-timeframe scalp model: D1 bias, H4 refinement, M5 trigger.

The cascade is enforced by construction rather than by convention. At each M5
bar the pipeline takes the most recent D1 and H4 bars **that have already
closed**, so a daily feature can only change once a day and a 4-hour feature
once every four hours -- which is exactly what "check D1 daily, H4 every four
hours" means once it is written down. The bar still forming is never visible:
its high and low are tomorrow's information, and reading them is the
look-ahead that makes a backtest lie.

Labels follow the triple-barrier method with a *structural* stop:

* profit barrier at ``+1.5 x ATR(M5)`` in the direction of the macro trend
* stop barrier at the invalidation level -- the local M5 swing low for a long,
  the swing high for a short, not a fixed ATR multiple
* time barrier at 24 M5 bars (two hours); scalping needs velocity, so an
  unresolved trade is closed and **labelled a loss**, not discarded

Because the stop is structural, the reward ratio is a property of each sample
rather than a constant, and so is the cost in R (``spread / stop_distance``).
Both are carried per row and fed to the gate.

Features are expressed **relative to the trade's direction**, so one model
covers both sides. A short's trend score, momentum and retracement are
mirrored and its support/resistance distances swapped; without this a linear
model would need opposite coefficients for longs and shorts and could fit
neither.

The classifier is L1-penalised logistic regression. Lasso is the point: a
feature that contributes nothing to the M5 outcome has its weight driven to
exactly zero, so the report below says which timeframes earned their place
instead of assuming they all did.

Usage:

    python scripts/train_scalp_model.py --history data/mt5_history.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from datetime import datetime, timedelta, timezone
from math import exp
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from forex_agent.models import Bar, Timeframe  # noqa: E402
from forex_agent.strategy.barriers import resolve  # noqa: E402
from forex_agent.strategy.expectancy import (  # noqa: E402
    breakeven_probability,
    brier_score,
    calibration_error,
)
from forex_agent.strategy.features import encode_features  # noqa: E402
from forex_agent.strategy.indicators import swing_points, true_range_atr  # noqa: E402

D1_WINDOW = 600
H4_WINDOW = 600
M5_WINDOW = 400

#: Trade-relative feature names, in matrix column order.
FEATURES = (
    "d1_trend",
    "d1_to_stop",
    "d1_to_target",
    "h4_trend",
    "h4_to_stop",
    "h4_to_target",
    "pullback",
    "momentum",
)

#: ADR-004 measured M5 spread drag at 0.095R against a 1.5xATR stop, so the
#: spread itself is about 0.1425 ATR. Structural stops vary in width, so cost
#: is recomputed per trade from that price rather than reused as a constant.
SPREAD_ATR = 0.095 * 1.5


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


#: Second encoding. Three defects in the first one are fixed here:
#:
#: 1. ``d1_trend`` was ``macro_trend_score * sign(macro_trend_score)`` -- the
#:    absolute value, with no negative anywhere in 73,872 rows, because the
#:    trade direction was *defined* by that score. A feature that cannot point
#:    against the trade cannot be shown to matter. v2 samples both directions
#:    at every bar, so the trend now varies in sign.
#: 2. Structural distance was handed to a linear model raw, ranging to +370
#:    M5-ATRs because D1 levels are huge next to a five-minute ATR. "Price is
#:    bouncing on support" is a threshold at |d| ~ 0, not a slope, so a
#:    bounded proximity ``exp(-|d|)`` is supplied alongside a clipped signed
#:    distance and L1 is left to choose between them.
#: 3. Tails are clipped. ``pullback`` reached 148 on degenerate swings, which
#:    alone decides what standardisation does to the other 99%.
FEATURES_V2 = (
    "d1_trend",
    "h4_trend",
    "d1_stop_prox",
    "d1_tgt_prox",
    "h4_stop_prox",
    "h4_tgt_prox",
    "d1_stop_dist",
    "d1_tgt_dist",
    "h4_stop_dist",
    "h4_tgt_dist",
    "pullback",
    "momentum",
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def encode_row_v2(daily, fourh, is_long: bool) -> list[float]:
    side = 1.0 if is_long else -1.0
    d1_stop = daily.distance_to_support if is_long else daily.distance_to_resistance
    d1_tgt = daily.distance_to_resistance if is_long else daily.distance_to_support
    h4_stop = fourh.distance_to_support if is_long else fourh.distance_to_resistance
    h4_tgt = fourh.distance_to_resistance if is_long else fourh.distance_to_support
    pull = daily.retracement_factor if is_long else 1.0 - daily.retracement_factor
    prox = lambda d: exp(-abs(_clip(d, -40.0, 40.0)))  # noqa: E731
    return [
        _clip(daily.macro_trend_score * side, -15.0, 15.0),
        _clip(fourh.macro_trend_score * side, -15.0, 15.0),
        prox(d1_stop), prox(d1_tgt), prox(h4_stop), prox(h4_tgt),
        _clip(d1_stop, -20.0, 20.0), _clip(d1_tgt, -20.0, 20.0),
        _clip(h4_stop, -20.0, 20.0), _clip(h4_tgt, -20.0, 20.0),
        _clip(pull, -1.0, 2.0),
        _clip(daily.micro_wave_momentum * side, -10.0, 10.0),
    ]


def build_dataset_v2(history: dict, args) -> dict:
    """Both directions at every bar, so direction is not a function of a feature."""
    rows: list[list[float]] = []
    labels: dict[int, list[int]] = {h: [] for h in args.holds}
    rewards: list[float] = []
    costs: list[float] = []
    stamps: list[float] = []
    sides: list[int] = []
    rejected = {"no_features": 0, "no_swing": 0, "geometry": 0}

    for symbol, frames in sorted(history.items()):
        d1 = load_bars(symbol, frames.get("1d") or [], Timeframe.D1)
        h4 = load_bars(symbol, frames.get("4h") or [], Timeframe.H4)
        m5 = load_bars(symbol, frames.get("5m") or [], Timeframe.M5)
        if len(d1) < D1_WINDOW or len(h4) < H4_WINDOW or len(m5) < M5_WINDOW:
            print(f"  {symbol}: skipped (insufficient history)")
            continue
        d1_ends = [bar.end for bar in d1]
        h4_ends = [bar.end for bar in h4]

        kept = 0
        for index in range(M5_WINDOW, len(m5) - max(args.holds) - 1, args.stride):
            now = m5[index].end
            d1_cut = bisect.bisect_right(d1_ends, now)
            h4_cut = bisect.bisect_right(h4_ends, now)
            if d1_cut < D1_WINDOW or h4_cut < H4_WINDOW:
                continue
            micro = m5[index + 1 - M5_WINDOW : index + 1]
            daily = encode_features(d1[d1_cut - D1_WINDOW : d1_cut], micro)
            fourh = encode_features(h4[h4_cut - H4_WINDOW : h4_cut], micro)
            if daily is None or fourh is None:
                rejected["no_features"] += 1
                continue
            entry = m5[index].close
            atr = true_range_atr(micro, args.atr_length)
            if atr <= 0:
                rejected["geometry"] += 1
                continue
            swing = swing_points(micro, strength=2, lookback=args.swing_lookback)

            for is_long in (True, False):
                level = swing["low"] if is_long else swing["high"]
                if level is None:
                    rejected["no_swing"] += 1
                    continue
                stop_distance = (entry - level) if is_long else (level - entry)
                if not (args.min_stop_atr * atr <= stop_distance <= args.max_stop_atr * atr):
                    rejected["geometry"] += 1
                    continue
                if args.target_mode == "r":
                    reward_r = args.target_atr
                else:
                    reward_r = (args.target_atr * atr) / stop_distance
                for hold in args.holds:
                    outcome = resolve(
                        m5, index, is_long=is_long, stop_distance=stop_distance,
                        reward_r=reward_r, max_hold=hold,
                    )
                    labels[hold].append(1 if outcome is True else 0)
                rows.append(encode_row_v2(daily, fourh, is_long))
                rewards.append(reward_r)
                costs.append(SPREAD_ATR * atr / stop_distance)
                stamps.append(now.timestamp())
                sides.append(1 if is_long else -1)
                kept += 1
        print(f"  {symbol}: {kept} samples")

    print(f"  rejected: {rejected}")
    order = np.argsort(np.asarray(stamps), kind="stable")
    return {
        "X": np.asarray(rows, dtype=float)[order],
        "y_by_hold": {h: np.asarray(v, dtype=int)[order] for h, v in labels.items()},
        "reward": np.asarray(rewards, dtype=float)[order],
        "cost": np.asarray(costs, dtype=float)[order],
        "stamp": np.asarray(stamps, dtype=float)[order],
        "side": np.asarray(sides, dtype=int)[order],
        "names": np.asarray(FEATURES_V2),
    }


def build_dataset(history: dict, args) -> dict[str, np.ndarray]:
    rows: list[list[float]] = []
    labels: dict[int, list[int]] = {h: [] for h in args.holds}
    rewards: list[float] = []
    costs: list[float] = []
    stamps: list[float] = []
    sides: list[int] = []
    rejected = {"no_features": 0, "no_swing": 0, "geometry": 0, "flat": 0}

    for symbol, frames in sorted(history.items()):
        d1 = load_bars(symbol, frames.get("1d") or [], Timeframe.D1)
        h4 = load_bars(symbol, frames.get("4h") or [], Timeframe.H4)
        m5 = load_bars(symbol, frames.get("5m") or [], Timeframe.M5)
        if len(d1) < D1_WINDOW or len(h4) < H4_WINDOW or len(m5) < M5_WINDOW:
            print(f"  {symbol}: skipped (insufficient history)")
            continue
        d1_ends = [bar.end for bar in d1]
        h4_ends = [bar.end for bar in h4]

        kept = 0
        for index in range(M5_WINDOW, len(m5) - max(args.holds) - 1, args.stride):
            now = m5[index].end
            # Point-in-time: only bars already closed at this instant.
            d1_cut = bisect.bisect_right(d1_ends, now)
            h4_cut = bisect.bisect_right(h4_ends, now)
            if d1_cut < D1_WINDOW or h4_cut < H4_WINDOW:
                continue
            micro = m5[index + 1 - M5_WINDOW : index + 1]

            daily = encode_features(d1[d1_cut - D1_WINDOW : d1_cut], micro)
            fourh = encode_features(h4[h4_cut - H4_WINDOW : h4_cut], micro)
            if daily is None or fourh is None:
                rejected["no_features"] += 1
                continue

            # Direction is the top of the cascade: the daily tide.
            if daily.macro_trend_score > 0:
                side, is_long = 1, True
            elif daily.macro_trend_score < 0:
                side, is_long = -1, False
            else:
                rejected["flat"] += 1
                continue

            entry = m5[index].close
            atr = true_range_atr(micro, args.atr_length)
            if atr <= 0:
                rejected["geometry"] += 1
                continue

            # Stop at structural invalidation, not an ATR multiple.
            swing = swing_points(micro, strength=2, lookback=args.swing_lookback)
            level = swing["low"] if is_long else swing["high"]
            if level is None:
                rejected["no_swing"] += 1
                continue
            stop_distance = (entry - level) if is_long else (level - entry)
            # A stop inside the spread or further than the day is not a scalp;
            # without these bounds the set fills with degenerate geometry whose
            # reward ratio alone would walk through any cost gate.
            if not (args.min_stop_atr * atr <= stop_distance <= args.max_stop_atr * atr):
                rejected["geometry"] += 1
                continue

            # "+1.5R" against a "-1R" structural stop reads as 1.5x the risk,
            # which fixes the reward ratio. Reading it as 1.5xATR instead lets
            # the ratio float with how far the swing happens to sit, and the
            # two give completely different books -- so it is a flag, not a
            # silent choice.
            if args.target_mode == "r":
                reward_r = args.target_atr
            else:
                reward_r = (args.target_atr * atr) / stop_distance
            # One encoding pass, several time-stops: the features do not
            # depend on the hold, so labelling each is nearly free and the
            # comparison is like-for-like on identical rows.
            for hold in args.holds:
                outcome = resolve(
                    m5, index, is_long=is_long, stop_distance=stop_distance,
                    reward_r=reward_r, max_hold=hold,
                )
                # Velocity is part of the thesis: unresolved is a loss.
                labels[hold].append(1 if outcome is True else 0)

            rows.append(
                [
                    daily.macro_trend_score * side,
                    daily.distance_to_support if is_long else daily.distance_to_resistance,
                    daily.distance_to_resistance if is_long else daily.distance_to_support,
                    fourh.macro_trend_score * side,
                    fourh.distance_to_support if is_long else fourh.distance_to_resistance,
                    fourh.distance_to_resistance if is_long else fourh.distance_to_support,
                    daily.retracement_factor if is_long else 1.0 - daily.retracement_factor,
                    daily.micro_wave_momentum * side,
                ]
            )
            rewards.append(reward_r)
            costs.append(SPREAD_ATR * atr / stop_distance)
            stamps.append(now.timestamp())
            sides.append(side)
            kept += 1
        print(f"  {symbol}: {kept} samples")

    print(f"  rejected: {rejected}")
    order = np.argsort(np.asarray(stamps))
    return {
        "X": np.asarray(rows, dtype=float)[order],
        "y_by_hold": {h: np.asarray(v, dtype=int)[order] for h, v in labels.items()},
        "reward": np.asarray(rewards, dtype=float)[order],
        "cost": np.asarray(costs, dtype=float)[order],
        "stamp": np.asarray(stamps, dtype=float)[order],
        "side": np.asarray(sides, dtype=int)[order],
        "names": np.asarray(FEATURES),
    }


def purged_split(data: dict, fraction: float, horizon_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Time-ordered split with the boundary purged.

    A training row whose 24-bar label window reaches past the split point has
    seen the test period. Dropping those is the difference between a backtest
    and a number that flatters itself.
    """
    stamps = data["stamp"]
    cut = stamps[int(len(stamps) * fraction)]
    train = np.where(stamps + horizon_seconds < cut)[0]
    test = np.where(stamps >= cut)[0]
    return train, test


def gate_report(p_win: np.ndarray, data: dict, index: np.ndarray, label: str) -> dict:
    """Apply the cost-adjusted veto and total the result in R."""
    reward, cost, y = data["reward"][index], data["cost"][index], data["y"][index]
    threshold = np.array(
        [breakeven_probability(r, c) for r, c in zip(reward, cost)]
    )
    taken = p_win > threshold
    if not taken.any():
        return {"label": label, "n": 0, "wins": 0, "win_rate": 0.0, "total_r": 0.0, "mean_r": 0.0}
    won, lost = y[taken] == 1, y[taken] == 0
    total = float((reward[taken][won] - cost[taken][won]).sum() - (1.0 + cost[taken][lost]).sum())
    return {
        "label": label,
        "n": int(taken.sum()),
        "wins": int(won.sum()),
        "win_rate": float(won.mean()),
        "total_r": total,
        "mean_r": total / int(taken.sum()),
    }


def train_and_report(data: dict, y: np.ndarray, hold: int, args) -> None:
    """Fit, evaluate out of sample, and run the cost gate for one time stop."""
    total = len(y)
    print()
    print("=" * 82)
    print(f"TIME STOP {hold} bars ({hold * 5 / 60:.1f}h)   base rate {y.mean():.4f}")
    print("=" * 82)

    train, test = purged_split(data, args.train_fraction, hold * 300)
    scaler = StandardScaler().fit(data["X"][train])
    x_train, x_test = scaler.transform(data["X"][train]), scaler.transform(data["X"][test])
    y_train, y_test = y[train], y[test]

    # A small, pre-declared C ladder, selected on the training split only.
    best, best_c = None, None
    inner = int(len(train) * 0.75)
    for c in (0.001, 0.01, 0.1, 1.0):
        trial = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=c, max_iter=2000)
        trial.fit(x_train[:inner], y_train[:inner])
        score = brier_score(list(trial.predict_proba(x_train[inner:])[:, 1]), list(y_train[inner:]))
        if best is None or score < best:
            best, best_c = score, c

    model = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=best_c, max_iter=2000)
    model.fit(x_train, y_train)
    p_test = model.predict_proba(x_test)[:, 1]

    # Per-feature accounting. A Lasso weight says what the fitted model leans
    # on; a univariate AUC says whether the column carries any ranking
    # information at all on its own. A feature can score on one and not the
    # other, and the pair is what separates a real input from a passenger.
    print(f"  C={best_c} | full per-feature accounting")
    print(f"    {'feature':<16}{'L1 weight':>12}{'|w| rank':>10}{'uni AUC':>10}{'verdict':>14}")
    weights = model.coef_[0]
    names = [str(n) for n in data['names']]
    order = sorted(range(len(names)), key=lambda i: -abs(weights[i]))
    rank = {i: r + 1 for r, i in enumerate(order)}
    for i, name in enumerate(names):
        column = x_test[:, i]
        try:
            auc = roc_auc_score(y_test, column)
        except ValueError:
            auc = 0.5
        # Direction-free: a column that ranks inversely is still informative.
        edge = abs(auc - 0.5)
        if weights[i] == 0.0:
            verdict = "ZEROED"
        elif edge >= 0.03 and abs(weights[i]) >= 0.05:
            verdict = "carries"
        elif edge >= 0.02:
            verdict = "marginal"
        else:
            verdict = "no signal"
        print(f"    {name:<16}{weights[i]:>+12.5f}{rank[i]:>10}{auc:>10.4f}{verdict:>14}")

    baseline = [float(y_train.mean())] * len(y_test)
    model_brier = brier_score(list(p_test), list(y_test))
    base_brier = brier_score(baseline, list(y_test))
    reward = data["reward"][test]
    cost = data["cost"][test]
    need = float(np.mean([breakeven_probability(r, c) for r, c in zip(reward, cost)]))
    print(f"  Brier {model_brier:.5f} vs {base_brier:.5f} base "
          f"(delta {base_brier - model_brier:+.5f}) | AUC {roc_auc_score(y_test, p_test):.4f} "
          f"| calib {calibration_error(list(p_test), list(y_test)):.4f}")
    print(f"  test base rate {y_test.mean():.4f} vs mean breakeven {need:.4f} "
          f"-> gap {need - y_test.mean():+.4f}")

    print(f"  {'strategy':<24}{'trades':>8}{'win rate':>10}{'total R':>11}{'R/trade':>10}")
    scoped = dict(data)
    scoped["y"] = y
    for row in (
        gate_report(np.ones(len(y_test)), scoped, test, "take every setup"),
        gate_report(p_test, scoped, test, "model + cost gate"),
    ):
        print(f"  {row['label']:<24}{row['n']:>8}{row['win_rate']:>10.4f}"
              f"{row['total_r']:>11.1f}{row['mean_r']:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/mt5_history.json"))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--holds", type=int, nargs="+", default=[24],
                        help="M5 bars for the time stop; several sweeps them")
    parser.add_argument("--target-atr", type=float, default=1.5)
    parser.add_argument("--target-mode", choices=("r", "atr"), default="r")
    parser.add_argument("--atr-length", type=int, default=14)
    parser.add_argument("--swing-lookback", type=int, default=60)
    parser.add_argument("--min-stop-atr", type=float, default=0.3)
    parser.add_argument("--max-stop-atr", type=float, default=5.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--encoding", choices=("v1", "v2"), default="v2",
                        help="v2 fixes the sign collapse, scaling and tails")
    parser.add_argument("--cache", type=Path, default=None,
                        help="npz path; reused if present, written if not")
    args = parser.parse_args()

    unit = "x stop (R)" if args.target_mode == "r" else "xATR"
    print("Point-in-time cascade: D1 bias -> H4 refinement -> M5 trigger")
    print(f"target {args.target_atr}{unit}, structural stop, timeout counts as a loss\n")

    if args.cache and args.cache.exists():
        blob = np.load(args.cache, allow_pickle=False)
        data = {k: blob[k] for k in blob.files if not k.startswith("y_")}
        data["y_by_hold"] = {
            int(k[2:]): blob[k] for k in blob.files if k.startswith("y_")
        }
        missing = [h for h in args.holds if h not in data["y_by_hold"]]
        if missing:
            raise SystemExit(f"cache lacks holds {missing}; delete {args.cache} and rebuild")
        print(f"loaded cached dataset from {args.cache}")
    else:
        builder = build_dataset_v2 if args.encoding == 'v2' else build_dataset
        data = builder(json.loads(args.history.read_text()), args)
        if args.cache:
            flat = {k: v for k, v in data.items() if k != "y_by_hold"}
            flat.update({f"y_{h}": v for h, v in data["y_by_hold"].items()})
            np.savez_compressed(args.cache, **flat)
            print(f"cached dataset to {args.cache}")
    total = len(data["X"])
    if total < 1000:
        print(f"\nOnly {total} samples; not enough to train.")
        return
    print(f"\n{total} samples | longs {int((data['side'] == 1).sum())} "
          f"shorts {int((data['side'] == -1).sum())}")
    print(f"reward ratio: median {np.median(data['reward']):.2f} "
          f"| cost in R: median {np.median(data['cost']):.4f}")

    for hold in args.holds:
        train_and_report(data, data["y_by_hold"][hold], hold, args)


if __name__ == "__main__":
    main()
