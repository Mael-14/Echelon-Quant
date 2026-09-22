"""Point-in-time multi-timeframe scalp model: macro analysis, micro trigger.

The cascade is a pair of flags. ``--macro-frames`` are the analysis frames and
``--exec-frame`` is the one the trade is triggered and resolved on; the default
is H4 analysis into an M1 trigger. Every macro frame must be strictly coarser
than the execution frame, which the encoder requires and ``main`` checks.

The cascade is enforced by construction rather than by convention. At each
execution bar the pipeline takes the most recent macro bars **that have already
closed**, so a 4-hour feature can only change once every four hours -- which is
exactly what "check H4 every four hours" means once it is written down. The bar
still forming is never visible: its high and low are tomorrow's information, and
reading them is the look-ahead that makes a backtest lie.

Read the cost line the run prints before reading its scores. Cost in R is
``spread / stop_distance`` and ATR grows as the square root of time, so drag
falls as ``1/sqrt(time)`` -- ADR-004 measures 0.095R at M5 against 0.017R at H4.
Dropping the execution frame raises that hurdle rather than lowering it, and the
M1 entry in ``SPREAD_DRAG_R`` is extrapolated from M5 rather than measured.

Labels follow the triple-barrier method with a *structural* stop:

* profit barrier at ``+1.5 x ATR`` on the execution frame, in the direction of
  the macro trend
* stop barrier at the invalidation level -- the local swing low for a long, the
  swing high for a short, not a fixed ATR multiple
* time barrier at ``--holds`` execution-frame bars; scalping needs velocity, so
  an unresolved trade is closed and **labelled a loss**, not discarded

Because the stop is structural, the reward ratio is a property of each sample
rather than a constant, and so is the cost in R (``spread / stop_distance``).
Both are carried per row and fed to the gate.

Features are expressed **relative to the trade's direction**, so one model
covers both sides. A short's trend score, momentum and retracement are
mirrored and its support/resistance distances swapped; without this a linear
model would need opposite coefficients for longs and shorts and could fit
neither.

The classifier is L1-penalised logistic regression. Lasso is the point: a
feature that contributes nothing to the outcome has its weight driven to
exactly zero, so the report below says which timeframes earned their place
instead of assuming they all did.

Usage:

    python scripts/train_scalp_model.py --history data/deriv_m1.json --macro-frames 4h --exec-frame 1m
"""
from __future__ import annotations

import argparse
import bisect
import json
from datetime import datetime, timedelta, timezone
from math import exp, sqrt
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from forex_agent import artifact
from forex_agent.models import Bar, Timeframe
from forex_agent.strategy.barriers import resolve
from forex_agent.strategy.expectancy import (
    breakeven_probability,
    brier_score,
    calibration_error,
)
from forex_agent.strategy.features import encode_features
from forex_agent.strategy.indicators import swing_points, true_range_atr

#: Defaults for --macro-window and --exec-window. Both were fixed module
#: constants pinned to a D1/H4 -> M5 cascade; the cascade is a flag now.
MACRO_WINDOW = 600
EXEC_WINDOW = 400


def default_exec_window(frame: Timeframe) -> int:
    """Execution bars to keep: at least one whole UTC session.

    ``momentum`` is the close's distance from the session VWAP, and
    ``features.vwap`` anchors to the newest UTC day *present in the window*. A
    window shorter than a session does not fail -- it silently re-anchors to a
    partial day, so the column means something different in the afternoon than
    it does at midnight.

    A UTC day is 288 M5 bars, so the historical 400 already covered a session at
    M5 and every coarser frame, and those defaults are unchanged. It is 1440 M1
    bars, which 400 does not cover -- at M1 this is the difference between a
    session reading and a six-hour one.
    """
    return max(EXEC_WINDOW, 86_400 // frame.seconds)

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


def feature_names_v1(macro_frames: tuple[Timeframe, ...]) -> tuple[str, ...]:
    """:data:`FEATURES` generalised. Grouped by frame, as v1 already was."""
    return tuple(
        [
            name
            for frame in macro_frames
            for name in (
                f"{frame.name.lower()}_trend",
                f"{frame.name.lower()}_to_stop",
                f"{frame.name.lower()}_to_target",
            )
        ]
        + ["pullback", "momentum"]
    )


#: Spread drag in R against a 1.5xATR stop, per execution frame.
#:
#: Cost in R is spread/stop_distance; the spread is fixed in price while ATR
#: grows with the square root of time, so drag falls as 1/sqrt(time). ADR-004
#: calls this the single largest lever on expectancy the toolkit found -- at
#: M5 a strategy must clear an 18-point handicap over the martingale rate, at
#: H4 it must clear 0.7.
#:
#: M5, M15, H4 and D1 are measured. **M1 is not.** It is extrapolated from M5
#: by that same law (0.095 * sqrt(300/60)), and the law only reproduces the
#: measured cells to within about 20% -- it predicts 0.0137R at H4 where the
#: measurement says 0.017R. Re-measure against a real Deriv spread before
#: trusting an M1 number, and expect it to be the dominant term: at M1 the
#: drag is roughly 2.2x M5, the frame the ADR already rejected.
SPREAD_DRAG_R = {
    Timeframe.M1: 0.095 * sqrt(300 / 60),
    Timeframe.M5: 0.095,
    Timeframe.M15: 0.045,
    Timeframe.H4: 0.017,
    Timeframe.D1: 0.007,
}

#: The stop the drag table was measured against. The --spread-atr flag is
#: carried in ATR units rather than R, matching backtest_scalp.py, because
#: cost is recomputed per trade from the stop actually taken.
STOP_ATR_MULTIPLE = 1.5


def default_spread_atr(frame: Timeframe) -> float:
    """Spread in ATR units for an execution frame, or 0 if it has no entry."""
    drag = SPREAD_DRAG_R.get(frame)
    return drag * STOP_ATR_MULTIPLE if drag is not None else 0.0


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


def feature_names_v2(macro_frames: tuple[Timeframe, ...]) -> tuple[str, ...]:
    """:data:`FEATURES_V2` generalised to any number of macro frames.

    Columns are grouped by kind rather than by frame -- every trend, then
    every proximity, then every signed distance, then the two micro columns --
    which is the order the fixed twelve-column tuple already used. Given
    ``(D1, H4)`` this reproduces ``FEATURES_V2`` exactly, and a test pins that
    so the generalisation cannot silently reorder a trained model's inputs.
    """
    tags = [frame.name.lower() for frame in macro_frames]
    return tuple(
        [f"{tag}_trend" for tag in tags]
        + [f"{tag}_{kind}_prox" for tag in tags for kind in ("stop", "tgt")]
        + [f"{tag}_{kind}_dist" for tag in tags for kind in ("stop", "tgt")]
        + ["pullback", "momentum"]
    )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def encode_row_v2(macro, is_long: bool) -> list[float]:
    """One row per trade direction, in :func:`feature_names_v2` order.

    ``macro`` holds one ``MarketFeatures`` per macro frame, coarsest first.
    The coarsest is the top of the cascade and supplies the two micro columns;
    the rest contribute their trend, proximity and distance readings.
    """
    side = 1.0 if is_long else -1.0
    prox = lambda d: exp(-abs(_clip(d, -40.0, 40.0)))  # noqa: E731
    stops = [f.distance_to_support if is_long else f.distance_to_resistance for f in macro]
    targets = [f.distance_to_resistance if is_long else f.distance_to_support for f in macro]
    top = macro[0]
    pull = top.retracement_factor if is_long else 1.0 - top.retracement_factor

    row = [_clip(f.macro_trend_score * side, -15.0, 15.0) for f in macro]
    for stop, target in zip(stops, targets):
        row += [prox(stop), prox(target)]
    for stop, target in zip(stops, targets):
        row += [_clip(stop, -20.0, 20.0), _clip(target, -20.0, 20.0)]
    row += [
        _clip(pull, -1.0, 2.0),
        _clip(top.micro_wave_momentum * side, -10.0, 10.0),
    ]
    return row


def load_cascade(symbol: str, frames: dict, args):
    """Macro bar lists (coarsest first) plus the execution bars, or ``None``.

    ``None`` means this symbol cannot support the requested cascade. The
    message names the frame and the shortfall rather than saying "insufficient
    history", because at M1 that is the failure everyone hits first: a
    download long enough to fill the execution window is usually still far too
    short to fill the macro windows behind it.
    """
    macro = []
    for frame in args.macro_frames:
        bars = load_bars(symbol, frames.get(frame.value) or [], frame)
        if len(bars) < args.macro_window:
            print(f"  {symbol}: skipped ({frame.value} has {len(bars)} bars, needs {args.macro_window})")
            return None
        macro.append(bars)
    micro = load_bars(symbol, frames.get(args.exec_frame.value) or [], args.exec_frame)
    if len(micro) < args.exec_window:
        print(f"  {symbol}: skipped ({args.exec_frame.value} has {len(micro)} bars, needs {args.exec_window})")
        return None
    return macro, micro


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
        loaded = load_cascade(symbol, frames, args)
        if loaded is None:
            continue
        macro_bars, exec_bars = loaded
        macro_ends = [[bar.end for bar in bars] for bars in macro_bars]

        kept = 0
        for index in range(args.exec_window, len(exec_bars) - max(args.holds) - 1, args.stride):
            now = exec_bars[index].end
            # Point-in-time: only bars already closed at this instant.
            cuts = [bisect.bisect_right(ends, now) for ends in macro_ends]
            if any(cut < args.macro_window for cut in cuts):
                continue
            micro = exec_bars[index + 1 - args.exec_window : index + 1]
            macro = [
                encode_features(bars[cut - args.macro_window : cut], micro)
                for bars, cut in zip(macro_bars, cuts)
            ]
            if any(f is None for f in macro):
                rejected["no_features"] += 1
                continue
            entry = exec_bars[index].close
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
                        exec_bars, index, is_long=is_long, stop_distance=stop_distance,
                        reward_r=reward_r, max_hold=hold,
                    )
                    labels[hold].append(1 if outcome is True else 0)
                rows.append(encode_row_v2(macro, is_long))
                rewards.append(reward_r)
                costs.append(args.spread_atr * atr / stop_distance)
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
        "names": np.asarray(feature_names_v2(args.macro_frames)),
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
        loaded = load_cascade(symbol, frames, args)
        if loaded is None:
            continue
        macro_bars, exec_bars = loaded
        macro_ends = [[bar.end for bar in bars] for bars in macro_bars]

        kept = 0
        for index in range(args.exec_window, len(exec_bars) - max(args.holds) - 1, args.stride):
            now = exec_bars[index].end
            # Point-in-time: only bars already closed at this instant.
            cuts = [bisect.bisect_right(ends, now) for ends in macro_ends]
            if any(cut < args.macro_window for cut in cuts):
                continue
            micro = exec_bars[index + 1 - args.exec_window : index + 1]

            macro = [
                encode_features(bars[cut - args.macro_window : cut], micro)
                for bars, cut in zip(macro_bars, cuts)
            ]
            if any(f is None for f in macro):
                rejected["no_features"] += 1
                continue

            # Direction is the top of the cascade: the coarsest frame's tide.
            top = macro[0]
            if top.macro_trend_score > 0:
                side, is_long = 1, True
            elif top.macro_trend_score < 0:
                side, is_long = -1, False
            else:
                rejected["flat"] += 1
                continue

            entry = exec_bars[index].close
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
                    exec_bars, index, is_long=is_long, stop_distance=stop_distance,
                    reward_r=reward_r, max_hold=hold,
                )
                # Velocity is part of the thesis: unresolved is a loss.
                labels[hold].append(1 if outcome is True else 0)

            row: list[float] = []
            for f in macro:
                row += [
                    f.macro_trend_score * side,
                    f.distance_to_support if is_long else f.distance_to_resistance,
                    f.distance_to_resistance if is_long else f.distance_to_support,
                ]
            row += [
                top.retracement_factor if is_long else 1.0 - top.retracement_factor,
                top.micro_wave_momentum * side,
            ]
            rows.append(row)
            rewards.append(reward_r)
            costs.append(args.spread_atr * atr / stop_distance)
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
        "names": np.asarray(feature_names_v1(args.macro_frames)),
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


#: scikit-learn renamed the knob that selects L1 mid-way through the range this
#: toolkit supports. Through 1.7, L1 is ``penalty="l1"`` and ``l1_ratio`` is read
#: only when ``penalty="elasticnet"``. From 1.8, ``penalty`` is deprecated in
#: favour of ``l1_ratio`` and is removed in 1.10; passing both raises an
#: inconsistency warning. So the spelling has to be chosen, not guessed.
_SKLEARN = tuple(int(part) for part in sklearn.__version__.split(".")[:2] if part.isdigit())
_L1_BY_RATIO = _SKLEARN >= (1, 8)


def lasso_logistic(c: float) -> LogisticRegression:
    """L1-penalised logistic regression, spelled for the installed sklearn.

    Lasso is the point of this script -- a feature that contributes nothing has
    its weight driven to exactly zero, which is what the ZEROED verdict below
    reports. The original call passed ``l1_ratio=1.0`` and left ``penalty`` at
    its default, so on sklearn before 1.8 the "Lasso" reported here was ridge
    and no weight could ever reach exactly zero.
    """
    knob = {"l1_ratio": 1.0} if _L1_BY_RATIO else {"penalty": "l1"}
    return LogisticRegression(solver="liblinear", C=c, max_iter=2000, **knob)


def train_and_report(data: dict, y: np.ndarray, hold: int, args) -> None:
    """Fit, evaluate out of sample, and run the cost gate for one time stop."""
    total = len(y)
    print()
    print("=" * 82)
    print(f"TIME STOP {hold} bars ({hold * 5 / 60:.1f}h)   base rate {y.mean():.4f}")
    print("=" * 82)

    train, test = purged_split(data, args.train_fraction, hold * args.exec_frame.seconds)
    scaler = StandardScaler().fit(data["X"][train])
    x_train, x_test = scaler.transform(data["X"][train]), scaler.transform(data["X"][test])
    y_train, y_test = y[train], y[test]

    # A small, pre-declared C ladder, selected on the training split only.
    best, best_c = None, None
    inner = int(len(train) * 0.75)
    for c in (0.001, 0.01, 0.1, 1.0):
        trial = lasso_logistic(c)
        trial.fit(x_train[:inner], y_train[:inner])
        score = brier_score(list(trial.predict_proba(x_train[inner:])[:, 1]), list(y_train[inner:]))
        if best is None or score < best:
            best, best_c = score, c

    model = lasso_logistic(best_c)
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
    auc_test = roc_auc_score(y_test, p_test)
    calib = calibration_error(list(p_test), list(y_test))
    print(f"  Brier {model_brier:.5f} vs {base_brier:.5f} base "
          f"(delta {base_brier - model_brier:+.5f}) | AUC {auc_test:.4f} "
          f"| calib {calib:.4f}")
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

    # Persist. Without this the fitted estimator and its scaler are locals that
    # die on return, which is why the platform had no model to serve: the whole
    # output of a training run was the text above.
    if args.save_model:
        destination = Path(str(args.save_model).replace("{hold}", str(hold)))
        saved = artifact.save(
            destination,
            model=model,
            scaler=scaler,
            feature_names=[str(n) for n in data["names"]],
            exec_frame=args.exec_frame.value,
            macro_frames=[f.value for f in args.macro_frames],
            spread_atr=args.spread_atr,
            encoding=args.encoding,
            hold=hold,
            metrics={
                "brier": float(model_brier),
                "auc": float(auc_test),
                "calibration_error": float(calib),
                "base_rate": float(y_test.mean()),
                "n_train": float(len(train)),
                "n_test": float(len(test)),
            },
        )
        print(f"  saved {saved}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/deriv_history.json"))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--holds", type=int, nargs="+", default=[24],
                        help="execution-frame bars for the time stop; several sweeps them")
    parser.add_argument("--exec-frame", type=Timeframe, default=Timeframe.M1,
                        choices=list(Timeframe), metavar="{1m,5m,15m,4h,1d,1w}",
                        help="frame the trade is triggered and resolved on (default 1m)")
    parser.add_argument("--macro-frames", type=Timeframe, nargs="+",
                        default=[Timeframe.H4], choices=list(Timeframe),
                        metavar="{1m,5m,15m,4h,1d,1w}",
                        help="analysis frames; each must be coarser than "
                             "--exec-frame (default 4h)")
    parser.add_argument("--macro-window", type=int, default=MACRO_WINDOW,
                        help="bars of history required per macro frame")
    parser.add_argument("--exec-window", type=int, default=None,
                        help="bars of history required on the execution frame; "
                             "defaults to one whole UTC session at --exec-frame "
                             "(1440 bars at 1m), never below 400")
    parser.add_argument("--spread-atr", type=float, default=None,
                        help="spread in ATR units; defaults per --exec-frame from the "
                             "ADR-004 drag table (1m is extrapolated, not measured)")
    parser.add_argument("--target-atr", type=float, default=1.5)
    parser.add_argument("--target-mode", choices=("r", "atr"), default="r")
    parser.add_argument("--atr-length", type=int, default=14)
    parser.add_argument("--swing-lookback", type=int, default=60)
    parser.add_argument("--min-stop-atr", type=float, default=0.3)
    parser.add_argument("--max-stop-atr", type=float, default=5.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--encoding", choices=("v1", "v2"), default="v2",
                        help="v2 fixes the sign collapse, scaling and tails")
    parser.add_argument("--save-model", type=Path, default=None,
                        help="joblib path for the fitted model, its scaler and the "
                             "assumptions it was fitted under; {hold} in the name is "
                             "substituted when sweeping several time stops")
    parser.add_argument("--cache", type=Path, default=None,
                        help="npz path; reused if present, written if not")
    args = parser.parse_args()

    # Coarsest first: the head of the cascade sets trade direction and supplies
    # the two micro columns.
    args.macro_frames = tuple(
        sorted(dict.fromkeys(args.macro_frames), key=lambda f: -f.seconds)
    )
    # encode_features returns None for a macro frame that is not strictly coarser
    # than the micro one, so a transposed cascade would land every row in
    # "no_features" -- an empty dataset rather than an error. Catch it here.
    too_fine = [f for f in args.macro_frames if f.seconds <= args.exec_frame.seconds]
    if too_fine:
        raise SystemExit(
            f"--macro-frames must all be coarser than --exec-frame "
            f"({args.exec_frame.value}); these are not: "
            + ", ".join(f.value for f in too_fine)
        )
    if args.exec_window is None:
        args.exec_window = default_exec_window(args.exec_frame)
    if args.spread_atr is None:
        args.spread_atr = default_spread_atr(args.exec_frame)
        if args.spread_atr <= 0.0:
            raise SystemExit(
                f"no spread drag on record for {args.exec_frame.value}; "
                "pass --spread-atr explicitly"
            )

    unit = "x stop (R)" if args.target_mode == "r" else "xATR"
    cascade = " -> ".join(f.value for f in args.macro_frames)
    print(f"Point-in-time cascade: {cascade} analysis -> {args.exec_frame.value} trigger")
    drag = args.spread_atr / STOP_ATR_MULTIPLE
    caveat = " EXTRAPOLATED, not measured" if args.exec_frame is Timeframe.M1 else ""
    print(f"cost: spread {args.spread_atr:.4f} ATR = {drag:.4f}R drag at a "
          f"{STOP_ATR_MULTIPLE}xATR stop;{caveat or ' measured (ADR-004)'}")
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
