"""Candles in, a Signal out -- or a stated reason for refusing.

The pipeline for one symbol at one instant:

    candles -> encode_features(macro, micro) -> encode_row -> scaler
            -> predict_proba -> evaluate_edge -> Signal

**This service fails closed.** No artifact, or an artifact whose cascade
disagrees with configuration, means `degraded` and zero signals rather than a
best effort. That mirrors ``expectancy.evaluate_edge`` itself, which catches a
malformed input and returns a rejecting verdict instead of raising: inside a
gate chain, not knowing is a reason not to trade.

A rejection is the ordinary outcome, not an error. At M1 the cost hurdle is
roughly 0.212R, so most bars have no tradeable edge and saying so is the
service working correctly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from backend.shared.candles import CandleWindow
from backend.shared.schemas.trading import Prediction, Signal

log = logging.getLogger("analysis-service")


class EngineDegraded(RuntimeError):
    """The engine has no usable model and must not emit signals."""


@dataclass(frozen=True)
class Setup:
    """The trade geometry read off the execution frame, before the model."""

    side: Literal["buy", "sell"]
    entry: float
    stop_distance: float
    reward_r: float
    cost_r: float
    atr: float


@dataclass(frozen=True)
class Evaluation:
    """What the engine concluded, whether or not it wants to trade."""

    symbol: str
    accepted: bool
    reason: str
    p_win: float = 0.0
    setup: Setup | None = None
    expected_value_r: float = 0.0
    breakeven: float = 0.0
    features: dict[str, float] | None = None


class AnalysisEngine:
    def __init__(
        self,
        *,
        artifact: Any,
        exec_frame: str,
        macro_frames: Sequence[str],
        macro_window: int = 600,
        exec_window: int = 1440,
        atr_length: int = 14,
        swing_lookback: int = 60,
        min_stop_atr: float = 0.3,
        max_stop_atr: float = 5.0,
        target_r: float = 1.5,
        min_expected_value_r: float = 0.0,
    ) -> None:
        self.artifact = artifact
        self.exec_frame = exec_frame
        self.macro_frames = tuple(macro_frames)
        self.macro_window = macro_window
        self.exec_window = exec_window
        self.atr_length = atr_length
        self.swing_lookback = swing_lookback
        self.min_stop_atr = min_stop_atr
        self.max_stop_atr = max_stop_atr
        self.target_r = target_r
        self.min_expected_value_r = min_expected_value_r

        # Refuse a cascade mismatch at construction, not at the first signal.
        artifact.check_serves(exec_frame=exec_frame, macro_frames=self.macro_frames)

    # ------------------------------------------------------------- features

    def encode(
        self, macro_windows: Sequence[CandleWindow], exec_window: CandleWindow
    ) -> list[Any] | None:
        """One MarketFeatures per macro frame, or None if state is unreadable.

        ``encode_features`` returns None rather than a defaulted row whenever
        the inputs cannot support the calculation -- too little history, a zero
        ATR, no confirmed swing. A defaulted column would be indistinguishable
        from a real reading of zero, and zero means something in every one of
        these columns.
        """
        from forex_agent.strategy.features import encode_features

        micro = exec_window.bars()
        if len(micro) < self.exec_window:
            return None

        encoded = []
        for window in macro_windows:
            bars = window.bars()
            if len(bars) < self.macro_window:
                return None
            features = encode_features(bars[-self.macro_window :], micro)
            if features is None:
                return None
            encoded.append(features)
        return encoded

    def read_setup(
        self, exec_window: CandleWindow, *, side: Literal["buy", "sell"]
    ) -> Setup | None:
        """Entry, structural stop and reward ratio for one direction.

        The stop is the local swing -- structural invalidation -- not an ATR
        multiple, so the reward ratio and the cost in R are properties of this
        particular setup rather than constants.
        """
        from forex_agent.strategy.indicators import swing_points, true_range_atr

        bars = exec_window.bars()
        if not bars:
            return None

        atr = true_range_atr(bars, self.atr_length)
        if atr <= 0:
            return None

        swing = swing_points(bars, strength=2, lookback=self.swing_lookback)
        is_long = side == "buy"
        level = swing["low"] if is_long else swing["high"]
        if level is None:
            return None

        entry = bars[-1].close
        stop_distance = (entry - level) if is_long else (level - entry)
        if not (self.min_stop_atr * atr <= stop_distance <= self.max_stop_atr * atr):
            return None

        return Setup(
            side=side,
            entry=entry,
            stop_distance=stop_distance,
            reward_r=self.target_r,
            cost_r=self.artifact.spread_atr * atr / stop_distance,
            atr=atr,
        )

    # ------------------------------------------------------------ evaluation

    def evaluate(
        self,
        symbol: str,
        macro_windows: Sequence[CandleWindow],
        exec_window: CandleWindow,
    ) -> Evaluation:
        """Score both directions and return the better one, if either clears."""
        from forex_agent.strategy.expectancy import evaluate_edge

        encoded = self.encode(macro_windows, exec_window)
        if encoded is None:
            return Evaluation(symbol, False, "insufficient history to encode features")

        best: Evaluation | None = None
        sides: tuple[Literal["buy", "sell"], ...] = ("buy", "sell")
        for side in sides:
            setup = self.read_setup(exec_window, side=side)
            if setup is None:
                continue

            row = self._row(encoded, is_long=side == "buy")
            p_win = self.artifact.predict_proba(row)
            verdict = evaluate_edge(
                p_win=p_win,
                reward_r=setup.reward_r,
                cost_r=setup.cost_r,
                min_expected_value_r=self.min_expected_value_r,
            )
            candidate = Evaluation(
                symbol=symbol,
                accepted=bool(verdict.accepted),
                reason=str(verdict.reason),
                p_win=p_win,
                setup=setup,
                expected_value_r=float(verdict.expected_value_r),
                breakeven=float(verdict.breakeven_p),
                features=dict(zip(self.artifact.feature_names, row, strict=True)),
            )
            # Prefer an accepted side; among equals, the higher expectancy.
            if (
                best is None
                or (candidate.accepted and not best.accepted)
                or (
                    candidate.accepted == best.accepted
                    and candidate.expected_value_r > best.expected_value_r
                )
            ):
                best = candidate

        if best is None:
            return Evaluation(symbol, False, "no tradeable geometry on either side")
        return best

    def _row(self, encoded: Sequence[Any], *, is_long: bool) -> list[float]:
        """Encode in the artifact's column order.

        The trainer's row builders live in ``scripts/``, which is not an
        installed package, so the ordering is rebuilt here from the same rule
        and checked against ``feature_names`` -- a mismatch raises rather than
        producing a confidently wrong probability.
        """
        from math import exp

        def clip(value: float, low: float, high: float) -> float:
            return max(low, min(high, value))

        def prox(distance: float) -> float:
            return exp(-abs(clip(distance, -40.0, 40.0)))

        sign = 1.0 if is_long else -1.0
        stops = [f.distance_to_support if is_long else f.distance_to_resistance for f in encoded]
        targets = [f.distance_to_resistance if is_long else f.distance_to_support for f in encoded]
        top = encoded[0]
        pull = top.retracement_factor if is_long else 1.0 - top.retracement_factor

        row = [clip(f.macro_trend_score * sign, -15.0, 15.0) for f in encoded]
        for stop, target in zip(stops, targets, strict=True):
            row += [prox(stop), prox(target)]
        for stop, target in zip(stops, targets, strict=True):
            row += [clip(stop, -20.0, 20.0), clip(target, -20.0, 20.0)]
        row += [
            clip(pull, -1.0, 2.0),
            clip(top.micro_wave_momentum * sign, -10.0, 10.0),
        ]

        if len(row) != len(self.artifact.feature_names):
            raise EngineDegraded(
                f"encoded {len(row)} columns but the artifact expects "
                f"{len(self.artifact.feature_names)}; the cascade does not match"
            )
        return row

    # --------------------------------------------------------------- output

    def to_signal(self, evaluation: Evaluation) -> Signal:
        """An accepted evaluation as the shared Signal schema."""
        if not evaluation.accepted or evaluation.setup is None:
            raise EngineDegraded("only an accepted evaluation becomes a signal")
        setup = evaluation.setup
        return Signal(
            signal_id=f"{evaluation.symbol}-{int(datetime.now(tz=timezone.utc).timestamp())}",
            symbol=evaluation.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            side=setup.side,
            # Strength is the margin over the hurdle, normalised into [0, 1] --
            # not the raw probability, which says nothing about whether the
            # trade pays once the spread is charged.
            strength=max(0.0, min(1.0, evaluation.p_win - evaluation.breakeven + 0.5)),
            reason=evaluation.reason,
        )

    def to_prediction(self, evaluation: Evaluation, *, horizon_minutes: int) -> Prediction:
        return Prediction(
            model_name=self.artifact.summary,
            symbol=evaluation.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            horizon_minutes=max(1, horizon_minutes),
            predicted_return=evaluation.expected_value_r,
            confidence=max(0.0, min(1.0, evaluation.p_win)),
        )
