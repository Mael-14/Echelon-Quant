"""Triple-barrier resolution, and the null it has to beat.

Kept after ADR-004 deleted the models that used it, because the null test
is the thing that should run before anyone builds another one.
"""
from __future__ import annotations

from typing import Sequence


def resolve(
    candles: Sequence, index: int, *, is_long: bool, stop_distance: float,
    reward_r: float, max_hold: int,
) -> bool | None:
    """Did the target come before the stop? ``None`` if neither, in ``max_hold``.

    Walks the high/low path. Where one bar spans both barriers the stop is
    assumed first: intrabar order is unknowable from OHLC, and only the
    pessimistic reading cannot flatter a model into recommending trades it
    would have lost.
    """
    entry = candles[index].close
    if is_long:
        stop, target = entry - stop_distance, entry + stop_distance * reward_r
    else:
        stop, target = entry + stop_distance, entry - stop_distance * reward_r

    for step in range(index + 1, min(index + 1 + max_hold, len(candles))):
        bar = candles[step]
        if is_long:
            if bar.low <= stop:
                return False
            if bar.high >= target:
                return True
        else:
            if bar.high >= stop:
                return False
            if bar.low <= target:
                return True
    return None


def martingale_probability(reward_r: float) -> float:
    """P(touch +b before -1) for a driftless walk: exactly ``1/(1+b)``.

    The number any directional model must beat. Measured against M15 majors
    it held to within 0.01 at every reward ratio tested, which is why
    ADR-004 deletes the models rather than tuning them.
    """
    if reward_r <= 0:
        raise ValueError("reward must be positive")
    return 1.0 / (1.0 + reward_r)
