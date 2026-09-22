"""Encode market state as numeric columns.

Five readings, one row. Each is normalised so the same number means the same
thing on a 1.10-handle currency pair and a 46,000-handle synthetic index, and
in a calm year as in a violent one: four divide by ATR, and the fifth is a
ratio by construction. That is the point of the encoding -- a raw "price is 40
above its baseline" says nothing until you know whether 40 is a day's range or
a month's.

Every column is **dimensionless**: multiply every price by a constant, or add
one to every price, and the row does not move. ``tests/test_features.py``
asserts both invariances directly, because that property is the only reason
one threshold can be shared across instruments.

The columns, in order:

* ``macro_trend_score``      (Close_H4 - EMA200_H4) / ATR14_H4
* ``distance_to_support``    (Close_M5 - nearest H4 level at or below) / ATR14_M5
* ``distance_to_resistance`` (nearest H4 level at or above - Close_M5) / ATR14_M5
* ``retracement_factor``     (Close_M5 - SwingLow_M5) / (SwingHigh_M5 - SwingLow_M5)
* ``micro_wave_momentum``    (Close_M5 - VWAP_M5) / ATR14_M5

Both proximity columns are written so **approaching zero means price is
sitting on the level**, which is why resistance is measured downward rather
than reusing the support subtraction. The sign then carries the break: a
negative reading means price has left the scanned structure on that side.

This module computes features and decides nothing. Nothing here predicts a
direction -- see ``docs/ADR-004-no-directional-model.md`` for why the
directional models were removed. These columns are measurements of where price
is, not forecasts of where it goes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone

from forex_agent.models import Bar
from forex_agent.strategy.indicators import ema, swing_points, true_range_atr

#: Column order. :meth:`MarketFeatures.as_row` is defined in terms of this
#: tuple rather than field order, so a row can never silently transpose two
#: columns when a field is added.
FEATURE_COLUMNS = (
    "macro_trend_score",
    "distance_to_support",
    "distance_to_resistance",
    "retracement_factor",
    "micro_wave_momentum",
)

#: The long-term baseline the macro tide is measured against.
MACRO_EMA_LENGTH = 200

#: Wilder's default, used for every ATR normaliser here.
ATR_LENGTH = 14

#: Bars either side of a candidate that must be lower (or higher) before it
#: counts as a pivot. Two is the usual fractal definition.
PIVOT_STRENGTH = 2

#: How far back the structural scan looks for key levels. 180 H4 bars is about
#: thirty trading days -- the "last N days" the level scan is specified over.
H4_PIVOT_LOOKBACK = 180

#: The active leg on the execution frame. The retracement factor is only
#: meaningful against the swing price is actually retracing.
M5_SWING_LOOKBACK = 60


@dataclass(frozen=True, slots=True)
class MarketFeatures:
    """One encoded row.

    ``retracement_factor`` is the only column with a natural range, 0.0 at the
    swing low and 1.0 at the swing high, with 0.50 the equilibrium
    retracement. It is deliberately **not** clamped: a value above 1.0 means
    price has taken out the swing high, and flattening that to 1.0 would erase
    the difference between a pullback that held and a breakout that did not.
    """

    macro_trend_score: float
    distance_to_support: float
    distance_to_resistance: float
    retracement_factor: float
    micro_wave_momentum: float

    def as_row(self) -> tuple[float, ...]:
        """The values in :data:`FEATURE_COLUMNS` order."""
        return tuple(float(getattr(self, name)) for name in FEATURE_COLUMNS)


def pivot_levels(
    bars: list[Bar], *, strength: int = PIVOT_STRENGTH, lookback: int = H4_PIVOT_LOOKBACK
) -> list[float]:
    """Every confirmed swing high and low in the window, as one sorted list.

    ``indicators.swing_points`` returns only the most recent of each, which
    answers "where is the active leg". Finding the *nearest* level needs all of
    them.

    Highs and lows are returned together because a level does not remember
    which it was: a swing high that price has closed above is where the next
    pullback tends to find support. Splitting them would count a broken
    resistance as if it no longer existed.
    """
    window = bars[-lookback:] if lookback > 0 else list(bars)
    if strength <= 0 or len(window) < strength * 2 + 1:
        return []
    levels: set[float] = set()
    # A pivot needs `strength` bars on both sides, so the last `strength` bars
    # can never be pivots. That lag is the price of confirmation: an
    # unconfirmed extreme is just the current price.
    for index in range(strength, len(window) - strength):
        candidate = window[index]
        neighbours = (*window[index - strength : index], *window[index + 1 : index + 1 + strength])
        if all(candidate.high >= other.high for other in neighbours):
            levels.add(candidate.high)
        if all(candidate.low <= other.low for other in neighbours):
            levels.add(candidate.low)
    return sorted(levels)


def vwap(bars: list[Bar]) -> float | None:
    """Volume-weighted average typical price, anchored to the UTC session.

    Live bars carry a tick count, and ``download_mt5_training_data.py`` now
    records ``tick_volume``, so replayed MT5 history is weighted too. **Deriv
    candle history has no volume at all** -- the API does not return one, and
    only the live tick stream can supply it -- so an unweighted mean of typical
    prices is used whenever a session carries no volume. That is a session
    average rather than a true VWAP; it keeps the sign and the rough scale of
    the reading, and it is why history downloaded before the ``tick_volume``
    change reads slightly differently from history downloaded after it.
    """
    if not bars:
        return None
    session = _session_bars(bars)
    if not session:
        return None
    typical = [(bar.high + bar.low + bar.close) / 3.0 for bar in session]
    weights = [float(bar.volume) for bar in session]
    total = sum(weights)
    if total <= 0:
        return sum(typical) / len(typical)
    return sum(price * weight for price, weight in zip(typical, weights)) / total


def _session_bars(bars: list[Bar]) -> list[Bar]:
    """The trailing run of bars sharing the last bar's UTC date."""
    last_day = bars[-1].start.astimezone(timezone.utc).date()
    collected: list[Bar] = []
    for bar in reversed(bars):
        if bar.start.astimezone(timezone.utc).date() != last_day:
            break
        collected.append(bar)
    return list(reversed(collected))


def encode_features(
    h4: list[Bar],
    m5: list[Bar],
    *,
    macro_ema_length: int = MACRO_EMA_LENGTH,
    atr_length: int = ATR_LENGTH,
    pivot_strength: int = PIVOT_STRENGTH,
    h4_pivot_lookback: int = H4_PIVOT_LOOKBACK,
    m5_swing_lookback: int = M5_SWING_LOOKBACK,
) -> MarketFeatures | None:
    """Encode the newest bar's market state, or ``None`` if it cannot be read.

    Both lists must end at the same moment -- the H4 bar containing the newest
    M5 bar -- which is how the agent holds them.

    Returns ``None`` rather than a partial or defaulted row whenever the inputs
    cannot support the calculation: too little history for the baseline, a zero
    ATR, or no confirmed swing to measure against. A defaulted column would be
    indistinguishable from a real reading of zero, and zero is a meaningful
    value in every one of these columns.
    """
    if not h4 or not m5:
        return None
    # Catch a swapped call. The math works on any pair of frames, so this does
    # not pin the arguments to H4 and M5 -- it only insists the macro frame is
    # the coarser one, which a transposed call never satisfies.
    if h4[-1].timeframe.seconds <= m5[-1].timeframe.seconds:
        return None
    # `_ema` seeds from the first value, so it returns a number from any length
    # of input -- a three-bar "EMA200" is just the third bar. Require the full
    # window rather than accept one.
    if len(h4) < macro_ema_length or len(h4) < atr_length + 1 or len(m5) < atr_length + 1:
        return None

    atr_h4 = true_range_atr(h4, atr_length)
    atr_m5 = true_range_atr(m5, atr_length)
    if atr_h4 <= 0.0 or atr_m5 <= 0.0:
        return None

    close_h4 = h4[-1].close
    close_m5 = m5[-1].close

    # 1. The macro tide: how far H4 price sits from its long-term baseline.
    baseline = ema([bar.close for bar in h4], macro_ema_length)
    macro_trend_score = (close_h4 - baseline) / atr_h4

    # 2. Structural proximity: the nearest H4 key level on each side.
    levels = pivot_levels(h4, strength=pivot_strength, lookback=h4_pivot_lookback)
    if not levels:
        return None
    below = [level for level in levels if level <= close_m5]
    above = [level for level in levels if level >= close_m5]
    # With no level on a side, price has left the scanned structure entirely.
    # Fall back to the outermost level so the distance stays defined and goes
    # negative, which is the reading: the level was broken, not absent.
    support = max(below) if below else levels[0]
    resistance = min(above) if above else levels[-1]
    distance_to_support = (close_m5 - support) / atr_m5
    distance_to_resistance = (resistance - close_m5) / atr_m5

    # 3. Where price sits inside the active M5 swing.
    swing = swing_points(m5, strength=pivot_strength, lookback=m5_swing_lookback)
    swing_high, swing_low = swing["high"], swing["low"]
    if swing_high is None or swing_low is None:
        return None
    span = swing_high - swing_low
    if span <= 0.0:
        return None
    retracement_factor = (close_m5 - swing_low) / span

    # 4. The immediate wave: distance from the M5 VWAP.
    session_vwap = vwap(m5)
    if session_vwap is None:
        return None
    micro_wave_momentum = (close_m5 - session_vwap) / atr_m5

    return MarketFeatures(
        macro_trend_score=macro_trend_score,
        distance_to_support=distance_to_support,
        distance_to_resistance=distance_to_resistance,
        retracement_factor=retracement_factor,
        micro_wave_momentum=micro_wave_momentum,
    )
