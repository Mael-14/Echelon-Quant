"""Technical indicators required by the PDF strategy specification.

The implementation stays dependency-light for Render's free tier.  When
``pandas-ta`` is installed the production dependency can use it, while these
scalar helpers remain deterministic and testable without importing a large
numeric stack.
"""
from __future__ import annotations

from math import isfinite, log10, sqrt

from forex_agent.models import Bar, Side


def _ema(values: list[float], length: int) -> list[float]:
    if not values or length <= 0:
        return []
    alpha = 2.0 / (length + 1)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1 - alpha) * result[-1])
    return result


def _pandas_ta():
    import pandas as pd

    try:
        import pandas_ta as ta
    except ImportError:
        # ``pandas-ta`` is unavailable on some Python/Render combinations;
        # pandas-ta-classic provides the same public indicator functions.
        import pandas_ta_classic as ta
    return pd, ta


def atr(prices: list[float], length: int = 14) -> float:
    if len(prices) < 2 or length <= 0:
        return 0.0
    try:
        pd, ta = _pandas_ta()

        series = pd.Series(prices, dtype="float64")
        value = ta.atr(high=series, low=series, close=series, length=length).iloc[-1]
        if isfinite(float(value)):
            return float(value)
    except (ImportError, AttributeError, IndexError, TypeError, ValueError):
        pass
    ranges = [abs(float(current) - float(previous)) for previous, current in zip(prices, prices[1:])]
    window = ranges[-length:]
    return sum(window) / len(window) if window else 0.0


def true_range_atr(bars: list[Bar], length: int = 14) -> float:
    """Wilder's ATR over real true ranges.

    ``atr`` above takes a list of closes and passes it as high, low *and*
    close, so every true range collapses to ``abs(close - previous_close)``
    and the intrabar range vanishes. That is a serviceable volatility proxy
    for a caller that only holds closes, and it is what the existing gates
    are calibrated against -- but it understates a market that ranges inside
    the bar and settles near its open, which matters whenever ATR is a
    *denominator* rather than a threshold. A ``Bar`` carries the real high and
    low, so use them.
    """
    if length <= 0 or len(bars) < length + 1:
        return 0.0
    ranges = [
        max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        for previous, current in zip(bars, bars[1:])
    ]
    # Seed with the simple mean of the first window, then smooth -- Wilder's
    # RMA, which is what "ATR(14)" means on a chart.
    value = sum(ranges[:length]) / length
    for current in ranges[length:]:
        value = (value * (length - 1) + current) / length
    return value


def rsi(prices: list[float], length: int = 14) -> float:
    if len(prices) < 2 or length <= 0:
        return 50.0
    try:
        pd, ta = _pandas_ta()

        value = ta.rsi(pd.Series(prices, dtype="float64"), length=length).iloc[-1]
        if isfinite(float(value)):
            return float(value)
    except (ImportError, AttributeError, IndexError, TypeError, ValueError):
        pass
    changes = [float(current) - float(previous) for previous, current in zip(prices, prices[1:])][-length:]
    gains = sum(change for change in changes if change > 0)
    losses = -sum(change for change in changes if change < 0)
    if losses == 0:
        return 100.0 if gains else 50.0
    return 100.0 - (100.0 / (1.0 + gains / losses))


def macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, float]:
    if not prices or min(fast, slow, signal) <= 0:
        return {"macd": 0.0, "signal": 0.0, "hist": 0.0}
    try:
        pd, ta = _pandas_ta()

        frame = ta.macd(pd.Series(prices, dtype="float64"), fast=fast, slow=slow, signal=signal)
        if frame is not None:
            row = frame.iloc[-1]
            values = {str(key).lower(): float(value) for key, value in row.items() if isfinite(float(value))}
            macd_value = next(value for key, value in values.items() if key.startswith("macd_") and "h" not in key and "s" not in key)
            hist_value = next(value for key, value in values.items() if "macdh" in key)
            signal_value = next(value for key, value in values.items() if "macds" in key)
            return {"macd": macd_value, "signal": signal_value, "hist": hist_value}
    except (ImportError, AttributeError, IndexError, StopIteration, TypeError, ValueError):
        pass
    fast_line = _ema(prices, fast)
    slow_line = _ema(prices, slow)
    line = [fast_value - slow_value for fast_value, slow_value in zip(fast_line, slow_line)]
    signal_line = _ema(line, signal)
    value = line[-1] if line else 0.0
    trigger = signal_line[-1] if signal_line else 0.0
    return {"macd": value, "signal": trigger, "hist": value - trigger}


def macd_cross_direction(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    *,
    within: int = 1,
) -> Side | None:
    """Return a genuine MACD signal-line crossover direction.

    The PDF requires crossover direction, not merely the sign of the current
    histogram.  A crossover is therefore only valid when two consecutive
    histogram values sit on opposite sides of zero.

    ``within`` widens that test to the most recent ``within`` bars and is the
    difference between a usable filter and an unusable one.  A crossover on
    one exact bar is roughly a 1-in-25 event; insisting it coincide with a
    Fibonacci touch, a candle pattern and an RSI band makes the conjunction
    vanishingly rare.  Allowing a cross that happened a few bars ago, provided
    the histogram still favours that side, matches how the crossover is
    actually traded.
    """
    if len(prices) < max(slow + signal, 3) or within < 1:
        return None
    fast_line = _ema(prices, fast)
    slow_line = _ema(prices, slow)
    macd_line = [a - b for a, b in zip(fast_line, slow_line)]
    signal_line = _ema(macd_line, signal)
    hist = [a - b for a, b in zip(macd_line, signal_line)]
    if len(hist) < 2:
        return None
    window = min(within, len(hist) - 1)
    for offset in range(window):
        index = len(hist) - 1 - offset
        previous, current = hist[index - 1], hist[index]
        if previous <= 0 < current:
            # The cross must not have been undone since it happened.
            return Side.BUY if hist[-1] > 0 else None
        if previous >= 0 > current:
            return Side.SELL if hist[-1] < 0 else None
    return None


def choppiness_index(prices: list[float], length: int = 14) -> float:
    if len(prices) < 2 or length <= 1:
        return 50.0
    window = [float(value) for value in prices[-(length + 1):]]
    ranges = [abs(current - previous) for previous, current in zip(window, window[1:])]
    span = max(window) - min(window)
    if span <= 0 or not ranges:
        return 50.0
    raw = 100.0 * log10(sum(ranges) / span) / log10(length)
    return max(0.0, min(100.0, raw))


def fibonacci_levels(prices: list[float]) -> dict[str, float]:
    if not prices:
        return {key: 0.0 for key in ("high", "low", "fib_21_4", "fib_23_6", "fib_38_2", "fib_50_0", "fib_61_8", "fib_78_6")}
    high, low = max(prices), min(prices)
    span = high - low
    return {
        "high": high,
        "low": low,
        "fib_21_4": high - 0.214 * span,
        "fib_23_6": high - 0.236 * span,
        "fib_38_2": high - 0.382 * span,
        "fib_50_0": high - 0.500 * span,
        "fib_61_8": high - 0.618 * span,
        "fib_78_6": high - 0.786 * span,
    }


def swing_fibonacci_levels(prices: list[float], side: Side, *, lookback: int = 60) -> dict[str, float] | None:
    """Fibonacci levels for the impulse leg that precedes the current pullback.

    :func:`fibonacci_levels` spans a window that *ends at the current bar*, so
    in any trending market the latest price is itself the window extreme and
    the measured retracement is pinned near 0% or 100%.  Price then never sits
    inside the 61.8%-78.6% band the location gate looks for, which is why a
    trend-aligned candidate could never also be "in discount".

    A retracement is only meaningful against a completed leg.  For a long, the
    leg runs from the lowest close in the window up to the highest close that
    comes *after* it; a short mirrors that.  Price retracing back into the band
    is then exactly the pullback the strategy intends to buy or sell.

    Returns ``None`` when no such leg exists.
    """
    window = [float(value) for value in prices[-lookback:] if isfinite(float(value))]
    if len(window) < 5:
        return None
    if side is Side.BUY:
        anchor = min(range(len(window)), key=window.__getitem__)
        leg = window[anchor:]
        if len(leg) < 2:
            return None
        low, high = window[anchor], max(leg)
    else:
        anchor = max(range(len(window)), key=window.__getitem__)
        leg = window[anchor:]
        if len(leg) < 2:
            return None
        high, low = window[anchor], min(leg)
    if high <= low:
        return None
    span = high - low
    return {
        "high": high,
        "low": low,
        "fib_21_4": high - 0.214 * span,
        "fib_23_6": high - 0.236 * span,
        "fib_38_2": high - 0.382 * span,
        "fib_50_0": high - 0.500 * span,
        "fib_61_8": high - 0.618 * span,
        "fib_78_6": high - 0.786 * span,
    }


#: The PDF's retracement windows, as (deep, shallow) percentages of the leg.
#: A buy looks for the discount window, a sell for its mirror.  Widening the
#: shallow edge toward 50% admits ordinary pullbacks as well as deep ones and
#: is the single largest control on how often the agent trades.
PDF_DISCOUNT_BAND = (0.786, 0.618)
PDF_PREMIUM_BAND = (0.214, 0.382)


def fibonacci_location(
    price: float,
    levels: dict[str, float],
    side: Side,
    *,
    discount_band: tuple[float, float] = PDF_DISCOUNT_BAND,
    premium_band: tuple[float, float] = PDF_PREMIUM_BAND,
) -> str | None:
    """Return the PDF's discount/premium location or ``None``.

    Buys must be in the 61.8%-78.6% discount retracement.  Sells use the
    mirrored 21.4%-38.2% premium retracement.  This keeps the direction of
    the Fibonacci map explicit instead of treating the 50% midpoint as a
    trade location.

    The bands are parameters so the window can be widened deliberately and
    measured, rather than by editing a constant.
    """
    high = levels.get("high", 0.0)
    low = levels.get("low", 0.0)
    if high <= low or not (low <= price <= high):
        return None
    span = high - low
    if side is Side.BUY:
        deep, shallow = discount_band
        if high - deep * span <= price <= high - shallow * span:
            return "discount"
        return None
    shallow, deep = premium_band
    if high - deep * span <= price <= high - shallow * span:
        return "premium"
    return None


def fibonacci_location_consensus(
    price: float,
    levels_by_timeframe: dict[str, dict[str, float]],
    side: Side,
) -> str | None:
    """Require the PDF location gate to agree across structural timeframes."""
    locations = [fibonacci_location(price, levels, side) for levels in levels_by_timeframe.values()]
    if not locations or any(location is None for location in locations):
        return None
    if len(set(locations)) != 1:
        return None
    return locations[0]


def candlestick_confirmation(bars: list[Bar], side: Side, *, within: int = 1) -> str | None:
    """Recognize structural candle confirmation for the final verification.

    The patterns are deliberately small and deterministic: bullish/bearish
    engulfing and hammer/shooting-star rejection.  The function returns the
    pattern name for Telegram/audit telemetry, or ``None`` when no pattern
    confirms the requested side.

    ``within`` scans that many of the most recent bar pairs rather than only
    the last one.  The reversal candle that ends a pullback commonly prints a
    bar or two before a MACD crossover confirms the turn, so pinning both to
    the same bar makes the conjunction far rarer than either condition implies.
    """
    if len(bars) < 2 or within < 1:
        return None
    for offset in range(min(within, len(bars) - 1)):
        pattern = _candlestick_pair(bars[-2 - offset], bars[-1 - offset], side)
        if pattern is not None:
            return pattern
    return None


def _candlestick_pair(previous: Bar, current: Bar, side: Side) -> str | None:
    body = abs(current.close - current.open)
    upper_wick = current.high - max(current.open, current.close)
    lower_wick = min(current.open, current.close) - current.low
    if side is Side.BUY:
        engulfing = previous.close < previous.open and current.close > current.open and current.open <= previous.close and current.close >= previous.open
        hammer = current.close > current.open and lower_wick >= max(body * 2.0, upper_wick * 1.5)
        if engulfing:
            return "bullish_engulfing"
        if hammer:
            return "hammer"
    else:
        engulfing = previous.close > previous.open and current.close < current.open and current.open >= previous.close and current.close <= previous.open
        shooting_star = current.close < current.open and upper_wick >= max(body * 2.0, lower_wick * 1.5)
        if engulfing:
            return "bearish_engulfing"
        if shooting_star:
            return "shooting_star"
    return None


def ema(values: list[float], length: int) -> float:
    """Exponential moving average of the series, or 0.0 when unavailable."""
    series = _ema([float(v) for v in values], length)
    return series[-1] if series else 0.0


def ema_series(values: list[float], length: int) -> list[float]:
    return _ema([float(v) for v in values], length)


def ema_slope(values: list[float], length: int, *, lookback: int = 5) -> float:
    """Fractional change in the EMA over ``lookback`` bars.

    Expressed relative to the EMA's own level so the reading means the same
    thing on a 1.10-handle currency pair and a 46,000-handle index.
    """
    series = _ema([float(v) for v in values], length)
    if len(series) <= lookback:
        return 0.0
    previous, current = series[-1 - lookback], series[-1]
    if previous <= 0:
        return 0.0
    return (current - previous) / previous


def bollinger_bands(values: list[float], length: int = 20, deviations: float = 2.0) -> dict[str, float]:
    """Mean and volatility envelope over the trailing window."""
    window = [float(v) for v in values[-length:]]
    if len(window) < 2:
        price = window[-1] if window else 0.0
        return {"middle": price, "upper": price, "lower": price, "width": 0.0}
    middle = sum(window) / len(window)
    variance = sum((value - middle) ** 2 for value in window) / len(window)
    spread = deviations * sqrt(variance)
    return {
        "middle": middle,
        "upper": middle + spread,
        "lower": middle - spread,
        # Width relative to the mean keeps squeeze detection scale-free.
        "width": (2 * spread / middle) if middle > 0 else 0.0,
    }


def donchian_channel(bars: list[Bar], length: int = 20) -> dict[str, float]:
    """Highest high and lowest low of the trailing window."""
    window = bars[-length:]
    if not window:
        return {"high": 0.0, "low": 0.0, "mid": 0.0}
    high = max(bar.high for bar in window)
    low = min(bar.low for bar in window)
    return {"high": high, "low": low, "mid": (high + low) / 2.0}


def swing_points(bars: list[Bar], *, strength: int = 2, lookback: int = 60) -> dict[str, float | None]:
    """Most recent confirmed swing high and low.

    A pivot is confirmed only when ``strength`` bars on *both* sides are lower
    (or higher), so the most recent bars cannot themselves be pivots.  That
    delay is deliberate: an unconfirmed extreme is just the current price.
    """
    window = bars[-lookback:]
    if len(window) < strength * 2 + 1:
        return {"high": None, "low": None}
    swing_high: float | None = None
    swing_low: float | None = None
    for index in range(len(window) - strength - 1, strength - 1, -1):
        left = window[index - strength : index]
        right = window[index + 1 : index + 1 + strength]
        if not left or not right:
            continue
        candidate = window[index]
        if swing_high is None and all(
            candidate.high >= other.high for other in (*left, *right)
        ):
            swing_high = candidate.high
        if swing_low is None and all(
            candidate.low <= other.low for other in (*left, *right)
        ):
            swing_low = candidate.low
        if swing_high is not None and swing_low is not None:
            break
    return {"high": swing_high, "low": swing_low}


def trend_slope(values: list[float], *, lookback: int = 20) -> float:
    """Least-squares slope over the window, normalised by price level.

    This is the algebraic equivalent of drawing a trendline through recent
    closes: positive means the fitted line rises.
    """
    window = [float(v) for v in values[-lookback:]]
    count = len(window)
    if count < 3:
        return 0.0
    mean_x = (count - 1) / 2.0
    mean_y = sum(window) / count
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(window))
    denominator = sum((index - mean_x) ** 2 for index in range(count))
    if denominator <= 0 or mean_y <= 0:
        return 0.0
    return (numerator / denominator) / mean_y


def adx(bars: list[Bar], length: int = 14) -> float:
    """Average Directional Index: how strongly the market is trending (0-100)."""
    if len(bars) < length + 1:
        return 0.0
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    true_range: list[float] = []
    for previous, current in zip(bars, bars[1:]):
        up = current.high - previous.high
        down = previous.low - current.low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        true_range.append(max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        ))
    window_tr = sum(true_range[-length:])
    if window_tr <= 0:
        return 0.0
    plus_di = 100.0 * sum(plus_dm[-length:]) / window_tr
    minus_di = 100.0 * sum(minus_dm[-length:]) / window_tr
    total = plus_di + minus_di
    if total <= 0:
        return 0.0
    return 100.0 * abs(plus_di - minus_di) / total
