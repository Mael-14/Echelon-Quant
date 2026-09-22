"""The encoder must mean the same thing on any instrument, or not answer."""
import math
from datetime import datetime, timedelta, timezone

from forex_agent.models import Bar, Timeframe
from forex_agent.strategy.features import (
    FEATURE_COLUMNS,
    MarketFeatures,
    encode_features,
    pivot_levels,
    vwap,
)
from forex_agent.strategy.indicators import atr as close_only_atr
from forex_agent.strategy.indicators import swing_points, true_range_atr

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _bar(
    close: float,
    *,
    index: int = 0,
    timeframe: Timeframe = Timeframe.M5,
    spread: float = 0.0,
    volume: int = 0,
) -> Bar:
    start = EPOCH + timedelta(seconds=timeframe.seconds * index)
    return Bar(
        "TEST", timeframe, start, start + timedelta(seconds=timeframe.seconds),
        close, close + spread, close - spread, close, volume,
    )


def _series(
    closes: list[float], *, timeframe: Timeframe = Timeframe.M5, spread: float = 0.0
) -> list[Bar]:
    return [_bar(c, index=i, timeframe=timeframe, spread=spread) for i, c in enumerate(closes)]


def _wave(
    n: int,
    *,
    base: float,
    unit: float,
    timeframe: Timeframe,
    drift: float = 0.5,
    swing: float = 4.0,
    period: int = 8,
) -> list[Bar]:
    """A drifting oscillation, described dimensionlessly and then scaled.

    The oscillation matters: a monotonic ramp confirms no pivot anywhere in it,
    and the encoder correctly refuses to read structure that is not there.
    """
    closes = [
        base + unit * (drift * i + swing * math.sin(i * 2 * math.pi / period)) for i in range(n)
    ]
    return _series(closes, timeframe=timeframe, spread=unit)


def _inputs(*, base: float, unit: float, drift: float = 0.5) -> tuple[list[Bar], list[Bar]]:
    """An H4 history and an M5 history sitting inside its range."""
    h4 = _wave(260, base=base, unit=unit, timeframe=Timeframe.H4, drift=drift, swing=4.0)
    m5 = _wave(
        80, base=base + unit * 65.0, unit=unit * 0.2, timeframe=Timeframe.M5, drift=0.3, swing=3.0
    )
    return h4, m5


def _remap(bars: list[Bar], fn) -> list[Bar]:
    return [
        Bar(
            b.symbol, b.timeframe, b.start, b.end,
            fn(b.open), fn(b.high), fn(b.low), fn(b.close), b.volume,
        )
        for b in bars
    ]


def _with_volumes(bars: list[Bar], volumes: list[int]) -> list[Bar]:
    return [
        Bar(b.symbol, b.timeframe, b.start, b.end, b.open, b.high, b.low, b.close, v)
        for b, v in zip(bars, volumes)
    ]


# --- the row -----------------------------------------------------------------


def test_the_row_is_the_five_specified_columns_in_order() -> None:
    assert FEATURE_COLUMNS == (
        "macro_trend_score",
        "distance_to_support",
        "distance_to_resistance",
        "retracement_factor",
        "micro_wave_momentum",
    )
    assert MarketFeatures(1.0, 2.0, 3.0, 4.0, 5.0).as_row() == (1.0, 2.0, 3.0, 4.0, 5.0)


def test_as_row_follows_the_declared_column_order() -> None:
    """The row is the contract; field order must not be able to reorder it."""
    features = MarketFeatures(1.0, 2.0, 3.0, 4.0, 5.0)
    assert features.as_row()[FEATURE_COLUMNS.index("retracement_factor")] == 4.0
    assert len(FEATURE_COLUMNS) == len(features.as_row())


# --- every column is dimensionless -------------------------------------------


def test_every_column_is_invariant_under_translation() -> None:
    """Add a constant to every price and the row must not move."""
    h4, m5 = _inputs(base=1.10, unit=0.001)
    plain = encode_features(h4, m5)
    shifted = encode_features(_remap(h4, lambda p: p + 10.0), _remap(m5, lambda p: p + 10.0))
    assert plain is not None and shifted is not None
    for column in FEATURE_COLUMNS:
        left, right = getattr(plain, column), getattr(shifted, column)
        assert abs(left - right) < 1e-6, f"{column}: {left} vs {right}"


def test_every_column_is_invariant_under_scaling() -> None:
    """Multiply every price by a constant and the row must not move.

    This is what makes ``micro_wave_momentum`` comparable to the rest: raw
    distance from the VWAP is a price, and a price is not a feature until it
    is divided by something with the same units.
    """
    h4, m5 = _inputs(base=1.10, unit=0.001)
    plain = encode_features(h4, m5)
    scaled = encode_features(_remap(h4, lambda p: p * 1000.0), _remap(m5, lambda p: p * 1000.0))
    assert plain is not None and scaled is not None
    for column in FEATURE_COLUMNS:
        left, right = getattr(plain, column), getattr(scaled, column)
        assert abs(left - right) < 1e-6, f"{column}: {left} vs {right}"


def test_the_encoding_is_scale_invariant_across_instruments() -> None:
    """A 1.10-handle pair and a 46,000-handle index at the same relative
    volatility must produce the same numbers."""
    pair = encode_features(*_inputs(base=1.10, unit=0.001))
    index = encode_features(*_inputs(base=46_000.0, unit=41.8))
    assert pair is not None and index is not None
    for column in FEATURE_COLUMNS:
        left, right = getattr(pair, column), getattr(index, column)
        assert abs(left - right) < 1e-6, f"{column}: {left} vs {right}"


# --- ATR ---------------------------------------------------------------------


def test_true_range_atr_sees_the_intrabar_range_that_close_only_atr_misses() -> None:
    """Flat closes with wide bars have real volatility; close-only ATR reads zero."""
    bars = [_bar(1.10, index=i, spread=0.005) for i in range(30)]
    assert close_only_atr([bar.close for bar in bars]) == 0.0
    assert true_range_atr(bars) > 0.009


def test_true_range_atr_needs_a_full_window() -> None:
    assert true_range_atr(_series([1.0, 1.1, 1.2]), 14) == 0.0


# --- the level scan ----------------------------------------------------------


def test_pivot_levels_returns_every_confirmed_level_not_just_the_latest() -> None:
    closes = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.5, 2.0, 1.0, 2.0, 3.0]
    levels = pivot_levels(_series(closes), strength=2, lookback=60)
    assert len(levels) >= 3
    assert levels == sorted(levels)
    assert max(levels) == 3.5
    # swing_points sees only the most recent of each; that is the difference
    # this helper exists for.
    latest = swing_points(_series(closes), strength=2, lookback=60)
    assert len([v for v in latest.values() if v is not None]) < len(levels)


def test_pivot_levels_scans_both_swing_highs_and_swing_lows() -> None:
    """A broken high becomes support; splitting the lists would lose it."""
    closes = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.5, 2.0, 1.0]
    levels = pivot_levels(_series(closes), strength=2, lookback=60)
    assert 3.5 in levels and 1.0 in levels


def test_pivot_levels_is_empty_when_the_window_cannot_confirm_one() -> None:
    assert pivot_levels(_series([1.0, 2.0, 3.0]), strength=2) == []


# --- VWAP --------------------------------------------------------------------


def test_vwap_weights_by_volume_when_volume_is_present() -> None:
    bars = [_bar(1.0, index=0, volume=1), _bar(2.0, index=1, volume=99)]
    weighted = vwap(bars)
    assert weighted is not None and abs(weighted - 2.0) < 0.02


def test_vwap_falls_back_to_an_unweighted_mean_when_history_carries_no_volume() -> None:
    """Deriv candle history has no volume at all; the column must survive it."""
    assert vwap([_bar(1.0, index=0), _bar(3.0, index=1)]) == 2.0


def test_vwap_anchors_to_the_session() -> None:
    """Yesterday's bars must not drag the anchor."""
    yesterday = [
        Bar(
            "TEST", Timeframe.M5,
            EPOCH - timedelta(days=1) + timedelta(minutes=5 * i),
            EPOCH - timedelta(days=1) + timedelta(minutes=5 * (i + 1)),
            100.0, 100.0, 100.0, 100.0,
        )
        for i in range(10)
    ]
    today = [_bar(1.0, index=i) for i in range(4)]
    assert vwap(yesterday + today) == 1.0


def test_recorded_volume_changes_the_micro_wave_column() -> None:
    """The tick_volume the MT5 downloader now records has to actually reach
    the column, or wiring it through was pointless."""
    h4, m5 = _inputs(base=1.10, unit=0.001)
    half = len(m5) // 2
    early = encode_features(h4, _with_volumes(m5, [100] * half + [1] * (len(m5) - half)))
    late = encode_features(h4, _with_volumes(m5, [1] * half + [100] * (len(m5) - half)))
    flat = encode_features(h4, m5)
    assert early is not None and late is not None and flat is not None
    assert early.micro_wave_momentum != late.micro_wave_momentum
    # Volume-free history takes the unweighted path, so it matches neither.
    assert flat.micro_wave_momentum != early.micro_wave_momentum


# --- 1. the macro tide -------------------------------------------------------


def test_macro_trend_score_is_positive_above_the_baseline_and_negative_below() -> None:
    rising = encode_features(*_inputs(base=1.10, unit=0.001, drift=0.5))
    assert rising is not None and rising.macro_trend_score > 0

    falling = encode_features(*_inputs(base=1.10, unit=0.001, drift=-0.5))
    assert falling is not None and falling.macro_trend_score < 0


# --- 2. structural proximity -------------------------------------------------


def test_distance_to_support_reads_zero_when_price_sits_on_the_level() -> None:
    """A value approaching 0.0 means price is bouncing on structural support."""
    h4, m5 = _inputs(base=1.10, unit=0.001)
    on_level = max(level for level in pivot_levels(h4) if level < m5[-1].close)
    parked = m5 + [_bar(on_level, index=len(m5), spread=0.0002)]
    features = encode_features(h4, parked)
    assert features is not None
    assert abs(features.distance_to_support) < 1e-9


def test_distance_to_resistance_reads_zero_when_price_sits_on_the_level() -> None:
    """Measured downward, so near-zero means the same thing on both sides."""
    h4, m5 = _inputs(base=1.10, unit=0.001)
    on_level = min(level for level in pivot_levels(h4) if level > m5[-1].close)
    parked = m5 + [_bar(on_level, index=len(m5), spread=0.0002)]
    features = encode_features(h4, parked)
    assert features is not None
    assert abs(features.distance_to_resistance) < 1e-9


def test_both_proximity_columns_are_positive_inside_the_scanned_structure() -> None:
    features = encode_features(*_inputs(base=1.10, unit=0.001))
    assert features is not None
    assert features.distance_to_support >= 0.0
    assert features.distance_to_resistance >= 0.0


def test_the_proximity_columns_go_negative_when_structure_breaks() -> None:
    """Outside every scanned level, the sign carries the break."""
    h4, m5 = _inputs(base=1.10, unit=0.001)
    levels = pivot_levels(h4)

    under = m5 + [_bar(min(levels) - 0.05, index=len(m5), spread=0.0002)]
    broken_down = encode_features(h4, under)
    assert broken_down is not None and broken_down.distance_to_support < 0.0

    over = m5 + [_bar(max(levels) + 0.05, index=len(m5), spread=0.0002)]
    broken_up = encode_features(h4, over)
    assert broken_up is not None and broken_up.distance_to_resistance < 0.0


# --- 3. the retracement factor -----------------------------------------------


def _retracement_inputs(final: float) -> tuple[list[Bar], list[Bar]]:
    """An M5 leg whose most recent confirmed swing is exactly 1.36 -> 1.42."""
    h4 = _wave(260, base=1.10, unit=0.001, timeframe=Timeframe.H4)
    lead = [1.40, 1.41, 1.40, 1.41, 1.40, 1.41, 1.40, 1.41]
    leg = [1.40, 1.39, 1.38, 1.36, 1.37, 1.38, 1.42, 1.39, 1.38, 1.37]
    return h4, _series(lead + leg + [final], timeframe=Timeframe.M5)


def test_the_retracement_swing_is_the_one_the_fixture_intends() -> None:
    """Pin the fixture, so the factor assertions below mean what they say."""
    _, m5 = _retracement_inputs(1.39)
    assert swing_points(m5, strength=2, lookback=60) == {"high": 1.42, "low": 1.36}


def test_retracement_factor_reads_zero_at_the_low_half_at_equilibrium_one_at_the_high() -> None:
    def factor_at(final: float) -> float:
        features = encode_features(*_retracement_inputs(final))
        assert features is not None
        return features.retracement_factor

    assert abs(factor_at(1.36) - 0.0) < 1e-9
    # 0.50 is the equilibrium retracement.
    assert abs(factor_at(1.39) - 0.5) < 1e-9
    assert abs(factor_at(1.42) - 1.0) < 1e-9


def test_retracement_factor_is_not_clamped_past_the_swing() -> None:
    """Above 1.0 means the swing high broke; flattening it would hide that."""
    features = encode_features(*_retracement_inputs(1.48))
    assert features is not None and features.retracement_factor > 1.0


# --- 4. micro wave momentum --------------------------------------------------


def test_micro_wave_momentum_signs_against_the_vwap() -> None:
    above = encode_features(*_retracement_inputs(1.90))
    below = encode_features(*_retracement_inputs(0.90))
    assert above is not None and below is not None
    assert above.micro_wave_momentum > 0
    assert below.micro_wave_momentum < 0


# --- fail closed -------------------------------------------------------------


def test_short_history_returns_none_rather_than_a_defaulted_row() -> None:
    """A twenty-bar 'EMA200' is not an EMA200; refuse instead of pretending."""
    h4, m5 = _inputs(base=1.10, unit=0.001)
    assert encode_features(h4[:20], m5) is None
    assert encode_features([], m5) is None
    assert encode_features(h4, []) is None
    assert encode_features(h4, m5[:5]) is None


def test_a_flat_market_has_no_atr_to_divide_by_and_returns_none() -> None:
    h4 = [_bar(1.10, index=i, timeframe=Timeframe.H4) for i in range(260)]
    m5 = [_bar(1.10, index=i, timeframe=Timeframe.M5) for i in range(80)]
    assert encode_features(h4, m5) is None


def test_structureless_history_returns_none() -> None:
    """A monotonic ramp confirms no pivot, so there is no level and no swing."""
    h4 = _series([1.10 + i * 0.001 for i in range(260)], timeframe=Timeframe.H4, spread=0.0005)
    m5 = _series([1.36 + i * 0.0001 for i in range(80)], timeframe=Timeframe.M5, spread=0.00005)
    assert pivot_levels(h4) == []
    assert encode_features(h4, m5) is None


def test_transposed_frames_are_refused() -> None:
    """Passing the frames the wrong way round is a caller bug, not a reading."""
    h4, m5 = _inputs(base=1.10, unit=0.001)
    assert encode_features(h4, m5) is not None
    assert encode_features(m5, h4) is None
