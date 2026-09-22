"""ICT chart patterns, as arithmetic.

Three learning attempts failed on the same six indicators -- a 16-unit GRU, a
32-unit GRU, and a logistic regression on a completely different label. All
three landed on the base rate, and the logistic model said why in a way the
others could not: with standardised coefficients no larger than 0.015, RSI,
ATR, choppiness and the timeframe offsets carry essentially no information
about whether a target is reached before a stop.

That is a statement about the *features*, not the algorithms. This module adds
genuinely different ones.

Everything here is a pure function of OHLC bars, returning numbers and
booleans rather than opinions. That matters more for ICT than for most
material, because the concepts are usually taught visually and two traders
will annotate the same chart differently. A definition that cannot be written
down cannot be backtested, and a backtest of a discretionary rule measures the
person, not the market.

Where the standard definition is ambiguous this module picks one reading and
says so in a comment. The choices are conservative by default: a pattern is
recognised only when it is unambiguous, because a feature that fires on
marginal cases teaches a learner to trade noise.

**Volume is deliberately absent.** The classic Market Structure Shift is
confirmed by displacement *or* volume, and the recorded history carries OHLC
only. Displacement is measured against ATR instead, which is the same idea
expressed in the data that exists. Adding tick volume is a downloader change,
not a definition change.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

#: ICT kill zones in UTC, as (start_hour, end_hour) half-open intervals.
#:
#: These are the New York-time windows converted to UTC and therefore
#: **shift by an hour with US daylight saving**. They are approximated here
#: rather than tracked exactly: the edge a kill zone is claimed to carry is not
#: a one-hour-sharp effect, and a DST table would imply a precision the concept
#: does not have.
KILL_ZONES: dict[str, tuple[int, int]] = {
    "asian": (0, 4),      # Tokyo
    "london": (7, 10),    # 02:00-05:00 New York
    "newyork": (12, 15),  # 07:00-10:00 New York, spanning the 13:30 open
}

#: The Silver Bullet windows, in UTC: one hour each, London and New York.
#:
#: 03:00-04:00 and 10:00-11:00 New York time. Used as a **pre-filter** rather
#: than a state field, which is the point -- as a field it costs a dimension
#: and dilutes every cell; as a filter it costs nothing and concentrates the
#: sample on the hours the claimed edge is supposed to live in.
#:
#: The cost is severe and has to be weighed rather than assumed: two hours in
#: twenty-four discards ~92% of bars, so a table that needs 100 observations
#: per cell needs twelve times the history to fill the same cells.
SILVER_BULLET_HOURS: tuple[int, ...] = (7, 14)


def in_silver_bullet(epoch: int) -> bool:
    """Whether ``epoch`` falls in a Silver Bullet hour."""
    if not epoch:
        return False
    return datetime.fromtimestamp(epoch, tz=timezone.utc).hour in SILVER_BULLET_HOURS


@dataclass(frozen=True, slots=True)
class Candle:
    """One bar, in the only four fields these patterns need.

    Defined here rather than reusing ``models.Bar`` so the pure functions can
    be exercised with plain tuples in tests, and so a caller replaying a
    recorded JSON candle does not have to build a validated domain object per
    bar.
    """

    high: float
    low: float
    open: float
    close: float
    epoch: int = 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low


def candles_from(rows: Sequence[dict]) -> list[Candle]:
    """Build candles from the recorded ``{epoch, open, high, low, close}`` shape."""
    return [
        Candle(
            high=float(row["high"]), low=float(row["low"]),
            open=float(row["open"]), close=float(row["close"]),
            epoch=int(row.get("epoch", 0)),
        )
        for row in rows
    ]


# ------------------------------------------------------------ fair value gaps


@dataclass(frozen=True, slots=True)
class FairValueGap:
    """An unfilled three-candle imbalance."""

    bullish: bool
    #: The gap's edges, low first.
    bottom: float
    top: float
    #: Index of the middle candle, which is the one that displaced.
    index: int

    @property
    def size(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def is_filled_by(self, candle: Candle) -> bool:
        """Whether a later candle has traded back through the gap.

        Full fill, not partial: a gap that price has merely tagged is still
        treated as live. The alternative -- retiring a gap on first touch --
        would discard exactly the retest the setup is built on.
        """
        return candle.low <= self.bottom and candle.high >= self.top


def find_fair_value_gaps(
    candles: Sequence[Candle], *, min_size: float = 0.0, lookback: int = 50
) -> list[FairValueGap]:
    """Unfilled FVGs in the most recent ``lookback`` candles, newest last.

    A bullish gap is a three-candle sequence where candle 3's low sits above
    candle 1's high, leaving a band of prices the market skipped. Bearish is
    the mirror.

    ``min_size`` filters noise. Left at zero every one-tick gap qualifies, and
    on M5 forex that is most of the chart -- which is precisely how a feature
    becomes useless by firing constantly.
    """
    if len(candles) < 3:
        return []
    window = candles[-lookback:] if lookback > 0 else list(candles)
    offset = len(candles) - len(window)

    gaps: list[FairValueGap] = []
    for index in range(len(window) - 2):
        first, third = window[index], window[index + 2]
        if third.low > first.high and third.low - first.high >= min_size:
            gaps.append(FairValueGap(True, first.high, third.low, offset + index + 1))
        elif first.low > third.high and first.low - third.high >= min_size:
            gaps.append(FairValueGap(False, third.high, first.low, offset + index + 1))

    # Retire the ones price has already traded back through. A filled gap is
    # not a level; treating it as one is the most common way this feature is
    # mis-implemented.
    live: list[FairValueGap] = []
    for gap in gaps:
        later = window[gap.index - offset + 2 :]
        if not any(gap.is_filled_by(candle) for candle in later):
            live.append(gap)
    return live


# ------------------------------------------------------------- swing structure


def swing_highs(candles: Sequence[Candle], *, strength: int = 2) -> list[int]:
    """Indices of pivot highs: a high with ``strength`` lower highs each side.

    ``strength`` is the whole definition of a "swing". At 1 every minor bump
    qualifies; at 5 only major turns do. It is a parameter rather than a
    constant because it is exactly the kind of threshold an optimiser should
    be choosing rather than a person guessing.
    """
    found: list[int] = []
    for index in range(strength, len(candles) - strength):
        pivot = candles[index].high
        if all(
            candles[index + step].high < pivot
            for step in range(-strength, strength + 1)
            if step != 0
        ):
            found.append(index)
    return found


def swing_lows(candles: Sequence[Candle], *, strength: int = 2) -> list[int]:
    found: list[int] = []
    for index in range(strength, len(candles) - strength):
        pivot = candles[index].low
        if all(
            candles[index + step].low > pivot
            for step in range(-strength, strength + 1)
            if step != 0
        ):
            found.append(index)
    return found


def average_true_range(candles: Sequence[Candle], *, period: int = 14) -> float:
    """Wilder's range, simple-averaged. Zero when there is not enough history."""
    if len(candles) < 2:
        return 0.0
    ranges = [
        max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        for previous, current in zip(candles[-period - 1 :], candles[-period:])
    ]
    return sum(ranges) / len(ranges) if ranges else 0.0


def market_structure_shift(
    candles: Sequence[Candle], *, strength: int = 2, displacement: float = 1.5
) -> int:
    """+1 for a bullish shift, -1 for bearish, 0 for none, on the last candle.

    A shift is a *close* beyond the most recent opposing pivot, with a body
    larger than ``displacement`` times ATR.

    Both halves matter. Requiring the close rather than the wick is what
    separates a structural break from a liquidity sweep -- the sweep pokes
    through and closes back inside, and calling that a break is the single
    most expensive misreading in this vocabulary. The displacement filter is
    the stand-in for the volume confirmation the recorded history cannot
    provide.
    """
    if len(candles) < strength * 2 + 3:
        return 0
    last = candles[-1]
    atr = average_true_range(candles[:-1])
    if atr <= 0 or last.body < displacement * atr:
        return 0

    # Pivots are searched excluding the final candle, which cannot yet be one.
    body = candles[:-1]
    highs = swing_highs(body, strength=strength)
    lows = swing_lows(body, strength=strength)
    if highs and last.close > body[highs[-1]].high:
        return 1
    if lows and last.close < body[lows[-1]].low:
        return -1
    return 0


def liquidity_sweep(candles: Sequence[Candle], *, strength: int = 2) -> int:
    """+1 when sell-side liquidity was swept, -1 for buy-side, 0 for neither.

    A sweep is the inverse of a structural break: the wick takes out a prior
    pivot and the candle closes back inside. Sell-side (a low swept) is
    bullish, because the stops resting below were taken and price rejected --
    which is why the sign convention here is by *implication*, not by which
    side was hit.
    """
    if len(candles) < strength * 2 + 3:
        return 0
    last = candles[-1]
    body = candles[:-1]
    lows = swing_lows(body, strength=strength)
    highs = swing_highs(body, strength=strength)

    if lows:
        level = body[lows[-1]].low
        if last.low < level <= last.close:
            return 1
    if highs:
        level = body[highs[-1]].high
        if last.high > level >= last.close:
            return -1
    return 0


# ---------------------------------------------------------------- order blocks


@dataclass(frozen=True, slots=True)
class OrderBlock:
    """The last opposing candle before a displacement move."""

    bullish: bool
    top: float
    bottom: float
    index: int

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def find_order_block(
    candles: Sequence[Candle], *, strength: int = 2, displacement: float = 1.5,
    lookback: int = 30,
) -> OrderBlock | None:
    """The most recent order block, or ``None``.

    Defined as the last down-candle before an up-move that shifted structure
    (bullish), or the mirror. Anchoring it to a structure shift rather than to
    any impulsive move is the conservative reading: without that anchor every
    pullback in a trend produces an "order block" and the feature stops
    discriminating.

    The block's range is the full candle, not just the body. Wick-to-wick is
    the more common convention and the more forgiving one; body-only would
    miss retests that everyone watching the chart would call a touch.
    """
    if len(candles) < strength * 2 + 4:
        return None
    window = candles[-lookback:] if lookback > 0 else list(candles)
    offset = len(candles) - len(window)

    # Walk back from the newest candle looking for the displacement leg.
    for end in range(len(window) - 1, strength * 2 + 2, -1):
        shift = market_structure_shift(
            window[: end + 1], strength=strength, displacement=displacement
        )
        if shift == 0:
            continue
        # The last candle of the opposite colour before the displacement.
        for index in range(end - 1, max(-1, end - 10), -1):
            candle = window[index]
            if shift == 1 and not candle.is_bullish:
                return OrderBlock(True, candle.high, candle.low, offset + index)
            if shift == -1 and candle.is_bullish:
                return OrderBlock(False, candle.high, candle.low, offset + index)
        return None
    return None


# ------------------------------------------------------------------ kill zones


def kill_zone(epoch: int) -> str:
    """Which ICT session window ``epoch`` falls in, or ``"none"``."""
    if not epoch:
        return "none"
    hour = datetime.fromtimestamp(epoch, tz=timezone.utc).hour
    for name, (start, end) in KILL_ZONES.items():
        if start <= hour < end:
            return name
    return "none"


# -------------------------------------------------------------- the feature set


def structural_bias(candles: Sequence[Candle], *, strength: int = 2) -> int:
    """Persistent directional read: +1 bullish, -1 bearish, 0 undecided.

    Not :func:`market_structure_shift`, which fires on one candle and is
    silent the rest of the time. A *bias* has to persist between breaks, so
    this walks back to the most recent confirmed break and reports its
    direction until the opposite one occurs.

    This is the function that makes the feature set multi-timeframe, and it
    is the one the first implementation was missing. ICT is a
    higher-timeframe-bias method: a bullish fair value gap on M5 means the
    opposite thing depending on whether the daily is bullish or bearish, and
    a state that cannot tell those apart is blind at exactly the level where
    direction is decided.
    """
    if len(candles) < strength * 2 + 3:
        return 0
    highs = swing_highs(candles, strength=strength)
    lows = swing_lows(candles, strength=strength)
    if not highs and not lows:
        return 0

    # Walk back from the newest candle; the first pivot a later close broke
    # is the one that set the current bias.
    for index in range(len(candles) - 1, strength, -1):
        close = candles[index].close
        prior_highs = [h for h in highs if h < index - strength]
        prior_lows = [low for low in lows if low < index - strength]
        if prior_highs and close > candles[prior_highs[-1]].high:
            return 1
        if prior_lows and close < candles[prior_lows[-1]].low:
            return -1
    return 0


@dataclass(frozen=True, slots=True)
class HigherTimeframes:
    """Bias from the frames above the one being traded.

    Supplied by the caller rather than derived here, because only the caller
    knows which coarse bars had actually *closed* by the execution bar being
    scored. Deriving them inside this module would invite exactly the
    look-ahead that shifts every frame by one bar.
    """

    daily: int = 0
    h4: int = 0
    m15: int = 0


@dataclass(frozen=True, slots=True)
class StateSpec:
    """Which fields make it into the Q-table key, and how coarse they are.

    A knob rather than a constant because the right answer is an empirical
    trade-off, not a matter of taste: every field included multiplies the
    number of cells, and the number of bars available to fill them is fixed
    by what the broker holds. The trainer sweeps these and reports coverage
    against held-out expectancy, so the choice is measured.
    """

    #: The daily and 4H bias -- the whole point of the multi-timeframe
    #: change. A flag rather than a constant so "did it help?" can be
    #: measured against an otherwise identical run, instead of inferred
    #: from two experiments that differed in two ways.
    include_htf_bias: bool = True
    #: Off by default: when M15 *is* the execution frame, its own bias is
    #: already implied by the structure fields.
    include_m15_bias: bool = False
    #: The weakest field conceptually -- volatility is already partly in the
    #: ATR-scaled stop -- and it triples the space.
    include_volatility: bool = True
    #: Collapse four sessions to in-zone/out. Kill zones are core to the
    #: method, so this is the last thing to give up. Redundant when the
    #: caller is already filtering to one session.
    coarse_kill_zone: bool = False
    #: Replace the four signed structure fields with two counts: how many
    #: bullish structures are active, and how many bearish.
    #:
    #: 3^4 = 81 combinations collapse to 4x4 = 16. The information given up
    #: is *which* structures are present; what is kept is how many agree,
    #: which is the thing a confluence-based method actually claims to
    #: trade. With 1,275 of 1,397 cells pruned as too rare at full
    #: cardinality, that trade is worth making.
    confluence: bool = False

    def cells(self) -> int:
        """Upper bound on distinct states, for sizing against the data."""
        total = 4 * 4 if self.confluence else 3 * 3 * 3 * 3
        if self.include_htf_bias:
            total *= 3 * 3              # d1, h4
        if self.include_m15_bias:
            total *= 3
        total *= 2 if self.coarse_kill_zone else 4
        if self.include_volatility:
            total *= 3
        return total

    def describe(self) -> str:
        bits = [
            "conf" if self.confluence else "full",
            "htf" if self.include_htf_bias else "-htf",
            "m15" if self.include_m15_bias else "-m15",
            "vol" if self.include_volatility else "-vol",
            "kz2" if self.coarse_kill_zone else "kz4",
        ]
        return f"{'/'.join(bits)} ({self.cells():,} cells)"


@dataclass(frozen=True, slots=True)
class ICTFeatures:
    """Everything this module can say about the newest candle.

    The fields are deliberately coarse -- signs, booleans and small buckets --
    because they feed a tabular learner whose state space is the product of
    their cardinalities. A continuous feature here would make the table
    unfillable.
    """

    #: -1, 0 or +1.
    structure_shift: int
    liquidity_sweep: int
    #: Price is inside an unfilled gap of that direction.
    in_bullish_fvg: bool
    in_bearish_fvg: bool
    in_bullish_ob: bool
    in_bearish_ob: bool
    kill_zone: str
    #: Bucketed 0-2: below average, average, above. Continuous volatility
    #: would explode the state space for information the sign already carries.
    volatility_bucket: int
    #: Direction on the frames above. The whole point of the method.
    daily_bias: int = 0
    h4_bias: int = 0
    m15_bias: int = 0

    @property
    def fvg(self) -> int:
        """Gap direction as one signed value: +1 bullish, -1 bearish, 0 none.

        Two booleans can encode "inside a bullish *and* a bearish gap", which
        is possible but rare and carries no usable meaning. Collapsing them
        costs almost nothing and removes a quarter of the state space.
        """
        if self.in_bullish_fvg and not self.in_bearish_fvg:
            return 1
        if self.in_bearish_fvg and not self.in_bullish_fvg:
            return -1
        return 0

    @property
    def order_block(self) -> int:
        if self.in_bullish_ob and not self.in_bearish_ob:
            return 1
        if self.in_bearish_ob and not self.in_bullish_ob:
            return -1
        return 0

    def state(self, spec: "StateSpec | None" = None) -> tuple:
        """The discrete key a Q-table is indexed by.

        Order is fixed and the values are small integers, so the key is
        hashable, printable, and stable across runs -- which matters because a
        Q-table trained under one key layout is meaningless under another.
        **Changing this order or the spec invalidates every saved table**,
        which is why the trainer stamps both.

        The higher-timeframe biases come first: they are the coarsest and
        most persistent part of the state, and putting them at the front
        makes a printed key readable top-down, like the chart analysis it
        stands for.

        ``spec`` exists because the state space is a hard constraint rather
        than a preference. Every field multiplies the number of cells, and
        the broker holds only ~89 days of M5 -- the full key is 46,656 cells
        against ~93,000 bars, which is two samples each and learns nothing.
        Coarsening is not a compromise here; it is what makes the table
        estimable at all.
        """
        spec = spec or StateSpec()
        parts: list = []
        if spec.include_htf_bias:
            parts += [self.daily_bias, self.h4_bias]
        if spec.include_m15_bias:
            parts.append(self.m15_bias)
        if spec.confluence:
            parts += [self.bullish_confluence, self.bearish_confluence]
        else:
            parts += [
                self.structure_shift, self.liquidity_sweep,
                self.fvg, self.order_block,
            ]
        if spec.coarse_kill_zone:
            parts.append(int(self.kill_zone != "none"))
        else:
            parts.append(self.kill_zone)
        if spec.include_volatility:
            parts.append(self.volatility_bucket)
        return tuple(parts)

    @property
    def bullish_confluence(self) -> int:
        """How many bullish structures are active, capped at 3.

        Capped because the fourth adds a state and almost no samples: the
        tail beyond three simultaneous structures is a handful of bars in a
        year, and a cell that rare is noise whatever its value says.
        """
        count = sum((
            self.structure_shift > 0,
            self.liquidity_sweep > 0,
            self.fvg > 0,
            self.order_block > 0,
        ))
        return min(count, 3)

    @property
    def bearish_confluence(self) -> int:
        count = sum((
            self.structure_shift < 0,
            self.liquidity_sweep < 0,
            self.fvg < 0,
            self.order_block < 0,
        ))
        return min(count, 3)

    @property
    def htf_agrees_with_buy(self) -> bool:
        """Whether the frames above lean long, with no disagreement."""
        return self.daily_bias >= 0 and self.h4_bias >= 0 and (
            self.daily_bias > 0 or self.h4_bias > 0
        )

    @property
    def htf_agrees_with_sell(self) -> bool:
        return self.daily_bias <= 0 and self.h4_bias <= 0 and (
            self.daily_bias < 0 or self.h4_bias < 0
        )

    @property
    def has_any_signal(self) -> bool:
        """Whether anything ICT-shaped is present at all.

        Most bars are empty of all of it, and a learner shown only empty
        states spends its samples learning that nothing usually happens.
        """
        return bool(
            self.structure_shift
            or self.liquidity_sweep
            or self.in_bullish_fvg
            or self.in_bearish_fvg
            or self.in_bullish_ob
            or self.in_bearish_ob
        )


def extract_features(
    candles: Sequence[Candle], *, strength: int = 2, displacement: float = 1.5,
    min_gap_atr: float = 0.25, higher: HigherTimeframes | None = None,
) -> ICTFeatures:
    """Read the ICT state of the newest candle.

    ``min_gap_atr`` scales the gap filter by volatility rather than fixing it
    in price, so one threshold works across EURUSD at 0.0001 and gold at 1.0.

    ``higher`` carries the bias from the frames above. It defaults to neutral
    so the pure patterns stay testable in isolation, but **a caller that omits
    it is reading ICT with its most important input removed** -- the first
    version of this module did exactly that, and every state it learned was
    blind to direction at the level where direction is decided.
    """
    higher = higher or HigherTimeframes()
    if not candles:
        return ICTFeatures(
            0, 0, False, False, False, False, "none", 1,
            daily_bias=higher.daily, h4_bias=higher.h4, m15_bias=higher.m15,
        )

    last = candles[-1]
    atr = average_true_range(candles)
    gaps = find_fair_value_gaps(candles, min_size=atr * min_gap_atr)
    block = find_order_block(candles, strength=strength, displacement=displacement)

    # A candle's own range counts as "in" the zone: the strategy enters on the
    # retest, and requiring the close inside would miss a wick that tagged the
    # level and reversed, which is the textbook entry.
    def touches(bottom: float, top: float) -> bool:
        return last.low <= top and last.high >= bottom

    recent = average_true_range(candles[:-1], period=50)
    if recent <= 0:
        bucket = 1
    elif atr < recent * 0.8:
        bucket = 0
    elif atr > recent * 1.2:
        bucket = 2
    else:
        bucket = 1

    return ICTFeatures(
        structure_shift=market_structure_shift(
            candles, strength=strength, displacement=displacement
        ),
        liquidity_sweep=liquidity_sweep(candles, strength=strength),
        in_bullish_fvg=any(g.bullish and touches(g.bottom, g.top) for g in gaps),
        in_bearish_fvg=any(not g.bullish and touches(g.bottom, g.top) for g in gaps),
        in_bullish_ob=bool(block and block.bullish and touches(block.bottom, block.top)),
        in_bearish_ob=bool(block and not block.bullish and touches(block.bottom, block.top)),
        kill_zone=kill_zone(last.epoch),
        volatility_bucket=bucket,
        daily_bias=higher.daily,
        h4_bias=higher.h4,
        m15_bias=higher.m15,
    )


# ------------------------------------------------- continuous feature vector

#: Columns for an estimator that generalises, rather than a lookup table.
#:
#: A Q-table can only index discrete cells, so every continuous quantity had
#: to be bucketed and every bucket multiplied the state space -- which is why
#: 1,275 of 1,397 cells ended up pruned as too rare. A tree or a regularised
#: linear model has no such constraint: it splits or weights a real number
#: directly, and shares what it learns between neighbouring values instead of
#: treating them as unrelated.
#:
#: So the quantities the table had to throw away are back, in full precision.
#: ``hour`` in particular is deliberately raw: a tree can discover which hours
#: matter, which is a direct test of the kill-zone hypothesis rather than an
#: assumption of it.
ICT_FEATURE_COLUMNS: tuple[str, ...] = (
    "side",
    # Discrete structure, unchanged from the table.
    "structure_shift", "liquidity_sweep", "fvg", "order_block",
    "daily_bias", "h4_bias",
    # Continuous, and new. Each was a bucket or absent before.
    "fvg_size_atr", "fvg_depth", "ob_height_atr", "ob_distance_atr",
    "atr_ratio", "body_atr", "upper_wick_atr", "lower_wick_atr",
    "hour",
)


def feature_vector(
    candles: Sequence[Candle], *, side: int, higher: HigherTimeframes | None = None,
    strength: int = 2, displacement: float = 1.5, min_gap_atr: float = 0.25,
) -> dict[str, float]:
    """ICT state as real numbers, keyed by name.

    ``side`` is +1 for a long candidate and -1 for a short. It is a feature
    rather than two separate models because the structures are not symmetric
    -- a bullish order block under a bearish daily is its own situation, and
    one model with a side column can represent that interaction while two
    models cannot share anything they learn.

    Everything volatility-relative is divided by ATR so one model spans
    EURUSD at 0.0001 and gold at 1.0 without rescaling.
    """
    higher = higher or HigherTimeframes()
    base = dict.fromkeys(ICT_FEATURE_COLUMNS, 0.0)
    base["side"] = float(side)
    if not candles:
        return base

    last = candles[-1]
    atr = average_true_range(candles)
    if atr <= 0:
        return base

    features = extract_features(
        candles, strength=strength, displacement=displacement,
        min_gap_atr=min_gap_atr, higher=higher,
    )
    base.update(
        structure_shift=float(features.structure_shift),
        liquidity_sweep=float(features.liquidity_sweep),
        fvg=float(features.fvg),
        order_block=float(features.order_block),
        daily_bias=float(higher.daily),
        h4_bias=float(higher.h4),
        atr_ratio=atr / (average_true_range(candles[:-1], period=50) or atr),
        body_atr=last.body / atr,
        upper_wick_atr=(last.high - max(last.open, last.close)) / atr,
        lower_wick_atr=(min(last.open, last.close) - last.low) / atr,
        hour=float(datetime.fromtimestamp(last.epoch, tz=timezone.utc).hour)
        if last.epoch else 0.0,
    )

    # The nearest live gap price is actually touching, if any. Size and depth
    # are what the bucketed version could never express: a gap twice the ATR
    # and a gap a tenth of it were the same cell.
    gaps = [
        g for g in find_fair_value_gaps(candles, min_size=atr * min_gap_atr)
        if last.low <= g.top and last.high >= g.bottom
    ]
    if gaps:
        gap = max(gaps, key=lambda g: g.size)
        base["fvg_size_atr"] = gap.size / atr
        if gap.size > 0:
            # How far through the gap price has travelled, 0 at the near edge.
            travelled = (last.close - gap.bottom) / gap.size
            base["fvg_depth"] = min(1.0, max(0.0, travelled))

    block = find_order_block(candles, strength=strength, displacement=displacement)
    if block is not None:
        base["ob_height_atr"] = (block.top - block.bottom) / atr
        midpoint = (block.top + block.bottom) / 2.0
        base["ob_distance_atr"] = (last.close - midpoint) / atr
    return base
