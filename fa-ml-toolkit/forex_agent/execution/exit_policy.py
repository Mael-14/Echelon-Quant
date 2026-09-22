"""Time- and progress-based exits applied while a contract is open.

The broker-side stop and target attached at entry are the only levels that
survive this process dying, so they stay exactly where ``submit`` put them.
Everything here tightens *inside* those levels, and only while the agent is
alive and receiving ticks.

Three rules, in the order they fire on a tick:

*Break-even.* Once a trade has earned ``breakeven_at_r`` of its own risk, the
stop moves to entry plus ``breakeven_offset_r``. A winner can then no longer
become a full-size loser, which is the single most expensive thing a fixed stop
allows. The offset exists because exiting exactly at entry still pays the
spread twice, and on a multiplier contract that spread is amplified.

*Trail.* Past ``trail_at_r`` the stop follows the best price seen by
``trail_distance_r``. A move that keeps going is not handed back waiting for a
fixed target, and a move that stalls is banked near its high.

*Time stop.* A scalp that has not resolved is not a scalp. After
``max_hold_seconds`` the contract is sold at market for whatever it is worth.
This is also what restores trade frequency: one position per symbol is enforced
at entry, so a contract held for twenty minutes silently costs every entry that
symbol would otherwise have taken in the meantime.

Progress is measured in ``R`` -- multiples of the trade's *original* risk. The
original distance is recorded at entry rather than recomputed, because once
break-even moves the stop onto the entry price the distance between them is
zero and every ratio built from it collapses.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite


#: How far past the round-trip cost a break-even stop is placed. At 1.0 the
#: stop sits exactly at the cost, which scratches; above it the exit is a real
#: if small gain. Anything armed at or beyond ``breakeven_at_r`` is skipped
#: instead, because a stop at its own trigger closes on the tick that arms it.
BREAKEVEN_COST_MARGIN = 1.5


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    """How aggressively an open contract is managed after entry.

    Every rule is disabled by setting its field to zero, so a profile can opt
    out of trailing without opting out of the time stop.
    """

    #: Sell at market once the contract has been open this long. 0 disables.
    max_hold_seconds: float = 0.0

    #: Favourable excursion, in R, at which the stop moves to break-even.
    breakeven_at_r: float = 0.0

    #: How far beyond entry break-even sits, in R, so the spread is covered.
    breakeven_offset_r: float = 0.0

    #: Favourable excursion, in R, at which trailing begins.
    trail_at_r: float = 0.0

    #: How far behind the best price the trailing stop sits, in R.
    trail_distance_r: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_hold_seconds", "breakeven_at_r", "breakeven_offset_r",
            "trail_at_r", "trail_distance_r",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative number")
        if self.breakeven_at_r and self.breakeven_offset_r >= self.breakeven_at_r:
            # Otherwise the "break-even" stop sits at or beyond the price that
            # triggers it, which closes the trade on the tick that arms it.
            raise ValueError("breakeven_offset_r must be smaller than breakeven_at_r")
        if self.trail_at_r and self.trail_distance_r >= self.trail_at_r:
            # A trail wider than the excursion that arms it would place the
            # first trailing stop below entry -- looser than break-even.
            raise ValueError("trail_distance_r must be smaller than trail_at_r")


#: M1 execution.
#:
#: Three minutes rather than one. Swept against real M1 history, the hold is by
#: far the most sensitive knob in the whole exit model:
#:
#:     hold   trades/day   win rate   expectancy   exits
#:     off          19.6      53.8%      -0.186R   stop 12, trail 12, target 2
#:     60s          26.0      31.4%      -0.125R   time 32, stop 3
#:     120s         20.2      44.4%      -0.160R   time 15, stop 6, trail 5
#:     180s         20.2      51.9%      -0.073R   time 9, trail 8, stop 8
#:     300s         19.6      53.8%      -0.130R   trail 11, stop 10, time 3
#:
#: At sixty seconds the clock fires before anything else can: 32 of 35 exits
#: are time stops, the trailing stop never arms once, and the entry is denied
#: the chance to be right. At three minutes the exit mix is balanced and
#: expectancy is the best measured anywhere in this file's history. Operators
#: who want a stricter clock set ``SCALP_MAX_HOLD_SECONDS``.
SCALP_EXIT = ExitPolicy(
    max_hold_seconds=180.0,
    breakeven_at_r=0.5,
    breakeven_offset_r=0.1,
    trail_at_r=0.7,
    trail_distance_r=0.4,
)

#: M5 execution. Three bars, which is the point past which a trigger bar's
#: information has been fully priced in.
SWING_EXIT = ExitPolicy(
    max_hold_seconds=900.0,
    breakeven_at_r=0.7,
    breakeven_offset_r=0.1,
    trail_at_r=1.2,
    trail_distance_r=0.6,
)

#: Used when nothing is known about a contract -- notably one adopted from a
#: previous run, whose levels the broker does not report. Managing it on
#: invented levels would be worse than not managing it, so only the time stop
#: applies, and generously.
UNMANAGED_EXIT = ExitPolicy(max_hold_seconds=900.0)


@dataclass(frozen=True, slots=True)
class ExitCheck:
    """What a tick concluded about an open contract.

    ``close_reason`` set means sell now. Otherwise the two prices are the
    updated state to persist; either may be ``None`` when it did not move.
    """

    close_reason: str | None = None
    stop_price: float | None = None
    peak_price: float | None = None


def _as_datetime(value: object) -> datetime | None:
    """Parse a recorded ISO timestamp, tolerating anything unparseable.

    The timestamp comes back from durable state that an older build wrote, so
    it may be missing or in a shape this build does not recognise. That must
    disable the time stop rather than raise inside the tick handler.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def evaluate_exit(
    *,
    side: str,
    entry_price: float,
    stop_price: float,
    take_profit: float,
    risk_distance: float,
    peak_price: float,
    opened_at: object,
    quote: float,
    now: datetime,
    policy: ExitPolicy,
    cost_r: float = 0.0,
) -> ExitCheck:
    """Decide whether ``quote`` closes the contract, and how the stop moves.

    Pure, so the whole exit model is testable with plain floats and without a
    broker. The order matters: the stop is tightened *before* the levels are
    tested, so a tick that both extends the trail and breaches it closes on the
    same tick rather than one tick later at a worse price.
    """
    long = side == "BUY"
    if not isfinite(quote) or quote <= 0:
        return ExitCheck()

    elapsed_exceeded = False
    opened = _as_datetime(opened_at)
    if policy.max_hold_seconds > 0 and opened is not None:
        elapsed_exceeded = (now - opened).total_seconds() >= policy.max_hold_seconds

    # A position restored from state an older build wrote, or one adopted from
    # the broker, has no entry or recorded risk. That disables *tightening*
    # only: the fixed levels it was opened under still hold, and the time stop
    # does not need to know where the trade started. Skipping the levels here
    # would leave such a position running past its own stop.
    manageable = (
        isfinite(entry_price) and entry_price > 0
        and isfinite(risk_distance) and risk_distance > 0
    )

    best = peak_price
    stop = stop_price
    if manageable:
        best = max(peak_price, quote) if long else min(peak_price, quote)
        if not isfinite(best) or best <= 0:
            best = quote
        progress = (best - entry_price) / risk_distance if long else (entry_price - best) / risk_distance

        tighter = max if long else min
        # The offset has to clear the cost of the contract, not just the entry
        # price. A "break-even" stop inside the commission is a guaranteed
        # loss dressed as a scratch: measured live, 61% of all contracts exited
        # this way for a median of -7.2% of stake while the code believed it
        # was banking +0.1R.
        offset_r = max(policy.breakeven_offset_r, cost_r * BREAKEVEN_COST_MARGIN)
        if policy.breakeven_at_r and progress >= policy.breakeven_at_r and offset_r < policy.breakeven_at_r:
            offset = offset_r * risk_distance
            stop = tighter(stop, entry_price + offset if long else entry_price - offset)
        if policy.trail_at_r and progress >= policy.trail_at_r:
            distance = policy.trail_distance_r * risk_distance
            stop = tighter(stop, best - distance if long else best + distance)

    # A level of zero means "not recorded", not "at zero". Testing it anyway
    # closes every long position on its first tick.
    if take_profit > 0 and (long and quote >= take_profit or not long and quote <= take_profit):
        return ExitCheck("take profit")
    if stop > 0 and (long and quote <= stop or not long and quote >= stop):
        # Naming which stop fired is what makes the closed-trade log readable:
        # a managed exit is a protected win, the original stop is a loss. The
        # original level is recovered from the recorded risk rather than from
        # ``stop_price``, which has already been overwritten on earlier ticks.
        original = entry_price - risk_distance if long else entry_price + risk_distance
        return ExitCheck("stop loss" if not manageable or stop == original else "trailing stop")
    if elapsed_exceeded:
        return ExitCheck("time stop")

    return ExitCheck(
        None,
        stop_price=stop if stop != stop_price else None,
        peak_price=best if best != peak_price else None,
    )
