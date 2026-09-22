"""Whether a trade is worth taking, in expectation.

Every gate before this one asks a *pattern* question: is the trend aligned, is
price in the discount band, did the candle confirm. None of them asks the only
question that decides whether an account grows, which is whether the edge is
larger than the cost of collecting it.

That gap is measurable and it is the whole of the current problem. Replayed
over 68 days of MT5 history the strategy wins 52.6% of trades at an average
win of 0.87R, and the spread costs 0.095R to enter. The probability needed to
break even at those terms is

    p* = (1 + cost) / (reward + 1) = 1.095 / 1.87 = 58.6%

so the book needs 58.6% and delivers 52.6%. Every individual trade looked fine
to the gate chain and the book still loses 0.047R a trade, because nothing in
the chain compared those two numbers.

This module is that comparison. It is pure arithmetic on plain floats -- no
model, no broker -- so it is equally usable by the live pipeline, the replay
and a notebook, and it is correct regardless of where ``p_win`` came from.

**On where ``p_win`` comes from.** A probability is only as good as its
calibration: a model that says 0.7 must be right about 70% of the time, or the
expected value computed from it is fiction. :func:`brier_score` and
:func:`calibration_error` exist so that claim is checked rather than assumed,
and they should be reported with any model that feeds this.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import isfinite

logger = logging.getLogger(__name__)


def breakeven_probability(reward_r: float, cost_r: float = 0.0) -> float:
    """The win rate at which a trade exactly breaks even.

    Risking 1R to make ``reward_r``, and paying ``cost_r`` to enter:

        p·reward − (1−p)·1 − cost = 0   ⟹   p = (1 + cost) / (reward + 1)

    The cost term is why this is not the textbook ``1/(1+b)``. At a 1:1
    reward the textbook figure is 50%; with a 0.095R spread it is 54.8%, and
    that 4.8 points is the difference between a book that compounds and one
    that bleeds.
    """
    if not isfinite(reward_r) or reward_r <= 0:
        raise ValueError("reward must be a positive multiple of risk")
    if not isfinite(cost_r) or cost_r < 0:
        raise ValueError("cost cannot be negative")
    return (1.0 + cost_r) / (reward_r + 1.0)


def expected_value_r(p_win: float, reward_r: float, cost_r: float = 0.0) -> float:
    """Expected return of one trade, in R.

    ``p·reward − (1−p)·1 − cost``. Negative means the trade loses money on
    average however good it looks, which is the failure the pattern gates
    cannot see.
    """
    if not isfinite(p_win) or not 0.0 <= p_win <= 1.0:
        raise ValueError("p_win must be a probability")
    if not isfinite(reward_r) or reward_r <= 0:
        raise ValueError("reward must be a positive multiple of risk")
    if not isfinite(cost_r) or cost_r < 0:
        raise ValueError("cost cannot be negative")
    return p_win * reward_r - (1.0 - p_win) - cost_r


def kelly_fraction(p_win: float, reward_r: float, cost_r: float = 0.0) -> float:
    """Share of the bankroll that maximises long-run growth.

        f* = (p·(b+1) − 1) / b,  with b reduced by the cost of entry

    Clamped at zero: a negative Kelly means "bet the other side", and this
    agent does not invert a signal it has already decided against.

    **Full Kelly is not a position size.** It maximises the median outcome of
    an infinite sequence of *independent* bets whose probability is known
    exactly. None of those hold here -- five USD-quoted instruments are
    correlated, the series is finite, and ``p_win`` is an estimate. Callers
    take a fraction of this (see ``KellySizing``), which is why the function
    returns the raw figure rather than pre-shrinking it.
    """
    net_reward = reward_r - cost_r
    if net_reward <= 0:
        return 0.0
    edge = p_win * (net_reward + 1.0) - 1.0
    return max(0.0, edge / net_reward)


@dataclass(frozen=True, slots=True)
class KellySizing:
    """How much of Kelly to actually stake, and the cap above it.

    ``fraction`` is deliberately well below 1. Half-Kelly gives about 75% of
    the growth rate for half the volatility, and quarter-Kelly is the usual
    choice when the probability is estimated rather than known -- which it
    always is here.
    """

    #: Share of full Kelly to stake. 0.25 is quarter-Kelly.
    fraction: float = 0.25
    #: Hard ceiling as a share of the bankroll, whatever Kelly says. The
    #: existing risk ladder is the real authority on size; this never raises
    #: risk above it, only lowers it.
    cap: float = 0.02

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError("the Kelly fraction must be in (0, 1]")
        if not 0 < self.cap <= 1:
            raise ValueError("the cap must be in (0, 1]")

    def stake_fraction(self, p_win: float, reward_r: float, cost_r: float = 0.0) -> float:
        """Bankroll share to risk on this trade, never above ``cap``."""
        return min(self.cap, kelly_fraction(p_win, reward_r, cost_r) * self.fraction)

    def scale_against(self, configured_pct: float, p_win: float, reward_r: float,
                      cost_r: float = 0.0) -> float:
        """Shrink an already-decided risk percentage by the measured edge.

        The risk ladder stays the authority on how much this account may
        stake; this only ever reduces that figure, so a miscalibrated model
        cannot enlarge a position. A trade with no edge sizes to zero, which
        the caller should treat as a rejection rather than a zero-size order.
        """
        kelly_pct = self.stake_fraction(p_win, reward_r, cost_r) * 100.0
        return min(configured_pct, kelly_pct)


@dataclass(frozen=True, slots=True)
class EdgeVerdict:
    """Why a candidate was worth taking, or was not."""

    accepted: bool
    expected_value_r: float
    breakeven_p: float
    p_win: float
    reward_r: float
    cost_r: float
    reason: str = ""

    def describe(self) -> str:
        return (
            f"p={self.p_win:.3f} vs breakeven {self.breakeven_p:.3f}, "
            f"reward {self.reward_r:.2f}R, cost {self.cost_r:.3f}R, "
            f"EV {self.expected_value_r:+.3f}R"
        )


def evaluate_edge(
    *,
    p_win: float,
    reward_r: float,
    cost_r: float = 0.0,
    min_expected_value_r: float = 0.0,
) -> EdgeVerdict:
    """Decide whether the edge clears the cost, with a margin.

    ``min_expected_value_r`` above zero is the honest setting. An estimated
    probability carries error, and accepting everything with EV marginally
    above zero accepts a population whose *true* mean is around zero once
    that error is accounted for. Requiring a margin buys tolerance for
    exactly the miscalibration the model is known to have.

    Fails closed: an unusable probability or reward rejects rather than
    raising, because this runs inside the gate chain and a malformed input is
    a reason not to trade.
    """
    try:
        breakeven = breakeven_probability(reward_r, cost_r)
        value = expected_value_r(p_win, reward_r, cost_r)
    except ValueError as exc:
        return EdgeVerdict(False, 0.0, 1.0, 0.0, 0.0, 0.0, f"edge is not computable: {exc}")

    if value < min_expected_value_r:
        return EdgeVerdict(
            False, value, breakeven, p_win, reward_r, cost_r,
            f"expected value {value:+.3f}R is below the {min_expected_value_r:+.3f}R "
            f"minimum; this setup needs a {breakeven:.1%} win rate and the model "
            f"gives it {p_win:.1%}",
        )
    return EdgeVerdict(True, value, breakeven, p_win, reward_r, cost_r)


# --------------------------------------------------------------- calibration


def brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    """Mean squared error of a probabilistic forecast; lower is better.

    The single number that says whether a probability means anything. A model
    that always says 0.5 scores 0.25; one that beats that is carrying real
    information, and one that does not is noise with a decimal point.

    Note it rewards *calibration and resolution together*, so compare it
    against the base-rate model's score rather than against an absolute.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be the same length")
    if not probabilities:
        raise ValueError("cannot score an empty forecast")
    return sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / len(probabilities)


def calibration_error(
    probabilities: list[float], outcomes: list[int], *, bins: int = 10
) -> float:
    """Expected calibration error: mean |predicted − observed| across bins.

    Brier conflates being *calibrated* with being *informative*. This isolates
    the first: of the trades the model called 70%, did about 70% win? If not,
    every expected value computed from those numbers is wrong by that much,
    and the EV gate is being fed fiction.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be the same length")
    if not probabilities:
        raise ValueError("cannot score an empty forecast")
    if bins < 2:
        raise ValueError("at least two bins are needed")

    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        # The final bin is closed so a forecast of exactly 1.0 is counted.
        members = [
            (p, o) for p, o in zip(probabilities, outcomes)
            if (low <= p < high) or (index == bins - 1 and p == 1.0)
        ]
        if not members:
            continue
        mean_p = sum(p for p, _ in members) / len(members)
        observed = sum(o for _, o in members) / len(members)
        error += (len(members) / total) * abs(mean_p - observed)
    return error


def reliability_table(
    probabilities: list[float], outcomes: list[int], *, bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin predicted-versus-observed, for printing next to a model.

    The table a person actually reads to decide whether to trust a
    probability, rather than the scalar summaries above.
    """
    rows: list[dict[str, float]] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (p, o) for p, o in zip(probabilities, outcomes)
            if (low <= p < high) or (index == bins - 1 and p == 1.0)
        ]
        if not members:
            continue
        rows.append({
            "low": low,
            "high": high,
            "count": float(len(members)),
            "predicted": sum(p for p, _ in members) / len(members),
            "observed": sum(o for _, o in members) / len(members),
        })
    return rows
