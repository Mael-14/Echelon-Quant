"""Mapping a toolkit signal onto a Deriv multiplier contract.

Pure functions, no I/O, because the conversion below is the single easiest
thing in this integration to get wrong and the consequence is silent.

**Deriv's limit_order amounts are money, not price levels.** `stop_loss` and
`take_profit` on a multiplier contract are amounts in account currency: how much
you are willing to lose or take, not the price at which to do it. A strategy
emitting a stop *distance in price* must therefore convert:

    stop_loss_amount = stake x multiplier x (stop_distance / entry_price)

`stop_distance / entry_price` is the move as a fraction of price;
`stake x multiplier` is the notional exposure; the product is what that move
costs. Passing a price level where an amount is expected is accepted by the API
and produces a position sized wrongly by orders of magnitude.

The take-profit is then `stop_loss_amount x reward_r`, which preserves the R
ratio the model was trained against -- the reward ratio is a property of each
setup here, not a constant, because the stop is structural.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Deriv's contract types for an up and a down multiplier.
MULT_UP = "MULTUP"
MULT_DOWN = "MULTDOWN"


class OrderMappingError(ValueError):
    """A signal cannot be expressed as a multiplier order."""


@dataclass(frozen=True)
class MultiplierOrder:
    symbol: str
    contract_type: str
    stake: float
    multiplier: int
    currency: str
    stop_loss_amount: float
    take_profit_amount: float

    def limit_order(self) -> dict[str, float]:
        return {
            "stop_loss": self.stop_loss_amount,
            "take_profit": self.take_profit_amount,
        }


def contract_type_for(side: str) -> str:
    """``MULTUP``/``MULTDOWN`` from a side.

    ``exit_policy.evaluate_exit`` takes ``side`` as a bare ``str`` "BUY" while
    the rest of the toolkit uses ``models.Side``; it works only because ``Side``
    is a str enum. Normalising at this boundary means the rest of the service
    never has to care which it was handed.
    """
    normalised = str(getattr(side, "value", side)).strip().lower()
    if normalised in {"buy", "long", "multup"}:
        return MULT_UP
    if normalised in {"sell", "short", "multdown"}:
        return MULT_DOWN
    raise OrderMappingError(f"cannot map side {side!r} to a multiplier contract")


def to_toolkit_side(side: str) -> str:
    """The uppercase ``"BUY"``/``"SELL"`` that ``evaluate_exit`` expects."""
    return "BUY" if contract_type_for(side) == MULT_UP else "SELL"


def stop_loss_amount(
    *, stake: float, multiplier: int, stop_distance: float, entry_price: float
) -> float:
    """Money at risk for a stop ``stop_distance`` away from ``entry_price``."""
    if stake <= 0:
        raise OrderMappingError(f"stake must be positive, got {stake}")
    if multiplier <= 0:
        raise OrderMappingError(f"multiplier must be positive, got {multiplier}")
    if entry_price <= 0:
        raise OrderMappingError(f"entry price must be positive, got {entry_price}")
    if stop_distance <= 0:
        raise OrderMappingError(f"stop distance must be positive, got {stop_distance}")
    return stake * multiplier * (stop_distance / entry_price)


def build_order(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    stop_distance: float,
    reward_r: float,
    stake: float,
    multiplier: int,
    currency: str,
) -> MultiplierOrder:
    if reward_r <= 0:
        raise OrderMappingError(f"reward ratio must be positive, got {reward_r}")

    risk = stop_loss_amount(
        stake=stake,
        multiplier=multiplier,
        stop_distance=stop_distance,
        entry_price=entry_price,
    )
    # Deriv takes money amounts to two decimals, and the ratio that matters is
    # the one between the two amounts it actually receives. So round the stop
    # first and derive the target from the rounded figure: rounding each
    # independently from the unrounded risk moves the served R ratio off the one
    # the model was trained against.
    stop = round(risk, 2)
    return MultiplierOrder(
        symbol=symbol,
        contract_type=contract_type_for(side),
        stake=stake,
        multiplier=multiplier,
        currency=currency,
        stop_loss_amount=stop,
        take_profit_amount=round(stop * reward_r, 2),
    )


def sized_stake(
    *,
    balance: float,
    configured_pct: float,
    p_win: float,
    reward_r: float,
    cost_r: float,
    minimum: float = 1.0,
) -> float:
    """Stake after the measured edge shrinks the configured risk.

    ``KellySizing.scale_against`` is quarter-Kelly with a 2% cap and only ever
    reduces: the configured percentage stays the authority on how much this
    account may risk, and a model barely above its cost hurdle simply gets less
    of it.
    """
    from forex_agent.strategy.expectancy import KellySizing

    fraction = KellySizing().scale_against(configured_pct, p_win, reward_r, cost_r)
    return max(minimum, round(balance * fraction, 2))
