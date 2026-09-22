"""Consuming signals and placing managed multiplier trades.

The flow for one accepted signal:

    contracts_for (preflight) -> proposal -> buy
        -> subscribe proposal_open_contract
        -> evaluate_exit on each update
        -> contract_update to move the stop, or sell to close

**Guards come first, every time.** `refuse` runs before anything reaches Deriv,
and the demo check is read from one shared property rather than re-compared at
each call site -- two sites spelling that check differently is exactly how such
a guard gets lost.

For v1 the risk check is folded in here rather than standing up risk-service.
The `RiskDecision` schema exists for when that is worth splitting out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.shared.deriv_v3_client import DerivV3Client, DerivV3Error

from .orders import MultiplierOrder, build_order, sized_stake, to_toolkit_side

log = logging.getLogger("execution-service")


class TradeRefused(RuntimeError):
    """A guard stopped this trade. The message is the reason."""


@dataclass
class OpenPosition:
    contract_id: str
    symbol: str
    side: str
    entry_price: float
    stop_price: float
    take_profit: float
    risk_distance: float
    cost_r: float
    opened_at: datetime
    peak_price: float
    order: MultiplierOrder


@dataclass
class TradingState:
    """What the guards need to know, held in one place so they agree."""

    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    realised_r_today: float = 0.0
    day: str = ""
    halted: bool = False
    halt_reason: str = ""

    def __post_init__(self) -> None:
        # Stamp the day on construction. Without this a state built with losses
        # already recorded has an empty day, so the first roll_day sees a day
        # change and zeroes the running total -- clearing the loss limit at
        # exactly the moment it should be enforcing it.
        if not self.day:
            self.day = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    def roll_day(self, now: datetime | None = None) -> None:
        today = (now or datetime.now(tz=timezone.utc)).strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            self.realised_r_today = 0.0
            # A new day clears a loss-limit halt but not a manual one.
            if self.halt_reason.startswith("daily loss"):
                self.halted = False
                self.halt_reason = ""


def refuse(
    *,
    settings: Any,
    state: TradingState,
    symbol: str,
) -> str | None:
    """The reason not to trade, or None. Checked in order of severity."""
    if not settings.is_demo_trading:
        return f"DERIV_TRADE_MODE is {settings.deriv_trade_mode!r}, refusing to place orders"
    if state.halted:
        return state.halt_reason or "trading is halted"
    if len(state.open_positions) >= settings.execution_max_open_positions:
        return (
            f"{len(state.open_positions)} positions already open "
            f"(cap {settings.execution_max_open_positions})"
        )
    if state.realised_r_today <= -abs(settings.execution_daily_loss_limit_r):
        return (
            f"daily loss limit reached: {state.realised_r_today:+.2f}R against a "
            f"{-abs(settings.execution_daily_loss_limit_r):+.2f}R limit"
        )
    if symbol in state.open_positions:
        return f"already holding {symbol}"
    return None


async def preflight(client: DerivV3Client, symbol: str, currency: str) -> None:
    """Confirm Deriv will sell a multiplier on this symbol at all.

    Multipliers are not offered on every instrument in every jurisdiction.
    Finding that out here costs one read; finding it out from a rejected buy
    costs a confusing failure in the middle of a trading loop.
    """
    offered = await client.contracts_for(symbol, currency=currency)
    available = offered.get("available") or []
    types = {str(entry.get("contract_type")) for entry in available}
    if not any(t.startswith("MULT") for t in types):
        raise TradeRefused(f"Deriv offers no multiplier contracts on {symbol}")


async def place(
    client: DerivV3Client,
    *,
    settings: Any,
    state: TradingState,
    symbol: str,
    side: str,
    entry_price: float,
    stop_distance: float,
    reward_r: float,
    cost_r: float,
    p_win: float,
    balance: float,
) -> OpenPosition:
    """Size, preflight, price and buy. Raises TradeRefused rather than trading."""
    state.roll_day()
    reason = refuse(settings=settings, state=state, symbol=symbol)
    if reason:
        raise TradeRefused(reason)

    await preflight(client, symbol, settings.deriv_currency)

    stake = sized_stake(
        balance=balance,
        configured_pct=settings.execution_risk_pct,
        p_win=p_win,
        reward_r=reward_r,
        cost_r=cost_r,
    )
    order = build_order(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        stop_distance=stop_distance,
        reward_r=reward_r,
        stake=stake,
        multiplier=settings.deriv_multiplier,
        currency=settings.deriv_currency,
    )

    priced = await client.proposal(
        symbol=order.symbol,
        contract_type=order.contract_type,
        amount=order.stake,
        currency=order.currency,
        multiplier=order.multiplier,
        limit_order=order.limit_order(),
    )
    proposal_id = priced.get("id")
    if not proposal_id:
        raise TradeRefused(f"Deriv returned no proposal id for {symbol}")

    bought = await client.buy(str(proposal_id), price=float(priced.get("ask_price", order.stake)))
    contract_id = bought.get("contract_id")
    if not contract_id:
        raise TradeRefused(f"Deriv returned no contract id for {symbol}")

    is_long = order.contract_type == "MULTUP"
    stop_price = entry_price - stop_distance if is_long else entry_price + stop_distance
    target = (
        entry_price + reward_r * stop_distance
        if is_long
        else entry_price - reward_r * stop_distance
    )
    position = OpenPosition(
        contract_id=str(contract_id),
        symbol=symbol,
        side=to_toolkit_side(side),
        entry_price=entry_price,
        stop_price=stop_price,
        take_profit=target,
        risk_distance=stop_distance,
        cost_r=cost_r,
        opened_at=datetime.now(tz=timezone.utc),
        peak_price=entry_price,
        order=order,
    )
    state.open_positions[symbol] = position
    return position


async def manage(
    client: DerivV3Client,
    position: OpenPosition,
    *,
    quote: float,
    policy: Any,
    now: datetime | None = None,
) -> str:
    """Apply the exit policy to one quote. Returns what was done.

    ``evaluate_exit`` tightens the stop *before* testing the levels, so a tick
    that both extends the trail and breaches it closes on that tick rather than
    one tick later at a worse price.
    """
    from forex_agent.execution.exit_policy import evaluate_exit

    moment = now or datetime.now(tz=timezone.utc)
    if position.side == "BUY":
        position.peak_price = max(position.peak_price, quote)
    else:
        position.peak_price = min(position.peak_price, quote)

    check = evaluate_exit(
        side=position.side,
        entry_price=position.entry_price,
        stop_price=position.stop_price,
        take_profit=position.take_profit,
        risk_distance=position.risk_distance,
        peak_price=position.peak_price,
        opened_at=position.opened_at,
        quote=quote,
        now=moment,
        policy=policy,
        cost_r=position.cost_r,
    )

    # ExitCheck's fields are close_reason / stop_price / peak_price. Read them
    # directly: a getattr with a default here silently swallows a renamed field
    # and the position simply never closes.
    if check.peak_price is not None:
        position.peak_price = float(check.peak_price)

    if check.close_reason:
        await client.sell(position.contract_id, price=0)
        return f"closed: {check.close_reason}"

    new_stop = check.stop_price
    if new_stop is not None and new_stop != position.stop_price:
        # Deriv wants the stop as an amount in account currency, so the moved
        # stop is re-derived through the same conversion as the original.
        from .orders import stop_loss_amount

        distance = abs(position.entry_price - float(new_stop))
        if distance > 0:
            await client.contract_update(
                position.contract_id,
                stop_loss=round(
                    stop_loss_amount(
                        stake=position.order.stake,
                        multiplier=position.order.multiplier,
                        stop_distance=distance,
                        entry_price=position.entry_price,
                    ),
                    2,
                ),
            )
        position.stop_price = float(new_stop)
        return "stop moved"

    return "held"


async def close_all(client: DerivV3Client, state: TradingState, *, reason: str) -> int:
    """Emergency stop: halt, then close everything open."""
    state.halted = True
    state.halt_reason = reason
    closed = 0
    for symbol, position in list(state.open_positions.items()):
        try:
            await client.sell(position.contract_id, price=0)
            closed += 1
        except DerivV3Error:
            log.exception("Failed to close %s during emergency stop", symbol)
        finally:
            state.open_positions.pop(symbol, None)
    return closed
