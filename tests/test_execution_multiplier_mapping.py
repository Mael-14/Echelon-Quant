"""The stop/take-profit money conversion, and the guards around it.

Deriv's limit_order amounts are money in account currency, not price levels.
Getting that wrong is accepted by the API and sizes every position wrongly, so
it is pinned here with worked numbers rather than round trips.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _load(module: str):
    """Import a module from the hyphenated execution-service directory.

    The service directory name is not a valid package name, and every service
    exposes a top-level package called `app`, so cached `app.*` modules are
    purged first: module imports are cached by name regardless of sys.path
    order, and without this a test picks up whichever service was imported
    first. Same dance as tests/test_market_data_service_api.py.
    """
    import importlib
    import sys
    from pathlib import Path

    service_dir = Path(__file__).resolve().parents[1] / "backend" / "services" / "execution-service"
    sys.path.insert(0, str(service_dir))
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    return importlib.import_module(f"app.{module}")


orders = _load("orders")
trader = _load("trader")

MULT_UP = orders.MULT_UP
MULT_DOWN = orders.MULT_DOWN
OrderMappingError = orders.OrderMappingError
build_order = orders.build_order
contract_type_for = orders.contract_type_for
stop_loss_amount = orders.stop_loss_amount
to_toolkit_side = orders.to_toolkit_side
TradingState = trader.TradingState
refuse = trader.refuse


class Settings:
    """Only the fields the guards read."""

    def __init__(self, **kwargs) -> None:
        self.deriv_trade_mode = kwargs.get("deriv_trade_mode", "demo")
        self.is_demo_trading = self.deriv_trade_mode == "demo"
        self.execution_max_open_positions = kwargs.get("execution_max_open_positions", 1)
        self.execution_daily_loss_limit_r = kwargs.get("execution_daily_loss_limit_r", 5.0)


# ------------------------------------------------------------------ direction


@pytest.mark.parametrize("side", ["buy", "BUY", "long", "Buy"])
def test_a_long_becomes_multup(side):
    assert contract_type_for(side) == MULT_UP


@pytest.mark.parametrize("side", ["sell", "SELL", "short"])
def test_a_short_becomes_multdown(side):
    assert contract_type_for(side) == MULT_DOWN


def test_an_unknown_side_is_refused():
    with pytest.raises(OrderMappingError):
        contract_type_for("hold")


def test_side_normalises_for_the_exit_policy():
    """evaluate_exit takes a bare "BUY" while the toolkit elsewhere uses Side."""
    assert to_toolkit_side("buy") == "BUY"
    assert to_toolkit_side("sell") == "SELL"


# ------------------------------------------------------------ the conversion


def test_stop_loss_is_money_not_a_price_level():
    """stake x multiplier x (stop_distance / entry_price).

    A 0.001 stop on a 1.10 entry is a 0.0909% move. On 10 USD at 100x the
    notional is 1000 USD, so the stop is worth 0.909 USD -- not 1.099, which is
    what passing the stop *price* would have sent.
    """
    amount = stop_loss_amount(stake=10.0, multiplier=100, stop_distance=0.001, entry_price=1.10)

    assert amount == pytest.approx(0.909090, rel=1e-4)


def test_take_profit_preserves_the_r_ratio():
    """The reward ratio is a property of each setup, so it must survive."""
    order = build_order(
        symbol="frxEURUSD",
        side="buy",
        entry_price=1.10,
        stop_distance=0.001,
        reward_r=1.5,
        stake=10.0,
        multiplier=100,
        currency="USD",
    )

    # Both amounts are derived from the same rounded stop, so the ratio Deriv
    # receives is the intended one to within the 0.01 money granularity.
    assert abs(order.take_profit_amount - order.stop_loss_amount * 1.5) <= 0.01
    assert order.contract_type == MULT_UP


def test_a_bigger_multiplier_risks_proportionally_more():
    ten = stop_loss_amount(stake=10.0, multiplier=10, stop_distance=0.001, entry_price=1.10)
    hundred = stop_loss_amount(stake=10.0, multiplier=100, stop_distance=0.001, entry_price=1.10)

    assert hundred == pytest.approx(ten * 10, rel=1e-9)


def test_limit_order_uses_deriv_s_field_names():
    order = build_order(
        symbol="frxEURUSD",
        side="sell",
        entry_price=1.10,
        stop_distance=0.002,
        reward_r=2.0,
        stake=25.0,
        multiplier=50,
        currency="USD",
    )

    assert set(order.limit_order()) == {"stop_loss", "take_profit"}
    assert order.contract_type == MULT_DOWN


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"stake": 0.0}, "stake"),
        ({"multiplier": 0}, "multiplier"),
        ({"entry_price": 0.0}, "entry price"),
        ({"stop_distance": 0.0}, "stop distance"),
    ],
)
def test_degenerate_geometry_is_refused_not_rounded(kwargs, match):
    base = {"stake": 10.0, "multiplier": 100, "stop_distance": 0.001, "entry_price": 1.10}
    with pytest.raises(OrderMappingError, match=match):
        stop_loss_amount(**{**base, **kwargs})


def test_a_negative_reward_is_refused():
    with pytest.raises(OrderMappingError, match="reward"):
        build_order(
            symbol="frxEURUSD",
            side="buy",
            entry_price=1.10,
            stop_distance=0.001,
            reward_r=-1.0,
            stake=10.0,
            multiplier=100,
            currency="USD",
        )


# ---------------------------------------------------------------- the guards


def test_real_money_mode_refuses_everything():
    """The guard that matters most, checked before any other."""
    reason = refuse(settings=Settings(deriv_trade_mode="real"), state=TradingState(), symbol="X")

    assert reason is not None
    assert "refusing to place orders" in reason


def test_a_clean_state_permits_the_trade():
    assert refuse(settings=Settings(), state=TradingState(), symbol="frxEURUSD") is None


def test_the_position_cap_is_enforced():
    state = TradingState(open_positions={"frxGBPUSD": object()})  # type: ignore[dict-item]

    reason = refuse(settings=Settings(), state=state, symbol="frxEURUSD")

    assert reason is not None and "cap 1" in reason


def test_the_same_symbol_is_not_doubled_up():
    state = TradingState(open_positions={"frxEURUSD": object()})  # type: ignore[dict-item]
    settings = Settings(execution_max_open_positions=5)

    assert refuse(settings=settings, state=state, symbol="frxEURUSD") == "already holding frxEURUSD"


def test_the_daily_loss_limit_halts_trading():
    state = TradingState(realised_r_today=-5.0)

    reason = refuse(settings=Settings(), state=state, symbol="frxEURUSD")

    assert reason is not None and "daily loss limit" in reason


def test_a_manual_halt_is_respected():
    state = TradingState(halted=True, halt_reason="emergency stop")

    assert refuse(settings=Settings(), state=state, symbol="frxEURUSD") == "emergency stop"


def test_a_new_day_clears_a_loss_halt_but_not_a_manual_one():
    state = TradingState(day="2026-09-21", realised_r_today=-9.0)
    state.halted = True
    state.halt_reason = "daily loss limit reached"

    state.roll_day(datetime(2026, 9, 22, tzinfo=timezone.utc))

    assert state.realised_r_today == 0.0
    assert state.halted is False

    manual = TradingState(day="2026-09-21")
    manual.halted = True
    manual.halt_reason = "emergency stop"
    manual.roll_day(datetime(2026, 9, 22, tzinfo=timezone.utc))
    assert manual.halted is True


def test_rolling_within_the_same_day_keeps_the_running_total():
    state = TradingState(day="2026-09-22", realised_r_today=-2.0)

    state.roll_day(datetime(2026, 9, 22, 23, 59, tzinfo=timezone.utc) - timedelta(hours=1))

    assert state.realised_r_today == -2.0
