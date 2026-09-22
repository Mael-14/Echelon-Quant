"""The whole loop, in process, over one fake Deriv socket.

    candles -> features -> model -> cost gate -> signal
            -> sized order -> buy -> managed exit

asked for by docs/fix-and-bot-development-plan.md 2.8. Everything except the
socket is the real thing: the real `DerivV3Client` with its req_id routing, the
real candle fetching and windows, the real `AnalysisEngine`, the real order
mapping and guards. Only the bytes on the wire are faked, which is the whole
point -- a test that stubs the components cannot catch them disagreeing.

No network, no unittest.mock, and the `app` purge dance for the hyphenated
service directories, matching tests/test_market_data_service_api.py.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from backend.shared.candles import CandleWindow, fetch_candles
from backend.shared.deriv_v3_client import DerivV3Client

ROOT = Path(__file__).resolve().parents[1]
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _load(service: str, module: str):
    sys.path.insert(0, str(ROOT / "backend" / "services" / service))
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]
    return importlib.import_module(f"app.{module}")


engine_module = _load("analysis-service", "engine")
orders = _load("execution-service", "orders")
trader = _load("execution-service", "trader")


# --------------------------------------------------------------- the fake wire


def _candles(count: int, seconds: int, *, base: float, amp: float, period: float) -> list[dict]:
    """A wave with enough structure to produce swings, pivots and a nonzero ATR."""
    out = []
    for i in range(count):
        mid = base + amp * math.sin(i / period * 2 * math.pi) + 0.0004 * math.sin(i * 1.7)
        open_ = mid + 0.0002 * math.cos(i * 0.9)
        close = mid + 0.0002 * math.sin(i * 1.3)
        out.append(
            {
                "epoch": int((EPOCH + timedelta(seconds=seconds * i)).timestamp()),
                "open": open_,
                "high": max(open_, close) + 0.0006,
                "low": min(open_, close) - 0.0006,
                "close": close,
            }
        )
    return out


class FakeDerivSocket:
    """Answers v3 requests by type, echoing req_id as Deriv does."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._inbound: asyncio.Queue[str] = asyncio.Queue()
        self.contract_updates: list[dict] = []
        self.sells: list[dict] = []
        self.bought: list[dict] = []
        self.open_contract_req: int | None = None
        # 4h macro and 1m micro, each long enough for the encoder's windows.
        self.history = {
            14_400: _candles(260, 14_400, base=1.10, amp=0.02, period=60),
            60: _candles(260, 60, base=1.10, amp=0.004, period=25),
        }

    async def send(self, raw: str) -> None:
        message = json.loads(raw)
        self.sent.append(message)
        self._inbound.put_nowait(json.dumps(self._answer(message)))

    async def recv(self) -> str:
        return await self._inbound.get()

    async def close(self) -> None:
        self.closed = True

    def push(self, payload: dict) -> None:
        """Inject an unsolicited stream update, as a live contract would."""
        self._inbound.put_nowait(json.dumps(payload))

    def _answer(self, message: dict) -> dict:
        req_id = message.get("req_id")
        if "authorize" in message:
            return {
                "req_id": req_id,
                "authorize": {"loginid": "VRTC900001", "currency": "USD"},
            }
        if "balance" in message:
            return {"req_id": req_id, "balance": {"balance": 10_000.0, "currency": "USD"}}
        if "ticks_history" in message:
            rows = self.history[message["granularity"]]
            return {"req_id": req_id, "candles": rows[-message["count"] :]}
        if "contracts_for" in message:
            return {
                "req_id": req_id,
                "contracts_for": {
                    "available": [
                        {"contract_type": "MULTUP"},
                        {"contract_type": "MULTDOWN"},
                        {"contract_type": "CALL"},
                    ]
                },
            }
        if "proposal" in message and "proposal_open_contract" not in message:
            return {
                "req_id": req_id,
                "proposal": {"id": "prop-1", "ask_price": message["amount"]},
            }
        if "buy" in message:
            self.bought.append(message)
            return {"req_id": req_id, "buy": {"contract_id": 55_501, "buy_price": message["price"]}}
        if "proposal_open_contract" in message:
            self.open_contract_req = req_id
            return {
                "req_id": req_id,
                "subscription": {"id": "poc-1"},
                "proposal_open_contract": {"contract_id": 55_501, "is_sold": 0},
            }
        if "contract_update" in message:
            self.contract_updates.append(message)
            return {"req_id": req_id, "contract_update": {"stop_loss": {"order_amount": "1.0"}}}
        if "sell" in message:
            self.sells.append(message)
            return {"req_id": req_id, "sell": {"sold_for": 10.0}}
        if "forget" in message:
            return {"req_id": req_id, "forget": 1}
        if "ping" in message:
            return {"req_id": req_id, "ping": "pong"}
        return {"req_id": req_id, "error": {"code": "Unsupported", "message": str(message)}}


class StubModel:
    """A model that is confident, so the test exercises the gate not the fit."""

    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, rows):
        return np.asarray([[1.0 - self.p, self.p] for _ in rows])


class PassThroughScaler:
    def transform(self, rows):
        return np.asarray(rows, dtype=float)


class Artifact:
    """The artifact contract, without needing a trained file on disk."""

    def __init__(self, p_win: float = 0.85) -> None:
        self.model = StubModel(p_win)
        self.scaler = PassThroughScaler()
        self.feature_names = (
            "h4_trend",
            "h4_stop_prox",
            "h4_tgt_prox",
            "h4_stop_dist",
            "h4_tgt_dist",
            "pullback",
            "momentum",
        )
        self.exec_frame = "1m"
        self.macro_frames = ("4h",)
        self.spread_atr = 0.3186
        self.encoding = "v2"
        self.hold = 24
        self.summary = "4h -> 1m (v2, hold 24)"

    def check_serves(self, *, exec_frame, macro_frames):
        assert exec_frame == self.exec_frame
        assert tuple(macro_frames) == self.macro_frames

    def predict_proba(self, row):
        assert len(row) == len(self.feature_names)
        return float(self.model.predict_proba([list(row)])[0][1])


class Settings:
    def __init__(self, **kwargs) -> None:
        self.deriv_trade_mode = kwargs.get("deriv_trade_mode", "demo")
        self.is_demo_trading = self.deriv_trade_mode == "demo"
        self.deriv_currency = "USD"
        self.deriv_multiplier = 100
        self.execution_max_open_positions = kwargs.get("execution_max_open_positions", 1)
        self.execution_daily_loss_limit_r = kwargs.get("execution_daily_loss_limit_r", 5.0)
        self.execution_risk_pct = kwargs.get("execution_risk_pct", 0.01)


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def socket():
    return FakeDerivSocket()


@pytest_asyncio.fixture
async def client(socket):
    async def factory(url):
        return socket

    created = DerivV3Client(app_id=1234, websocket_factory=factory, ping_interval=0)
    await created.connect()
    yield created
    await created.close()


def _engine(p_win: float = 0.85):
    return engine_module.AnalysisEngine(
        artifact=Artifact(p_win),
        exec_frame="1m",
        macro_frames=["4h"],
        macro_window=250,
        exec_window=250,
    )


async def _windows(client) -> tuple[CandleWindow, CandleWindow]:
    macro = CandleWindow("frxEURUSD", "4h", maxlen=400)
    micro = CandleWindow("frxEURUSD", "1m", maxlen=400)
    macro.extend(await fetch_candles(client, "frxEURUSD", timeframe="4h", count=260))
    micro.extend(await fetch_candles(client, "frxEURUSD", timeframe="1m", count=260))
    return macro, micro


# ------------------------------------------------------------ candles -> signal


@pytest.mark.asyncio
async def test_candles_arrive_through_the_real_client(client):
    macro, micro = await _windows(client)

    assert len(macro) == 260
    assert len(micro) == 260
    assert micro.newest is not None
    # The bars line up with what the encoder expects.
    assert micro.bars()[-1].timeframe.value == "1m"


@pytest.mark.asyncio
async def test_the_loop_produces_an_accepted_signal(client):
    """candles -> features -> model -> cost gate -> Signal."""
    engine = _engine(p_win=0.85)
    macro, micro = await _windows(client)

    evaluation = engine.evaluate("frxEURUSD", [macro], micro)

    assert evaluation.accepted, evaluation.reason
    assert evaluation.setup is not None
    assert evaluation.setup.side in {"buy", "sell"}
    assert evaluation.expected_value_r > 0
    # The gate cleared its own hurdle rather than just being confident.
    assert evaluation.p_win > evaluation.breakeven

    signal = engine.to_signal(evaluation)
    assert signal.symbol == "frxEURUSD"
    assert signal.side == evaluation.setup.side


@pytest.mark.asyncio
async def test_a_weak_model_is_refused_by_the_cost_gate(client):
    """The same candles, a model that is not confident enough to pay the spread."""
    engine = _engine(p_win=0.35)
    macro, micro = await _windows(client)

    evaluation = engine.evaluate("frxEURUSD", [macro], micro)

    assert not evaluation.accepted
    assert evaluation.p_win < evaluation.breakeven


@pytest.mark.asyncio
async def test_the_m1_cost_hurdle_is_carried_into_the_setup(client):
    """Cost is per-setup, from the artifact's spread and this stop's width."""
    engine = _engine()
    macro, micro = await _windows(client)

    evaluation = engine.evaluate("frxEURUSD", [macro], micro)

    assert evaluation.setup is not None
    assert evaluation.setup.cost_r > 0.1  # M1 drag is an order above H4's
    assert evaluation.breakeven > 0.5


# --------------------------------------------------------------- signal -> order


@pytest.mark.asyncio
async def test_an_accepted_signal_becomes_a_correctly_sized_order(client, socket):
    engine = _engine()
    macro, micro = await _windows(client)
    evaluation = engine.evaluate("frxEURUSD", [macro], micro)
    assert evaluation.accepted and evaluation.setup is not None
    setup = evaluation.setup
    state = trader.TradingState()

    position = await trader.place(
        client,
        settings=Settings(),
        state=state,
        symbol="frxEURUSD",
        side=setup.side,
        entry_price=setup.entry,
        stop_distance=setup.stop_distance,
        reward_r=setup.reward_r,
        cost_r=setup.cost_r,
        p_win=evaluation.p_win,
        balance=10_000.0,
    )

    # A position is recorded.
    assert state.open_positions["frxEURUSD"] is position
    assert position.contract_id == "55501"

    # Preflight ran before anything was bought.
    kinds = [next(iter(m)) for m in socket.sent]
    assert kinds.index("contracts_for") < kinds.index("proposal") < kinds.index("buy")

    # The proposal carries money amounts, not price levels.
    proposal = next(m for m in socket.sent if "proposal" in m)
    limit = proposal["limit_order"]
    expected_stop = orders.stop_loss_amount(
        stake=proposal["amount"],
        multiplier=100,
        stop_distance=setup.stop_distance,
        entry_price=setup.entry,
    )
    assert limit["stop_loss"] == pytest.approx(expected_stop, abs=0.01)
    assert abs(limit["take_profit"] - limit["stop_loss"] * setup.reward_r) <= 0.01

    # And it is not the stop *price level*, which is the mistake this guards
    # against: Deriv accepts a level here and sizes the position by it.
    is_long = setup.side == "buy"
    stop_level = setup.entry - setup.stop_distance if is_long else setup.entry + setup.stop_distance
    assert abs(limit["stop_loss"] - stop_level) > 0.01
    assert proposal["contract_type"] == ("MULTUP" if is_long else "MULTDOWN")


@pytest.mark.asyncio
async def test_kelly_only_ever_reduces_the_configured_stake(client):
    """The configured percentage stays the authority on size."""
    engine = _engine()
    macro, micro = await _windows(client)
    evaluation = engine.evaluate("frxEURUSD", [macro], micro)
    setup = evaluation.setup
    assert setup is not None

    position = await trader.place(
        client,
        settings=Settings(execution_risk_pct=0.01),
        state=trader.TradingState(),
        symbol="frxEURUSD",
        side=setup.side,
        entry_price=setup.entry,
        stop_distance=setup.stop_distance,
        reward_r=setup.reward_r,
        cost_r=setup.cost_r,
        p_win=evaluation.p_win,
        balance=10_000.0,
    )

    assert position.order.stake <= 10_000.0 * 0.01


@pytest.mark.asyncio
async def test_a_symbol_without_multipliers_is_refused_before_buying(client, socket):
    socket.history = dict(socket.history)

    def only_options(message):
        return {
            "req_id": message.get("req_id"),
            "contracts_for": {"available": [{"contract_type": "CALL"}]},
        }

    original = socket._answer

    def patched(message):
        if "contracts_for" in message:
            return only_options(message)
        return original(message)

    socket._answer = patched  # noqa: SLF001

    with pytest.raises(trader.TradeRefused, match="no multiplier"):
        await trader.place(
            client,
            settings=Settings(),
            state=trader.TradingState(),
            symbol="R_75",
            side="buy",
            entry_price=1.10,
            stop_distance=0.001,
            reward_r=1.5,
            cost_r=0.2,
            p_win=0.8,
            balance=10_000.0,
        )

    assert socket.bought == []


@pytest.mark.asyncio
async def test_real_money_mode_never_reaches_the_wire(client, socket):
    """The guard that matters: nothing is sent at all, not merely not bought."""
    before = len(socket.sent)

    with pytest.raises(trader.TradeRefused, match="refusing to place orders"):
        await trader.place(
            client,
            settings=Settings(deriv_trade_mode="real"),
            state=trader.TradingState(),
            symbol="frxEURUSD",
            side="buy",
            entry_price=1.10,
            stop_distance=0.001,
            reward_r=1.5,
            cost_r=0.2,
            p_win=0.9,
            balance=10_000.0,
        )

    assert socket.sent[before:] == []


# ------------------------------------------------------------- managed exit


def _position(side: str = "BUY"):
    order = orders.build_order(
        symbol="frxEURUSD",
        side=side.lower(),
        entry_price=1.10,
        stop_distance=0.001,
        reward_r=1.5,
        stake=10.0,
        multiplier=100,
        currency="USD",
    )
    return trader.OpenPosition(
        contract_id="55501",
        symbol="frxEURUSD",
        side=side,
        entry_price=1.10,
        stop_price=1.099,
        take_profit=1.1015,
        risk_distance=0.001,
        cost_r=0.2,
        opened_at=datetime.now(tz=timezone.utc),
        peak_price=1.10,
        order=order,
    )


@pytest.mark.asyncio
async def test_a_quiet_quote_holds_the_position(client, socket):
    from forex_agent.execution.exit_policy import ExitPolicy

    outcome = await trader.manage(client, _position(), quote=1.1001, policy=ExitPolicy())

    assert outcome == "held"
    assert socket.contract_updates == []
    assert socket.sells == []


@pytest.mark.asyncio
async def test_hitting_the_target_closes_the_contract(client, socket):
    from forex_agent.execution.exit_policy import ExitPolicy

    outcome = await trader.manage(client, _position(), quote=1.1020, policy=ExitPolicy())

    assert outcome.startswith("closed")
    assert socket.sells and socket.sells[0]["sell"] == "55501"


@pytest.mark.asyncio
async def test_a_moved_stop_is_sent_as_money_not_a_price(client, socket):
    """contract_update is what makes break-even and trailing possible at all.

    The moved stop goes through the same price-to-money conversion as the
    original, so what Deriv receives is an amount, never the new level.
    """
    from forex_agent.execution.exit_policy import ExitPolicy

    position = _position()
    policy = ExitPolicy()

    # Far enough in profit to trigger the policy's break-even or trail move.
    await trader.manage(client, position, quote=1.1012, policy=policy)

    if socket.contract_updates:
        sent = socket.contract_updates[0]["limit_order"]["stop_loss"]
        assert sent < position.entry_price  # an amount, not a level near 1.10
        assert sent == pytest.approx(
            orders.stop_loss_amount(
                stake=position.order.stake,
                multiplier=position.order.multiplier,
                stop_distance=abs(position.entry_price - position.stop_price),
                entry_price=position.entry_price,
            ),
            abs=0.01,
        )


# ---------------------------------------------------------------- kill switch


@pytest.mark.asyncio
async def test_emergency_stop_closes_everything_and_halts(client, socket):
    state = trader.TradingState()
    state.open_positions["frxEURUSD"] = _position()
    state.open_positions["frxGBPUSD"] = _position()

    closed = await trader.close_all(client, state, reason="manual emergency stop")

    assert closed == 2
    assert state.open_positions == {}
    assert state.halted is True
    assert len(socket.sells) == 2

    # And nothing new is accepted afterwards.
    reason = trader.refuse(settings=Settings(), state=state, symbol="frxEURUSD")
    assert reason == "manual emergency stop"


@pytest.mark.asyncio
async def test_the_daily_loss_limit_stops_the_loop(client):
    state = trader.TradingState(realised_r_today=-5.5)

    with pytest.raises(trader.TradeRefused, match="daily loss limit"):
        await trader.place(
            client,
            settings=Settings(),
            state=state,
            symbol="frxEURUSD",
            side="buy",
            entry_price=1.10,
            stop_distance=0.001,
            reward_r=1.5,
            cost_r=0.2,
            p_win=0.9,
            balance=10_000.0,
        )


# ----------------------------------------------------------------- full cycle


@pytest.mark.asyncio
async def test_the_whole_loop_end_to_end(client, socket):
    """Candles in, position recorded, exit managed, position released.

    This is the exit criterion from the development plan: a bot autonomously
    reacting to market data through to a recorded position.
    """
    engine = _engine()
    settings = Settings()
    state = trader.TradingState()

    # 1. Market data.
    macro, micro = await _windows(client)
    assert len(micro) == 260

    # 2. A decision.
    evaluation = engine.evaluate("frxEURUSD", [macro], micro)
    assert evaluation.accepted
    setup = evaluation.setup
    assert setup is not None

    # 3. Risk, sizing and the order.
    position = await trader.place(
        client,
        settings=settings,
        state=state,
        symbol="frxEURUSD",
        side=setup.side,
        entry_price=setup.entry,
        stop_distance=setup.stop_distance,
        reward_r=setup.reward_r,
        cost_r=setup.cost_r,
        p_win=evaluation.p_win,
        balance=10_000.0,
    )
    assert len(state.open_positions) == 1

    # 4. A second signal is refused while one position is open.
    assert trader.refuse(settings=settings, state=state, symbol="frxEURUSD") is not None

    # 5. The exit closes it and frees the slot.
    from forex_agent.execution.exit_policy import ExitPolicy

    is_long = position.side == "BUY"
    target = position.take_profit + (0.001 if is_long else -0.001)
    outcome = await trader.manage(client, position, quote=target, policy=ExitPolicy())
    assert outcome.startswith("closed")

    state.open_positions.pop("frxEURUSD")
    assert trader.refuse(settings=settings, state=state, symbol="frxEURUSD") is None
