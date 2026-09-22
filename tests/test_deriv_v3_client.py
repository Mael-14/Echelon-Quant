"""The v3 client exists for one reason: req_id correlation. Test that hardest.

Fakes are hand-rolled and the socket is never real, matching
tests/test_deriv_client.py. The fake here has to do more than that one's,
because this client reads in a background task rather than on demand: it blocks
on ``recv`` until a test pushes, instead of raising on an empty queue.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from backend.shared.deriv_v3_client import (
    DEFAULT_V3_WS_URL,
    DerivV3Client,
    DerivV3Error,
)


class FakeWebSocket:
    """A socket whose inbound side the test drives message by message."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._inbound: asyncio.Queue[str | Exception] = asyncio.Queue()

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        item = await self._inbound.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True

    # -- test-side helpers ---------------------------------------------------

    def push(self, payload: dict) -> None:
        self._inbound.put_nowait(json.dumps(payload))

    def drop(self) -> None:
        self._inbound.put_nowait(ConnectionResetError("socket closed"))

    async def next_sent(self, index: int) -> dict:
        """Wait until the client has sent at least ``index + 1`` messages.

        Yields with ``sleep(0)`` first, which is enough for a message the client
        sends on the current turn, then falls back to real sleeps -- the ping
        keepalive is on a timer, and ``sleep(0)`` never advances a timer.
        """
        for attempt in range(120):
            if len(self.sent) > index:
                return self.sent[index]
            await asyncio.sleep(0 if attempt < 10 else 0.01)
        raise AssertionError(f"client never sent message {index}")


def _client(socket: FakeWebSocket, **kwargs) -> DerivV3Client:
    async def factory(url: str) -> FakeWebSocket:
        socket.url = url  # type: ignore[attr-defined]
        return socket

    kwargs.setdefault("ping_interval", 0)  # no keepalive noise unless asked for
    return DerivV3Client(app_id=1234, websocket_factory=factory, **kwargs)


@pytest.mark.asyncio
async def test_app_id_is_attached_to_the_url() -> None:
    socket = FakeWebSocket()
    client = _client(socket)

    await client.connect()

    assert socket.url == f"{DEFAULT_V3_WS_URL}?app_id=1234"
    await client.close()


@pytest.mark.asyncio
async def test_app_id_appends_when_the_url_already_has_a_query() -> None:
    socket = FakeWebSocket()

    async def factory(url: str) -> FakeWebSocket:
        socket.url = url  # type: ignore[attr-defined]
        return socket

    client = DerivV3Client(
        app_id=7,
        ws_url="wss://example.test/v3?lang=EN",
        websocket_factory=factory,
        ping_interval=0,
    )
    await client.connect()

    assert socket.url == "wss://example.test/v3?lang=EN&app_id=7"
    await client.close()


@pytest.mark.asyncio
async def test_authorize_sends_the_token_and_marks_the_session() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    task = asyncio.ensure_future(client.authorize("demo-token"))
    sent = await socket.next_sent(0)
    socket.push(
        {
            "msg_type": "authorize",
            "req_id": sent["req_id"],
            "authorize": {"loginid": "VRTC1234", "currency": "USD"},
        }
    )

    result = await task

    assert sent["authorize"] == "demo-token"
    assert result["loginid"] == "VRTC1234"
    assert client.authorized is True
    await client.close()


@pytest.mark.asyncio
async def test_responses_are_matched_by_req_id_not_arrival_order() -> None:
    """The whole reason this client exists.

    Two calls are in flight and Deriv answers the second one first. A client
    that returned "whatever arrived next" would hand the balance reply to the
    portfolio caller.
    """
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    first = asyncio.ensure_future(client.balance())
    second = asyncio.ensure_future(client.portfolio())
    balance_req = (await socket.next_sent(0))["req_id"]
    portfolio_req = (await socket.next_sent(1))["req_id"]

    # Answer out of order, deliberately.
    socket.push(
        {"msg_type": "portfolio", "req_id": portfolio_req, "portfolio": {"contracts": ["p"]}}
    )
    socket.push({"msg_type": "balance", "req_id": balance_req, "balance": {"balance": 10000.0}})

    assert (await first)["balance"] == 10000.0
    assert (await second)["contracts"] == ["p"]
    await client.close()


@pytest.mark.asyncio
async def test_a_request_resolves_while_a_subscription_is_streaming() -> None:
    """Interleaving is the case the Options client structurally cannot serve."""
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    ticks = asyncio.ensure_future(client.ticks("frxEURUSD"))
    tick_req = (await socket.next_sent(0))["req_id"]
    socket.push(
        {
            "msg_type": "tick",
            "req_id": tick_req,
            "subscription": {"id": "sub-1"},
            "tick": {"quote": 1.10},
        }
    )
    subscription = await ticks

    updates = subscription.__aiter__()
    balance = asyncio.ensure_future(client.balance())
    balance_req = (await socket.next_sent(1))["req_id"]

    # A tick lands between the request and its answer.
    socket.push({"msg_type": "tick", "req_id": tick_req, "tick": {"quote": 1.11}})
    socket.push({"msg_type": "balance", "req_id": balance_req, "balance": {"balance": 9000.0}})

    assert (await balance)["balance"] == 9000.0
    assert (await updates.__anext__())["tick"]["quote"] == 1.11
    assert subscription.id == "sub-1"
    await client.close()


@pytest.mark.asyncio
async def test_an_api_error_raises_with_deriv_s_code() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    task = asyncio.ensure_future(client.buy("proposal-1", price=10.0))
    req_id = (await socket.next_sent(0))["req_id"]
    socket.push(
        {
            "req_id": req_id,
            "error": {"code": "InsufficientBalance", "message": "Not enough balance"},
        }
    )

    with pytest.raises(DerivV3Error) as caught:
        await task

    assert caught.value.code == "InsufficientBalance"
    assert "Not enough balance" in str(caught.value)
    await client.close()


@pytest.mark.asyncio
async def test_an_error_on_a_stream_surfaces_to_the_iterator() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    task = asyncio.ensure_future(client.proposal_open_contract(4242))
    req_id = (await socket.next_sent(0))["req_id"]
    socket.push(
        {
            "req_id": req_id,
            "subscription": {"id": "poc-1"},
            "proposal_open_contract": {"contract_id": 4242},
        }
    )
    subscription = await task

    socket.push({"req_id": req_id, "error": {"code": "GetProposalFailure", "message": "gone"}})

    with pytest.raises(DerivV3Error):
        await subscription.__aiter__().__anext__()
    await client.close()


@pytest.mark.asyncio
async def test_a_dropped_socket_fails_everything_in_flight() -> None:
    """Otherwise a caller waits out the full request timeout for a dead socket."""
    socket = FakeWebSocket()
    client = _client(socket, request_timeout=30.0)
    await client.connect()

    pending = asyncio.ensure_future(client.balance())
    await socket.next_sent(0)
    socket.drop()

    with pytest.raises(DerivV3Error):
        await asyncio.wait_for(pending, timeout=2.0)
    await client.close()


@pytest.mark.asyncio
async def test_a_silent_deriv_times_out_rather_than_hanging() -> None:
    socket = FakeWebSocket()
    client = _client(socket, request_timeout=0.05)
    await client.connect()

    with pytest.raises(DerivV3Error) as caught:
        await client.balance()

    assert "balance" in str(caught.value)
    await client.close()


@pytest.mark.asyncio
async def test_ticks_history_asks_for_candles_and_returns_them() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    task = asyncio.ensure_future(client.ticks_history("frxEURUSD", granularity=60, count=2))
    sent = await socket.next_sent(0)
    socket.push(
        {
            "msg_type": "candles",
            "req_id": sent["req_id"],
            "candles": [{"epoch": 1, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05}],
        }
    )

    candles = await task

    assert sent["style"] == "candles"
    assert sent["granularity"] == 60
    assert sent["ticks_history"] == "frxEURUSD"
    assert len(candles) == 1
    await client.close()


@pytest.mark.asyncio
async def test_proposal_carries_the_multiplier_and_limit_order() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    task = asyncio.ensure_future(
        client.proposal(
            symbol="frxEURUSD",
            contract_type="MULTUP",
            amount=10.0,
            currency="USD",
            multiplier=100,
            limit_order={"stop_loss": 2.5, "take_profit": 3.75},
        )
    )
    sent = await socket.next_sent(0)
    socket.push({"req_id": sent["req_id"], "proposal": {"id": "prop-1", "ask_price": 10.0}})

    proposal = await task

    assert sent["contract_type"] == "MULTUP"
    assert sent["multiplier"] == 100
    assert sent["basis"] == "stake"
    # Amounts in account currency, not price levels -- the easiest thing here
    # to get wrong, so the shape is pinned.
    assert sent["limit_order"] == {"stop_loss": 2.5, "take_profit": 3.75}
    assert proposal["id"] == "prop-1"
    await client.close()


@pytest.mark.asyncio
async def test_contract_update_sends_only_the_fields_given() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    task = asyncio.ensure_future(client.contract_update(99, stop_loss=1.25))
    sent = await socket.next_sent(0)
    socket.push({"req_id": sent["req_id"], "contract_update": {"stop_loss": {"value": "1.25"}}})
    await task

    # take_profit is absent rather than null: sending null would clear it.
    assert sent["limit_order"] == {"stop_loss": 1.25}
    assert sent["contract_id"] == 99
    await client.close()


@pytest.mark.asyncio
async def test_contract_update_needs_at_least_one_field() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    with pytest.raises(ValueError):
        await client.contract_update(99)
    await client.close()


@pytest.mark.asyncio
async def test_cancelling_a_subscription_forgets_it_at_deriv() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    task = asyncio.ensure_future(client.ticks("frxEURUSD"))
    req_id = (await socket.next_sent(0))["req_id"]
    socket.push({"req_id": req_id, "subscription": {"id": "sub-9"}, "tick": {"quote": 1.0}})
    subscription = await task

    cancelling = asyncio.ensure_future(subscription.cancel())
    forget = await socket.next_sent(1)
    socket.push({"req_id": forget["req_id"], "forget": 1})
    await cancelling

    assert forget["forget"] == "sub-9"
    # The stream terminates rather than hanging on the next update.
    with pytest.raises(StopAsyncIteration):
        await subscription.__aiter__().__anext__()
    await client.close()


@pytest.mark.asyncio
async def test_keepalive_pings_without_being_asked() -> None:
    """Deriv drops an idle v3 session after two minutes."""
    socket = FakeWebSocket()
    client = _client(socket, ping_interval=0.01)
    await client.connect()

    sent = await socket.next_sent(0)

    assert sent["ping"] == 1
    await client.close()


@pytest.mark.asyncio
async def test_close_is_safe_to_call_twice() -> None:
    socket = FakeWebSocket()
    client = _client(socket)
    await client.connect()

    await client.close()
    await client.close()

    assert socket.closed is True
    assert client.connected is False
