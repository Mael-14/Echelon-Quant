from __future__ import annotations

import pytest

from backend.shared.deriv_client import DerivClient, DerivClientError


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []
        self.incoming_messages: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent_messages.append(message)

    async def recv(self) -> str:
        if not self.incoming_messages:
            raise RuntimeError("No queued message")
        return self.incoming_messages.pop(0)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_connect_uses_public_socket_by_default() -> None:
    urls: list[str] = []
    websocket = FakeWebSocket()

    async def websocket_factory(url: str) -> FakeWebSocket:
        urls.append(url)
        return websocket

    client = DerivClient(websocket_factory=websocket_factory)

    connected = await client.connect()

    assert connected is websocket
    assert urls == ["wss://api.derivws.com/trading/v1/options/ws/public"]
    assert client.connected is True


@pytest.mark.asyncio
async def test_authenticate_fetches_fresh_otp_and_connects() -> None:
    otp_calls: list[tuple[str, str, int | None]] = []
    urls: list[str] = []
    websocket = FakeWebSocket()

    async def otp_resolver(account_id: str, token: str, app_id: int | None) -> str:
        otp_calls.append((account_id, token, app_id))
        return f"wss://example.test/{account_id}/otp-1"

    async def websocket_factory(url: str) -> FakeWebSocket:
        urls.append(url)
        return websocket

    client = DerivClient(websocket_factory=websocket_factory, otp_resolver=otp_resolver)

    connected = await client.authenticate("token-123", "CR123456", 777)

    assert connected is websocket
    assert otp_calls == [("CR123456", "token-123", 777)]
    assert urls == ["wss://example.test/CR123456/otp-1"]
    assert client.authenticated is True
    assert client.connection_generation == 1


@pytest.mark.asyncio
async def test_subscribe_sends_payload_and_receives_first_update() -> None:
    websocket = FakeWebSocket()
    websocket.incoming_messages.append('{"msg_type":"ticks","subscription":{"id":"sub-1"}}')

    async def websocket_factory(url: str) -> FakeWebSocket:
        return websocket

    client = DerivClient(websocket_factory=websocket_factory)
    await client.connect()

    response = await client.subscribe({"ticks": "R_100"})

    assert response["subscription"]["id"] == "sub-1"
    assert websocket.sent_messages == ['{"ticks": "R_100", "subscribe": 1}']


@pytest.mark.asyncio
async def test_receive_decodes_json_payloads() -> None:
    websocket = FakeWebSocket()
    websocket.incoming_messages.append('{"ping":"pong"}')

    async def websocket_factory(url: str) -> FakeWebSocket:
        return websocket

    client = DerivClient(websocket_factory=websocket_factory)
    await client.connect()

    message = await client.receive()

    assert message == {"ping": "pong"}


@pytest.mark.asyncio
async def test_reconnect_uses_fresh_authenticated_url() -> None:
    otp_calls: list[int] = []
    urls: list[str] = []
    sockets = [FakeWebSocket(), FakeWebSocket()]

    async def otp_resolver(account_id: str, token: str, app_id: int | None) -> str:
        otp_calls.append(len(otp_calls) + 1)
        return f"wss://example.test/{account_id}/otp-{len(otp_calls)}"

    async def websocket_factory(url: str) -> FakeWebSocket:
        urls.append(url)
        return sockets[len(urls) - 1]

    client = DerivClient(websocket_factory=websocket_factory, otp_resolver=otp_resolver)
    await client.authenticate("token-123", "CR123456", 777)

    first_socket = client._websocket
    await client.reconnect()

    assert otp_calls == [1, 2]
    assert urls == ["wss://example.test/CR123456/otp-1", "wss://example.test/CR123456/otp-2"]
    assert first_socket.closed is True
    assert client._websocket is sockets[1]
    assert client.connection_generation == 2


@pytest.mark.asyncio
async def test_disconnect_raises_when_not_connected() -> None:
    client = DerivClient()

    with pytest.raises(DerivClientError, match="not connected"):
        await client.send({"ping": 1})
