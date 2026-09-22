"""Client for Deriv's legacy v3 WebSocket API.

This sits *alongside* :mod:`backend.shared.deriv_client` rather than replacing
it. That one targets the Options API (fixed payout, fixed expiry), which
market-data-service and its tests depend on. This one targets the v3 API,
because that is where multiplier contracts live -- and multipliers are the only
contract type that accepts a stop-loss, a take-profit and a moving stop, which
is what a strategy emitting a stop distance and an R-multiple target needs.

**The one structural difference.** ``DerivClient.receive()`` returns whichever
message arrives next. That is fine for a single subscription and impossible for
anything else: once ticks are streaming, a ``buy`` response and a tick update
are indistinguishable to a caller waiting on ``recv()``, and whichever arrives
first is handed to whoever asked last. Deriv's answer is ``req_id`` -- every
request may carry one and every response echoes it -- so this client runs a
single reader task that routes each message by ``req_id``: to a waiting future
for a one-shot call, or to a queue for a subscription. Callers never touch the
socket directly.

**Keepalive is mandatory.** The v3 session is dropped after two minutes of
silence. A background ping task holds it open; without it a bot that is merely
waiting for the next H4 close is disconnected long before the close arrives.

Sources: https://legacy-docs.deriv.com/docs/websockets ,
https://developers.deriv.com/docs/intro/api-overview/
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any, AsyncIterator, Awaitable, Callable

import websockets

DEFAULT_V3_WS_URL = "wss://ws.derivws.com/websockets/v3"

#: Deriv closes an idle v3 session after two minutes. Ping well inside that:
#: the margin covers a slow round trip without making the socket chatty.
DEFAULT_PING_INTERVAL = 30.0

WebSocketFactory = Callable[[str], Awaitable[Any]]


class DerivV3Error(RuntimeError):
    """A Deriv v3 call failed, or the transport did.

    ``code`` is Deriv's own error code when the failure came back as an API
    error rather than a dropped connection -- callers branch on it (for example
    to tell a symbol that is not offered from a token that is not authorised).
    """

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class Subscription:
    """A live stream of updates for one subscribed request.

    Iterating yields every update after the first. ``first`` is the initial
    response, which Deriv sends before the stream proper and which carries the
    subscription id used to cancel it.
    """

    def __init__(self, client: "DerivV3Client", req_id: int, first: dict[str, Any]) -> None:
        self._client = client
        self._req_id = req_id
        self.first = first
        self.id: str | None = (first.get("subscription") or {}).get("id")
        self._queue: asyncio.Queue[dict[str, Any] | Exception | None] = asyncio.Queue()

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def _push(self, message: dict[str, Any] | Exception | None) -> None:
        self._queue.put_nowait(message)

    async def cancel(self) -> None:
        """Stop the stream, both at Deriv and locally."""
        self._client._subscriptions.pop(self._req_id, None)
        self._push(None)
        if self.id:
            # A forget for a stream Deriv has already dropped is not worth
            # failing a caller's cleanup path over.
            with contextlib.suppress(DerivV3Error):
                await self._client.request({"forget": self.id})


class DerivV3Client:
    def __init__(
        self,
        *,
        app_id: int | str | None = None,
        ws_url: str = DEFAULT_V3_WS_URL,
        websocket_factory: WebSocketFactory | None = None,
        connect_timeout: float = 30.0,
        request_timeout: float = 30.0,
        ping_interval: float = DEFAULT_PING_INTERVAL,
    ) -> None:
        self.app_id = app_id
        self.ws_url = ws_url
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout
        self.ping_interval = ping_interval

        # The same seam tests/test_deriv_client.py uses, kept deliberately so
        # the existing hand-rolled fakes work here too.
        self._websocket_factory = websocket_factory or self._default_websocket_factory

        self._websocket: Any | None = None
        self._reader: asyncio.Task[None] | None = None
        self._pinger: asyncio.Task[None] | None = None
        self._next_req_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._subscriptions: dict[int, Subscription] = {}
        self._authorized = False

    # ---------------------------------------------------------------- state

    @property
    def connected(self) -> bool:
        return self._websocket is not None

    @property
    def authorized(self) -> bool:
        return self._authorized

    @property
    def url(self) -> str:
        """Socket URL with ``app_id`` attached, which Deriv requires."""
        if self.app_id is None:
            return self.ws_url
        separator = "&" if "?" in self.ws_url else "?"
        return f"{self.ws_url}{separator}app_id={self.app_id}"

    # ----------------------------------------------------------- lifecycle

    async def connect(self) -> Any:
        if self._websocket is not None:
            return self._websocket

        try:
            websocket = await self._websocket_factory(self.url)
        except Exception as exc:
            raise DerivV3Error(f"Failed to connect to Deriv v3 at {self.url}") from exc

        self._websocket = websocket
        # The reader is handed its socket rather than reading the attribute, so
        # a close() that nils the attribute cannot be observed mid-loop.
        self._reader = asyncio.create_task(self._read_loop(websocket))
        if self.ping_interval > 0:
            self._pinger = asyncio.create_task(self._ping_loop())
        return websocket

    async def close(self) -> None:
        websocket, self._websocket = self._websocket, None
        self._authorized = False

        for task in (self._pinger, self._reader):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._pinger = self._reader = None

        self._fail_all(DerivV3Error("Deriv v3 connection closed"))

        if websocket is not None:
            with contextlib.suppress(Exception):
                await websocket.close()

    async def __aenter__(self) -> "DerivV3Client":
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    # ------------------------------------------------------------ dispatch

    async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and wait for the response that echoes its req_id."""
        await self.connect()
        req_id = self._allocate_req_id()
        message = {**payload, "req_id": req_id}

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        try:
            await self._send(message)
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError as exc:
            raise DerivV3Error(
                f"Deriv v3 did not answer {self._describe(payload)} within {self.request_timeout}s"
            ) from exc
        finally:
            self._pending.pop(req_id, None)

    async def subscribe(self, payload: dict[str, Any]) -> Subscription:
        """Open a stream. The first response is awaited; the rest are queued."""
        await self.connect()
        req_id = self._allocate_req_id()
        message = {**payload, "subscribe": 1, "req_id": req_id}

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        try:
            await self._send(message)
            first = await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError as exc:
            raise DerivV3Error(
                f"Deriv v3 did not answer {self._describe(payload)} within {self.request_timeout}s"
            ) from exc
        finally:
            self._pending.pop(req_id, None)

        subscription = Subscription(self, req_id, first)
        self._subscriptions[req_id] = subscription
        return subscription

    # ------------------------------------------------------------- methods

    async def authorize(self, token: str) -> dict[str, Any]:
        """Exchange an API token for a session.

        Unlike the Options API this needs no OTP round trip -- the token is
        enough, which is most of why pointing this client at a demo account is
        simple where the other one is not.
        """
        response = await self.request({"authorize": token})
        self._authorized = True
        return response.get("authorize", response)

    async def ping(self) -> dict[str, Any]:
        return await self.request({"ping": 1})

    async def balance(self) -> dict[str, Any]:
        response = await self.request({"balance": 1})
        return response.get("balance", response)

    async def portfolio(self) -> dict[str, Any]:
        response = await self.request({"portfolio": 1})
        return response.get("portfolio", response)

    async def contracts_for(
        self, symbol: str, *, currency: str | None = None, product_type: str = "basic"
    ) -> dict[str, Any]:
        """What Deriv will actually sell on this symbol.

        Worth calling before the first order on a new symbol: multipliers are
        not offered on every instrument in every jurisdiction, and this is the
        difference between finding that out here and finding it out from a
        rejected buy.
        """
        payload: dict[str, Any] = {"contracts_for": symbol, "product_type": product_type}
        if currency:
            payload["currency"] = currency
        response = await self.request(payload)
        return response.get("contracts_for", response)

    async def ticks_history(
        self,
        symbol: str,
        *,
        granularity: int,
        count: int = 1000,
        end: int | str = "latest",
        start: int | None = None,
    ) -> list[dict[str, Any]]:
        """OHLC candles, newest last.

        ``granularity`` is in seconds and must be one Deriv offers (60, 120,
        180, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400).
        """
        payload: dict[str, Any] = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": granularity,
            "count": count,
            "end": end,
        }
        if start is not None:
            payload["start"] = start
        response = await self.request(payload)
        return list(response.get("candles") or [])

    async def ticks(self, symbol: str) -> Subscription:
        return await self.subscribe({"ticks": symbol})

    async def proposal(
        self,
        *,
        symbol: str,
        contract_type: str,
        amount: float,
        currency: str,
        multiplier: int,
        limit_order: dict[str, float] | None = None,
        basis: str = "stake",
    ) -> dict[str, Any]:
        """Price a multiplier contract before buying it.

        ``limit_order`` carries ``stop_loss`` and ``take_profit`` as **amounts
        in account currency**, not price levels. That conversion is the caller's
        job and is the single easiest thing to get wrong in this API.
        """
        payload: dict[str, Any] = {
            "proposal": 1,
            "symbol": symbol,
            "contract_type": contract_type,
            "amount": amount,
            "basis": basis,
            "currency": currency,
            "multiplier": multiplier,
        }
        if limit_order:
            payload["limit_order"] = limit_order
        response = await self.request(payload)
        return response.get("proposal", response)

    async def buy(self, proposal_id: str, *, price: float) -> dict[str, Any]:
        """Buy a priced proposal. ``price`` is the maximum acceptable stake."""
        response = await self.request({"buy": proposal_id, "price": price})
        return response.get("buy", response)

    async def sell(self, contract_id: int | str, *, price: float = 0) -> dict[str, Any]:
        """Close a contract. ``price`` 0 means "at market, any price"."""
        response = await self.request({"sell": contract_id, "price": price})
        return response.get("sell", response)

    async def proposal_open_contract(self, contract_id: int | str) -> Subscription:
        """Stream an open contract's state: this is what drives managed exits."""
        return await self.subscribe({"proposal_open_contract": 1, "contract_id": contract_id})

    async def contract_update(
        self,
        contract_id: int | str,
        *,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """Move the stop or target on an open contract.

        This is what makes break-even and trailing possible, and it is the
        capability the Options API does not have at all. Passing ``None`` for a
        field leaves it untouched; Deriv clears one with the string ``"null"``.
        """
        limit_order: dict[str, Any] = {}
        if stop_loss is not None:
            limit_order["stop_loss"] = stop_loss
        if take_profit is not None:
            limit_order["take_profit"] = take_profit
        if not limit_order:
            raise ValueError("contract_update needs a stop_loss or a take_profit")
        response = await self.request(
            {"contract_update": 1, "contract_id": contract_id, "limit_order": limit_order}
        )
        return response.get("contract_update", response)

    # -------------------------------------------------------------- internals

    def _allocate_req_id(self) -> int:
        self._next_req_id += 1
        return self._next_req_id

    async def _send(self, message: dict[str, Any]) -> None:
        websocket = self._websocket
        if websocket is None:
            raise DerivV3Error("DerivV3Client is not connected")
        try:
            await websocket.send(json.dumps(message))
        except Exception as exc:
            raise DerivV3Error("Failed to send a message to Deriv v3") from exc

    async def _read_loop(self, websocket: Any) -> None:
        try:
            while True:
                raw = await websocket.recv()
                message = self._decode(raw)
                if isinstance(message, dict):
                    self._dispatch(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A dropped socket must not leave callers waiting for the request
            # timeout, and must not leave a subscription iterating forever.
            self._fail_all(DerivV3Error("Deriv v3 connection lost"), cause=exc)

    def _dispatch(self, message: dict[str, Any]) -> None:
        req_id = message.get("req_id")
        error = message.get("error")
        failure: DerivV3Error | None = None
        if isinstance(error, dict):
            failure = DerivV3Error(
                f"Deriv v3 error [{error.get('code', 'UnknownError')}]: "
                f"{error.get('message', 'no message')}",
                code=error.get("code"),
            )

        future = self._pending.get(req_id) if req_id is not None else None
        if future is not None and not future.done():
            if failure is not None:
                future.set_exception(failure)
            else:
                future.set_result(message)
            return

        subscription = self._subscriptions.get(req_id) if req_id is not None else None
        if subscription is not None:
            subscription._push(failure if failure is not None else message)
            return

        # Anything else is an update for a stream nobody is holding any more --
        # a forget racing an in-flight tick. Dropping it is correct.

    def _fail_all(self, error: DerivV3Error, *, cause: BaseException | None = None) -> None:
        if cause is not None:
            error.__cause__ = cause
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(error)
        self._pending.clear()
        for subscription in list(self._subscriptions.values()):
            subscription._push(error)
        self._subscriptions.clear()

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(self.ping_interval)
            try:
                await self.ping()
            except asyncio.CancelledError:
                raise
            except Exception:
                # The reader task owns connection failure; a failed keepalive is
                # a symptom of it, not a second thing to report.
                return

    async def _default_websocket_factory(self, url: str) -> Any:
        return await websockets.connect(url, open_timeout=self.connect_timeout)

    @staticmethod
    def _describe(payload: dict[str, Any]) -> str:
        for key in payload:
            if key not in {"req_id", "subscribe"}:
                return key
        return "request"

    @staticmethod
    def _decode(message: Any) -> Any:
        if isinstance(message, (bytes, bytearray)):
            message = message.decode("utf-8")
        if isinstance(message, str):
            try:
                return json.loads(message)
            except ValueError:
                return message
        return message
