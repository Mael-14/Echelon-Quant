from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import websockets

DEFAULT_DERIV_API_BASE_URL = "https://api.derivws.com"
DEFAULT_PUBLIC_WEBSOCKET_URL = "wss://api.derivws.com/trading/v1/options/ws/public"
DEFAULT_OTP_ENDPOINT_TEMPLATE = (
    "https://api.derivws.com/trading/v1/options/accounts/{account_id}/otp"
)

WebSocketFactory = Callable[[str], Awaitable[Any]]
OtpResolver = Callable[[str, str, int | None], Awaitable[str]]


class DerivClientError(RuntimeError):
    """Raised when the Deriv client cannot connect or exchange messages."""


class DerivClient:
    def __init__(
        self,
        *,
        public_websocket_url: str = DEFAULT_PUBLIC_WEBSOCKET_URL,
        otp_endpoint_template: str = DEFAULT_OTP_ENDPOINT_TEMPLATE,
        websocket_factory: WebSocketFactory | None = None,
        otp_resolver: OtpResolver | None = None,
        connect_timeout: float = 30.0,
        request_timeout: float = 30.0,
    ) -> None:
        self.public_websocket_url = public_websocket_url
        self.otp_endpoint_template = otp_endpoint_template
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout

        self._websocket_factory = websocket_factory or self._default_websocket_factory
        self._otp_resolver = otp_resolver or self._default_otp_resolver

        self._websocket: Any | None = None
        self._current_websocket_url: str | None = None
        self._auth_token: str | None = None
        self._account_id: str | None = None
        self._app_id: int | None = None
        self._authenticated = False
        self._connection_generation = 0

    @property
    def connected(self) -> bool:
        return self._websocket is not None and not self._is_closed(self._websocket)

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    @property
    def connection_generation(self) -> int:
        return self._connection_generation

    async def connect(self, websocket_url: str | None = None) -> Any:
        url = websocket_url or self._current_websocket_url or self.public_websocket_url
        await self._open_websocket(url)
        return self._websocket

    async def authenticate(self, token: str, account_id: str, app_id: int | None = None) -> Any:
        self._auth_token = token
        self._account_id = account_id
        self._app_id = app_id
        self._authenticated = True
        return await self.reconnect()

    async def subscribe(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        payload["subscribe"] = 1
        await self.send(payload)
        response = await self.receive()

        if not isinstance(response, dict):
            raise DerivClientError("Subscription response was not a JSON object")

        # Deriv reports a rejected subscription (e.g. an unknown symbol) as a normal JSON
        # object with an "error" field, not a transport-level failure - without this check
        # a bad symbol looks like a successful subscribe and then silently never ticks.
        error = response.get("error")
        if isinstance(error, dict):
            code = error.get("code", "UnknownError")
            message = error.get("message", "Deriv rejected the subscription")
            raise DerivClientError(f"Deriv subscription error [{code}]: {message}")

        return response

    async def send(self, message: dict[str, Any] | str) -> str:
        websocket = self._require_websocket()

        payload = message if isinstance(message, str) else json.dumps(message)
        await websocket.send(payload)
        return payload

    async def receive(self) -> Any:
        websocket = self._require_websocket()

        try:
            message = await websocket.recv()
        except Exception as exc:  # pragma: no cover - defensive transport guard
            raise DerivClientError("Failed to receive a message from Deriv") from exc

        return self._decode_message(message)

    async def reconnect(self) -> Any:
        if self._authenticated:
            if not self._auth_token or not self._account_id:
                raise DerivClientError("Authentication details are missing")

            self._current_websocket_url = await self._otp_resolver(
                self._account_id,
                self._auth_token,
                self._app_id,
            )

        url = self._current_websocket_url or self.public_websocket_url
        await self.disconnect()
        await self._open_websocket(url)
        return self._websocket

    async def disconnect(self) -> None:
        if self._websocket is None:
            return

        websocket = self._websocket
        self._websocket = None

        try:
            await websocket.close()
        except Exception:  # pragma: no cover - transport cleanup should not fail the caller
            pass

    async def _open_websocket(self, websocket_url: str) -> Any:
        await self.disconnect()

        try:
            websocket = await self._websocket_factory(websocket_url)
        except Exception as exc:  # pragma: no cover - connection factory errors bubble up as client errors
            raise DerivClientError(f"Failed to connect to Deriv at {websocket_url}") from exc

        self._websocket = websocket
        self._current_websocket_url = websocket_url
        self._connection_generation += 1
        return websocket

    async def _default_websocket_factory(self, websocket_url: str) -> Any:
        return await websockets.connect(websocket_url, open_timeout=self.connect_timeout)

    async def _default_otp_resolver(self, account_id: str, token: str, app_id: int | None) -> str:
        return await asyncio.to_thread(self._fetch_otp_url, account_id, token, app_id)

    def _fetch_otp_url(self, account_id: str, token: str, app_id: int | None) -> str:
        encoded_account_id = quote(account_id, safe="")
        endpoint = self.otp_endpoint_template.format(account_id=encoded_account_id)
        request = Request(endpoint, method="POST")
        request.add_header("Authorization", f"Bearer {token}")

        if app_id is not None:
            request.add_header("Deriv-App-ID", str(app_id))

        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise DerivClientError("Failed to retrieve an OTP WebSocket URL") from exc

        otp_url = payload.get("data", {}).get("url") if isinstance(payload, dict) else None
        if not isinstance(otp_url, str) or not otp_url:
            raise DerivClientError("OTP response did not include a WebSocket URL")

        return otp_url

    def _require_websocket(self) -> Any:
        if self._websocket is None:
            raise DerivClientError("DerivClient is not connected")

        return self._websocket

    @staticmethod
    def _is_closed(websocket: Any) -> bool:
        ready_state = getattr(websocket, "closed", None)
        if isinstance(ready_state, bool):
            return ready_state

        state_value = getattr(websocket, "state", None)
        if isinstance(state_value, int):
            return state_value == 3

        return False

    @staticmethod
    def _decode_message(message: Any) -> Any:
        if isinstance(message, (bytes, bytearray)):
            message = message.decode("utf-8")

        if isinstance(message, str):
            try:
                return json.loads(message)
            except ValueError:
                return message

        return message