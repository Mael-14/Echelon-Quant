from fastapi import FastAPI, HTTPException, Path, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import asyncio
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json

import asyncpg
import httpx

from backend.shared.config import get_settings
from backend.shared.health import build_health_response
from backend.shared.observability import configure_logging, metrics_response
from backend.shared.schemas import BotConfig
from backend.shared.token_crypto import TokenDecryptionError, decrypt_token, encrypt_token

settings = get_settings()
configure_logging("api-gateway")
app = FastAPI(title=settings.app_name, version="0.1.0", debug=settings.debug)

_db_pool_lock = asyncio.Lock()


async def _ensure_db_pool() -> asyncpg.Pool:
    # Schema (the `deriv_tokens` table) is managed by Alembic migrations
    # (backend/alembic/versions/0001_initial_schema.py) - run `alembic upgrade head`
    # before starting this service. Unlike market-data-service/bot-service, a failed
    # connection here is allowed to raise: token storage has no degraded fallback mode.
    pool: asyncpg.Pool | None = getattr(app.state, "pg_pool", None)
    if pool is not None:
        return pool
    async with _db_pool_lock:
        pool = getattr(app.state, "pg_pool", None)
        if pool is None:
            dsn = settings.database_url.replace("+asyncpg", "")
            pool = await asyncpg.create_pool(dsn)
            app.state.pg_pool = pool
    return pool


def _ensure_bot_service_client() -> httpx.AsyncClient:
    client: httpx.AsyncClient | None = getattr(app.state, "bot_service_client", None)
    if client is None:
        client = httpx.AsyncClient(base_url=settings.bot_service_url, timeout=10.0)
        app.state.bot_service_client = client
    return client


async def _proxy_bot_request(method: str, path: str, *, json_body: Any = None) -> JSONResponse:
    client = _ensure_bot_service_client()
    try:
        resp = await client.request(method, path, json=json_body)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="bot-service unavailable"
        ) from exc
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.on_event("shutdown")
async def _shutdown_db() -> None:
    pool: asyncpg.Pool | None = getattr(app.state, "pg_pool", None)
    if pool is not None:
        await pool.close()


@app.on_event("shutdown")
async def _shutdown_bot_service_client() -> None:
    client: httpx.AsyncClient | None = getattr(app.state, "bot_service_client", None)
    if client is not None:
        await client.aclose()


@app.get("/health")
async def health():
    return build_health_response("api-gateway", settings.environment)


@app.get("/metrics")
async def metrics():
    return metrics_response()


class DerivTokenIn(BaseModel):
    user_id: str
    token: str
    account_id: str | None = None
    app_id: int | None = None


@app.post("/api/v1/deriv/token", status_code=status.HTTP_201_CREATED)
async def store_deriv_token(payload: DerivTokenIn) -> dict:
    pool = await _ensure_db_pool()
    enc = encrypt_token(payload.token, settings=settings)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO deriv_tokens(user_id, token, account_id, app_id) VALUES($1,$2,$3,$4) "
            "ON CONFLICT(user_id) DO UPDATE SET token=EXCLUDED.token, account_id=EXCLUDED.account_id, app_id=EXCLUDED.app_id, created_at=now()",
            payload.user_id,
            enc,
            payload.account_id,
            payload.app_id,
        )
    return {"status": "ok"}


@app.get("/api/v1/deriv/accounts/{user_id}")
async def list_deriv_accounts(user_id: str = Path(min_length=1)) -> Any:
    pool = await _ensure_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT token FROM deriv_tokens WHERE user_id = $1", user_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No token for user")

    try:
        token = decrypt_token(row["token"], settings=settings)
    except TokenDecryptionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stored Deriv token could not be decrypted; please re-authenticate",
        ) from exc

    def _fetch_accounts() -> Any:
        endpoint = "https://api.derivws.com/trading/v1/options/accounts"
        req = Request(endpoint, method="GET")
        req.add_header("Authorization", f"Bearer {token}")
        try:
            with urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            raise RuntimeError("Failed to fetch accounts") from exc

    try:
        result = await asyncio.to_thread(_fetch_accounts)
    except Exception:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to fetch accounts from Deriv")

    return result



class OAuthExchangeIn(BaseModel):
    user_id: str
    client_id: str
    client_secret: str
    code: str
    redirect_uri: str


@app.post("/api/v1/deriv/oauth/exchange", status_code=status.HTTP_201_CREATED)
async def deriv_oauth_exchange(payload: OAuthExchangeIn) -> dict:
    # Exchange authorization code for token (simple implementation)
    def _exchange() -> Any:
        endpoint = "https://oauth.deriv.com/oauth2/token"
        body = (
            f"grant_type=authorization_code&code={payload.code}&redirect_uri={payload.redirect_uri}"
        )
        data = body.encode("utf-8")
        req = Request(endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        # Basic auth with client_id:client_secret
        import base64

        auth = base64.b64encode(f"{payload.client_id}:{payload.client_secret}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")
        try:
            with urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("OAuth token exchange failed") from exc

    try:
        result = await asyncio.to_thread(_exchange)
    except Exception:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="OAuth exchange failed")

    access_token = result.get("access_token") if isinstance(result, dict) else None
    if not access_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No access_token in response")

    # store token for user
    pool = await _ensure_db_pool()
    enc = encrypt_token(access_token, settings=settings)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO deriv_tokens(user_id, token) VALUES($1,$2) ON CONFLICT(user_id) DO UPDATE SET token=EXCLUDED.token, created_at=now()",
            payload.user_id,
            enc,
        )

    return {"status": "ok"}


# Bot lifecycle is owned by bot-service (BotManager's state machine); api-gateway is a
# thin proxy so there's a single source of truth for bot state.
@app.get("/api/v1/bots")
async def list_bots() -> JSONResponse:
    return await _proxy_bot_request("GET", "/api/v1/bots")


@app.post("/api/v1/bots")
async def create_bot(bot_config: BotConfig) -> JSONResponse:
    return await _proxy_bot_request(
        "POST", "/api/v1/bots", json_body=bot_config.model_dump(mode="json")
    )


@app.get("/api/v1/bots/{id}")
async def get_bot(id: str = Path(min_length=1)) -> JSONResponse:
    return await _proxy_bot_request("GET", f"/api/v1/bots/{id}")


@app.put("/api/v1/bots/{id}")
async def update_bot(bot_config: BotConfig, id: str = Path(min_length=1)) -> JSONResponse:
    return await _proxy_bot_request(
        "PUT", f"/api/v1/bots/{id}", json_body=bot_config.model_dump(mode="json")
    )


@app.post("/api/v1/bots/{id}/start")
async def start_bot(id: str = Path(min_length=1)) -> JSONResponse:
    return await _proxy_bot_request("POST", f"/api/v1/bots/{id}/start")


@app.post("/api/v1/bots/{id}/stop")
async def stop_bot(id: str = Path(min_length=1)) -> JSONResponse:
    return await _proxy_bot_request("POST", f"/api/v1/bots/{id}/stop")


@app.post("/api/v1/bots/{id}/pause")
async def pause_bot(id: str = Path(min_length=1)) -> JSONResponse:
    return await _proxy_bot_request("POST", f"/api/v1/bots/{id}/pause")
