from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Path, status
from pydantic import BaseModel

import asyncio
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json

import asyncpg
from cryptography.fernet import Fernet, InvalidToken

from backend.shared.config import get_settings
from backend.shared.health import build_health_response
from backend.shared.schemas import BotConfig, BotLifecycleState, BotStatus

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", debug=settings.debug)


async def _ensure_db_pool() -> asyncpg.Pool:
    pool: asyncpg.Pool | None = getattr(app.state, "pg_pool", None)
    if pool is None:
        dsn = settings.database_url.replace("+asyncpg", "")
        pool = await asyncpg.create_pool(dsn)
        app.state.pg_pool = pool
    return pool


async def _create_deriv_tokens_table() -> None:
    pool = await _ensure_db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS deriv_tokens (
                user_id TEXT PRIMARY KEY,
                token TEXT NOT NULL,
                account_id TEXT,
                app_id INTEGER,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )


def _get_fernet() -> Fernet | None:
    key = settings.deriv_token_key
    if not key:
        return None
    try:
        return Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    except Exception:
        return None


def _encrypt_token(plain: str) -> str:
    f = _get_fernet()
    if f is None:
        return plain
    return f.encrypt(plain.encode("utf-8")).decode("utf-8")


def _decrypt_token(enc: str) -> str:
    f = _get_fernet()
    if f is None:
        return enc
    try:
        return f.decrypt(enc.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        # If decryption fails, return as-is to avoid blocking; caller should handle failures.
        return enc


@app.on_event("startup")
async def _startup_db() -> None:
    try:
        await _create_deriv_tokens_table()
    except Exception:
        # DB might be unavailable in dev; fail loudly would crash service.
        pass


@app.on_event("shutdown")
async def _shutdown_db() -> None:
    pool: asyncpg.Pool | None = getattr(app.state, "pg_pool", None)
    if pool is not None:
        await pool.close()


class BotState(BaseModel):
    config: BotConfig
    status: BotStatus


bots_store: dict[str, BotState] = {}


@app.get("/health")
async def health():
    return build_health_response("api-gateway", settings.environment)


class DerivTokenIn(BaseModel):
    user_id: str
    token: str
    account_id: str | None = None
    app_id: int | None = None


@app.post("/api/v1/deriv/token", status_code=status.HTTP_201_CREATED)
async def store_deriv_token(payload: DerivTokenIn) -> dict:
    pool = await _ensure_db_pool()
    enc = _encrypt_token(payload.token)
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

    token = row["token"]
    # decrypt if encrypted
    token = _decrypt_token(token)

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
    enc = _encrypt_token(access_token)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO deriv_tokens(user_id, token) VALUES($1,$2) ON CONFLICT(user_id) DO UPDATE SET token=EXCLUDED.token, created_at=now()",
            payload.user_id,
            enc,
        )

    return {"status": "ok"}


@app.get("/api/v1/bots", response_model=list[BotState])
async def list_bots() -> list[BotState]:
    return list(bots_store.values())


@app.post("/api/v1/bots", response_model=BotState, status_code=status.HTTP_201_CREATED)
async def create_bot(bot_config: BotConfig) -> BotState:
    if bot_config.bot_id in bots_store:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Bot '{bot_config.bot_id}' already exists",
        )

    bot_state = BotState(
        config=bot_config,
        status=BotStatus(
            bot_id=bot_config.bot_id,
            state=BotLifecycleState.CREATED,
            uptime_seconds=0,
        ),
    )
    bots_store[bot_config.bot_id] = bot_state
    return bot_state


@app.get("/api/v1/bots/{id}", response_model=BotState)
async def get_bot(id: str = Path(min_length=1)) -> BotState:
    bot_state = bots_store.get(id)
    if bot_state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot '{id}' not found")
    return bot_state


@app.put("/api/v1/bots/{id}", response_model=BotState)
async def update_bot(bot_config: BotConfig, id: str = Path(min_length=1)) -> BotState:
    existing = bots_store.get(id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot '{id}' not found")
    if bot_config.bot_id != id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path id must match body bot_id",
        )

    updated_status = existing.status.model_copy(update={"last_heartbeat": datetime.now(timezone.utc)})
    updated = BotState(config=bot_config, status=updated_status)
    bots_store[id] = updated
    return updated


@app.post("/api/v1/bots/{id}/start", response_model=BotStatus)
async def start_bot(id: str = Path(min_length=1)) -> BotStatus:
    bot_state = bots_store.get(id)
    if bot_state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot '{id}' not found")

    bot_state.status = bot_state.status.model_copy(
        update={
            "state": BotLifecycleState.RUNNING,
            "error": None,
            "last_heartbeat": datetime.now(timezone.utc),
        }
    )
    return bot_state.status


@app.post("/api/v1/bots/{id}/stop", response_model=BotStatus)
async def stop_bot(id: str = Path(min_length=1)) -> BotStatus:
    bot_state = bots_store.get(id)
    if bot_state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot '{id}' not found")

    bot_state.status = bot_state.status.model_copy(
        update={
            "state": BotLifecycleState.STOPPED,
            "last_heartbeat": datetime.now(timezone.utc),
        }
    )
    return bot_state.status


@app.post("/api/v1/bots/{id}/pause", response_model=BotStatus)
async def pause_bot(id: str = Path(min_length=1)) -> BotStatus:
    bot_state = bots_store.get(id)
    if bot_state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot '{id}' not found")

    bot_state.status = bot_state.status.model_copy(
        update={
            "state": BotLifecycleState.PAUSED,
            "last_heartbeat": datetime.now(timezone.utc),
        }
    )
    return bot_state.status
