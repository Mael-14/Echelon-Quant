from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Path, status
from pydantic import BaseModel

from backend.shared.config import get_settings
from backend.shared.health import build_health_response
from backend.shared.schemas import BotConfig, BotLifecycleState, BotStatus

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", debug=settings.debug)


class BotState(BaseModel):
    config: BotConfig
    status: BotStatus


bots_store: dict[str, BotState] = {}


@app.get("/health")
async def health():
    return build_health_response("api-gateway", settings.environment)


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
