from fastapi import FastAPI, HTTPException, Path, status

from backend.shared.config import get_settings
from backend.shared.health import build_health_response
from backend.shared.schemas import BotConfig, BotStatus

from .service import BotManager, BotRecord

settings = get_settings()
app = FastAPI(title="Bot Service", version="0.1.0", debug=settings.debug)
bot_manager = BotManager()


@app.get("/health")
async def health():
    return build_health_response("bot-service", settings.environment)


@app.get("/api/v1/bots", response_model=list[BotRecord])
async def list_bots() -> list[BotRecord]:
    return bot_manager.list_bots()


@app.post("/api/v1/bots", response_model=BotRecord, status_code=status.HTTP_201_CREATED)
async def create_bot(bot_config: BotConfig) -> BotRecord:
    try:
        return bot_manager.create_bot(bot_config)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.get("/api/v1/bots/{id}", response_model=BotRecord)
async def get_bot(id: str = Path(min_length=1)) -> BotRecord:
    record = bot_manager.get_bot(id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot '{id}' not found")
    return record


@app.put("/api/v1/bots/{id}", response_model=BotRecord)
async def update_bot(bot_config: BotConfig, id: str = Path(min_length=1)) -> BotRecord:
    try:
        return bot_manager.update_bot(id, bot_config)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/api/v1/bots/{id}/start", response_model=BotStatus)
async def start_bot(id: str = Path(min_length=1)) -> BotStatus:
    try:
        return bot_manager.start_bot(id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.post("/api/v1/bots/{id}/stop", response_model=BotStatus)
async def stop_bot(id: str = Path(min_length=1)) -> BotStatus:
    try:
        return bot_manager.stop_bot(id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.post("/api/v1/bots/{id}/pause", response_model=BotStatus)
async def pause_bot(id: str = Path(min_length=1)) -> BotStatus:
    try:
        return bot_manager.pause_bot(id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
