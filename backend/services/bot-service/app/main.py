import logging

from fastapi import FastAPI, HTTPException, Path, status

from backend.shared.config import get_settings
from backend.shared.health import build_health_response
from backend.shared.observability import configure_logging, metrics_response
from backend.shared.schemas import BotConfig, BotStatus

from .service import BotManager, BotRecord
from .store import BotStore

log = logging.getLogger("bot-service")

settings = get_settings()
configure_logging("bot-service")
app = FastAPI(title="Bot Service", version="0.1.0", debug=settings.debug)
bot_manager = BotManager()
bot_store = BotStore(settings=settings)


async def _persist(bot_id: str) -> None:
    record = bot_manager.get_bot(bot_id)
    if record is None:
        return
    try:
        await bot_store.save(record)
    except Exception:
        # Persistence is best-effort: a DB hiccup shouldn't fail an otherwise
        # successful in-memory bot mutation or the HTTP request that made it.
        log.exception("Failed to persist bot '%s'", bot_id)


@app.on_event("startup")
async def _load_persisted_bots() -> None:
    try:
        records = await bot_store.load_all()
    except Exception:
        log.exception("Failed to load persisted bots; starting with an empty store")
        return
    for record in records:
        bot_manager.load_bot(record)


@app.on_event("shutdown")
async def _shutdown_bot_store() -> None:
    await bot_store.close()


@app.get("/health")
async def health():
    return build_health_response("bot-service", settings.environment)


@app.get("/metrics")
async def metrics():
    return metrics_response()


@app.get("/api/v1/bots", response_model=list[BotRecord])
async def list_bots() -> list[BotRecord]:
    return bot_manager.list_bots()


@app.post("/api/v1/bots", response_model=BotRecord, status_code=status.HTTP_201_CREATED)
async def create_bot(bot_config: BotConfig) -> BotRecord:
    try:
        record = bot_manager.create_bot(bot_config)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    await _persist(record.config.bot_id)
    return record


@app.get("/api/v1/bots/{id}", response_model=BotRecord)
async def get_bot(id: str = Path(min_length=1)) -> BotRecord:
    record = bot_manager.get_bot(id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot '{id}' not found")
    return record


@app.put("/api/v1/bots/{id}", response_model=BotRecord)
async def update_bot(bot_config: BotConfig, id: str = Path(min_length=1)) -> BotRecord:
    try:
        record = bot_manager.update_bot(id, bot_config)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    await _persist(id)
    return record


@app.post("/api/v1/bots/{id}/start", response_model=BotStatus)
async def start_bot(id: str = Path(min_length=1)) -> BotStatus:
    try:
        result = bot_manager.start_bot(id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    await _persist(id)
    return result


@app.post("/api/v1/bots/{id}/stop", response_model=BotStatus)
async def stop_bot(id: str = Path(min_length=1)) -> BotStatus:
    try:
        result = bot_manager.stop_bot(id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    await _persist(id)
    return result


@app.post("/api/v1/bots/{id}/pause", response_model=BotStatus)
async def pause_bot(id: str = Path(min_length=1)) -> BotStatus:
    try:
        result = bot_manager.pause_bot(id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    await _persist(id)
    return result
