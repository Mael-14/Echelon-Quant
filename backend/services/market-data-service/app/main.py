from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel

from backend.shared.config import get_settings
from backend.shared.health import build_health_response
from backend.shared.observability import configure_logging, metrics_response

from .pipeline import MarketDataPipeline

log = logging.getLogger("market-data-service")

settings = get_settings()
configure_logging("market-data-service")
app = FastAPI(title="Market Data Service", version="0.1.0", debug=settings.debug)


class SubscribeRequest(BaseModel):
    user_id: str
    symbol: str
    bot_id: str
    account_id: str | None = None
    app_id: int | None = None


class UnsubscribeRequest(BaseModel):
    user_id: str
    symbol: str
    bot_id: str


@app.get("/health")
async def health():
    return build_health_response("market-data-service", settings.environment)


@app.get("/metrics")
async def metrics():
    return metrics_response()


@app.post("/api/v1/market/subscribe")
async def subscribe(req: SubscribeRequest):
    pipeline = getattr(app.state, "_market_data_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="market-data pipeline not running")
    try:
        key = await pipeline.subscribe_for_bot(user_id=req.user_id, symbol=req.symbol, bot_id=req.bot_id, account_id=req.account_id, app_id=req.app_id)
        log.info("Started subscription %s for user=%s symbol=%s bot=%s", key, req.user_id, req.symbol, req.bot_id)
        return {"subscription_key": key}
    except Exception:
        log.exception("Failed to start subscription for user=%s symbol=%s bot=%s", req.user_id, req.symbol, req.bot_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to start subscription")


@app.post("/api/v1/market/unsubscribe", status_code=status.HTTP_204_NO_CONTENT)
async def unsubscribe(req: UnsubscribeRequest):
    pipeline = getattr(app.state, "_market_data_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="market-data pipeline not running")
    try:
        await pipeline.unsubscribe_for_bot(user_id=req.user_id, symbol=req.symbol, bot_id=req.bot_id)
        log.info("Stopped subscription for user=%s symbol=%s bot=%s", req.user_id, req.symbol, req.bot_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception:
        log.exception("Failed to stop subscription for user=%s symbol=%s bot=%s", req.user_id, req.symbol, req.bot_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="failed to stop subscription")


@app.on_event("startup")
async def startup_event() -> None:
    # Start market data pipeline in background
    symbols = ["XAUUSD", "V75_2S"]
    pipeline = MarketDataPipeline(settings=settings, symbols=symbols)
    task: Any = asyncio.create_task(pipeline.run(), name="market-data-pipeline")
    app.state._market_data_pipeline_task = task
    app.state._market_data_pipeline = pipeline
    log.info("Market data pipeline started for symbols: %s", symbols)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    # Attempt graceful shutdown of the pipeline
    task: asyncio.Task | None = getattr(app.state, "_market_data_pipeline_task", None)
    pipeline: MarketDataPipeline | None = getattr(app.state, "_market_data_pipeline", None)
    if pipeline is not None:
        await pipeline.stop()
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    log.info("Market data pipeline stopped")
