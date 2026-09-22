"""analysis-service: features, model, signals.

Was a 19-line /health stub. It is the service the toolkit was always meant to
become: candles arrive, features encode, the model scores, the cost gate
accepts or refuses, and an accepted setup is published to `signal-events` for
execution-service to act on.

**Degraded is a first-class state.** With no artifact, or an artifact trained
on a different cascade, /health reports `degraded` and the signal loop emits
nothing. A service that guesses here produces plausible numbers that are wrong
in a way nothing downstream can detect.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from backend.shared.candles import CandleWindow, fetch_candles, next_close
from backend.shared.config import get_settings
from backend.shared.deriv_v3_client import DerivV3Client, DerivV3Error
from backend.shared.observability import (
    ANALYSIS_LAST_SIGNAL_PROBABILITY,
    ANALYSIS_MODEL_LOADED,
    ANALYSIS_SIGNALS_EMITTED,
    ANALYSIS_SIGNALS_REJECTED,
    configure_logging,
    metrics_response,
)

from .engine import AnalysisEngine, Evaluation

configure_logging("analysis-service")
log = logging.getLogger("analysis-service")

app = FastAPI(title="Analysis Service", version="0.1.0")

state: dict[str, Any] = {
    "engine": None,
    "degraded_reason": "not started",
    "windows": {},
    "loop": None,
    "redis": None,
    "last": {},
}


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "analysis-service"
    environment: str = "development"
    model_loaded: bool = False
    cascade: str | None = None
    detail: str | None = None


class FeaturesRequest(BaseModel):
    symbol: str = Field(min_length=1)


def _settings():
    return get_settings()


def _symbols(settings) -> list[str]:
    raw = getattr(settings, "analysis_symbols", "") or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _load_engine(settings) -> tuple[Any, str | None]:
    """Build the engine, or explain why there isn't one.

    Returns ``(engine, None)`` or ``(None, reason)``. Every failure path here is
    a reason to emit nothing, so none of them raise.
    """
    path = getattr(settings, "analysis_model_path", "")
    if not path:
        return None, "ANALYSIS_MODEL_PATH is not set"

    try:
        from forex_agent import artifact as artifact_module
    except ImportError as exc:
        return None, f"fa-ml-toolkit is not installed: {exc}"

    try:
        loaded = artifact_module.load(path)
    except Exception as exc:
        return None, str(exc)

    exec_frame = settings.analysis_exec_frame
    macro_frames = [f.strip() for f in settings.analysis_macro_frames.split(",") if f.strip()]
    try:
        engine = AnalysisEngine(
            artifact=loaded,
            exec_frame=exec_frame,
            macro_frames=macro_frames,
            macro_window=settings.analysis_macro_window,
            exec_window=settings.analysis_exec_window,
            min_expected_value_r=settings.analysis_min_ev_r,
        )
    except Exception as exc:
        return None, str(exc)

    return engine, None


async def _refresh(client: DerivV3Client, engine: AnalysisEngine, symbol: str) -> None:
    """Top up every window this symbol needs."""
    windows: dict[str, CandleWindow] = state["windows"].setdefault(symbol, {})
    for frame, needed in [
        *[(f, engine.macro_window) for f in engine.macro_frames],
        (engine.exec_frame, engine.exec_window),
    ]:
        window = windows.get(frame)
        if window is None:
            window = CandleWindow(symbol, frame, maxlen=max(needed * 2, 100))
            windows[frame] = window
        # A full backfill on the first pass, then just the tail.
        count = needed if len(window) < needed else 5
        candles = await fetch_candles(client, symbol, timeframe=frame, count=count)
        window.extend(candles)


def _evaluate(engine: AnalysisEngine, symbol: str) -> Evaluation:
    windows = state["windows"].get(symbol, {})
    macro = [windows[f] for f in engine.macro_frames if f in windows]
    micro = windows.get(engine.exec_frame)
    if len(macro) != len(engine.macro_frames) or micro is None:
        return Evaluation(symbol, False, "no candles yet")
    return engine.evaluate(symbol, macro, micro)


async def _publish(evaluation: Evaluation, engine: AnalysisEngine) -> None:
    client = state.get("redis")
    if client is None:
        return
    settings = _settings()
    signal = engine.to_signal(evaluation)
    prediction = engine.to_prediction(evaluation, horizon_minutes=max(1, engine.artifact.hold))
    setup = evaluation.setup
    assert setup is not None  # guarded by to_signal
    await client.xadd(
        settings.redis_stream_signal_events,
        {
            "signal": signal.model_dump_json(),
            "prediction": prediction.model_dump_json(),
            # Geometry execution needs and the schema does not carry.
            "entry": str(setup.entry),
            "stop_distance": str(setup.stop_distance),
            "reward_r": str(setup.reward_r),
            "cost_r": str(setup.cost_r),
            "p_win": str(evaluation.p_win),
        },
    )


async def _signal_loop() -> None:
    """Poll on frame close, evaluate, publish what clears the gate."""
    settings = _settings()
    engine = state["engine"]
    symbols = _symbols(settings)
    if engine is None or not symbols:
        return

    client = DerivV3Client(app_id=settings.deriv_app_id, ws_url=settings.deriv_v3_ws_url)
    try:
        await client.connect()
        if settings.deriv_token:
            await client.authorize(settings.deriv_token)

        while True:
            for symbol in symbols:
                try:
                    await _refresh(client, engine, symbol)
                    evaluation = _evaluate(engine, symbol)
                    state["last"][symbol] = evaluation
                    ANALYSIS_LAST_SIGNAL_PROBABILITY.labels(symbol=symbol).set(evaluation.p_win)

                    if evaluation.accepted and evaluation.setup is not None:
                        await _publish(evaluation, engine)
                        ANALYSIS_SIGNALS_EMITTED.labels(
                            symbol=symbol, side=evaluation.setup.side
                        ).inc()
                        log.info("signal %s %s", symbol, evaluation.setup.side)
                    else:
                        ANALYSIS_SIGNALS_REJECTED.labels(
                            symbol=symbol, reason=evaluation.reason[:48]
                        ).inc()
                except DerivV3Error:
                    log.exception("Deriv call failed for %s", symbol)
                except Exception:
                    log.exception("Evaluation failed for %s", symbol)

            # Sleep to the next close of the execution frame, plus a moment:
            # Deriv publishes the closed bar once its own clock has passed the
            # boundary, so asking exactly on it returns the bar still forming.
            target = next_close(engine.exec_frame)
            delay = (target - datetime.now(tz=timezone.utc)).total_seconds() + 2.0
            await asyncio.sleep(max(1.0, delay))
    except asyncio.CancelledError:
        raise
    finally:
        await client.close()


@app.on_event("startup")
async def startup_event() -> None:
    settings = _settings()
    engine, reason = _load_engine(settings)
    state["engine"] = engine
    state["degraded_reason"] = reason
    ANALYSIS_MODEL_LOADED.set(1 if engine is not None else 0)

    if engine is None:
        log.warning("analysis-service is degraded: %s", reason)
        return

    log.info("loaded model: %s", engine.artifact.summary)
    with contextlib.suppress(Exception):
        state["redis"] = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
    state["loop"] = asyncio.create_task(_signal_loop())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    task = state.get("loop")
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    client = state.get("redis")
    if client is not None:
        with contextlib.suppress(Exception):
            await client.aclose()


@app.get("/health")
async def health() -> HealthResponse:
    settings = _settings()
    engine = state.get("engine")
    if engine is None:
        return HealthResponse(
            status="degraded",
            environment=settings.environment,
            model_loaded=False,
            detail=state.get("degraded_reason"),
        )
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        model_loaded=True,
        cascade=engine.artifact.summary,
    )


@app.get("/metrics")
async def metrics():
    return metrics_response()


@app.post("/api/v1/analysis/features")
async def features(request: FeaturesRequest) -> dict[str, Any]:
    engine = state.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.get("degraded_reason") or "no model loaded",
        )
    evaluation = _evaluate(engine, request.symbol)
    if evaluation.features is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=evaluation.reason,
        )
    return {"symbol": request.symbol, "features": evaluation.features}


@app.get("/api/v1/analysis/signal/{symbol}")
async def signal(symbol: str) -> dict[str, Any]:
    """The current verdict for a symbol.

    A refusal is a 200 with ``accepted: false``, not an error: at M1 most bars
    have no tradeable edge, and that is the answer rather than a failure.
    """
    engine = state.get("engine")
    if engine is None:
        return {
            "symbol": symbol,
            "status": "degraded",
            "detail": state.get("degraded_reason"),
        }

    evaluation = state["last"].get(symbol) or _evaluate(engine, symbol)
    payload: dict[str, Any] = {
        "symbol": symbol,
        "status": "ok",
        "accepted": evaluation.accepted,
        "reason": evaluation.reason,
        "p_win": evaluation.p_win,
        "breakeven": evaluation.breakeven,
        "expected_value_r": evaluation.expected_value_r,
        "cascade": engine.artifact.summary,
    }
    if evaluation.setup is not None:
        payload["setup"] = {
            "side": evaluation.setup.side,
            "entry": evaluation.setup.entry,
            "stop_distance": evaluation.setup.stop_distance,
            "reward_r": evaluation.setup.reward_r,
            "cost_r": evaluation.setup.cost_r,
        }
    if evaluation.accepted:
        payload["signal"] = json.loads(engine.to_signal(evaluation).model_dump_json())
    return payload
