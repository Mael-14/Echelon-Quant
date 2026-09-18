from fastapi import FastAPI
from pydantic import BaseModel

from backend.shared.observability import configure_logging, metrics_response

configure_logging("analysis-service")
app = FastAPI(title="Analysis Service", version="0.1.0")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "analysis-service"


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()


@app.get("/metrics")
async def metrics():
    return metrics_response()
