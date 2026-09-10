from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Analysis Service", version="0.1.0")


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "analysis-service"


@app.get("/health")
async def health() -> HealthResponse:
    return HealthResponse()
