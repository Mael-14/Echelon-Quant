from fastapi import FastAPI

from backend.shared.config import get_settings
from backend.shared.health import build_health_response
from backend.shared.observability import configure_logging, metrics_response

settings = get_settings()
configure_logging("strategy-ai-service")
app = FastAPI(title="Strategy AI Service", version="0.1.0", debug=settings.debug)


@app.get("/health")
async def health():
    return build_health_response("strategy-ai-service", settings.environment)


@app.get("/metrics")
async def metrics():
    return metrics_response()
