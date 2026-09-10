from fastapi import FastAPI

from backend.shared.config import get_settings
from backend.shared.health import build_health_response

settings = get_settings()
app = FastAPI(title="Market Data Service", version="0.1.0", debug=settings.debug)


@app.get("/health")
async def health():
    return build_health_response("market-data-service", settings.environment)
