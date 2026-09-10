from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str
    environment: str = "development"


def build_health_response(service_name: str, environment: str = "development") -> HealthResponse:
    return HealthResponse(service=service_name, environment=environment)
