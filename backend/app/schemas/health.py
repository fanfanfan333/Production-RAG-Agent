"""
Health-check response schemas.
"""

from typing import Literal
from pydantic import BaseModel


ServiceStatus = Literal["connected", "not_connected"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: ServiceStatus
    postgres: ServiceStatus
    qdrant: ServiceStatus

    model_config = {"json_schema_extra": {
        "example": {
            "status": "ok",
            "ollama": "connected",
            "postgres": "connected",
            "qdrant": "connected",
        }
    }}