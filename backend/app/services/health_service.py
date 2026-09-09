"""
Health-check service.

Performs real connectivity checks against:
  - Ollama          → GET /api/tags (lists locally available models)
  - PostgreSQL      → raw asyncpg ping (SELECT 1)
  - Qdrant          → cluster info endpoint
"""

import asyncio

import asyncpg
import httpx

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.schemas.health import HealthResponse, ServiceStatus
from app.utils.logging import get_logger

logger = get_logger(__name__)


async def _check_ollama(settings) -> ServiceStatus:
    """
    Verify the local Ollama server is reachable by hitting its /api/tags
    endpoint, which lists locally available models. This only confirms the
    Ollama daemon is up — it does not verify that OLLAMA_MODEL specifically
    has been pulled.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            resp.raise_for_status()
        logger.debug("Ollama check OK")
        return "connected"
    except Exception as exc:
        logger.warning("Ollama check failed: %s", exc)
        return "not_connected"


async def _check_postgres(settings) -> ServiceStatus:
    """Open a raw asyncpg connection and execute SELECT 1."""
    try:
        conn: asyncpg.Connection = await asyncio.wait_for(
            asyncpg.connect(
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                database=settings.POSTGRES_DB,
            ),
            timeout=5,
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        logger.debug("PostgreSQL check OK")
        return "connected"
    except Exception as exc:
        logger.warning("PostgreSQL check failed: %s", exc)
        return "not_connected"


async def _check_qdrant() -> ServiceStatus:
    """Call Qdrant's cluster info endpoint via the async client."""
    try:
        client = get_qdrant_client()
        info = await asyncio.wait_for(client.get_collections(), timeout=5)
        logger.debug(
            "Qdrant check OK — %d collection(s)", len(info.collections)
        )
        return "connected"
    except Exception as exc:
        logger.warning("Qdrant check failed: %s", exc)
        return "not_connected"


async def get_health() -> HealthResponse:
    """
    Run all connectivity checks concurrently and return an aggregated result.
    overall status is 'ok' only when all three services are connected.
    """
    settings = get_settings()

    ollama_status, postgres_status, qdrant_status = await asyncio.gather(
        _check_ollama(settings),
        _check_postgres(settings),
        _check_qdrant(),
    )

    overall = (
        "ok"
        if all(
            s == "connected"
            for s in (ollama_status, postgres_status, qdrant_status)
        )
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        postgres=postgres_status,
        qdrant=qdrant_status,
    )