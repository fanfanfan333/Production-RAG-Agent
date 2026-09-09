"""
Application-edge gateway middleware.

This module is the service's policy enforcement point when a full external API
gateway (Kong / APISIX / Envoy) is not yet deployed. It adds safe defaults:

- request correlation IDs
- trusted-proxy aware client IP handling (configured allowlist)
- request body-size guard before expensive parsers run
- per-route edge rate limits (public auth, upload, admin)
- production Origin check for browser write requests
- standard security response headers
- uniform JSON errors for gateway rejections

It intentionally does not replace an external gateway for multi-replica rate
limiting, WAF or TLS termination. Those remain deployment-layer concerns.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from app.api.deps import client_ip, limiter
from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_PUBLIC_PREFIXES = ("/health", "/docs", "/redoc", "/openapi.json")
_AUTH_WRITE_PATHS = {"/auth/login", "/auth/register", "/auth/change-password"}


def _origin_allowed(origin: str, allowed_origins: list[str]) -> bool:
    return "*" in allowed_origins or origin in allowed_origins


def _security_headers(response: Response, request_id: str) -> None:
    # API responses do not need browser embedding, MIME sniffing or referrer
    # propagation. CSP is intentionally API-focused and does not affect Next.js.
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    response.headers["Cache-Control"] = "no-store"


def _error(status_code: int, detail: str, request_id: str) -> JSONResponse:
    response = JSONResponse(
        status_code=status_code,
        content={"detail": detail, "request_id": request_id},
    )
    _security_headers(response, request_id)
    return response


async def gateway_middleware(request: Request, call_next) -> Response:
    """FastAPI middleware callback configured in ``main.create_app``."""
    settings = get_settings()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
    request.state.request_id = request_id
    start = time.perf_counter()

    # Public liveness/docs endpoints are deliberately light-weight.
    if request.url.path not in _PUBLIC_PREFIXES:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                return _error(400, "无效的 Content-Length 请求头", request_id)
            if declared_size > settings.GATEWAY_MAX_REQUEST_BYTES:
                return _error(413, "请求体超过网关允许的大小", request_id)

        # In production require an explicit allowed browser Origin for state-
        # changing requests. Non-browser callers may omit Origin and use bearer
        # auth; this remains compatible with service-to-service clients.
        origin = request.headers.get("origin")
        if (
            settings.GATEWAY_ENFORCE_ORIGIN
            and origin
            and request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and not _origin_allowed(origin, settings.CORS_ORIGINS)
        ):
            logger.warning("Gateway rejected origin=%s path=%s", origin, request.url.path)
            return _error(403, "请求来源不在允许列表中", request_id)

        # Edge rate limits by endpoint risk class. This reduces unnecessary
        # auth/embedding work before route dependencies execute.
        scope: str | None = None
        limit = 0
        if request.url.path in _AUTH_WRITE_PATHS:
            scope, limit = "gateway-auth", settings.GATEWAY_AUTH_RATE_LIMIT
        elif request.url.path == "/upload":
            scope, limit = "gateway-upload", settings.GATEWAY_UPLOAD_RATE_LIMIT
        elif request.url.path.startswith("/admin"):
            scope, limit = "gateway-admin", settings.GATEWAY_ADMIN_RATE_LIMIT

        if scope:
            allowed = await limiter.allow(
                f"{scope}:{client_ip(request)}",
                limit,
                settings.RATE_LIMIT_WINDOW_SECONDS,
            )
            if not allowed:
                response = _error(429, "请求过于频繁，请稍后再试", request_id)
                response.headers["Retry-After"] = str(settings.RATE_LIMIT_WINDOW_SECONDS)
                return response

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled gateway exception request_id=%s", request_id)
        return _error(500, "服务器内部错误", request_id)

    elapsed_ms = (time.perf_counter() - start) * 1000
    _security_headers(response, request_id)
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "gateway %s %s -> %d (%.2f ms) [%s]",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response
