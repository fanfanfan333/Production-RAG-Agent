import asyncio
import uuid
from types import SimpleNamespace

from starlette.requests import Request

from app.api import query
from app.middleware.gateway import _security_headers
from app.schemas.query import QueryRequest


def test_query_stream_prevents_proxy_compression_buffering(monkeypatch):
    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(query, "enforce_rate_limit", noop)
    monkeypatch.setattr(query, "record_audit", noop)
    request = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1234)})
    user = SimpleNamespace(id=uuid.uuid4(), username="qa")
    response = asyncio.run(query.query_endpoint(QueryRequest(query="设备功率是多少？"), request, user))
    assert response.media_type == "text/event-stream"
    assert "no-transform" in response.headers["cache-control"]
    assert "no-cache" in response.headers["cache-control"]
    assert response.headers["x-accel-buffering"] == "no"
    _security_headers(response, "stream-request")
    assert response.headers["cache-control"] == "no-store, no-transform"


def test_gateway_still_disallows_caching_normal_responses():
    from fastapi.responses import JSONResponse

    response = JSONResponse({"private": True}, headers={"Cache-Control": "public, max-age=3600"})
    _security_headers(response, "normal-request")
    assert response.headers["cache-control"] == "no-store"
