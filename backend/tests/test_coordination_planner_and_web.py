"""Pure tests for coordinator decomposition and controlled web access.

These do not touch PostgreSQL, Qdrant or Ollama, so they stay meaningful on the
host and in CI. The network is a deterministic httpx MockTransport; no test
claims that a live provider was reachable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from app.coordination.planner import MAX_SUBTASKS, plan_request  # noqa: E402
from app.coordination.web_research import (  # noqa: E402
    ControlledWebResearch,
    ResearchUnavailable,
    UnsafeUrl,
    validate_public_url,
)


async def _public(host: str, port: int):
    return [(2, 1, 6, "", ("93.184.216.34", port))]


async def _private(host: str, port: int):
    return [(2, 1, 6, "", ("10.0.0.7", port))]


def test_document_question_keeps_the_existing_default():
    plan = plan_request("这份合同的违约条款是什么？")
    assert plan.kinds == ("document",)
    assert plan.subtasks[0].objective == "这份合同的违约条款是什么？"


def test_complex_request_fans_out_to_bounded_specialists():
    plan = plan_request("先查文档里的接口说明，再联网搜索最新规范，然后改代码并运行测试")
    assert plan.kinds == ("document", "research", "code")
    assert len(plan.subtasks) <= MAX_SUBTASKS
    assert len({item.kind for item in plan.subtasks}) == len(plan.subtasks)
    assert plan.subtasks[0].depends_on == ()
    assert plan.subtasks[1].depends_on == ()
    assert plan.subtasks[2].depends_on == ("document", "research")


def test_explicit_kind_bypasses_marker_guessing():
    assert plan_request("修复登录页", kind="research").kinds == ("research",)


def test_unknown_kind_and_empty_query_are_rejected():
    with pytest.raises(ValueError, match="未知任务类型"):
        plan_request("x", kind="shell")
    with pytest.raises(ValueError, match="不能为空"):
        plan_request("   ")


def test_url_guard_blocks_private_literal_and_private_dns():
    async def case() -> None:
        with pytest.raises(UnsafeUrl):
            await validate_public_url("http://127.0.0.1/admin", _public)
        with pytest.raises(UnsafeUrl, match="内网"):
            await validate_public_url("https://docs.example.com/a", _private)
        assert (
            await validate_public_url("https://93.184.216.34/a", _public)
            == "https://93.184.216.34/a"
        )

    asyncio.run(case())


def test_unconfigured_search_is_unavailable_not_empty_success():
    async def case() -> None:
        client = ControlledWebResearch(search_endpoint="", resolver=_public)
        with pytest.raises(ResearchUnavailable, match="未配置"):
            await client.search("anything")

    asyncio.run(case())


def test_mock_search_payload_is_rejected():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"mock": True, "results": [{"title": "x", "url": "https://example.com"}]}
        )
    )
    client = ControlledWebResearch(
        search_endpoint="https://search.example.com",
        resolver=_public,
        transport=transport,
    )
    with pytest.raises(ResearchUnavailable, match="模拟数据"):
        asyncio.run(client.search("q"))


def test_search_filters_private_result_and_reports_real_provider():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "provider": "real-test-provider",
                "results": [
                    {"title": "private", "url": "http://10.0.0.9/secret"},
                    {"title": "public", "url": "https://example.com/doc", "snippet": "ok"},
                ],
            },
        )
    )
    client = ControlledWebResearch(
        search_endpoint="https://search.example.com",
        resolver=_public,
        transport=transport,
    )
    results, provider = asyncio.run(client.search("q"))
    assert provider == "real-test-provider"
    assert [item.url for item in results] == ["https://example.com/doc"]


def test_redirect_to_private_host_is_blocked():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    client = ControlledWebResearch(resolver=_public, transport=httpx.MockTransport(handler))
    with pytest.raises(UnsafeUrl):
        asyncio.run(client.read_url("https://example.com/start"))
    assert calls == ["https://93.184.216.34/start"]


def test_page_read_is_bounded_and_extracts_text():
    body = "<html><title>Doc</title><body><script>bad()</script><p>Hello &amp; world</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=body)

    client = ControlledWebResearch(resolver=_public, transport=httpx.MockTransport(handler))
    page = asyncio.run(client.read_url("https://example.com/doc"))
    assert page.title == "Doc"
    assert "Hello & world" in page.text
    assert "bad()" not in page.text


def test_page_connection_uses_checked_address_and_original_tls_hostname():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="ok")

    client = ControlledWebResearch(resolver=_public, transport=httpx.MockTransport(handler))
    assert asyncio.run(client.read_url("https://example.com/doc")).text == "ok"


def test_real_tls_checks_original_hostname(tmp_path, monkeypatch):
    import datetime
    import ipaddress
    import ssl
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import app.coordination.web_research as web

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "valid.example")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=1))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("valid.example")]), critical=False)
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    monkeypatch.setattr(web, "_is_public_address", lambda address: ipaddress.ip_address(address).is_loopback)

    async def resolver(host, port):
        return [(2, 1, 6, "", ("127.0.0.1", port))]

    async def serve(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def case():
        try:
            server = await asyncio.start_server(serve, "127.0.0.1", 443, ssl=context)
        except OSError:
            pytest.skip("local HTTPS port 443 unavailable")
        try:
            transport = httpx.AsyncHTTPTransport(verify=ssl.create_default_context(cafile=str(cert_path)))
            client = ControlledWebResearch(resolver=resolver, transport=transport)
            assert (await client.read_url("https://valid.example/")).text == "ok"
            with pytest.raises(ResearchUnavailable):
                await client.read_url("https://wrong.example/")
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(case())


def test_search_timeout_is_reported_as_unavailable():
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={"results": []})

    client = ControlledWebResearch(
        search_endpoint="https://search.example.com",
        timeout_seconds=0.01,
        resolver=_public,
        transport=httpx.MockTransport(slow),
    )
    with pytest.raises(ResearchUnavailable, match="请求失败"):
        asyncio.run(client.search("q"))
