"""Controlled web search and page reading.

No browser, no arbitrary proxy, no "pretend success":

* search requires an explicitly configured provider; unconfigured or failing
  search raises :class:`ResearchUnavailable`;
* every candidate URL, and every redirect hop, is resolved and checked against
  loopback / private / link-local / reserved address space;
* reads are bounded by timeout, redirect count, content type and byte size;
* a provider response marked as mock/simulated is rejected rather than shown as
  real evidence.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

MAX_REDIRECTS = 3
MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_SEARCH_RESULTS = 5
_ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")


class ResearchUnavailable(RuntimeError):
    """The configured research capability is absent or failed."""


class UnsafeUrl(ValueError):
    """A URL points at a non-public or otherwise forbidden destination."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str
    truncated: bool


Resolver = Callable[[str, int], Awaitable[list[tuple[Any, ...]]]]


async def _default_resolver(host: str, port: int) -> list[tuple[Any, ...]]:
    return await asyncio.to_thread(
        socket.getaddrinfo, host, port, type=socket.SOCK_STREAM
    )


def _is_public_address(raw: str) -> bool:
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return False
    return bool(
        addr.is_global
        and not addr.is_private
        and not addr.is_loopback
        and not addr.is_link_local
        and not addr.is_multicast
        and not addr.is_reserved
        and not addr.is_unspecified
    )


async def _validated_address(url: str, resolver: Resolver) -> tuple[str, str]:
    """Validate one absolute http(s) URL and all of its resolved addresses.

    Returns the normalized URL. Rejects credentials in the URL, non-standard
    ports, localhost-style names and any host that resolves to a non-public IP.
    Every redirect target must pass through this function again.
    """
    parts = urlsplit((url or "").strip())
    if parts.scheme not in {"http", "https"}:
        raise UnsafeUrl("仅允许 http/https 地址")
    if parts.username or parts.password:
        raise UnsafeUrl("地址不得包含凭据")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeUrl("地址缺少主机名")
    if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UnsafeUrl("不允许访问本机或内网主机")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeUrl("端口非法") from exc
    if port not in {80, 443}:
        raise UnsafeUrl("仅允许标准 HTTP(S) 端口")

    # An IP literal needs no DNS lookup, but still needs the public-address rule.
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        try:
            infos = await resolver(host, port)
        except OSError as exc:
            raise ResearchUnavailable(f"域名解析失败: {host}") from exc
        addresses = [str(info[4][0]) for info in infos]
        if not addresses:
            raise ResearchUnavailable(f"域名未解析到地址: {host}")

    if any(not _is_public_address(address) for address in addresses):
        raise UnsafeUrl("地址解析到本机、内网或保留网段")
    return parts.geturl(), addresses[0]


async def validate_public_url(url: str, resolver: Resolver = _default_resolver) -> str:
    return (await _validated_address(url, resolver))[0]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self._title: list[str] = []
        self._in_title = False
        self._body: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self._body.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self._title.append(data)
        self._body.append(data)

    def result(self) -> tuple[str, str]:
        title = " ".join("".join(self._title).split())
        text = " ".join("".join(self._body).split())
        return title, text


def _extract(content: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(content)
    title, text = parser.result()
    return title, html.unescape(text)


class ControlledWebResearch:
    """Search + read, with injectable resolver/transport for tests."""

    def __init__(
        self,
        *,
        search_endpoint: str = "",
        search_api_key: str = "",
        timeout_seconds: float = 20.0,
        resolver: Resolver = _default_resolver,
        transport: Any | None = None,
    ) -> None:
        self.search_endpoint = (search_endpoint or "").strip()
        self.search_api_key = (search_api_key or "").strip()
        self.timeout_seconds = max(0.01, float(timeout_seconds))
        self.resolver = resolver
        self.transport = transport

    def _client(self):
        # Imported lazily so this module remains importable in the lightweight
        # host test shim; calling research without httpx is an honest failure.
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency is in requirements
            raise ResearchUnavailable("HTTP 客户端未安装，联网研究不可用") from exc
        return httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=self.transport,
        )

    async def search(self, query: str) -> tuple[list[SearchResult], str]:
        """Call the configured provider. Never synthesizes results."""
        if not self.search_endpoint:
            raise ResearchUnavailable(
                "未配置联网搜索服务（WEB_SEARCH_ENDPOINT），联网研究不可用"
            )
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.search_api_key:
            headers["Authorization"] = f"Bearer {self.search_api_key}"
        try:
            async with self._client() as client:
                response = await asyncio.wait_for(
                    client.post(
                        self.search_endpoint,
                        json={"query": query, "max_results": MAX_SEARCH_RESULTS},
                        headers=headers,
                    ),
                    timeout=self.timeout_seconds,
                )
        except Exception as exc:  # noqa: BLE001 - transport failure is reported honestly
            raise ResearchUnavailable(f"联网搜索请求失败: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise ResearchUnavailable(f"联网搜索服务返回 HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ResearchUnavailable("联网搜索服务返回的 JSON 无法解析") from exc

        if not isinstance(payload, dict):
            raise ResearchUnavailable("联网搜索服务返回了非对象结果")
        if payload.get("mock") is True or payload.get("simulated") is True:
            raise ResearchUnavailable("联网搜索服务返回模拟数据，已拒绝作为真实来源")

        raw_items = payload.get("results")
        if not isinstance(raw_items, list):
            raise ResearchUnavailable("联网搜索服务未返回 results 列表")
        results: list[SearchResult] = []
        for item in raw_items[:MAX_SEARCH_RESULTS]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = " ".join(str(item.get("title") or "").split())[:300]
            snippet = " ".join(str(item.get("snippet") or item.get("content") or "").split())[:800]
            if not url:
                continue
            try:
                await validate_public_url(url, self.resolver)
            except (UnsafeUrl, ResearchUnavailable):
                continue
            results.append(SearchResult(title=title or url, url=url, snippet=snippet))
        if not results:
            raise ResearchUnavailable("联网搜索没有返回可安全访问的真实结果")
        provider = str(payload.get("provider") or "configured provider")
        return results, provider

    async def read_url(self, url: str) -> Page:
        """Read one page with manual, revalidated redirects and a byte ceiling."""
        async with self._client() as client:
            current = url
            for hop in range(MAX_REDIRECTS + 1):
                current, address = await _validated_address(current, self.resolver)
                parts = urlsplit(current)
                host = parts.hostname or ""
                netloc = f"[{address}]" if ":" in address else address
                if parts.port:
                    netloc += f":{parts.port}"
                target = urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
                host_header = parts.netloc
                try:
                    async with client.stream("GET", target, headers={"Host": host_header}, extensions={"sni_hostname": host}) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location", "")
                            if not location:
                                raise ResearchUnavailable("重定向缺少 Location")
                            if hop >= MAX_REDIRECTS:
                                raise ResearchUnavailable("重定向次数超过上限")
                            current = urljoin(current, location)
                            continue
                        if response.status_code >= 400:
                            raise ResearchUnavailable(
                                f"网页读取返回 HTTP {response.status_code}"
                            )
                        content_type = response.headers.get("content-type", "").lower()
                        if not any(kind in content_type for kind in _ALLOWED_CONTENT_TYPES):
                            raise ResearchUnavailable(
                                f"网页内容类型不受支持: {content_type or '未知'}"
                            )
                        chunks: list[bytes] = []
                        total = 0
                        truncated = False
                        async for chunk in response.aiter_bytes():
                            total += len(chunk)
                            if total > MAX_PAGE_BYTES:
                                chunks.append(chunk[: max(0, MAX_PAGE_BYTES - (total - len(chunk)))])
                                truncated = True
                                break
                            chunks.append(chunk)
                        raw = b"".join(chunks)
                except (UnsafeUrl, ResearchUnavailable):
                    raise
                except Exception as exc:  # noqa: BLE001 - timeout/connection are typed
                    raise ResearchUnavailable(
                        f"网页读取失败: {type(exc).__name__}"
                    ) from exc
                charset = response.encoding or "utf-8"
                try:
                    content = raw.decode(charset, errors="replace")
                except LookupError:
                    content = raw.decode("utf-8", errors="replace")
                title, text = _extract(content)
                return Page(
                    url=current,
                    title=title[:300],
                    text=text[:20000],
                    truncated=truncated or len(text) > 20000,
                )
        raise ResearchUnavailable("网页读取未完成")
