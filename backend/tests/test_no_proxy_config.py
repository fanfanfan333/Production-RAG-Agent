"""出网代理豁免（``HTTP_NO_PROXY`` → ``NO_PROXY`` / ``no_proxy``）的回归。

## 为什么需要这份测试

本机实测过一条**极难反查**的故障链：Ollama 明明活着（.NET / 裸 socket 打
``http://localhost:11434`` 都是 200），但 Python 侧全都失败 —— `httpx` 拿回
**502 空响应体**，表现出来是 ``responseError('')``、"模型不可用"、"链路不稳定"。

根因：``httpx`` **只认 ``NO_PROXY`` 环境变量**，不读 Windows 注册表的
``ProxyOverride``（本机代理软件在里面写了 ``localhost;127.*`` 例外，
``urllib.proxy_bypass()`` 也认，但 httpx 认为不存在）→ 发给本机/内网服务的请求
被送到代理。后果不止测试：``vision_service._probe_sync()`` 探活失败 →
``vision`` 被判不可用 → 整条多模态链路**静默降级为纯 OCR**，启动阶段还不报错。

修复点是 ``app/config.py`` 的 ``apply_no_proxy_env()``：把 ``HTTP_NO_PROXY``
**只增不减**地并入 ``NO_PROXY`` / ``no_proxy``。

## 这份测试守住的契约

1. **合并语义**：只增不减、保序、大小写不敏感去重、接受分号分隔。
2. **不能吃掉运维配置**：已存在的条目一律保留（这是最容易写错、后果最难查的一条）。
3. **配置不能是死配置**：``get_settings()`` 必须真的把值物化进 ``os.environ``
   —— 本项目踩过"定义了但全仓没人读"的坑（``DOC_SUMMARY_TIMEOUT_SECONDS``）。
4. **真的对 httpx 有效**：用 httpx 自己的环境代理解析函数验证豁免主机确实变成
   "不走代理"，并配**对照组**证明这条断言不是空跑。
5. **配置真的写进文件了**：``.env`` / ``.env.example`` / ``docker-compose.yml``。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config import (
    Settings,
    _NO_PROXY_ENV_KEYS,
    _split_no_proxy,
    apply_no_proxy_env,
    merge_no_proxy,
)

_BACKEND = Path(__file__).resolve().parents[1]

#: 默认清单里必须覆盖的"必经之路"主机 —— httpx 少认一个就会静默降级。
_REQUIRED_HOSTS = ("localhost", "127.0.0.1", "::1", "host.docker.internal")
#: 容器内 compose 网络的服务名（非 Python 客户端也会用）
_CONTAINER_HOSTS = ("postgres", "qdrant", "keycloak")


def _class_default() -> str:
    """``HTTP_NO_PROXY`` 的**类默认值**（出厂契约，不受本机 .env 影响）。"""
    field = Settings.model_fields["HTTP_NO_PROXY"]
    return str(field.default or "")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 切分
# ─────────────────────────────────────────────────────────────────────────────


def test_split_accepts_comma_and_semicolon():
    """分号也要认 —— 注册表 ProxyOverride 惯用分号，运维复制过来很常见。"""
    assert _split_no_proxy("a.com;b.com, c.com") == ["a.com", "b.com", "c.com"]


def test_split_drops_blank_items_and_trims():
    assert _split_no_proxy(" a.com ,,  , b.com ") == ["a.com", "b.com"]


def test_split_dedupes_case_insensitively_keeping_first():
    assert _split_no_proxy("LocalHost,localhost,LOCALHOST") == ["LocalHost"]


@pytest.mark.parametrize("raw", [None, "", "   ", ",,,"])
def test_split_empty_input_gives_empty_list(raw):
    assert _split_no_proxy(raw) == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. 合并语义（只增不减）
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_appends_missing_keeping_existing_order():
    merged = merge_no_proxy("ops.internal", "localhost,127.0.0.1")
    assert merged == "ops.internal,localhost,127.0.0.1"


def test_merge_never_drops_operator_entries():
    """★ 最容易写错的契约：运维已配的条目必须一个不少。

    如果应用启动时把 NO_PROXY 整体覆盖成自己那份清单，运维配的内网域名、
    堡垒机、镜像站就会被**静默吃掉** —— 比原问题更难查。
    """
    ops = "registry.corp:5000,gitlab.corp,10.0.0.0/8"
    merged = merge_no_proxy(ops, "localhost,127.0.0.1")
    kept = merged.split(",")
    for entry in _split_no_proxy(ops):
        assert entry in kept, f"运维条目 {entry} 被吃掉了"
    assert merged.startswith(ops), "运维条目应保持在前"


def test_merge_dedupes_case_insensitively_without_reordering():
    merged = merge_no_proxy("LOCALHOST,a.com", "localhost,b.com")
    assert merged == "LOCALHOST,a.com,b.com"


def test_merge_required_empty_returns_existing_untouched():
    assert merge_no_proxy("ops.internal", "") == "ops.internal"
    assert merge_no_proxy("ops.internal", None) == "ops.internal"


def test_merge_both_empty_returns_empty_string():
    assert merge_no_proxy(None, None) == ""
    assert merge_no_proxy("", "  ") == ""


def test_merge_existing_none_returns_required():
    assert merge_no_proxy(None, "localhost,::1") == "localhost,::1"


def test_merge_is_idempotent():
    once = merge_no_proxy("ops.internal", "localhost,127.0.0.1")
    assert merge_no_proxy(once, "localhost,127.0.0.1") == once


# ─────────────────────────────────────────────────────────────────────────────
# 3. 写入 os.environ（httpx 真正读的地方）
# ─────────────────────────────────────────────────────────────────────────────


def _clean_env(monkeypatch) -> None:
    for key in _NO_PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_apply_writes_both_env_keys(monkeypatch):
    """``NO_PROXY`` 与 ``no_proxy`` 都要写。

    requests / urllib 读小写 ``no_proxy``，httpx 读大写 ``NO_PROXY`` ——
    只写一个，另一条链路仍然走代理。
    """
    _clean_env(monkeypatch)
    final = apply_no_proxy_env(_Settings(HTTP_NO_PROXY="localhost,127.0.0.1"))
    assert final == "localhost,127.0.0.1"
    for key in _NO_PROXY_ENV_KEYS:
        assert os.environ[key] == "localhost,127.0.0.1", key


def test_apply_preserves_operator_entries(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("NO_PROXY", "ops.internal")
    monkeypatch.setenv("no_proxy", "ops.internal")
    final = apply_no_proxy_env(_Settings(HTTP_NO_PROXY="localhost"))
    assert "ops.internal" in final and "localhost" in final
    assert os.environ["no_proxy"] == final


def test_apply_is_idempotent(monkeypatch):
    _clean_env(monkeypatch)
    first = apply_no_proxy_env(_Settings(HTTP_NO_PROXY="localhost,127.0.0.1"))
    second = apply_no_proxy_env(_Settings(HTTP_NO_PROXY="localhost,127.0.0.1"))
    assert first == second
    assert os.environ["NO_PROXY"].count("localhost") == 1


def test_apply_empty_config_writes_nothing(monkeypatch):
    """置空 = 关闭本机制：不得凭空造出 NO_PROXY。"""
    _clean_env(monkeypatch)
    assert apply_no_proxy_env(_Settings(HTTP_NO_PROXY="")) == ""
    for key in _NO_PROXY_ENV_KEYS:
        assert key not in os.environ, key


def test_apply_empty_config_leaves_existing_alone(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("NO_PROXY", "ops.internal")
    apply_no_proxy_env(_Settings(HTTP_NO_PROXY=""))
    assert os.environ["NO_PROXY"] == "ops.internal"


def test_apply_tolerates_settings_without_the_field(monkeypatch):
    """字段缺失（旧配置对象 / 替身）时不能抛 —— 用 getattr 兜底。"""
    _clean_env(monkeypatch)
    assert apply_no_proxy_env(object()) == ""  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# 4. 出厂默认值确实覆盖必经之路
# ─────────────────────────────────────────────────────────────────────────────


def test_class_default_covers_host_and_container_hosts():
    default = _class_default()
    assert default, "HTTP_NO_PROXY 出厂默认值不能为空"
    hosts = {h.lower() for h in _split_no_proxy(default)}
    for host in _REQUIRED_HOSTS:
        assert host in hosts, f"出厂默认值缺少 {host}"
    for host in _CONTAINER_HOSTS:
        assert host in hosts, f"出厂默认值缺少容器内服务名 {host}"


def test_shipped_env_default_is_not_narrower_than_class_default():
    """本机 .env 若覆盖了该字段，不得比出厂值更窄（否则等于偷偷关掉豁免）。"""
    env_file = _BACKEND / ".env"
    if not env_file.exists():
        pytest.skip(".env 不在仓库里（CI / 新克隆）")
    value = ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("HTTP_NO_PROXY="):
            value = line.split("=", 1)[1].strip()
    if not value:
        pytest.skip(".env 未显式设置 HTTP_NO_PROXY（走出厂默认值）")
    hosts = {h.lower() for h in _split_no_proxy(value)}
    for host in _REQUIRED_HOSTS:
        assert host in hosts, f".env 的 HTTP_NO_PROXY 缺少 {host}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. 死配置守卫：get_settings() 必须真的物化到 os.environ
# ─────────────────────────────────────────────────────────────────────────────


def test_get_settings_materialises_no_proxy_into_environ(monkeypatch):
    """★ 本项目踩过的坑：定义了却全仓没人读（``DOC_SUMMARY_TIMEOUT_SECONDS``）。

    只断言"字段存在"是不够的 —— 必须证明**启动路径真的把它写进了
    ``os.environ``**，否则 httpx 一个字都读不到，字段就是个摆设。
    """
    import app.config as cfg

    _clean_env(monkeypatch)
    cfg.get_settings.cache_clear()
    try:
        settings = cfg.get_settings()
        assert settings.HTTP_NO_PROXY, "HTTP_NO_PROXY 不应为空"
        for key in _NO_PROXY_ENV_KEYS:
            assert os.environ.get(key), (
                f"get_settings() 没有把 HTTP_NO_PROXY 物化到 os.environ[{key!r}] "
                "—— httpx 读不到，豁免失效"
            )
        effective = os.environ["NO_PROXY"]
        for host in _REQUIRED_HOSTS:
            assert host in effective, f"生效的 NO_PROXY 里没有 {host}"
    finally:
        cfg.get_settings.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# 6. 真的对 httpx 有效（用 httpx 自己的解析函数）+ 对照组
# ─────────────────────────────────────────────────────────────────────────────

try:  # httpx 内部函数（0.28.1 实测存在）
    from httpx._utils import get_environment_proxies as _get_environment_proxies
except Exception:  # noqa: BLE001
    _get_environment_proxies = None  # type: ignore[assignment]
#: 注：它被 skip 而**不是**直接失败，是因为一旦 httpx 改名，本文件的价值会降到
#: "只验证了合并逻辑"；上面的死配置守卫与文件守卫仍然独立成立。见报告说明。
_httpx_ok = _get_environment_proxies is not None


def _bypass_hosts(proxies: dict[str, str | None]) -> set[str]:
    """httpx 约定：值为 ``None`` 的条目 = **不走代理**。"""
    return {key for key, value in proxies.items() if value is None}


def _with_fake_proxy(monkeypatch, *, no_proxy: str | None):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7892")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7892")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:7892")
    if no_proxy is None:
        _clean_env(monkeypatch)
    else:
        monkeypatch.setenv("NO_PROXY", no_proxy)
        monkeypatch.setenv("no_proxy", no_proxy)
    return _get_environment_proxies()


@pytest.mark.skipif(not _httpx_ok, reason="httpx 内部 API 变动（get_environment_proxies 不存在）")
def test_httpx_bypasses_proxy_for_every_exempt_host(monkeypatch):
    """★ 证明豁免对 httpx 真生效 —— 不是"我们写了环境变量"就算数。"""
    proxies = _with_fake_proxy(monkeypatch, no_proxy=_class_default())
    bypass = _bypass_hosts(proxies)
    assert bypass, f"httpx 没有识别出任何豁免主机: {proxies!r}"
    for host in _REQUIRED_HOSTS:
        assert any(host in entry for entry in bypass), (
            f"httpx 不会豁免 {host}（仍会走代理）: {sorted(bypass)}"
        )
    # 代理本身仍对**其他**主机生效 —— 别把代理整体关了
    assert any(value for value in proxies.values()), "外部主机仍应走代理"


@pytest.mark.skipif(not _httpx_ok, reason="httpx 内部 API 变动（get_environment_proxies 不存在）")
def test_control_without_no_proxy_everything_goes_through_proxy(monkeypatch):
    """对照组：**不设** NO_PROXY 时，本机地址也会被送去代理。

    这条证明上一个用例不是空跑 —— 豁免确实来自我们的环境变量，
    而不是 Windows 注册表里那份 httpx 看不见的例外清单。
    """
    proxies = _with_fake_proxy(monkeypatch, no_proxy=None)
    assert _bypass_hosts(proxies) == set(), (
        f"没有 NO_PROXY 时不该有任何豁免: {proxies!r}"
    )
    assert any(value for value in proxies.values()), "此时应有代理生效（否则对照无效）"


# ─────────────────────────────────────────────────────────────────────────────
# 7. 配置真的写进了交付文件（防止被误删）
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [".env", ".env.example"])
def test_env_files_declare_http_no_proxy(name: str):
    path = _BACKEND / name
    if not path.exists():
        pytest.skip(f"{name} 不在仓库里")
    text = path.read_text(encoding="utf-8")
    assert "HTTP_NO_PROXY=" in text, f"{name} 里少了 HTTP_NO_PROXY"
    assert "localhost" in text


def test_compose_passes_no_proxy_to_backend():
    compose = _BACKEND / "docker-compose.yml"
    if not compose.exists():
        pytest.skip("docker-compose.yml 不在仓库里")
    text = compose.read_text(encoding="utf-8")
    assert "NO_PROXY:" in text, "compose 未给 backend 声明 NO_PROXY"
    assert "host.docker.internal" in text
    # ⚠️ 必须是可被运维接管的插值，而不是字面量：`environment` 优先级高于
    # `env_file`，写字面量会把运维在 .env 里配的 NO_PROXY 静默吃掉。
    assert "${NO_PROXY:-" in text, "compose 里应写成 ${NO_PROXY:-...} 以允许运维接管"


# ─────────────────────────────────────────────────────────────────────────────
# 替身
# ─────────────────────────────────────────────────────────────────────────────


class _Settings:
    """最小替身：只带 ``HTTP_NO_PROXY``，避免动到真实的 lru_cache 单例。"""

    def __init__(self, *, HTTP_NO_PROXY: str) -> None:  # noqa: N803 - 对齐字段名
        self.HTTP_NO_PROXY = HTTP_NO_PROXY
