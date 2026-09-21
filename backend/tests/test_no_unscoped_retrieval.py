"""**AST 静态门禁**：任何检索调用都必须显式绑定 Scope（决策 10 第③④层）.

用户红线原文：「任何用户的 Query 在进入检索器时就必须绑定 UserScope，禁止先
检索全库再依赖 Prompt 做权限隔离」。

这条红线靠什么**机制**成立，而不是靠"约定"：

    ① 签名强制     ``retrieve_chunks`` 的既有 fail-closed（无权限上下文即
                   ``return []``）继续生效 ⇒ 漏传的后果是空结果，不是全库
    ② 运行时不可变  ``UserScope`` / ``ScopePredicate`` 均 frozen + frozenset
    ③ CI 静态门禁   本文件：AST 遍历 ``app/**``，断言每个检索调用都带
                   ``scope=`` 或 ``unrestricted=True``（或落在显式白名单里）
    ④ 禁止链路内重签发  同一门禁的第二条断言：``request_scope`` /
                   ``request_security_scope`` 只允许出现在 ``app/api/**``
                   与白名单内 —— 链路内不得重新查库放宽 scope

为什么必须带**阳性对照**
────────────────────────
没有阳性对照的门禁是假门禁：一个"只收集、从不断言"或"路径写错扫不到任何文件"
的测试永远是绿的。本文件因此有两条方向的断言：

    ``test_gate_catches_missing_scope``   故意构造**漏传**的样例 ⇒ 必须被抓住
    ``test_gate_accepts_scoped_call``     合规样例 ⇒ 必须不被误报

只有这两条同时成立，"当前门禁通过"才是有信息的。

白名单纪律
──────────
``UNSCOPED_ALLOWLIST`` 是**显式写在测试里**的常量。任何人要往里加一行，
都必须改这个测试、并在 PR 里被看见 —— 这是把"禁止绕过"变成可执行门禁的
关键。当前条目全部是 **T3 接入之前就存在的存量调用点**；T3 完成后应逐条清空
（``test_allowlist_paths_still_exist`` 会保证白名单不腐化成空指针）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
APP_ROOT = BACKEND_ROOT / "app"

#: 需要绑定 scope 的**函数名**（直接调用）
TARGET_FUNCTIONS = frozenset(
    {
        "retrieve_chunks",
        "retrieve_chunks_scoped",
        "keyword_search",
    }
)

#: 需要绑定 scope 的**向量库方法名**。判定时要求宿主名里含 "client"
#: （``client.`` / ``self.client.`` / ``_client.``），否则 ``re.search()`` 这类
#: 正则调用会被大面积误报；反过来，任何以它调用向量库的形式都逃不掉。
TARGET_CLIENT_METHODS = frozenset({"search", "scroll", "query_points"})

#: 签发 Scope 的函数 —— **只允许在 ``app/api/**``（或白名单）出现**（决策 10-④）。
#:
#: 精确等于设计文档 §10-④「`request_scope(` / `request_security_scope(` 只允许出现在
#: `app/api/**` 与白名单内」+ §验收表补入的 ``scope_for``（它是 ``request_scope``
#: 的纯函数内核）。
#:
#: ⚠️ **``content_scope`` 不在其中**：它返回的是既有三维 ``DocumentScope``（基座），
#: 是**唯一签发函数 ``request_security_scope`` 内部合法调用**的一环，不是"签发点"。
#: 把它列进来只会制造一处自伤式误报（``security_scope.py`` 自己）。真正要拦的是
#: "链路里有人**重新组装并放宽** scope"，那三个函数才是入口。
SCOPE_ISSUERS = frozenset({"request_scope", "request_security_scope", "scope_for"})

#: 检测到下列任一关键字参数即视为"已绑定 Scope"
BINDING_KWARGS = frozenset({"scope", "unrestricted", "pred"})

#: 【存量】当前仍未绑定 Scope 的调用点。**只允许减少，不允许增加。**
#:
#: 全部是"T3 接入之前就存在的底层调用"：向量库的两个原始方法（``scroll`` /
#: ``search``，它们的过滤器由**上层的** ``to_qdrant(pred)`` 编译后传入，因此在
#: 它们身上看不到 ``scope=``）与评测端点的兼容调用。T3 完成后应逐条清空。
#: 任何人要加一行，都必须在 PR 里被看见。
UNSCOPED_ALLOWLIST = frozenset(
    {
        "backend/app/api/evaluation.py::retrieve_chunks",
        "backend/app/services/document_query_service.py::client.scroll",
        "backend/app/services/relation_service.py::client.scroll",
        "backend/app/services/retrieval_service.py::client.scroll",
        "backend/app/services/retrieval_service.py::client.search",
    }
)

#: 【存量 / 模块内组合】链路内签发 Scope 的例外（收敛到 api 层之前）。
#: ``app/api/**`` 由扫描器自动豁免，不需要写在这里。
#:
#: 两类条目：
#:   1. **issuer 模块内部的合法组合**（``tenancy`` 自己）：``content_scope`` → ``request_scope``
#:      → ``scope_for``，这是三维 scope 的既有组装链，不是"链路内放宽"；
#:   2. **存量服务层签发点**：它们**先于 T2 就存在**，本轮不动（零行为变化红线），
#:      但按决策 10-④ 应改为"从 api 层透传 scope"。责任人不是 T2，已在回传里点出。
#:
#: 任何人要加一行，都必须在 PR 里被看见。
SCOPE_ISSUER_ALLOWLIST = frozenset(
    {
        # ① issuer 模块内部组合（tenancy.py 自己）：request_scope → scope_for
        "backend/app/services/tenancy.py::scope_for",
        # content_scope() 内部复用 request_scope()（同一模块内的既有组合）
        "backend/app/services/tenancy.py::request_scope",
        # ② 存量服务层签发点（T5 应改为从 api 层透传）
        # 共享审批链路自己签发 reviewer scope
        "backend/app/services/share_service.py::request_scope",
        # 文档代理链路为"源文档访问复核"重新签发三维 scope
        "backend/app/services/document_agent_service.py::request_scope",
    }
)


# ═══════════════════════════════════════════════════════════════════════════════
# 扫描器（纯函数，可对任意源码字符串运行 ⇒ 可做阳性对照）
# ═══════════════════════════════════════════════════════════════════════════════


def dotted_name(node: ast.AST) -> str:
    """``client.scroll`` / ``retrieve_chunks`` 这类调用目标的点号名。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def is_target_call(name: str) -> bool:
    """是否属于"必须绑定 Scope"的检索调用。"""
    if name in TARGET_FUNCTIONS:
        return True
    head, _, tail = name.rpartition(".")
    if tail in TARGET_CLIENT_METHODS and "client" in head.lower():
        return True
    return False


def scan_source(source: str, rel_path: str) -> list[dict]:
    """扫描一段源码，返回"未绑定 Scope 的检索调用"清单。"""
    offenders: list[dict] = []
    tree = ast.parse(source, filename=rel_path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted_name(node.func)
        if not is_target_call(name):
            continue
        kwargs = {k.arg for k in node.keywords if k.arg}
        if not (BINDING_KWARGS & kwargs):
            offenders.append({"path": rel_path, "name": name, "lineno": node.lineno})
    return offenders


def scan_issuers(source: str, rel_path: str) -> list[dict]:
    """扫描一段源码，返回"链路内签发 Scope"的清单（api 层除外）。"""
    if "/api/" in f"/{rel_path}":
        return []
    offenders: list[dict] = []
    tree = ast.parse(source, filename=rel_path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted_name(node.func)
        if name in SCOPE_ISSUERS:
            offenders.append({"path": rel_path, "name": name, "lineno": node.lineno})
    return offenders


def iter_app_files() -> list[Path]:
    return sorted(p for p in APP_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def scan_repo() -> tuple[list[dict], list[dict]]:
    unscoped: list[dict] = []
    issuers: list[dict] = []
    for path in iter_app_files():
        source = path.read_text(encoding="utf-8")
        rel = _rel(path)
        unscoped.extend(scan_source(source, rel))
        issuers.extend(scan_issuers(source, rel))
    return unscoped, issuers


# ═══════════════════════════════════════════════════════════════════════════════
# 断言 ①②④：真实仓库
# ═══════════════════════════════════════════════════════════════════════════════


def test_no_unscoped_retrieval_calls() -> None:
    """每个检索调用点都必须带 ``scope=`` / ``unrestricted=`` / ``pred=``。"""
    unscoped, _ = scan_repo()
    offenders = [
        f"{o['path']}:{o['lineno']} {o['name']}()"
        for o in unscoped
        if f"{o['path']}::{o['name']}" not in UNSCOPED_ALLOWLIST
    ]
    assert not offenders, (
        "检索调用必须显式绑定 Scope（scope= / pred=）或显式声明 unrestricted=True。\n"
        "要么补参数，要么把它加进 UNSCOPED_ALLOWLIST 并在 PR 里被看见：\n  "
        + "\n  ".join(offenders)
    )


def test_scope_is_issued_only_at_the_api_layer() -> None:
    """决策 10-④：链路内不得重新签发 Scope（不得中途查库放宽范围）。"""
    _, issuers = scan_repo()
    offenders = [
        f"{o['path']}:{o['lineno']} {o['name']}()"
        for o in issuers
        if f"{o['path']}::{o['name']}" not in SCOPE_ISSUER_ALLOWLIST
    ]
    assert not offenders, (
        "Scope 只能在 app/api/** 层签发（AST 门禁）。\n"
        "服务层需要 Scope 时应当**接收**它，而不是自己重新签发：\n  "
        + "\n  ".join(offenders)
    )


def test_allowlist_paths_still_exist() -> None:
    """白名单防腐化：条目指向的文件必须还在（改名/删除后必须同步清理）。"""
    missing = []
    for entry in UNSCOPED_ALLOWLIST | SCOPE_ISSUER_ALLOWLIST:
        path_part = entry.split("::")[0]
        if not (REPO_ROOT / path_part).exists():
            missing.append(entry)
    assert not missing, f"白名单指向了不存在的文件，请清理：{missing}"


def test_allowlist_does_not_grow_silently() -> None:
    """
    **棘轮**：真正未绑定的调用点（按 ``文件::函数`` 去重）不得超过白名单规模。

    口径必须与 :func:`test_no_unscoped_retrieval_calls` **一致** —— 后者是按
    ``文件::函数`` 判白名单的，所以一个函数体里的第 2、第 3 个调用点本就同属一条
    白名单条目（例如 ``tenancy.py::scope_for`` 在 ``request_scope`` 里有 3 个调用点）。
    这里同样按去重键计数，否则"同一函数多调几次"会平白触发棘轮。

    新增一个**未绑定 Scope 的函数**，就必须同时删掉一条白名单条目 —— 净增会被
    这条断言拦下（逼着人正视"为什么又多了一处"）。
    """
    unscoped, issuers = scan_repo()
    unscoped_keys = {(o["path"], o["name"]) for o in unscoped}
    issuer_keys = {(o["path"], o["name"]) for o in issuers}
    assert len(unscoped_keys) <= len(UNSCOPED_ALLOWLIST), (
        f"未绑定 Scope 的函数 {len(unscoped_keys)} 个 > 白名单 {len(UNSCOPED_ALLOWLIST)} 条 —— "
        "白名单是**只减不增**的棘轮，请补 scope= 而不是加白名单"
    )
    assert len(issuer_keys) <= len(SCOPE_ISSUER_ALLOWLIST), (
        f"链路内签发 Scope 的函数 {len(issuer_keys)} 个 > 白名单 {len(SCOPE_ISSUER_ALLOWLIST)} 条"
    )


def test_scanner_actually_sees_call_sites() -> None:
    """
    扫描器自检：必须真的扫到东西，否则"门禁通过"毫无信息量。

    这是一个**阴性对照**：如果扫描路径写错（比如 ``APP_ROOT`` 指错目录、
    ``TARGET_CALLS`` 拼错），``scan_repo()`` 会返回空 —— 此时所有门禁都恒绿。
    """
    assert APP_ROOT.is_dir(), f"扫描根目录不存在: {APP_ROOT}"
    files = iter_app_files()
    assert len(files) > 50, f"扫描到的 app 源码文件太少（{len(files)}）—— 路径配错了吧"

    unscoped, issuers = scan_repo()
    assert unscoped, "一个检索调用点都没扫到 —— 扫描器失效了（假门禁）"
    assert issuers, "一个 Scope 签发点都没扫到 —— 扫描器失效了（假门禁）"


# ═══════════════════════════════════════════════════════════════════════════════
# 断言③：**阳性对照**（故意漏传 ⇒ 门禁必须抓到）
# ═══════════════════════════════════════════════════════════════════════════════

_MISSING_SCOPE_SAMPLE = '''
async def bad_handler(request, user_scope):
    """故意漏传 scope：门禁必须抓到它。"""
    return await retrieve_chunks(
        "query", top_k=5, owner_id="u1", tenant_ids=frozenset({"t"}),
    )


async def bad_scroll(client):
    return await client.scroll(collection_name="documents", limit=10)


async def bad_self_client(self):
    return await self.client.search(collection_name="documents", limit=10)
'''

_SCOPED_SAMPLE = '''
async def good_handler(pred):
    """合规形态：显式绑定 pred（= ScopePredicate）。"""
    return await retrieve_chunks("query", top_k=5, pred=pred)


async def good_scoped(scope):
    return await retrieve_chunks_scoped("query", scope=scope, top_k=5)


async def good_scroll(client, scope, flt):
    return await client.scroll(
        collection_name="documents", limit=10, scroll_filter=flt, scope=scope,
    )


def not_a_retrieval_call(text):
    """正则调用**不得**被误报（宿主名里没有 client）。"""
    return _PATTERN.search(text)
'''

_ISSUER_IN_SERVICE_SAMPLE = '''
async def leaky_service(user):
    """链路内重新签发 Scope —— 必须被 ④ 抓到。"""
    scope = await request_security_scope(user)
    return scope
'''

_ISSUER_IN_API_SAMPLE = '''
async def endpoint(user):
    scope = await request_security_scope(user)
    return scope
'''


def test_gate_catches_missing_scope() -> None:
    """**阳性对照**：漏传 scope 的样例必须被判为 offender。"""
    offenders = scan_source(_MISSING_SCOPE_SAMPLE, "backend/app/services/fake_bad.py")
    names = {o["name"] for o in offenders}
    assert "retrieve_chunks" in names, "门禁没抓到漏传 scope 的 retrieve_chunks —— 假门禁"
    assert "client.scroll" in names, "门禁没抓到漏传的 client.scroll —— 假门禁"
    assert "self.client.search" in names, "门禁没抓到 self.client.search —— 存在盲区"


def test_gate_accepts_scoped_call() -> None:
    """合规样例不得被误报（否则门禁会被人绕过性地关掉）。"""
    offenders = scan_source(_SCOPED_SAMPLE, "backend/app/services/fake_good.py")
    assert offenders == [], f"合规调用被误报: {offenders}"


def test_gate_catches_scope_reissuance_inside_services() -> None:
    """**阳性对照**：服务层里签发 Scope 必须被 ④ 抓到。"""
    offenders = scan_issuers(_ISSUER_IN_SERVICE_SAMPLE, "backend/app/services/fake_bad.py")
    assert {o["name"] for o in offenders} == {"request_security_scope"}


def test_gate_allows_scope_issuance_in_api() -> None:
    offenders = scan_issuers(_ISSUER_IN_API_SAMPLE, "backend/app/api/fake_good.py")
    assert offenders == []


def test_positive_control_is_not_vacuous() -> None:
    """
    元断言：阳性对照样例**必须**真的能被当前扫描器区分开。

    如果哪天有人把 ``TARGET_FUNCTIONS`` / ``TARGET_CLIENT_METHODS`` 改成空集，
    上面两条会同时"通过"（一个抓到 0 个、一个也 0 个）—— 这条断言把那种情况
    变成失败。
    """
    assert TARGET_FUNCTIONS, "TARGET_FUNCTIONS 为空 ⇒ 门禁恒绿"
    assert TARGET_CLIENT_METHODS, "TARGET_CLIENT_METHODS 为空 ⇒ 门禁恒绿"
    assert "scope" in BINDING_KWARGS, "必须支持 scope= 这一绑定形态"
    bad = scan_source(_MISSING_SCOPE_SAMPLE, "x.py")
    good = scan_source(_SCOPED_SAMPLE, "x.py")
    assert len(bad) == 3, f"阳性样例应产生 3 个 offender，实际 {len(bad)}"
    assert good == [], f"阴性样例不应产生 offender，实际 {good}"


@pytest.mark.parametrize(
    "target",
    ["retrieve_chunks", "retrieve_chunks_scoped", "keyword_search",
     "client.search", "client.scroll", "client.query_points",
     "self.client.scroll", "_client.query_points"],
)
def test_every_target_is_detectable(target: str) -> None:
    """每个受管调用名都必须可被检出（防止漏配某个名字造成盲区）。"""
    source = f"async def f(x):\n    return await {target}('q')\n"
    offenders = scan_source(source, "backend/app/services/fake.py")
    assert [o["name"] for o in offenders] == [target]


@pytest.mark.parametrize(
    "innocent",
    ["re.search(pattern)", "self._PATTERN.search(text)", "regex.match(s)",
     "index.search(query, top_n=5)", "text.scroll(10)"],
)
def test_innocent_attribute_calls_are_not_flagged(innocent: str) -> None:
    """宿主名里没有 ``client`` 的同名方法不得被误报（否则门禁会被关掉）。"""
    source = f"def f(self, pattern, text, s, query, regex, index):\n    return {innocent}\n"
    assert scan_source(source, "backend/app/services/fake.py") == []
