"""
Document Agent 产物下载端的越权回归测试（P0 旁路）.

漏洞现场
────────
``GET /documents/generated/{filename}`` 原先**只验文件是否存在**：任何登录用户
只要拿到/猜到产物文件名就能下载别人的 Word —— 而产物内含 N 条来源片段与 N 张
原图，等于绕过「检索前租户过滤 → 检索后 ACL → LLM 输入前校验」整条隔离链路。

修复后的判定：调用者必须是产物所有者，**或**对产物引用的**全部**源文档都可访问；
归属无法确定时 fail-closed。

护栏（为什么这两个方向都要测）
──────────────────────────────
- **阴性**（B 请求 A 的产物 → 必须失败）
- **阳性对照**（A 请求自己的产物 → 必须成功）

只写阴性是**假测试**：如果端点整个坏掉（谁都下载不了），阴性测试照样"通过"。
阳性对照负责证明合法路径仍然可用，堵死"修完把正常功能弄没了"这条退路。

运行方式（容器内）：
    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python -m pytest tests/test_generated_doc_acl.py -v"
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    import pytest
    from fastapi import FastAPI, HTTPException, status
    from fastapi.testclient import TestClient

    from app.api import deps as api_deps
    from app.api.document_management import (
        _GENERATED_404_DETAIL,
        router as documents_router,
    )
    from app.config import get_settings
    from app.services import document_agent_service as das
    from app.services.document_agent_service import (
        authorize_generated_file,
        generate_document,
        meta_path_for,
        read_generated_meta,
        source_document_ids_of,
    )
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    # 不能用 sys.exit()：pytest 在收集阶段导入本模块，抛 SystemExit 会让整个
    # 会话 INTERNALERROR，同目录其它用例全部跑不了。
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


_TENANT = "company_A"


# ── 测试替身 ──────────────────────────────────────────────────────────────────


class _StubUser:
    """替身用户：只需满足 tenancy 判定用到的字段."""

    def __init__(self, username: str, *, tenant_id: str = _TENANT,
                 department_id: str | None = None, role: str = "user"):
        self.id = uuid.uuid4()
        self.username = username
        self.tenant_id = tenant_id
        self.department_id = department_id
        self.role = role

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass
class _StubChunk:
    """替身检索片段：字段对齐 RetrievedChunk 中被 _build_docx 消费的部分."""

    document_id: str
    text: str = "来自受限文档的正文片段。"
    filename: str = "受限来源.pdf"
    page_number: int = 1
    chunk_index: int = 0
    content_type: str = "text"
    score: float = 0.91
    heading: str | None = None
    parent_text: str | None = None
    image_path: str | None = None
    image_caption: str | None = None


@dataclass
class _StubDoc:
    """替身 Document 行：字段对齐 tenancy.can_access_document 读取的部分."""

    owner_id: uuid.UUID
    access_level: str = "private"
    tenant_id: str = _TENANT
    department_id: str | None = None


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """把产物输出目录重定向到临时目录，避免污染 uploads/_generated."""
    target = tmp_path / "_generated"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        get_settings(), "DOCUMENT_OUTPUT_DIR", str(target), raising=False
    )
    return target


@pytest.fixture
def party():
    alice = _StubUser("alice")
    bob = _StubUser("bob")
    return SimpleNamespace(
        alice=alice,
        bob=bob,
        token_alice="jwt-alice",
        token_bob="jwt-bob",
    )


@pytest.fixture
def client(party, monkeypatch):
    """
    只挂载 documents 路由的最小 FastAPI app.

    **不**用 ``dependency_overrides`` 顶掉 ``get_current_user_media`` —— 那样会
    连同 token 解析一起被替换掉，"``?token=`` 是否与 Authorization 走同一套判定"
    就测不到了。这里只把最底层的 ``api_deps._resolve_user`` 换成按 token 查用户
    的替身：鉴权依赖、路由、端点、归属判定全部是真实代码。
    """
    async def _fake_resolve_user(token: str | None):
        mapping = {party.token_alice: party.alice, party.token_bob: party.bob}
        user = mapping.get(token or "")
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="未登录或缺少访问令牌",
            )
        return user

    monkeypatch.setattr(api_deps, "_resolve_user", _fake_resolve_user)

    app = FastAPI()
    app.include_router(documents_router)
    return TestClient(app)


def _patch_source_loader(monkeypatch, docs_by_id: dict[str, object]) -> None:
    """把源文档的 DB 查询换成内存查表（loads-= 走真实的 ACL 判定）."""
    async def _fake_load(document_ids: list[str]) -> list:
        return [docs_by_id.get(str(i)) for i in document_ids]

    monkeypatch.setattr(das, "_load_source_documents", _fake_load)


def _build_artifact(owner: _StubUser, source_doc_ids: list[uuid.UUID]) -> str:
    """真实生成一份 .docx（含 sidecar 归属记录），返回产物文件名."""
    chunks = [_StubChunk(document_id=str(doc_id)) for doc_id in source_doc_ids]
    info = generate_document(
        "生成一份调研报告", chunks, title="调研报告", owner_id=owner.id
    )
    assert not info.error, f"Document Agent 生成失败: {info.error}"
    assert info.filename, "未返回产物文件名"
    return info.filename


def _get(client: TestClient, filename: str, *, token: str | None = None,
         bearer: bool = True) -> "object":
    """按通道请求产物：bearer=True 走 Authorization 头，False 走 ?token=."""
    if bearer:
        return client.get(
            f"/documents/generated/{filename}",
            headers={"Authorization": f"Bearer {token}"},
        )
    return client.get(f"/documents/generated/{filename}?token={token}")


# ── 阳性对照：合法路径必须仍然可用 ────────────────────────────────────────────


def test_owner_can_download_own_artifact(out_dir, party, client, monkeypatch) -> None:
    """阳性对照①：所有者请求自己的产物 → 200 且拿到真实 .docx 字节.

    没有这一条，"越权被拦住"可能只是因为端点整个坏了。
    """
    _patch_source_loader(monkeypatch, {})
    filename = _build_artifact(party.alice, [uuid.uuid4()])

    resp = _get(client, filename, token=party.token_alice)

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument"
    )
    on_disk = das.resolve_generated_file(filename)
    assert on_disk is not None and on_disk.read_bytes() == resp.content
    assert len(resp.content) > 0, "产物内容不应为空"


def test_collaborator_with_all_source_access_can_download(
    out_dir, party, client, monkeypatch
) -> None:
    """阳性对照②：协作者能访问产物的**全部**源文档 → 仍可正常下载.

    防止"为了修越权把正常协作一起堵死"。
    """
    shared_a = uuid.uuid4()
    shared_b = uuid.uuid4()
    _patch_source_loader(
        monkeypatch,
        {
            str(shared_a): _StubDoc(owner_id=party.alice.id, access_level="tenant"),
            str(shared_b): _StubDoc(owner_id=party.alice.id, access_level="tenant"),
        },
    )
    filename = _build_artifact(party.alice, [shared_a, shared_b])

    # bob 与 alice 同公司，公司库文档对其可见
    resp = _get(client, filename, token=party.token_bob)

    assert resp.status_code == 200, resp.text


# ── 阴性：越权必须失败，且与"文件不存在"不可区分 ──────────────────────────────


def test_other_user_cannot_download_foreign_artifact(
    out_dir, party, client, monkeypatch
) -> None:
    """阴性①：B 请求 A 的产物（源文档对 B 不可见）→ 404.

    返回 403 / "无权访问"都会确认"该文件存在"，可被批量探测，因此必须是 404。
    """
    private_doc = uuid.uuid4()
    _patch_source_loader(
        monkeypatch,
        {str(private_doc): _StubDoc(owner_id=party.alice.id, access_level="private")},
    )
    filename = _build_artifact(party.alice, [private_doc])

    resp = _get(client, filename, token=party.token_bob)

    assert resp.status_code == 404, resp.text
    assert resp.json() == {"detail": _GENERATED_404_DETAIL}


def test_partial_source_access_still_denied(
    out_dir, party, client, monkeypatch
) -> None:
    """阴性②：只有部分源文档可见 → 仍然拒绝.

    产物是多份来源的合订本，放行就等于把不可见那一份的内容泄露出去。
    """
    private_doc = uuid.uuid4()
    shared_doc = uuid.uuid4()
    _patch_source_loader(
        monkeypatch,
        {
            str(private_doc): _StubDoc(
                owner_id=party.alice.id, access_level="private"
            ),
            str(shared_doc): _StubDoc(owner_id=party.alice.id, access_level="tenant"),
        },
    )
    filename = _build_artifact(party.alice, [private_doc, shared_doc])

    resp = _get(client, filename, token=party.token_bob)

    assert resp.status_code == 404, resp.text
    assert resp.json() == {"detail": _GENERATED_404_DETAIL}


def test_unauthorized_is_indistinguishable_from_missing(
    out_dir, party, client, monkeypatch
) -> None:
    """越权响应必须与"文件不存在"**逐字一致**（状态码 + 响应体）.

    这是硬性安全要求：任何一点差异都等于在验证"这个别人的文件确实存在"。
    """
    private_doc = uuid.uuid4()
    _patch_source_loader(
        monkeypatch,
        {str(private_doc): _StubDoc(owner_id=party.alice.id, access_level="private")},
    )
    filename = _build_artifact(party.alice, [private_doc])

    missing_resp = _get(
        client, "rag_doc_20990101_000000_000000.docx", token=party.token_alice
    )
    denied_resp = _get(client, filename, token=party.token_bob)

    assert missing_resp.status_code == 404
    assert denied_resp.status_code == missing_resp.status_code
    assert denied_resp.json() == missing_resp.json(), (
        "越权响应与'不存在'响应不一致 —— 会泄露文件存在性"
    )
    # 口令里不得出现任何"权限"相关表述
    assert "权限" not in denied_resp.text and "无权" not in denied_resp.text


# ── fail-closed ───────────────────────────────────────────────────────────────


def test_missing_ownership_record_denies_everyone(
    out_dir, party, client, monkeypatch
) -> None:
    """归属记录缺失（历史产物 / 元数据损坏）→ 连所有者也拒（fail-closed）.

    「无法确定归属就放行」正是原漏洞的形态，这里是它的反向护栏。
    """
    _patch_source_loader(monkeypatch, {})
    filename = _build_artifact(party.alice, [uuid.uuid4()])

    meta = meta_path_for(filename)
    assert meta is not None and meta.is_file()
    meta.unlink()          # 制造"无法确定归属"的场景
    assert read_generated_meta(filename) is None

    resp = _get(client, filename, token=party.token_alice)

    assert resp.status_code == 404, resp.text
    assert resp.json() == {"detail": _GENERATED_404_DETAIL}


def test_authorize_rejects_anonymous(out_dir, party) -> None:
    """未登录 → 拒绝；决策函数不依赖任何鉴权副作用."""
    filename = _build_artifact(party.alice, [uuid.uuid4()])

    # 决策函数是 coroutine —— 直接 asyncio.run，不依赖 pytest-asyncio 插件
    resolved, reason = asyncio.run(authorize_generated_file(filename, None))

    assert resolved is None
    assert reason == "anonymous"


# ── ?token= 通道必须与 Authorization 同一套判定 ───────────────────────────────


def test_query_token_channel_uses_same_decision(
    out_dir, party, client, monkeypatch
) -> None:
    """``?token=``（浏览器 <a download> 用的通道）不得成为旁路.

    两条通道都收敛到 ``get_current_user_media`` → ``_resolve_user``，归属判定
    只依赖解出来的 user，因此行为必须一致：A 能下、B 不能下。
    """
    _patch_source_loader(monkeypatch, {})
    filename = _build_artifact(party.alice, [uuid.uuid4()])

    ok = _get(client, filename, token=party.token_alice, bearer=False)
    denied = _get(client, filename, token=party.token_bob, bearer=False)

    assert ok.status_code == 200, ok.text
    assert denied.status_code == 404, denied.text
    assert denied.json() == {"detail": _GENERATED_404_DETAIL}


# ── sidecar 内容正确性 ────────────────────────────────────────────────────────


def test_meta_records_owner_and_source_bindings(out_dir, party) -> None:
    """sidecar 必须落 owner_id 与**完整**的源文档列表（后续判定的唯一依据）."""
    doc_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
    filename = _build_artifact(party.alice, doc_ids)

    meta = read_generated_meta(filename)

    assert meta is not None
    assert meta["owner_id"] == str(party.alice.id)
    assert meta["source_document_ids"] == [str(d) for d in doc_ids]
    assert meta["filename"] == filename
    # 同一份 chunks 推导出的绑定必须与落盘记录一致
    assert source_document_ids_of(
        [_StubChunk(document_id=str(d)) for d in doc_ids]
    ) == [str(d) for d in doc_ids]
