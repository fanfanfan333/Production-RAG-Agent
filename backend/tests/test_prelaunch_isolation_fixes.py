"""
T5 预发布 —— 五项数据隔离旁路缺陷的回归测试（FIX-A ~ FIX-E）。

设计目标：尽量不依赖真实 Postgres / Qdrant（本机 Windows 不一定有），用
替身 session / 纯函数 / 表达式编译 / 替身 Qdrant client 来钉死「漏洞已关闭」，
同时原有通过行为仍然通过。

覆盖：
  FIX-A  文档总结 / 文档关联路径绕过五维权限（🔴 阻塞项）
  FIX-B  JWT 弱密钥仅告警不阻断
  FIX-C  sync_doc_row 把对象行 excluded 静默覆写为 False
  FIX-D  图片提级时未与父文档密级取 max（越权放宽）
  FIX-E  Qdrant 缺失安全字段 payload 索引
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy import select, update
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.dml import Update as SqlUpdate

try:
    from qdrant_client.http import models as qmodels

    from app.config import (
        is_jwt_secret_weak,
        jwt_secret_must_fail_startup,
    )
    from app.db.postgres import get_db_session
    from app.services.image_security import escalate_image_object
    from app.services.master_graph import _summary_digests_node
    from app.services.relation_service import (
        collect_document_digests,
        list_accessible_documents,
    )
    from app.services.security_policy import ObjectACLView, ScopePredicate, to_sql
    from app.services.security_scope import DocumentScope, UserScope
except ImportError as exc:  # pragma: no cover - 宿主机缺依赖 → 整份跳过
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")

# 复用项目自带的「两路同源」内存求值器（PG 侧 to_sql 的纯 Python 镜像）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scope_filter_equivalence import sql_clause_matches  # noqa: E402

TENANT = "company_prelaunch"
FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)


# ═══════════════════════════════════════════════════════════════════════════════
# 替身 / 工具
# ═══════════════════════════════════════════════════════════════════════════════


def _user_scope(*, clearance: int = 1, tenant: str = TENANT, user_id: str | None = None):
    uid = user_id or str(uuid.uuid4())
    base = DocumentScope(
        owner_id=uid,
        tenant_ids=frozenset({tenant}),
        owns_tenant_ids=frozenset(),
        department_id=None,
        tenant_wide=False,
    )
    return UserScope(
        base=base,
        user_id=uid,
        role="employee",
        clearance=clearance,
        project_ids=frozenset(),
        principals=frozenset({f"user:{uid}"}),
    )


class _FakeResult:
    def __init__(self, *, scalar=None, all_rows=None):
        self._scalar = scalar
        self._all = all_rows or []

    def scalar_one_or_none(self):
        # 产品对对象行读的是 `.scalar_one_or_none()`（security_cascade / image_security），
        # 不是 `.all()`。只返回 _scalar 会让产品误判"行不存在"从而走错分支。
        if self._scalar is not None:
            return self._scalar
        return self._all[0] if self._all else None

    def scalars(self):
        return self

    def all(self):
        return self._all


class _FakeSession:
    """记录所有 UPDATE 的 (表名, 参数)；SELECT 按表名返回替身行。"""

    def __init__(self, *, doc=None, obj_rows=None):
        self._doc = doc
        self._obj_rows = obj_rows or []
        self.updates = []  # (table_name, dict(parameters))
        self.added: list = []
        self.deleted: list = []
        # 捕获到的 SELECT 语句对象（供"编译后看 WHERE 是否下推"的断言使用）。
        # worker 版本漏了这个字段，_selects_of() 只能永远拿到空列表。
        self._selects: list = []

    async def execute(self, stmt):
        if isinstance(stmt, SqlUpdate):
            table = getattr(stmt, "table", None)
            name = getattr(table, "name", None)
            # ⚠️ 不要读 ``stmt.parameters``：实测在本环境（SQLAlchemy 2.x）里，对
            # "只带 values、没有 WHERE" 的 UPDATE，``parameters`` 是 **None**，
            # 赋值实际存在 ``stmt._values`` 里、且键是 **Column 对象**而不是字符串。
            # 上一版读 ``.parameters`` 并直接 ``p["excluded"]``，4 个用例全部
            # KeyError（夹具口径错，不是产品缺陷）。这里统一归一化成列名。
            raw = getattr(stmt, "_values", None)
            if not raw:
                raw = stmt.compile(dialect=postgresql.dialect()).params
            # ``_values`` 里每个值还被包成 ``BindParameter``（实测形如
            # ``BindParameter('%(... param)s', True)``），要取 ``.value`` 才是真值；
            # 普通标量没有 ``.value``，getattr 兜底即可。
            self.updates.append(
                (
                    name,
                    {
                        getattr(k, "name", k): getattr(v, "value", v)
                        for k, v in dict(raw).items()
                    },
                )
            )
            return _FakeResult()
        # SELECT
        self._selects.append(stmt)
        froms = stmt.get_final_froms()
        names = [getattr(f, "name", None) for f in froms]
        if any(n == "documents" for n in names):
            return _FakeResult(scalar=self._doc)
        if any(n == "document_objects" for n in names):
            return _FakeResult(all_rows=self._obj_rows)
        return _FakeResult()

    # ── 替身补齐 ──────────────────────────────────────────────────────────────
    # worker 版本漏了下面这几个方法：产品代码走到 session.add() 时直接
    # AttributeError: '_FakeSession' object has no attribute 'add'
    # ——属**夹具不完整**，不是产品缺陷（这也是该文件此前从未真跑通的证据）。
    def add(self, obj):        # noqa: D401
        self.added.append(obj)

    def add_all(self, objs):   # noqa: D401
        self.added.extend(objs)

    def delete(self, obj):     # noqa: D401
        self.deleted.append(obj)

    async def commit(self):    # noqa: D401
        return None

    async def rollback(self):  # noqa: D401
        return None

    async def refresh(self, obj):  # noqa: D401 - 无操作替身
        return None

    async def flush(self):  # noqa: D401
        return None


class _FakeDB:
    def __init__(self, session: _FakeSession):
        self._s = session

    async def __aenter__(self):
        return self._s

    async def __aexit__(self, *exc):
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-A：文档总结 / 关联路径绕过五维权限（🔴 阻塞项）
# ═══════════════════════════════════════════════════════════════════════════════


def test_clearance_1_summary_excludes_level3_document():
    """阻塞项：clearance=1 用户调文档总结/关联，必须 0 命中 level=3 文档。

    直接验证 summary 路径实际下推的判定 ``to_sql(pred, Document)`` —— 与检索链路
    同源。同一租户/owner（三维通过）但密级 3 的文档，对 clearance=1 用户必须被排除。
    """
    scope = _user_scope(clearance=1, tenant=TENANT)
    pred = scope.predicate()

    level3_doc = ObjectACLView(
        document_id="doc-secret",
        tenant_id=TENANT,
        owner_id=scope.user_id,
        access_level="tenant",
        visibility_mode="tier",
        security_level=3,
        parent_security_level=None,
        effective_security_level=3,
        acl_allow=frozenset(),
        acl_deny=frozenset(),
        excluded=False,
    )
    # clearance=1 看不到密级 3 —— 漏洞已关闭
    assert sql_clause_matches(pred, level3_doc) is False

    # 对照组：同 clearance=3 的用户应当可见（证明不是"一刀切全拒"）
    scope3 = _user_scope(clearance=3, tenant=TENANT)
    assert sql_clause_matches(scope3.predicate(), level3_doc) is True

    # 对照组：clearance=1 看密级 1 的文档应当可见（原有通过行为不变）
    level1_doc = ObjectACLView(
        document_id="doc-public",
        tenant_id=TENANT,
        owner_id=scope.user_id,
        access_level="tenant",
        visibility_mode="tier",
        security_level=1,
        parent_security_level=None,
        effective_security_level=1,
        acl_allow=frozenset(),
        acl_deny=frozenset(),
        excluded=False,
    )
    assert sql_clause_matches(pred, level1_doc) is True


def _compiled_list_select(scope, *, with_scope: bool) -> str:
    """跑一次 ``list_accessible_documents``（替身 session）并返回它编译出的 SELECT。

    ⚠️ 补丁必须打在 ``relation_service.get_db_session`` 上：产品在
    ``relation_service.py`` 里写的是 ``from app.db.postgres import get_db_session``
    的**直接绑定**，改 ``app.db.postgres.get_db_session`` 对它完全无效 —— 上一版
    正是这么写的，于是跑进了真实 DB 会话（asyncpg 被宿主机垫片顶替 → 报
    "object _Permissive can't be used in 'await' expression"），这条用例从未真正
    验证过五维下推。补丁打对后，替身 session 才真正接管。
    """
    import app.services.relation_service as rs

    sess = _FakeSession()
    orig = rs.get_db_session
    rs.get_db_session = lambda: _FakeDB(sess)
    try:

        async def _invoke():
            kwargs: dict = {
                "owner_id": scope.user_id,
                "tenant_ids": frozenset({TENANT}),
            }
            if with_scope:
                kwargs["security_scope"] = scope
            await list_accessible_documents(**kwargs)

        asyncio.run(_invoke())
    finally:
        rs.get_db_session = orig

    assert sess.updates == [], "不应产生 UPDATE"
    assert sess._selects, "应捕获到至少一次 SELECT"
    return str(sess._selects[-1].compile(dialect=postgresql.dialect()))


def test_list_accessible_documents_applies_five_dimensional_scope():
    """文档列举路径在透传 security_scope 时，必须下推五维密级条件（含 security_level）。"""
    scope = _user_scope(clearance=1, tenant=TENANT)

    with_scope = _compiled_list_select(scope, with_scope=True)
    assert "security_level" in with_scope, f"五维路径未下推密级条件: {with_scope}"

    # 阴性对照：不透传 security_scope 时走三维旧路径，不应出现密级条件
    # （证明"差异确实存在"，而不是两条路径都恰好含 security_level）。
    without_scope = _compiled_list_select(scope, with_scope=False)
    assert "security_level" not in without_scope, \
        f"三维旧路径不应含密级条件: {without_scope}"


def test_summary_node_forwards_security_scope():
    """master graph 的 _summary_digests_node 必须透传 security_scope 给下游两条路径。"""
    import app.services.master_graph as mg
    import app.services.relation_service as rs
    import app.services.nodes.document_summary_node as dsn

    scope = _user_scope(clearance=1, tenant=TENANT)

    calls = {"list": [], "collect": []}

    async def _fake_list(**kwargs):
        calls["list"].append(kwargs)
        return []

    async def _fake_collect(**kwargs):
        calls["collect"].append(kwargs)
        return []

    orig_list = rs.list_accessible_documents
    orig_collect = dsn.collect_summary_digests
    # ⚠️ 补丁必须打在**节点所在模块的全局名**上：``master_graph.py`` 用的是
    #   ``from app.services.relation_service import list_accessible_documents`` /
    #   ``from ...document_summary_node import collect_summary_digests`` 这类**直接
    #   绑定**（master_graph.py:886 调用的是模块全局名）。只改 rs / dsn 的属性，
    #   节点里的 ``collect_summary_digests`` 仍指向真实函数 —— 上一版正是只改了
    #   dsn，于是断言在"应被调用"处失败，这条用例此前从未真跑通过。
    targets = [
        (rs, "list_accessible_documents", _fake_list, orig_list),
        (mg, "list_accessible_documents", _fake_list, orig_list),
        (dsn, "collect_summary_digests", _fake_collect, orig_collect),
        (mg, "collect_summary_digests", _fake_collect, orig_collect),
    ]
    for _mod, _attr, _fake, _orig in targets:
        setattr(_mod, _attr, _fake)
    try:
        state = {
            "query": "总结所有文档",
            "owner_id": scope.user_id,
            "tenant_ids": frozenset({TENANT}),
            "owns_tenant_ids": frozenset(),
            "department_id": None,
            "tenant_wide": False,
            "user_scope": scope,
        }
        result = asyncio.run(_summary_digests_node(state))
        assert result is not None
    finally:
        for _mod, _attr, _fake, _orig in targets:
            setattr(_mod, _attr, _orig)

    assert calls["list"], "list_accessible_documents 应被调用"
    assert calls["collect"], "collect_summary_digests 应被调用"
    assert any(c.get("security_scope") is scope for c in calls["list"]), \
        "list_accessible_documents 未收到透传的 security_scope"
    assert any(c.get("security_scope") is scope for c in calls["collect"]), \
        "collect_summary_digests 未收到透传的 security_scope"


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-B：JWT 弱密钥仅告警不阻断
# ═══════════════════════════════════════════════════════════════════════════════


def test_is_jwt_secret_weak_classification():
    # 弱：空 / 空白 / 过短 / 已知弱值
    assert is_jwt_secret_weak(None) is True
    assert is_jwt_secret_weak("") is True
    assert is_jwt_secret_weak("   ") is True
    assert is_jwt_secret_weak("short") is True
    assert is_jwt_secret_weak("changeme") is True
    assert is_jwt_secret_weak("secret") is True
    assert is_jwt_secret_weak("your-secret-key") is True
    assert is_jwt_secret_weak("dev-insecure-secret-change-me-0123456789abcdef0123456789abcdef") is True
    assert is_jwt_secret_weak("dev-insecure-secret-change-me-XXXX") is True  # 默认前缀

    # 强：≥32 字符且不在白名单（即便恰好含 "secret" 字样也不应误伤）
    strong = "a" * 32
    assert is_jwt_secret_weak(strong) is False
    strong2 = "prod-shared-secret-" + "f" * 40
    assert is_jwt_secret_weak(strong2) is False
    strong3 = "9f2c7b1e4a8d6c3f0e5b9a7d2c4f6e8b1a3c5d7e9f0b2a4c6e8d0f1b3a5c7e9f0"
    assert is_jwt_secret_weak(strong3) is False


def test_jwt_secret_startup_policy():
    weak = "dev-insecure-secret-change-me-0123456789abcdef0123456789abcdef"
    strong = "a" * 48

    # 弱 + 无 escape ⇒ 必须终止启动
    assert jwt_secret_must_fail_startup(weak) is True
    assert jwt_secret_must_fail_startup(weak, allow_insecure_jwt=False, debug=False) is True

    # 弱 + escape 开关 ⇒ 放行（不打断）
    assert jwt_secret_must_fail_startup(weak, allow_insecure_jwt=True) is False
    assert jwt_secret_must_fail_startup(weak, debug=True) is False

    # 强密钥 ⇒ 永不因该检查终止
    assert jwt_secret_must_fail_startup(strong) is False
    assert jwt_secret_must_fail_startup(strong, allow_insecure_jwt=True) is False


def test_current_env_bootable_with_escape_hatch():
    """本仓库 .env 使用弱默认密钥 + ALLOW_INSECURE_JWT=true，启动不应被阻断。"""
    from app.config import get_settings

    s = get_settings()
    assert jwt_secret_must_fail_startup(
        s.JWT_SECRET, allow_insecure_jwt=s.ALLOW_INSECURE_JWT, debug=s.DEBUG
    ) is False, "开发环境（弱默认密钥 + 显式 escape）必须仍可启动"


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-C：sync_doc_row 把对象行 excluded 静默覆写为 False
# ═══════════════════════════════════════════════════════════════════════════════


def _doc_row(level=1):
    d = SimpleNamespace()
    d.id = str(uuid.uuid4())
    d.tenant_id = TENANT
    d.owner_id = str(uuid.uuid4())
    d.department_id = None
    d.access_level = "tenant"
    d.security_level = level
    d.visibility_mode = "tier"
    d.project_ids = []
    d.acl_allow = []
    d.acl_deny = []
    return d


def _derived_obj(object_id, *, excluded=True, level=1):
    o = SimpleNamespace()
    o.object_id = object_id
    o.security_level = level
    o.effective_security_level = level
    o.parent_security_level = level
    # 级联会读这些列（security_cascade.sync_doc_row 需要它们来重算行）——
    # 替身缺字段会 AttributeError，属夹具不完整，不是产品缺陷。
    o.visibility_mode = "tier"
    o.project_ids = []
    o.tenant_id = TENANT
    o.owner_id = None
    o.department_id = None
    o.access_level = "tenant"
    o.acl_allow = []
    o.acl_deny = []
    o.excluded = excluded
    return o


def test_sync_doc_row_preserves_excluded_when_not_passed():
    """excluded 未传 ⇒ doc 镜像行的 UPDATE 不得写入 excluded（保留原值）。"""
    from app.services import security_cascade

    doc = _doc_row(level=1)
    derived = [_derived_obj("obj-1", excluded=True, level=1)]
    sess = _FakeSession(doc=doc, obj_rows=derived)

    orig = security_cascade._push_document_payload
    async def _noop_push(*a, **k):
        return None

    security_cascade._push_document_payload = _noop_push
    try:
        result = asyncio.run(
            security_cascade.sync_doc_row(doc.id, security_level=2, excluded=None, session=sess)
        )
    finally:
        security_cascade._push_document_payload = orig

    assert result.get("object_rows", 0) >= 1
    mirror = [p for n, p in sess.updates if n == "document_objects"]
    assert mirror, "应产生 document_objects 的 UPDATE"
    # doc 镜像行是第一个 document_objects UPDATE
    mirror_values = mirror[0]
    assert "excluded" not in mirror_values, \
        f"excluded 未传时不应覆写（撤销剔除）: {mirror_values}"


def test_sync_doc_row_writes_excluded_when_passed():
    """excluded 显式传入 ⇒ 应如实写入（True / False 都生效）。"""
    from app.services import security_cascade

    doc = _doc_row(level=1)
    derived = [_derived_obj("obj-1", excluded=False, level=1)]
    sess = _FakeSession(doc=doc, obj_rows=derived)

    orig = security_cascade._push_document_payload
    async def _noop_push(*a, **k):
        return None

    security_cascade._push_document_payload = _noop_push
    try:
        result = asyncio.run(
            security_cascade.sync_doc_row(doc.id, security_level=2, excluded=True, session=sess)
        )
    finally:
        security_cascade._push_document_payload = orig

    mirror = [p for n, p in sess.updates if n == "document_objects"]
    assert mirror[0]["excluded"] is True

    # 反之：显式传 False 也写入 False（不是被忽略）
    sess2 = _FakeSession(doc=doc, obj_rows=derived)
    async def _noop_push(*a, **k):
        return None

    security_cascade._push_document_payload = _noop_push
    try:
        asyncio.run(
            security_cascade.sync_doc_row(doc.id, security_level=2, excluded=False, session=sess2)
        )
    finally:
        security_cascade._push_document_payload = orig
    mirror2 = [p for n, p in sess2.updates if n == "document_objects"]
    assert mirror2[0]["excluded"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-D：图片提级时未与父文档密级取 max（越权放宽）
# ═══════════════════════════════════════════════════════════════════════════════


def _image_row(*, parent_level=3, level=2, eff=2):
    o = SimpleNamespace()
    o.object_id = "doc::img1"
    o.parent_security_level = parent_level
    o.security_level = level
    o.effective_security_level = eff
    o.excluded = False
    o.acl_deny = []
    o.acl_allow = []
    return o


def _run_escalate(image_row, *, security_level=2, doc_level_for_fallback=None):
    from app.services import image_security

    sess = _FakeSession(doc=_doc_row(level=doc_level_for_fallback or 3), obj_rows=[])
    # escalate_image_object 在图片行存在时只查 DocumentObject（已有行）
    sess_obj = _FakeSession(doc=_doc_row(level=doc_level_for_fallback or 3), obj_rows=[image_row])
    # 用同一个 session：select(DocumentObject) 返回 image_row；fallback 时 select(documents) 返回 doc
    sess_obj._obj_rows = [image_row]

    orig_cascade = image_security.cascade_image_derived
    orig_audit = image_security._audit_image_change
    # ⚠️ 产品侧是 `await cascade_image_derived(...)` / `await _audit_image_change(...)`，
    # 替身必须是**协程函数**；原先的同步 lambda 会炸
    # "object types.SimpleNamespace can't be used in 'await' expression"。
    async def _fake_cascade(*a, **k):
        return SimpleNamespace(cascaded=0)

    async def _fake_audit(*a, **k):
        return None

    image_security.cascade_image_derived = _fake_cascade
    image_security._audit_image_change = _fake_audit
    try:
        result = asyncio.run(
            escalate_image_object(
                # ⚠️ 必须是合法 UUID：escalate_image_object 内部走 uuid.UUID(document_id)，
                # 传 "doc" 这类占位串会在任何环境下直接 ValueError（夹具 bug，非产品缺陷）。
                str(uuid.uuid4()), image_id="img1", security_level=security_level, session=sess_obj
            )
        )
    finally:
        image_security.cascade_image_derived = orig_cascade
        image_security._audit_image_change = orig_audit

    obj_updates = [p for n, p in sess_obj.updates if n == "document_objects"]
    return result, obj_updates


def test_escalate_image_preserves_parent_max_when_raising():
    """父文档绝密(3)、管理员把图片设为机密(2) ⇒ effective 必须为 3（不得放宽）。"""
    row = _image_row(parent_level=3, level=2, eff=2)
    result, updates = _run_escalate(row, security_level=2)
    assert result.get("ok") is True
    assert updates, "应产生图片行的 UPDATE"
    assert updates[0]["security_level"] == 2
    assert updates[0]["effective_security_level"] == 3, \
        f"图片提级未与父文档密级取 max，effective={updates[0].get('effective_security_level')}"


def test_escalate_image_raise_to_top_level():
    """设为密级 3 ⇒ effective 也为 3。"""
    row = _image_row(parent_level=2, level=1, eff=1)
    result, updates = _run_escalate(row, security_level=3)
    assert updates[0]["security_level"] == 3
    assert updates[0]["effective_security_level"] == 3


def test_escalate_image_fallback_to_document_level():
    """图片行缺 parent_security_level ⇒ 回退父文档 security_level 取 max。"""
    row = _image_row(parent_level=None, level=2, eff=2)
    result, updates = _run_escalate(row, security_level=2, doc_level_for_fallback=3)
    assert updates[0]["effective_security_level"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# FIX-E：Qdrant 缺失安全字段 payload 索引
# ═══════════════════════════════════════════════════════════════════════════════


def test_ensure_payload_indexes_creates_security_fields():
    from app.services import vector_service

    fake = _FakeQdrant()

    orig = vector_service.get_qdrant_client
    vector_service.get_qdrant_client = lambda: fake
    try:
        asyncio.run(vector_service._ensure_payload_indexes("documents"))
    finally:
        vector_service.get_qdrant_client = orig

    schema_map = {name: sch for name, sch in fake.indexes}
    required = {
        "security_level": qmodels.PayloadSchemaType.INTEGER,
        "parent_security_level": qmodels.PayloadSchemaType.INTEGER,
        "effective_security_level": qmodels.PayloadSchemaType.INTEGER,
        "visibility_mode": qmodels.PayloadSchemaType.KEYWORD,
        "project_ids": qmodels.PayloadSchemaType.KEYWORD,
        "acl_allow": qmodels.PayloadSchemaType.KEYWORD,
        "acl_deny": qmodels.PayloadSchemaType.KEYWORD,
        "excluded": qmodels.PayloadSchemaType.BOOL,
        "acl_expires_at_ts": qmodels.PayloadSchemaType.FLOAT,
    }
    for field, expected_schema in required.items():
        assert field in schema_map, f"缺失安全字段索引: {field}"
        assert schema_map[field] == expected_schema, \
            f"字段 {field} 索引类型错误：期望 {expected_schema}，实际 {schema_map[field]}"


class _FakeQdrant:
    def __init__(self):
        self.indexes = []

    async def create_payload_index(self, collection_name, field_name, field_schema):
        self.indexes.append((field_name, field_schema))
