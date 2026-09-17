"""
ACL 载荷「发布早于入库」竞态与收尾追平的回归测试.

这是审计 `_audit_916/audit_iso.py` 第 3.5 组抓出来的真实缺陷的固化测试：

    [FAIL] A 员工问本部门库金标 → 应可见  ← sources=1 answer_len=41
    [FAIL] A 员工问本公司库金标 → 应可见  ← sources=1 answer_len=41

`answer_len=41` 恰好是 evidence_gate 的拒答文案长度，**看起来**像"生成侧没答出来"，
实际是检索根本没取到那份文档。而 3.1 的列表接口对同一份文档是**可见**的 ——
两份事实分叉：

    PostgreSQL  access_level=department   ← 列表 / 详情 / 删除都读这份
    Qdrant      access_level=private      ← 检索的 ACL 前置过滤读这份

成因是**上传异步 + 传完立刻发布**这条正常路径上的竞态：

    1. ``POST /upload`` 受理即返回 202，管线丢后台，``access_level`` 此刻固定为 private
    2. ``PATCH /documents/{id}/visibility`` → Qdrant ``set_payload``，
       按 ``document_id`` 过滤更新 —— **此时向量点还没写入**
    3. ``set_payload`` 匹配 0 个点，却不报错；旧实现也无条件 ``return 1`` 并记
       一条 "Updated vector ACL payload" 的成功日志 → 谁都看不出更新落空了
    4. 入库按上传时刻的快照 private 写点 → 载荷永久停在 private

结果：owner 自己检索得到（private 对 owner 本就放行），**其他人全部检索不到** ——
所以那两条 FAIL 恰好是"非 owner 检索共享文档"。

本文件把四处行为钉死：

    1. 点不存在时 ``update_document_access_payload`` 必须返回 0（不得谎报成功）
    2. 点不存在时**不得**发起没有任何对象的写入
    3. 入库收尾的 ``resync_document_acl_payload`` 必须用 **PG 真值**覆盖入库快照，
       把丢失的层级变更追平
    4. PG 真值与入库快照一致时必须**跳过**（99% 未发布的文档不该多付一次写）
    5. 发布接口不得因为"向量点还没写入"而失败（PG 是判定权威）

不连数据库、不连向量库：DB 会话与 Qdrant 客户端都是本文件里的极小替身。

运行：
    docker exec -e PYTHONPATH=/app/.cache/.local/lib/python3.12/site-packages \
        rag_backend python -m pytest tests/test_acl_payload_resync.py -q
"""

from __future__ import annotations

import asyncio
import types
import uuid

import pytest

from app.services import knowledge_tier_service as kt
from app.services import vector_service


# ═════════════════════════════════════════════════════════════════════════════
# 极小替身：Qdrant 客户端
# ═════════════════════════════════════════════════════════════════════════════

def _condition_value(filt, key: str):
    """从 ``Filter`` 里取出 ``must=[FieldCondition(key=..., match=MatchValue)]`` 的值."""
    for cond in filt.must or []:
        if getattr(cond, "key", None) == key:
            return getattr(cond.match, "value", None)
    return None


class _FakeQdrant:
    """只实现被测路径用到的 ``count`` / ``set_payload``，并记录调用痕迹."""

    def __init__(self) -> None:
        self.points: dict[str, dict] = {}
        self.count_calls = 0
        self.set_payload_calls: list[tuple[str, dict]] = []

    # 入库侧模拟：写入一个带 ACL 载荷的向量点
    def upsert(self, point_id: str, payload: dict) -> None:
        self.points[point_id] = dict(payload)

    def points_of(self, document_id: str) -> list[dict]:
        return [
            pl for pl in self.points.values()
            if str(pl.get("document_id")) == document_id
        ]

    async def count(self, collection_name, count_filter, exact=True):  # noqa: D401
        self.count_calls += 1
        target = _condition_value(count_filter, "document_id")
        matched = sum(
            1 for pl in self.points.values()
            if str(pl.get("document_id")) == str(target)
        )
        return types.SimpleNamespace(count=matched)

    async def set_payload(self, collection_name, payload, points, wait=True):
        # points 既可能是 FilterSelector（按条件更新），也可能是纯 id 列表
        selector_filter = getattr(points, "filter", None)
        if selector_filter is None:
            for point_id in points:
                self.points.setdefault(point_id, {}).update(payload)
            self.set_payload_calls.append((f"ids:{len(points)}", dict(payload)))
            return

        target = _condition_value(selector_filter, "document_id")
        self.set_payload_calls.append((str(target), dict(payload)))
        for pl in self.points.values():
            if str(pl.get("document_id")) == str(target):
                pl.update(payload)


# ═════════════════════════════════════════════════════════════════════════════
# 极小替身：PG 会话（只服务 `select(Document.access_level, department_id)`)
# ═════════════════════════════════════════════════════════════════════════════

class _FakeResult:
    def __init__(self, row) -> None:
        self._row = row

    def first(self):
        return self._row

    def scalar_one_or_none(self):
        return self._row


class _FakeSession:
    def __init__(self, row) -> None:
        self._row = row

    async def execute(self, _stmt):
        return _FakeResult(self._row)

    async def flush(self) -> None:
        return None

    async def refresh(self, _obj) -> None:
        return None


class _FakeSessionCM:
    def __init__(self, row) -> None:
        self._row = row

    async def __aenter__(self) -> _FakeSession:
        return _FakeSession(self._row)

    async def __aexit__(self, *_exc) -> bool:
        return False


@pytest.fixture()
def fake_qdrant(monkeypatch):
    fake = _FakeQdrant()
    monkeypatch.setattr(vector_service, "get_qdrant_client", lambda: fake)
    return fake


def _bind_pg_truth(monkeypatch, row):
    """让 ``resync_document_acl_payload`` 读到指定的 PG 真值."""
    monkeypatch.setattr(kt, "get_db_session", lambda: _FakeSessionCM(row))


DOC_ID = str(uuid.uuid4())
OWNER_ID = str(uuid.uuid4())
PRIVATE_SNAPSHOT = ("private", None)


# ═════════════════════════════════════════════════════════════════════════════
# 1. 点不存在时不得谎报成功
# ═════════════════════════════════════════════════════════════════════════════

def test_publish_before_vectors_is_not_reported_as_success(fake_qdrant):
    """上传后立刻发布：向量点还不存在 —— 必须返回 0，而不是"成功"."""
    applied = asyncio.run(
        vector_service.update_document_access_payload(
            DOC_ID, access_level="department", department_id="eng-916"
        )
    )
    assert applied == 0, (
        "点不存在时 update_document_access_payload 必须返回 0；"
        "返回非 0 会让「发布成功但检索不到」永远静默"
    )


def test_publish_before_vectors_does_not_issue_a_pointless_write(fake_qdrant):
    """确认没有可更新的点时，连写请求都不该发出（避免制造"写过了"的假象）."""
    asyncio.run(
        vector_service.update_document_access_payload(
            DOC_ID, access_level="tenant", department_id=None
        )
    )
    assert fake_qdrant.set_payload_calls == [], "匹配 0 个点时不应发起 set_payload"


def test_publish_after_vectors_reports_real_point_count(fake_qdrant):
    """对照组：点已存在时必须真的改到，且返回真实点数."""
    fake_qdrant.upsert("p1", {"document_id": DOC_ID, "access_level": "private"})
    fake_qdrant.upsert("p2", {"document_id": DOC_ID, "access_level": "private"})

    applied = asyncio.run(
        vector_service.update_document_access_payload(
            DOC_ID, access_level="department", department_id="eng-916"
        )
    )
    assert applied == 2
    for payload in fake_qdrant.points_of(DOC_ID):
        assert payload["access_level"] == "department"
        assert payload["department_id"] == "eng-916"


# ═════════════════════════════════════════════════════════════════════════════
# 2. 入库收尾追平：完整复现审计那两条 FAIL 的时序
# ═════════════════════════════════════════════════════════════════════════════

def test_ingestion_tail_resync_recovers_lost_level_change(fake_qdrant, monkeypatch):
    """时序复现：发布（无点）→ 入库写入 private → 收尾对齐 → 载荷 = PG 真值.

    这正是审计里 "A 员工问本部门库金标 → 应可见" 失败的成因链。修复后，
    收尾对齐必须把 department 追平，检索才可能取到这份文档。
    """
    # ── 第 1 步：上传受理，管线在后台跑，快照 private ──────────────────────
    # ── 第 2 步：用户立刻点"发布到部门库" —— 此时还没有任何向量点 ──────────
    applied = asyncio.run(
        vector_service.update_document_access_payload(
            DOC_ID, access_level="department", department_id="eng-916"
        )
    )
    assert applied == 0, "竞态前提：发布时点尚未写入"

    # ── 第 3 步：入库 upsert，用的是上传时刻的快照 private ──────────────────
    fake_qdrant.upsert(
        "p1",
        {
            "document_id": DOC_ID,
            "user_id": OWNER_ID,
            "access_level": PRIVATE_SNAPSHOT[0],
            "department_id": PRIVATE_SNAPSHOT[1],
        },
    )
    assert fake_qdrant.points_of(DOC_ID)[0]["access_level"] == "private"

    # ── 第 4 步：入库收尾对齐（读 PG 真值 = department）────────────────────
    _bind_pg_truth(monkeypatch, ("department", "eng-916"))
    fixed = asyncio.run(
        kt.resync_document_acl_payload(
            DOC_ID, reason="ingestion_completion", expected=PRIVATE_SNAPSHOT
        )
    )

    assert fixed == 1, "收尾对齐必须真的改写那份落后的载荷"
    payload = fake_qdrant.points_of(DOC_ID)[0]
    assert payload["access_level"] == "department", (
        "入库收尾必须用 PG 真值覆盖入库快照，否则该文档对非 owner 永久检索不到"
    )
    assert payload["department_id"] == "eng-916"
    assert payload["user_id"] == OWNER_ID, "只改 ACL 字段，不得动其它载荷"


def test_resync_skips_when_pg_matches_ingestion_snapshot(fake_qdrant, monkeypatch):
    """从未发布的文档（占绝大多数）：PG 与快照一致时必须跳过，零额外开销."""
    _bind_pg_truth(monkeypatch, ("private", None))

    applied = asyncio.run(
        kt.resync_document_acl_payload(
            DOC_ID, reason="ingestion_completion", expected=PRIVATE_SNAPSHOT
        )
    )

    assert applied == 0
    assert fake_qdrant.count_calls == 0, "跳过时不该去查 Qdrant"
    assert fake_qdrant.set_payload_calls == [], "跳过时不该写 Qdrant"


def test_resync_normalises_department_for_non_department_levels(fake_qdrant, monkeypatch):
    """公司库/个人库必须清空部门归属，避免旧部门值残留导致日后误判.

    走的是与 ``set_document_access_level`` 完全相同的口径 —— 两条路径对
    "什么层级该带 department_id" 若有分歧，检索过滤会按陈旧部门误判可见性。
    """
    fake_qdrant.upsert("p1", {"document_id": DOC_ID, "access_level": "private"})
    # PG 里残留着部门归属，但层级是公司库
    _bind_pg_truth(monkeypatch, ("tenant", "eng-916"))

    applied = asyncio.run(
        kt.resync_document_acl_payload(DOC_ID, reason="repair_script", expected=None)
    )

    assert applied == 1
    payload = fake_qdrant.points_of(DOC_ID)[0]
    assert payload["access_level"] == "tenant"
    assert payload["department_id"] is None, "非部门库不得保留部门归属"


def test_resync_reports_zero_when_document_row_is_gone(fake_qdrant, monkeypatch):
    """文档行不存在（已删除）：返回 0，且不碰向量库."""
    _bind_pg_truth(monkeypatch, None)

    applied = asyncio.run(
        kt.resync_document_acl_payload(DOC_ID, reason="repair_script", expected=None)
    )

    assert applied == 0
    assert fake_qdrant.set_payload_calls == []


# ═════════════════════════════════════════════════════════════════════════════
# 3. 发布接口的用户可见契约：不因向量侧未就绪而失败
# ═════════════════════════════════════════════════════════════════════════════

def test_publish_still_succeeds_when_vectors_not_yet_written(
    fake_qdrant, monkeypatch
):
    """发布必须照常成功（PG 是判定权威）+ 更新审计 + 把落空写成告警留痕.

    若这里抛异常，用户在"上传后立刻发布"时会直接看到 500 —— 而这条恰恰是
    最正常的操作顺序。落空必须由收尾对齐兜底，不能变成用户可见的失败。
    """
    audits: list[tuple] = []

    async def _record_audit(action, **kwargs):
        audits.append((action, kwargs))

    monkeypatch.setattr(kt, "record_audit", _record_audit)

    doc = types.SimpleNamespace(
        id=uuid.UUID(DOC_ID),
        access_level="private",
        department_id=None,
    )
    monkeypatch.setattr(kt, "get_db_session", lambda: _FakeSessionCM(doc))

    updated = asyncio.run(
        kt.set_document_access_level(
            uuid.UUID(DOC_ID),
            level="department",
            department_id="eng-916",
            actor_username="a_mgr_eng@audit.local",
        )
    )

    assert updated.access_level == "department"
    assert updated.department_id == "eng-916"
    assert len(audits) == 1 and audits[0][0] == "document.publish"
    assert fake_qdrant.set_payload_calls == [], "无点可改时不应发起写入"
