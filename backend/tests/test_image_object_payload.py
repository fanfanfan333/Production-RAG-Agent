"""
图片对象的 ACL 物化载荷门禁（缺陷 A 的固化测试）.

背景（QA 端到端定位的真实缺陷）
────────────────────────────────
``build_object_rows``（``security_cascade.py``）为五种对象建 ``document_objects``
行，并把权限字段冗余进**行内嵌的** ``_payload``；随后 ``push_payload_async`` 逐行
``set_payload`` 推给 Qdrant。但 **image 行**（``_ensure_image_row``）建行时**从不写**
``row["_point_id"]`` / ``row["_payload"]``，于是 ``push_payload_async`` 里

    if not point_id or not payload:
        continue

逐行跳过 image ⇒ image 点的 Qdrant 载荷永远缺 ``object_id`` / ``visibility_mode`` /
``security_level`` / ``acl_sync_state`` / ``excluded`` / ``acl_allow`` /
``effective_security_level`` 这 7 个字段。第 11 环 ``allows()`` 读的正是这份载荷
（``ObjectACLView.from_payload``），``object_id`` 缺失 → ``missing_object_view`` →
**所有账号都检索不到图片**（能力丧失，不是越权）。

本文件钉死三条：
  1. image 行必须带 ``_point_id`` / ``_payload``，载荷含全部 7 键且与 PG 行同源；
  2. ``push_payload_async`` 必须真的把 image 点写进 Qdrant（不再被跳过）；
  3. 物化后 ``allows()`` 对 **owner 与同部门非 owner** 都放行（且异部门仍拒）。

红-绿：把 ``_ensure_image_row`` 里 ``row["_payload"] = _payload_from_row(row)`` 注释掉，
本文件必须失败（见交付报告）。

不连库 / 不连向量库：Qdrant 客户端是本文件里的极小替身。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from app.db.security_models import (
    OBJECT_TYPE_IMAGE,
    OBJECT_TYPE_TABLE,
    make_object_id,
)
from app.services.security_cascade import (
    build_object_rows,
    push_payload_async,
)
from app.services.security_policy import (
    DEFAULT_CLEARANCE_BY_ROLE,
    P_ACL_EXPIRES_AT_TS,
    ObjectACLView,
    ScopePredicate,
    allows,
)

FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)

DOC_ID = str(uuid.uuid4())
TENANT = "company_a"
DEPT = "dd6104e61cf30"
U_OWNER = str(uuid.uuid4())
U_MATE = str(uuid.uuid4())
U_OTHER_DEPT = str(uuid.uuid4())
DEPT_OTHER = "dept_other"

IMG_ID = "img_1"
IMG_PID = "imgpoint"
TBL_PID = "tblpoint"

#: QA 明确点名的 7 个"对象级 ACL 物化字段"——缺任意一个都会让第 11 环误判。
MATERIALIZED_KEYS = (
    "object_id",
    "visibility_mode",
    "security_level",
    "acl_sync_state",
    "excluded",
    "acl_allow",
    "effective_security_level",
)

#: 载荷与分块行**逐字对齐**的完整键集（防止有人只补 7 键、漏掉其余）。
FULL_PAYLOAD_KEYS = {
    "object_id",
    "object_type",
    "parent_object_id",
    "inherited_from",
    "visibility_mode",
    "project_ids",
    "security_level",
    "parent_security_level",
    "effective_security_level",
    "acl_allow",
    "acl_deny",
    "acl_expires_at",
    P_ACL_EXPIRES_AT_TS,
    "excluded",
    "acl_sync_state",
    "share_status",
}


# ── 替身 ──────────────────────────────────────────────────────────────────────


class _Doc:
    """文档权限快照（对齐 build_object_rows 读取的字段）."""

    def __init__(self, **kw):
        self.id = kw.get("id", DOC_ID)
        self.tenant_id = kw.get("tenant_id", TENANT)
        self.owner_id = kw.get("owner_id", U_OWNER)
        self.department_id = kw.get("department_id", DEPT)
        self.access_level = kw.get("access_level", "department")
        self.security_level = kw.get("security_level", DEFAULT_CLEARANCE_BY_ROLE["employee"])
        self.visibility_mode = kw.get("visibility_mode", "tier")
        self.project_ids = kw.get("project_ids", [])
        self.acl_allow = kw.get("acl_allow", [])
        self.acl_deny = kw.get("acl_deny", [])
        self.acl_expires_at = kw.get("acl_expires_at", None)
        self.share_status = kw.get("share_status", "none")
        self.share_grant_scope = kw.get("share_grant_scope", None)


def _point(pid: str, **payload) -> dict:
    return {"id": pid, "payload": payload}


class _FakeQdrant:
    """只实现 ``set_payload``，按点 id 合并载荷，并记录调用痕迹."""

    def __init__(self) -> None:
        self.points: dict[str, dict] = {}
        self.set_payload_calls: list[tuple[list[str], dict]] = []

    def seed_upsert(self, point_id: str, payload: dict) -> None:
        """模拟入库侧 ``upsert_vectors`` 写入的基础载荷（含 tenant/user/level/department）."""
        self.points.setdefault(point_id, {}).update(payload)

    async def set_payload(self, collection_name, payload, points, wait=True):
        for pid in points:
            self.points.setdefault(pid, {}).update(payload)
        self.set_payload_calls.append((list(points), dict(payload)))
        return True


@pytest.fixture()
def fake_qdrant(monkeypatch):
    import types

    import app.config as config_mod
    import app.db.qdrant as qdrant_mod

    fake = _FakeQdrant()
    monkeypatch.setattr(qdrant_mod, "get_qdrant_client", lambda: fake)
    # _set_point_payload 会读 settings.QDRANT_COLLECTION；宿主机无 .env，
    # 用一个最小替身顶掉，避免因缺 POSTGRES_PASSWORD 而伪装成"推送失败"。
    monkeypatch.setattr(
        config_mod,
        "get_settings",
        lambda: types.SimpleNamespace(QDRANT_COLLECTION="documents"),
    )
    return fake


def _pred(*, user_id: str, department_id: str = DEPT, principals=None) -> ScopePredicate:
    """真实维度的 predicate：员工密级（不是默认 SECURITY_LEVEL_MIN，否则先被密级闸门拒）."""
    return ScopePredicate(
        user_id=user_id,
        tenant_ids=frozenset({TENANT}),
        owns_tenant_ids=frozenset(),
        department_id=department_id,
        tenant_wide=False,
        clearance=DEFAULT_CLEARANCE_BY_ROLE["employee"],
        principals=frozenset(
            principals if principals is not None else {f"user:{user_id}"}
        ),
        now=FIXED_NOW,
    )


def _image_and_table_points() -> list[dict]:
    return [
        _point(
            IMG_PID,
            content_type="image",
            image_id=IMG_ID,
            image_path="images/a.png",
            page_number=1,
        ),
        _point(TBL_PID, content_type="table", image_id=IMG_ID, chunk_index=1),
    ]


def _seed_base_payload(fake: _FakeQdrant) -> None:
    """写入 image / table 两点由 ``upsert_vectors`` 提供的"基础层"载荷。

    真实 Qdrant 点的载荷 = upsert_vectors 写入的基础层 ∪ 物化层（set_payload 合并）。
    ``allows()`` 需要基础层的 tenant/user/access_level/department 才能判定。
    """
    base = {
        "document_id": DOC_ID,
        "tenant_id": TENANT,
        "user_id": U_OWNER,
        "access_level": "department",
        "department_id": DEPT,
        "image_id": IMG_ID,
    }
    fake.seed_upsert(IMG_PID, {**base, "content_type": "image"})
    fake.seed_upsert(TBL_PID, {**base, "content_type": "table"})


# ═══════════════════════════════════════════════════════════════════════════════
# ① 行构造：image 行必须带 _point_id / _payload
# ═══════════════════════════════════════════════════════════════════════════════


def test_image_row_carries_point_id_and_payload():
    """image 行（缺陷 A 的漏点）必须与分块行一样带 ``_point_id`` / ``_payload``."""
    doc = _Doc()
    rows, _ = build_object_rows(doc, _image_and_table_points(), now=FIXED_NOW)

    img_row = next(r for r in rows if r["object_type"] == OBJECT_TYPE_IMAGE)
    assert img_row["_point_id"] == IMG_PID, (
        "image 行的 _point_id 必须来自图片本体块的点 id；缺失会让 push_payload_async 跳过它"
    )
    assert "_payload" in img_row, "image 行必须构造 _payload，否则第 11 环读不到 object_id"

    payload = img_row["_payload"]
    assert payload["object_id"] == make_object_id(
        DOC_ID, IMG_ID, object_type=OBJECT_TYPE_IMAGE
    )
    # 值与 PG 行同源（逐键相等，不是另起一套推导）
    for key in ("visibility_mode", "security_level", "acl_sync_state", "excluded"):
        assert payload[key] == img_row[key], f"{key} 必须与 PG 行同值"
    assert payload["acl_allow"] == sorted(img_row["acl_allow"])
    assert payload["effective_security_level"] == img_row["effective_security_level"]


def test_image_point_id_is_the_image_body_not_the_ocr_derived_chunk():
    """OCR 派生块（table，带 image_id）也有自己的点；image 行的点 id 不得被它顶替."""
    doc = _Doc()
    # 故意让派生块排在图片本体之前，逼出"首次建行者是谁"的分支
    points = [
        _point(TBL_PID, content_type="table", image_id=IMG_ID, chunk_index=1),
        _point(IMG_PID, content_type="image", image_id=IMG_ID, page_number=1),
    ]
    rows, _ = build_object_rows(doc, points, now=FIXED_NOW)

    img_row = next(r for r in rows if r["object_type"] == OBJECT_TYPE_IMAGE)
    assert img_row["_point_id"] == IMG_PID, (
        "image 对象的点 id 必须是图片本体块；用派生块的点去 set_payload 会写错对象"
    )
    # 派生块自身的点 id 保持不变（它走分块行那条路径）
    tbl_row = next(r for r in rows if r["object_type"] == OBJECT_TYPE_TABLE)
    assert tbl_row["_point_id"] == TBL_PID


# ═══════════════════════════════════════════════════════════════════════════════
# ② 推送：image 点不再被 push_payload_async 跳过
# ═══════════════════════════════════════════════════════════════════════════════


def test_push_payload_materializes_image_point(fake_qdrant):
    """推送后 image 点载荷必须含全部 7 个物化键（对照：分块点也照常收到）."""
    _seed_base_payload(fake_qdrant)
    doc = _Doc()
    rows, _ = build_object_rows(doc, _image_and_table_points(), now=FIXED_NOW)

    pushed = asyncio.run(push_payload_async(rows))

    # doc 镜像行无点、被跳过；image + table 两点都应被推
    assert pushed == 2, (
        "image 点必须真的被推送（缺陷 A 下这里会是 1 —— 图片点被整行跳过）"
    )
    img_payload = fake_qdrant.points[IMG_PID]
    for key in MATERIALIZED_KEYS:
        assert key in img_payload, f"image 点载荷缺物化字段：{key}"
    assert set(FULL_PAYLOAD_KEYS) <= set(img_payload)
    assert img_payload["object_id"] == make_object_id(
        DOC_ID, IMG_ID, object_type=OBJECT_TYPE_IMAGE
    )
    # 合并式写入不得抹掉基础层字段
    assert img_payload["user_id"] == U_OWNER
    assert img_payload["access_level"] == "department"
    # 分块点也收到（回归：别为修图片而误伤分块）
    assert "object_id" in fake_qdrant.points[TBL_PID]


# ═══════════════════════════════════════════════════════════════════════════════
# ③ 第 11 环对象级复核：owner 与同部门非 owner 都放行
# ═══════════════════════════════════════════════════════════════════════════════


def test_image_payload_allows_owner_and_department_mate(fake_qdrant):
    """物化后 ``allows()`` 对 owner 与同部门员工都放行，异部门仍拒（非永真断言）."""
    _seed_base_payload(fake_qdrant)
    doc = _Doc(access_level="department", department_id=DEPT)
    rows, _ = build_object_rows(doc, _image_and_table_points(), now=FIXED_NOW)
    asyncio.run(push_payload_async(rows))

    view = ObjectACLView.from_payload(fake_qdrant.points[IMG_PID])

    owner_decision = allows(_pred(user_id=U_OWNER), view)
    assert owner_decision.allowed, f"owner 必须能检索到图片：{owner_decision}"

    mate_decision = allows(_pred(user_id=U_MATE), view)
    assert mate_decision.allowed, f"同部门员工必须能检索到图片：{mate_decision}"

    # 负向对照：异部门员工仍被拒（证明不是无脑放行）
    other_decision = allows(
        _pred(user_id=U_OTHER_DEPT, department_id=DEPT_OTHER), view
    )
    assert not other_decision.allowed, "异部门员工不应看到该图片"
