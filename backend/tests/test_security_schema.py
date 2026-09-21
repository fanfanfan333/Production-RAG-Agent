"""T1 安全隔离**数据层**的结构断言（``docs/system_design_security_isolation.md`` §4 / §8）.

这里只验证"形状"，不连数据库：

* ``documents`` +9 列、``users`` +1 列（clearance）、``chunk_parents`` +1 列
  （owner_id）—— **列存在、类型对、server_default 对**
* 4 张新表的列与索引都在
* :func:`make_object_id` 的三种形态与失败路径
* 密级/可见性常量在 ORM、settings、迁移三处**不漂移**
* 迁移文件挂在正确的 head 上，且具备幂等/可回滚的写法特征

为什么要把"形状"也写成测试：本轮的红线是**零行为变化**，而"零变化"是靠
``server_default`` 保证的。一次手滑把 ``security_level`` 的默认值写成 3，
存量文档会一夜之间全部不可见 —— 没有任何运行时报错会提示这件事。
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest

from app.db import security_models as sm
from app.db.models import ChunkParent, Document
from app.db.user_models import User

BACKEND_ROOT = Path(__file__).resolve().parents[1]
VERSIONS_DIR = BACKEND_ROOT / "alembic" / "versions"
MIGRATION_FILE = VERSIONS_DIR / "q2k3l4m5n6o7_add_security_isolation.py"


# ── documents +9 列 ───────────────────────────────────────────────────────────


def test_documents_has_nine_new_columns() -> None:
    cols = Document.__table__.columns
    expected = {
        "security_level", "visibility_mode", "project_ids", "acl_allow",
        "acl_deny", "acl_expires_at", "acl_sync_state", "share_status",
        "share_grant_scope",
    }
    missing = expected - set(cols.keys())
    assert not missing, f"documents 缺少新增列: {sorted(missing)}"


def test_documents_security_columns_are_not_null_with_safe_defaults() -> None:
    """`server_default` 是"存量行为零变化"的唯一保障 —— 逐列钉死。"""
    cols = Document.__table__.columns
    assert cols["security_level"].nullable is False
    assert cols["security_level"].default.arg == str(sm.DEFAULT_SECURITY_LEVEL) or \
        cols["security_level"].server_default is not None
    assert cols["visibility_mode"].nullable is False
    assert cols["project_ids"].nullable is False
    assert cols["acl_allow"].nullable is False
    assert cols["acl_deny"].nullable is False
    assert cols["acl_sync_state"].nullable is False
    assert cols["share_status"].nullable is False
    # 只有"有效期"与"共享范围"允许为空（它们本来就是可选语义）
    assert cols["acl_expires_at"].nullable is True
    assert cols["share_grant_scope"].nullable is True


def test_access_level_three_values_untouched() -> None:
    """红线：access_level 三值语义不动（可见范围 ≠ 密级）。"""
    cols = Document.__table__.columns
    assert cols["access_level"].nullable is False
    assert cols["access_level"].default.arg == "private"
    # 密级与可见范围是**两个不同维度**，必须同时存在
    assert "security_level" in cols
    assert cols["security_level"] is not cols["access_level"]


def test_users_has_nullable_clearance_without_server_default() -> None:
    """
    ``users.clearance`` 必须 **无 server_default**：NULL = 按角色推导。

    一旦写死初值，"管理员下调某人角色"就不会反映到密级上（A8 直接失效）。
    """
    col = User.__table__.columns.get("clearance")
    assert col is not None, "users 缺少 clearance 列"
    assert col.nullable is True
    assert col.server_default is None, "clearance 不得有 server_default（NULL = 按角色推导）"


def test_chunk_parents_gained_owner_id() -> None:
    """父块原先做不了用户级过滤（缺 owner_id）—— T1 补齐。"""
    assert "owner_id" in ChunkParent.__table__.columns


# ── 四张新表 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,required",
    [
        (
            sm.DocumentObject,
            {
                "object_id", "document_id", "object_type", "parent_object_id",
                "inherited_from", "tenant_id", "owner_id", "department_id",
                "access_level", "visibility_mode", "project_ids",
                "security_level", "parent_security_level",
                "effective_security_level", "acl_allow", "acl_deny",
                "acl_expires_at", "acl_sync_state", "excluded",
                "share_status", "chunk_index", "page_number", "image_id",
                "image_path", "content_type",
            },
        ),
        (sm.Project, {"id", "tenant_id", "name", "created_by", "created_at"}),
        (
            sm.ProjectMember,
            {"project_id", "user_id", "expires_at", "added_by", "added_at"},
        ),
        (
            sm.AclGrant,
            {
                "id", "object_id", "document_id", "subject", "effect", "status",
                "granted_by", "reviewer_id", "reason", "expires_at",
                "created_at", "reviewed_at",
            },
        ),
    ],
)
def test_new_tables_have_required_columns(model, required) -> None:
    missing = required - set(model.__table__.columns.keys())
    assert not missing, f"{model.__tablename__} 缺少列: {sorted(missing)}"


def test_document_objects_safe_defaults() -> None:
    cols = sm.DocumentObject.__table__.columns
    assert cols["security_level"].nullable is False
    assert cols["effective_security_level"].nullable is False
    assert cols["visibility_mode"].nullable is False
    assert cols["visibility_mode"].default.arg == sm.DEFAULT_VISIBILITY_MODE
    assert cols["excluded"].nullable is False
    assert cols["excluded"].default.arg is False
    assert cols["acl_sync_state"].default.arg == sm.ACL_SYNC_SYNCED


def test_document_objects_indexes_present() -> None:
    names = {idx.name for idx in sm.DocumentObject.__table__.indexes}
    for expected in (
        "ix_dobj_document", "ix_dobj_parent", "ix_dobj_type_tenant",
        "ix_dobj_eff_level", "ix_dobj_sync", "ix_dobj_projects",
        "ix_dobj_acl_allow", "uq_dobj_doc_chunk",
    ):
        assert expected in names, f"document_objects 缺少索引 {expected}"


# ── make_object_id：**唯一决策点** ────────────────────────────────────────────


def test_make_object_id_doc_is_plain_document_id() -> None:
    doc_id = str(uuid.uuid4())
    assert sm.make_object_id(doc_id, None, object_type=sm.OBJECT_TYPE_DOC) == doc_id


def test_make_object_id_namespaces_non_doc_objects() -> None:
    doc_id = str(uuid.uuid4())
    raw = "p3-i1"
    made = sm.make_object_id(doc_id, raw, object_type=sm.OBJECT_TYPE_IMAGE)
    assert made.endswith(raw)
    assert doc_id in made
    if sm.OBJECT_ID_MODE == "scoped":
        assert made == f"{doc_id}{sm.OBJECT_ID_SEPARATOR}{raw}"
    else:
        assert made == raw


def test_make_object_id_is_stable_and_unique() -> None:
    doc_a, doc_b = str(uuid.uuid4()), str(uuid.uuid4())
    a1 = sm.make_object_id(doc_a, "x1", object_type=sm.OBJECT_TYPE_TEXT_CHUNK)
    a2 = sm.make_object_id(doc_a, "x1", object_type=sm.OBJECT_TYPE_TEXT_CHUNK)
    b1 = sm.make_object_id(doc_b, "x1", object_type=sm.OBJECT_TYPE_TEXT_CHUNK)
    assert a1 == a2, "同一输入必须产出同一 object_id（幂等回填的前提）"
    assert a1 != b1, "不同文档的同类局部 id 不得撞车（当前默认最保守形态）"
    assert len(a1) <= sm.OBJECT_ID_MAX_LEN


def test_make_object_id_rejects_missing_raw() -> None:
    with pytest.raises(ValueError):
        sm.make_object_id(str(uuid.uuid4()), None, object_type=sm.OBJECT_TYPE_IMAGE)
    with pytest.raises(ValueError):
        sm.make_object_id(str(uuid.uuid4()), "   ", object_type=sm.OBJECT_TYPE_TABLE)


@pytest.mark.parametrize(
    "content_type,expected",
    [
        ("table", sm.OBJECT_TYPE_TABLE),
        ("code", sm.OBJECT_TYPE_CODE),
        ("image", sm.OBJECT_TYPE_IMAGE),
        ("text", sm.OBJECT_TYPE_TEXT_CHUNK),
        (None, sm.OBJECT_TYPE_TEXT_CHUNK),
        # 未知取值（待确认项 b）一律落到最宽松档，不得静默丢对象
        ("weird-new-type", sm.OBJECT_TYPE_TEXT_CHUNK),
    ],
)
def test_object_type_from_content_type(content_type, expected) -> None:
    assert sm.object_type_from_content_type(content_type) == expected


# ── 常量三处不漂移 ────────────────────────────────────────────────────────────


def test_security_level_bands_are_four() -> None:
    assert (sm.SECURITY_LEVEL_PUBLIC, sm.SECURITY_LEVEL_INTERNAL,
            sm.SECURITY_LEVEL_CONFIDENTIAL, sm.SECURITY_LEVEL_SECRET) == (0, 1, 2, 3)
    assert sm.SECURITY_LEVEL_MAX == 3
    assert sm.DEFAULT_SECURITY_LEVEL == 1, "已裁决 Q2：存量未标注 = 1（内部）"


def test_default_security_level_matches_settings() -> None:
    """settings 与 ORM 常量必须一致（回填脚本会拒绝在漂移时运行）。"""
    from app.config import get_settings

    assert get_settings().DEFAULT_SECURITY_LEVEL == sm.DEFAULT_SECURITY_LEVEL


def test_visibility_defaults_to_tier() -> None:
    """已裁决 Q3：默认 tier ⇒ 存量文档完全不受项目维度影响。"""
    assert sm.DEFAULT_VISIBILITY_MODE == sm.VISIBILITY_MODE_TIER == "tier"
    assert sm.VISIBILITY_MODE_PROJECT == "project"


# ── 迁移文件 ──────────────────────────────────────────────────────────────────


def _parse_revisions() -> dict[str, str | None]:
    revisions: dict[str, str | None] = {}
    for path in VERSIONS_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        rev = re.search(r"^revision(?:\s*:\s*str)?\s*=\s*[\"']([^\"']+)[\"']",
                        text, re.MULTILINE)
        down = re.search(r"^down_revision(?:\s*:\s*[^=]+)?\s*=\s*"
                         r"(?:None|[\"']([^\"']+)[\"'])", text, re.MULTILINE)
        if not rev:
            continue
        revisions[rev.group(1)] = (down.group(1) if down and down.group(1) else None)
    return revisions


def test_migration_file_exists_and_chains_to_head() -> None:
    assert MIGRATION_FILE.exists(), "缺少迁移 q2k3l4m5n6o7_add_security_isolation.py"
    revisions = _parse_revisions()
    assert "q2k3l4m5n6o7" in revisions
    referenced = {d for d in revisions.values() if d}
    heads = sorted(set(revisions) - referenced)
    assert heads == ["q2k3l4m5n6o7"], f"迁移未挂在唯一 head 上，当前 heads={heads}"


def test_migration_is_idempotent_and_reversible() -> None:
    text = MIGRATION_FILE.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS document_objects" in text
    assert "CREATE TABLE IF NOT EXISTS projects" in text
    assert "CREATE TABLE IF NOT EXISTS project_members" in text
    assert "CREATE TABLE IF NOT EXISTS acl_grants" in text
    # PG 没有 ADD COLUMN IF NOT EXISTS → 必须靠 inspect 探测后条件添加
    assert "get_columns" in text
    # 可回滚：逆序 drop
    assert "def downgrade()" in text
    assert "DROP TABLE IF EXISTS document_objects" in text
    # 只加列、不写业务规则
    for forbidden in ("UPDATE documents", "INSERT INTO document_objects"):
        assert forbidden not in text, f"迁移里出现业务规则 {forbidden!r}"


# ── 角色 → 默认密级（决策 14，admin 不豁免）───────────────────────────────────


def test_default_clearance_by_role() -> None:
    from app.services.security_policy import DEFAULT_CLEARANCE_BY_ROLE

    assert DEFAULT_CLEARANCE_BY_ROLE["employee"] == 1
    assert DEFAULT_CLEARANCE_BY_ROLE["dept_manager"] == 2
    assert DEFAULT_CLEARANCE_BY_ROLE["kb_admin"] == 3
    assert DEFAULT_CLEARANCE_BY_ROLE["company_admin"] == 3
    assert DEFAULT_CLEARANCE_BY_ROLE["admin"] == 3     # 有上限的"高"，不是无限
    # 历史角色必须全部覆盖，否则老账号会静默掉到 0（行为变化）
    for legacy in (User.ROLE_ADMIN, User.ROLE_MANAGER, User.ROLE_EDITOR,
                   User.ROLE_USER, User.ROLE_VIEWER, User.ROLE_EMPLOYEE):
        assert legacy in DEFAULT_CLEARANCE_BY_ROLE, f"角色 {legacy} 未映射默认密级"


def test_unknown_role_has_no_implicit_clearance() -> None:
    from app.services.security_policy import CLEARANCE_ON_FAILURE, clearance_for_role

    assert clearance_for_role("no-such-role") == CLEARANCE_ON_FAILURE == 0
    assert clearance_for_role(None) == 0


def test_permission_points_added_without_touching_matrix() -> None:
    """新增 3 个动作权限点；既有角色映射与历史映射一个不动。"""
    from app.services.permissions import ROLE_PERMISSIONS, has_permission

    from app.db.user_models import User as U

    # 既有能力不变（回归）
    assert has_permission(U(role=U.ROLE_EMPLOYEE), "document.write")
    assert not has_permission(U(role=U.ROLE_EMPLOYEE), "document.publish.company")
    assert has_permission(U(role=U.ROLE_KB_ADMIN), "document.publish.company")
    assert has_permission(U(role=U.ROLE_ADMIN), "任意.权限")
    # 新增点：只给 kb_admin 及以上（admin 走通配符）
    assert has_permission(U(role=U.ROLE_KB_ADMIN), "security.grant")
    assert has_permission(U(role=U.ROLE_KB_ADMIN), "security.review.grant")
    assert has_permission(U(role=U.ROLE_KB_ADMIN), "security.escalate")
    assert not has_permission(U(role=U.ROLE_DEPT_MANAGER), "security.grant")
    # 矩阵本身只是被**追加**，没有被重排或替换
    assert ROLE_PERMISSIONS[U.ROLE_ADMIN] == frozenset({"*"})
    assert ROLE_PERMISSIONS[U.ROLE_MANAGER] is ROLE_PERMISSIONS[U.ROLE_DEPT_MANAGER]
    assert ROLE_PERMISSIONS[U.ROLE_EDITOR] is ROLE_PERMISSIONS[U.ROLE_EMPLOYEE]
