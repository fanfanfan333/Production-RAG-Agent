"""add security isolation (五维权限模型的存储层：密级 / 项目 / 对象级 ACL)

本迁移是 **T1（数据层基础设施）** 的一半，对应
``docs/system_design_security_isolation.md`` §4 与 §8。

做了什么
────────
    documents      +9 列（security_level / visibility_mode / project_ids /
                    acl_allow / acl_deny / acl_expires_at / acl_sync_state /
                    share_status / share_grant_scope）+ 6 个索引
    users          +1 列（clearance，**无 server_default**：NULL = 按角色推导）
    chunk_parents  +1 列（owner_id —— 父块原先做不了用户级过滤）
    新表           document_objects / projects / project_members / acl_grants

**刻意不做的事**
────────────────
* **不写任何业务规则**（沿用上游决策 5 的纪律）。存量默认值全部交给
  ``server_default``：

      security_level   = 1      （已裁决 Q2：存量未标注 = 内部）
      visibility_mode  = 'tier' （已裁决 Q3：存量行为零变化）
      project_ids/acl_allow/acl_deny = '[]'
      acl_sync_state   = 'synced'（存量视为已同步，避免启动即误报 stale）

  因此**本迁移跑完之后，现有查询一行的行为都不变** —— 这是不可退化基线 B1 的
  落点。业务规则（谁给谁授予、派生对象怎么取严）一律放幂等脚本
  ``scripts/backfill_security_level.py`` 与 T4 的 cascade。

* **users.clearance 不加 server_default**：一旦写死初值，"管理员下调某人角色"
  就不会反映到密级上（A8 直接失效）。NULL 是有意义的默认（按角色推导）。

* **document_objects 的存量行不在这里生成**（需要按 Qdrant payload 逐块物化，
  属于业务规则），由回填脚本负责。

幂等与可回滚
────────────
* 全部 ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS``；
  PG 没有 ``ADD COLUMN IF NOT EXISTS``，改用 ``sa.inspect(bind).get_columns()``
  探测后条件添加 —— 与上游 ``h3c4d5e6f7g8`` 等迁移的写法一致。
  连跑两次无报错（与启动时 ``Base.metadata.create_all`` 两种建表机制任一先跑都不冲突）。
* ``downgrade()`` 按 ``upgrade()`` 的**严格逆序** drop index → drop column →
  drop table。

  ⚠️ **downgrade 会丢数据**：新列被删（密级 / 项目 / ACL 全部丢失），
  ``document_objects`` / ``acl_grants`` 整表被删。
  **回滚前请先导出**：

      pg_dump -t document_objects -t acl_grants -t projects -t project_members ...

  这与上游迁移的 downgrade 风格一致（它们也直接 drop），不做"假装安全"的软删除。

Revision ID: q2k3l4m5n6o7
Revises: p1j2k3l4m5n6
Create Date: 2026-09-20 02:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "q2k3l4m5n6o7"
down_revision: Union[str, None] = "p1j2k3l4m5n6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ── 新增列的定义（upgrade / downgrade 共用同一份，避免两处各写一遍而漂移）────────

_DOCUMENT_COLUMNS: list[tuple[str, sa.Column]] = [
    ("security_level", sa.Column("security_level", sa.SmallInteger(), nullable=False, server_default="1")),
    ("visibility_mode", sa.Column("visibility_mode", sa.String(length=16), nullable=False, server_default="tier")),
    ("project_ids", sa.Column("project_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb"))),
    ("acl_allow", sa.Column("acl_allow", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb"))),
    ("acl_deny", sa.Column("acl_deny", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb"))),
    ("acl_expires_at", sa.Column("acl_expires_at", sa.DateTime(timezone=True), nullable=True)),
    ("acl_sync_state", sa.Column("acl_sync_state", sa.String(length=16), nullable=False, server_default="synced")),
    ("share_status", sa.Column("share_status", sa.String(length=16), nullable=False, server_default="none")),
    ("share_grant_scope", sa.Column("share_grant_scope", sa.String(length=16), nullable=True)),
]

_OTHER_COLUMNS: dict[str, list[tuple[str, sa.Column]]] = {
    "users": [
        # 无 server_default —— NULL = 按角色推导（见模块 docstring）
        ("clearance", sa.Column("clearance", sa.SmallInteger(), nullable=True)),
    ],
    "chunk_parents": [
        ("owner_id", sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True)),
    ],
}

# (index_name, table, DDL) —— 逆序 drop 即 downgrade
_INDEXES: list[tuple[str, str, str]] = [
    ("ix_documents_security_level", "documents",
     "CREATE INDEX IF NOT EXISTS ix_documents_security_level ON documents(security_level)"),
    ("ix_documents_visibility_mode", "documents",
     "CREATE INDEX IF NOT EXISTS ix_documents_visibility_mode ON documents(visibility_mode)"),
    ("ix_documents_acl_sync_state", "documents",
     "CREATE INDEX IF NOT EXISTS ix_documents_acl_sync_state ON documents(acl_sync_state)"),
    ("ix_documents_project_ids", "documents",
     "CREATE INDEX IF NOT EXISTS ix_documents_project_ids ON documents USING gin (project_ids)"),
    ("ix_documents_acl_allow", "documents",
     "CREATE INDEX IF NOT EXISTS ix_documents_acl_allow ON documents USING gin (acl_allow)"),
    ("ix_documents_acl_deny", "documents",
     "CREATE INDEX IF NOT EXISTS ix_documents_acl_deny ON documents USING gin (acl_deny)"),
    ("ix_chunk_parents_owner", "chunk_parents",
     "CREATE INDEX IF NOT EXISTS ix_chunk_parents_owner ON chunk_parents(owner_id)"),
    ("ix_users_clearance", "users",
     "CREATE INDEX IF NOT EXISTS ix_users_clearance ON users(clearance)"),

    # ── document_objects ──────────────────────────────────────────────────────
    ("ix_dobj_document", "document_objects",
     "CREATE INDEX IF NOT EXISTS ix_dobj_document ON document_objects(document_id)"),
    ("ix_dobj_parent", "document_objects",
     "CREATE INDEX IF NOT EXISTS ix_dobj_parent ON document_objects(parent_object_id)"),
    ("ix_dobj_type_tenant", "document_objects",
     "CREATE INDEX IF NOT EXISTS ix_dobj_type_tenant ON document_objects(object_type, tenant_id)"),
    ("ix_dobj_eff_level", "document_objects",
     "CREATE INDEX IF NOT EXISTS ix_dobj_eff_level ON document_objects(effective_security_level)"),
    ("ix_dobj_sync", "document_objects",
     "CREATE INDEX IF NOT EXISTS ix_dobj_sync ON document_objects(acl_sync_state) "
     "WHERE acl_sync_state <> 'synced'"),
    ("ix_dobj_projects", "document_objects",
     "CREATE INDEX IF NOT EXISTS ix_dobj_projects ON document_objects USING gin (project_ids)"),
    ("ix_dobj_acl_allow", "document_objects",
     "CREATE INDEX IF NOT EXISTS ix_dobj_acl_allow ON document_objects USING gin (acl_allow)"),
    ("uq_dobj_doc_chunk", "document_objects",
     "CREATE UNIQUE INDEX IF NOT EXISTS uq_dobj_doc_chunk "
     "ON document_objects(document_id, chunk_index) WHERE chunk_index IS NOT NULL"),

    # ── projects / project_members ────────────────────────────────────────────
    ("ix_projects_tenant", "projects",
     "CREATE INDEX IF NOT EXISTS ix_projects_tenant ON projects(tenant_id)"),
    ("ix_pmember_user", "project_members",
     "CREATE INDEX IF NOT EXISTS ix_pmember_user ON project_members(user_id)"),

    # ── acl_grants ────────────────────────────────────────────────────────────
    ("ix_aclgrants_object_status", "acl_grants",
     "CREATE INDEX IF NOT EXISTS ix_aclgrants_object_status ON acl_grants(object_id, status)"),
    ("ix_aclgrants_subject", "acl_grants",
     "CREATE INDEX IF NOT EXISTS ix_aclgrants_subject ON acl_grants(subject, status)"),
]


def _has_column(table: str, column: str) -> bool:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    try:
        existing = {c["name"] for c in insp.get_columns(table)}
    except Exception:      # noqa: BLE001 — 表不存在时 get_columns 会抛
        return False
    return column in existing


def upgrade() -> None:
    # ── 1. 新表（全部 IF NOT EXISTS，与 create_all 并存）───────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_objects (
            object_id           VARCHAR(128) PRIMARY KEY,
            document_id         UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            object_type         VARCHAR(16)  NOT NULL DEFAULT 'doc',
            parent_object_id    VARCHAR(128) NULL,
            inherited_from      VARCHAR(128) NULL,
            inherited_at        TIMESTAMPTZ  NULL,

            tenant_id           VARCHAR(64)  NOT NULL DEFAULT 'default',
            owner_id            UUID         NULL,
            department_id       VARCHAR(64)  NULL,

            access_level        VARCHAR(20)  NOT NULL DEFAULT 'private',
            visibility_mode     VARCHAR(16)  NOT NULL DEFAULT 'tier',
            project_ids         JSONB        NOT NULL DEFAULT '[]'::jsonb,
            visible_scope       VARCHAR(16)  NULL,

            security_level           SMALLINT NOT NULL DEFAULT 1,
            parent_security_level    SMALLINT NULL,
            effective_security_level SMALLINT NOT NULL DEFAULT 1,

            acl_allow           JSONB        NOT NULL DEFAULT '[]'::jsonb,
            acl_deny            JSONB        NOT NULL DEFAULT '[]'::jsonb,
            acl_expires_at      TIMESTAMPTZ  NULL,

            acl_sync_state      VARCHAR(16)  NOT NULL DEFAULT 'synced',
            excluded            BOOLEAN      NOT NULL DEFAULT false,
            share_status        VARCHAR(16)  NOT NULL DEFAULT 'none',
            share_grant_scope   VARCHAR(16)  NULL,

            chunk_index         INTEGER      NULL,
            page_number         INTEGER      NULL,
            image_id            VARCHAR(128) NULL,
            image_path          VARCHAR(512) NULL,
            content_type        VARCHAR(32)  NULL,

            created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id          VARCHAR(64) PRIMARY KEY,
            tenant_id   VARCHAR(64) NOT NULL DEFAULT 'default',
            name        VARCHAR(128) NOT NULL,
            created_by  UUID NULL REFERENCES users(id) ON DELETE SET NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS project_members (
            project_id  VARCHAR(64) NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at  TIMESTAMPTZ NULL,
            added_by    UUID NULL REFERENCES users(id) ON DELETE SET NULL,
            added_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (project_id, user_id)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS acl_grants (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            object_id     VARCHAR(128) NOT NULL,
            document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            subject       VARCHAR(128) NOT NULL,
            effect        VARCHAR(8)   NOT NULL,
            status        VARCHAR(16)  NOT NULL DEFAULT 'pending',
            granted_by    UUID NULL REFERENCES users(id) ON DELETE SET NULL,
            reviewer_id   UUID NULL REFERENCES users(id) ON DELETE SET NULL,
            reason        TEXT NULL,
            expires_at    TIMESTAMPTZ NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            reviewed_at   TIMESTAMPTZ NULL
        )
        """
    )

    # ── 2. 加列（PG 无 ADD COLUMN IF NOT EXISTS → 探测后条件添加）──────────────
    for table, cols in (("documents", _DOCUMENT_COLUMNS), *_OTHER_COLUMNS.items()):
        for name, col in cols:
            if _has_column(table, name):
                continue
            op.add_column(table, col)

    # ── 3. 索引 ────────────────────────────────────────────────────────────────
    for _name, _table, ddl in _INDEXES:
        op.execute(ddl)


def downgrade() -> None:
    # 严格逆序：索引 → 列 → 表
    for name, _table, _ddl in reversed(_INDEXES):
        op.execute(f"DROP INDEX IF EXISTS {name}")

    for table, cols in (("documents", _DOCUMENT_COLUMNS), *_OTHER_COLUMNS.items()):
        for name, _col in cols:
            if not _has_column(table, name):
                continue
            op.drop_column(table, name)

    op.execute("DROP TABLE IF EXISTS acl_grants")
    op.execute("DROP TABLE IF EXISTS project_members")
    op.execute("DROP TABLE IF EXISTS projects")
    op.execute("DROP TABLE IF EXISTS document_objects")
