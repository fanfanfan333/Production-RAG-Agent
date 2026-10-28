"""Agent Runtime 管理面：技能清单 + 个人长期记忆.

设计边界：

* **只暴露自己的记忆**：读取与撤销都以 `(home_tenant_id(user), user.id)` 为键，
  没有管理员跨用户分支 —— 平台管理员审核他人的训练样本，不代表能读他人的
  长期设定。
* **不暴露技能内容以外的东西**：技能是只读常量，接口只回传清单，不允许注册。
* **写记忆必须显式**：`POST /agent/memories` 只接受显式提交的内容，并复用
  `sanitize_memory_text()` 做注入/凭据/长度校验；不合格返回 400，不静默丢弃。
* **撤销不是删除**：`revoke` 只置位，历史仍可查，符合"可审计可回退"的要求。
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.agent_runtime.memory_store import (
    MAX_MEMORY_CHARS,
    MEMORY_KINDS,
    load_memories,
    normalize_kind,
    revoke_memory,
    save_memory,
)
from app.agent_runtime.skills import SKILL_REGISTRY_LOCKED, list_skills
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.permissions import require_permission
from app.services.tenancy import home_tenant_id

router = APIRouter(tags=["Agent runtime"])


class MemoryCreate(BaseModel):
    kind: str = Field(default="note", max_length=32)
    content: str = Field(min_length=1, max_length=MAX_MEMORY_CHARS)


def _serialize(row) -> dict:
    return {
        "id": str(row.id),
        "kind": row.kind,
        "content": row.content,
        "source": row.source,
        "version": row.version,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


@router.get("/agent/skills")
async def list_agent_skills(
    user: Annotated[User, Depends(require_permission("conversation.read"))],
):
    """内置只读技能清单（不含可执行信息，注册表锁定状态一并回传）."""
    return {"skills": list_skills(), "registry_locked": SKILL_REGISTRY_LOCKED}


@router.get("/agent/memories")
async def list_agent_memories(
    user: Annotated[User, Depends(require_permission("conversation.read"))],
    limit: int = Query(default=8, ge=1, le=50),
):
    """列出当前用户未撤销的长期记忆。"""
    async with get_db_session() as session:
        rows = await load_memories(
            session, tenant_id=home_tenant_id(user), user_id=user.id, limit=limit
        )
    return {"memories": [_serialize(row) for row in rows]}


@router.post("/agent/memories")
async def create_agent_memory(
    payload: MemoryCreate,
    # 写记忆是"写"操作：用 conversation.write 而不是 read。用读权限放行写入
    # 是权限语义错配 —— 将来若出现"只读账号"，它会莫名其妙获得持久化能力。
    user: Annotated[User, Depends(require_permission("conversation.write"))],
):
    """显式写入一条个人长期记忆（内容会过注入/凭据/长度校验）."""
    # kind 会被拼进 system 层的行首，只接受封闭枚举。非法值直接 422，
    # 而不是静默降级成 note —— 静默降级会让客户端以为"设置成功了"。
    if normalize_kind(payload.kind) != str(payload.kind or "").strip().lower():
        raise HTTPException(
            status_code=422,
            detail=f"kind 必须是 {list(MEMORY_KINDS)} 之一",
        )
    async with get_db_session() as session:
        row = await save_memory(
            session,
            tenant_id=home_tenant_id(user),
            user_id=user.id,
            kind=payload.kind,
            content=payload.content,
            source="manual_api",
        )
    if row is None:
        raise HTTPException(
            status_code=400,
            detail="记忆内容不合格：疑似提示注入/凭据，或长度超出限制",
        )
    await record_audit(
        "agent.memory.create",
        user_id=user.id,
        username=user.username,
        resource_type="agent_memory",
        resource_id=str(row.id),
        detail=f"kind={row.kind};chars={len(row.content)}",
    )
    return {"memory": _serialize(row)}


@router.post("/agent/memories/{memory_id}/revoke")
async def revoke_agent_memory(
    memory_id: uuid.UUID,
    user: Annotated[User, Depends(require_permission("conversation.delete"))],
    reason: str = Query(default="", max_length=500),
):
    """撤销一条记忆（不删除）；作用域不匹配一律 404，不泄漏存在性。"""
    async with get_db_session() as session:
        changed = await revoke_memory(
            session,
            memory_id=memory_id,
            tenant_id=home_tenant_id(user),
            user_id=user.id,
            revoked_by=user.username,
            reason=reason,
        )
    if not changed:
        raise HTTPException(status_code=404, detail="记忆不存在、已撤销或无权操作")
    await record_audit(
        "agent.memory.revoke",
        user_id=user.id,
        username=user.username,
        resource_type="agent_memory",
        resource_id=str(memory_id),
        detail=f"reason={reason[:200]}",
    )
    return {"status": "revoked", "id": str(memory_id)}
