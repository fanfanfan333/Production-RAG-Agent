"""
Central RBAC policy for enterprise RAG resources.

Roles:
- admin: platform administration and all knowledge resources
- manager: manage users' operational content, collections, documents and review
           audit/feedback, but no raw infrastructure or account-role changes
- editor / legacy user: create and manage own knowledge resources; query them
- viewer: read/query access only; cannot upload, edit or delete

Existing ``user`` accounts are intentionally treated as editors to preserve
backward compatibility during the role migration.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, status

from app.api.deps import get_current_user
from app.db.user_models import User

# Canonical permission names are deliberately resource.action strings. This
# keeps dependency declarations readable and makes audits easy to aggregate.
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    User.ROLE_ADMIN: frozenset({"*"}),
    User.ROLE_MANAGER: frozenset(
        {
            "knowledge.read", "knowledge.write", "knowledge.delete",
            "document.read", "document.write", "document.delete",
            "conversation.read", "conversation.write", "conversation.delete",
            "feedback.write", "feedback.read",
            "audit.read",
        }
    ),
    User.ROLE_EDITOR: frozenset(
        {
            "knowledge.read", "knowledge.write", "knowledge.delete",
            "document.read", "document.write", "document.delete",
            "conversation.read", "conversation.write", "conversation.delete",
            "feedback.write",
        }
    ),
    # Legacy user role maps exactly to editor behavior.
    User.ROLE_USER: frozenset(
        {
            "knowledge.read", "knowledge.write", "knowledge.delete",
            "document.read", "document.write", "document.delete",
            "conversation.read", "conversation.write", "conversation.delete",
            "feedback.write",
        }
    ),
    User.ROLE_VIEWER: frozenset(
        {
            "knowledge.read", "document.read", "conversation.read",
            "conversation.write", "feedback.write",
        }
    ),
}


def has_permission(user: User, permission: str) -> bool:
    """Return whether *user* owns an exact named permission or wildcard."""
    permissions = ROLE_PERMISSIONS.get(user.role, frozenset())
    return "*" in permissions or permission in permissions


def require_permission(permission: str) -> Callable:
    """Build a FastAPI dependency enforcing one RBAC permission."""

    async def dependency(user: User = Depends(get_current_user)) -> User:
        if not has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"当前角色无权执行此操作（需要权限：{permission}）",
            )
        return user

    return dependency


async def require_platform_admin(user: User = Depends(get_current_user)) -> User:
    """Platform-only operations: role management and raw Qdrant administration."""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="该操作需要平台管理员权限",
        )
    return user
