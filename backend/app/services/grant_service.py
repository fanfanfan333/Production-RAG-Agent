"""
need-to-know 授予服务（T5，设计文档 §4.5 / 决策 13 / 共享知识 9）.

一句话口径：**``acl_grants`` 是权威源，``document_objects.acl_allow`` 是物化副本**。
本模块负责把"谁在什么对象上、对谁、授予到什么时候"这条权威事实，物化进
``documents`` 与 ``document_objects`` 的 ``acl_allow`` / ``acl_expires_at``，
并把副本推给 Qdrant（payload 是副本，推失败不阻断主流程）。

例外规则（PRD Q5=A，缺一不可）
────────────────────────────
1. **必须对象级显式授予**：只能在 ``doc`` / ``image`` 两类对象上授予
   （共享知识 9：派生对象不得通过 ``acl_allow`` 获得父之外的可见性）。
2. **必须带有效期**：``expires_at`` 为空直接拒绝（DB 列允许 NULL 只为向前兼容，
   业务入口不允许）。
3. **必须走审批链**：``pending`` → 审批（``security.review.grant``）→ ``approved``。
   与 ``share_service`` 的审核思路一致：申请人提交、他人审批，**pending 期间不写
   ``acl_allow``** ⇒ A7（中间态他人不可见）天然成立。
4. **禁止自我授予**：把主体设成自己（``user:<自己 id>``）→ **403**；审批人也不得
   审批"授予自己"的申请。这条是**真正的代码拦截**（不是注释），见
   :func:`request_grant` / :func:`review_grant`。
5. **角色不产生隐式 need-to-know**：``principals`` 里带 ``role:<role>`` 只是让
   ``acl_allow`` 能**按角色**授予；判定内核里没有任何基于 role 的放行分支。

物化副本的取严（决策 13）
────────────────────────
一个对象可能被**多个主体、各自不同有效期**地授予 —— ``acl_allow`` 是数组、
``acl_expires_at`` 是单值，表达不了。因此：

    权威源：``acl_grants``（per-subject 一条，带各自 ``expires_at``）
    物化副本：``acl_allow`` = 全部"已批准且未过期"的主体并集
              ``acl_expires_at`` = 这些主体里**最早的**到期时间

判定侧（第 11 环）再取严一次：若命中 ``acl_allow``，追加一次 ``acl_grants`` 的
未过期/未撤销确认（第 11 环由 T3 负责，本模块只保证副本形态正确）。
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.db.models import Document
from app.db.postgres import get_db_session
from app.db.security_models import (
    ACL_SYNC_PENDING,
    ACL_SYNC_SYNCED,
    GRANT_EFFECT_ALLOW,
    GRANT_EFFECT_DENY,
    GRANT_STATUS_APPROVED,
    GRANT_STATUS_PENDING,
    GRANT_STATUS_REJECTED,
    GRANT_STATUS_REVOKED,
    OBJECT_TYPE_DOC,
    OBJECT_TYPE_IMAGE,
    SECURITY_LEVEL_MAX,
    SECURITY_LEVEL_MIN,
    VISIBILITY_MODE_PROJECT,
    VISIBILITY_MODE_TIER,
    AclGrant,
    DocumentObject,
)
from app.services.audit_service import record_audit
from app.services.tenancy import effective_tenant_id, is_platform_admin
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: 允许的授予主体类型（共享知识 10；``group`` 预留）
SUBJECT_KINDS: frozenset[str] = frozenset({"user", "dept", "role", "project", "group"})

#: need-to-know 只能在对象级授予的两种对象（共享知识 9）
GRANTABLE_OBJECT_TYPES: frozenset[str] = frozenset({OBJECT_TYPE_DOC, OBJECT_TYPE_IMAGE})

_VALID_EFFECTS: frozenset[str] = frozenset({GRANT_EFFECT_ALLOW, GRANT_EFFECT_DENY})
_VALID_VISIBILITY: frozenset[str] = frozenset({VISIBILITY_MODE_TIER, VISIBILITY_MODE_PROJECT})


class GrantError(Exception):
    """need-to-know / 密级管理失败（用户可读中文 + HTTP 状态码）。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ═══════════════════════════════════════════════════════════════════════════════
# 小工具（纯）
# ═══════════════════════════════════════════════════════════════════════════════


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _g(obj: Any, key: str, default: Any = None) -> Any:
    """从 Mapping 或 ORM/替身对象读字段（两处来源共用一套读取）。"""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise GrantError("非法 id", status_code=400) from exc


async def _with_session(session: Any, factory: Any) -> Any:
    if session is not None:
        return await factory(session)
    async with get_db_session() as sess:
        return await factory(sess)


# ═══════════════════════════════════════════════════════════════════════════════
# 主体（principal）解析 —— 「禁止自我授予」与格式校验的唯一落点
# ═══════════════════════════════════════════════════════════════════════════════


def parse_subject(subject: str | None) -> tuple[str, str]:
    """``"user:<uuid>"`` → ``("user", "<uuid>")``；非法格式 → :class:`GrantError`。"""
    raw = (subject or "").strip()
    if ":" not in raw:
        raise GrantError("主体格式非法（应为 kind:value，如 user:<uuid>）", status_code=400)
    kind, _, value = raw.partition(":")
    kind = kind.strip().lower()
    value = value.strip()
    if kind not in SUBJECT_KINDS:
        raise GrantError(
            f"主体类型非法：{kind!r}（允许：{sorted(SUBJECT_KINDS)}）", status_code=400
        )
    if not value:
        raise GrantError("主体标识不能为空", status_code=400)
    return kind, value


def normalize_subject(subject: str | None) -> str:
    kind, value = parse_subject(subject)
    return f"{kind}:{value}"


def is_self_grant(subject: str | None, actor_id: Any) -> bool:
    """该主体是否是"授予给操作者自己"（禁止自我授予的判定式）。"""
    if not actor_id:
        return False
    try:
        kind, value = parse_subject(subject)
    except GrantError:
        return False
    return kind == "user" and value == str(actor_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 物化计算（纯函数）—— "哪些主体该进 acl_allow、到期时间是哪个" 的唯一算法
# ═══════════════════════════════════════════════════════════════════════════════


def _active_allow_grants(
    grants: Sequence[Any], *, now: datetime
) -> list[tuple[str, str, datetime | None]]:
    """筛出"已批准、未过期、effect=allow"的授予 → ``(object_id, subject, expires)``。"""
    out: list[tuple[str, str, datetime | None]] = []
    for g in grants:
        if str(_g(g, "effect")) != GRANT_EFFECT_ALLOW:
            continue
        if str(_g(g, "status")) != GRANT_STATUS_APPROVED:
            continue
        exp = _aware(_g(g, "expires_at"))
        if exp is not None and exp <= now:
            continue      # 到期即失效（即使定时任务还没把它标成 expired）
        out.append((str(_g(g, "object_id")), str(_g(g, "subject")), exp))
    return out


def allowed_subjects(
    grants: Sequence[Any], *, object_id: str, now: datetime | None = None
) -> frozenset[str]:
    """某对象上"当前有效"的被授予主体集合（纯函数）。"""
    now = now or _now()
    target = str(object_id)
    return frozenset(
        s for oid, s, _ in _active_allow_grants(grants, now=now) if oid == target
    )


def _earliest_expiry(
    active: Sequence[tuple[str, str, datetime | None]], object_ids: Sequence[str]
) -> datetime | None:
    """若干对象贡献的到期时间取**最早**；只要有一个无到期 ⇒ 不设对象级上限（None）。"""
    wanted = {str(o) for o in object_ids}
    exps = [exp for oid, _, exp in active if oid in wanted]
    if not exps:
        return None
    if any(e is None for e in exps):
        return None
    return min(exps)  # type: ignore[type-var]


def compute_object_acl(
    rows: Sequence[Any],
    grants: Sequence[Any],
    *,
    doc_object_id: str,
    now: datetime | None = None,
) -> dict[str, tuple[list[str], datetime | None]]:
    """
    由**权威授予**计算每一条 ``document_objects`` 行该写什么 ``acl_allow``（纯函数）.

    规则（与 ``build_object_rows`` / ``derive_child_fields`` 的取严口径对齐）：

        doc 行           → 文档级主体（object_id == 文档 id 的授予）
        普通分块         → 文档级主体（继承文档；与入库物化一致）
        image 行         → 文档级主体 ∪ 该图片自己的授予
        **OCR 派生块**   → 恒空集（`image_id` 非空且 `object_type != image`）

    Returns:
        ``{object_id: (sorted_subjects, expires_at)}``。
    """
    now = now or _now()
    doc_id = str(doc_object_id)
    active = _active_allow_grants(grants, now=now)
    doc_subs = sorted(s for oid, s, _ in active if oid == doc_id)
    doc_exp = _earliest_expiry(active, [doc_id])

    mapping: dict[str, tuple[list[str], datetime | None]] = {}
    for row in rows:
        oid = str(_g(row, "object_id"))
        otype = str(_g(row, "object_type") or OBJECT_TYPE_DOC)
        image_id = _g(row, "image_id")
        if otype == OBJECT_TYPE_IMAGE:
            subs = set(doc_subs) | {s for i, s, _ in active if i == oid}
            mapping[oid] = (sorted(subs), _earliest_expiry(active, [doc_id, oid]))
        elif image_id:
            # OCR 派生块：绝不通过 acl_allow 获得父之外的可见性（共享知识 9）
            mapping[oid] = ([], None)
        else:
            mapping[oid] = (list(doc_subs), doc_exp)
    return mapping


# ═══════════════════════════════════════════════════════════════════════════════
# 物化落库
# ═══════════════════════════════════════════════════════════════════════════════


async def _apply(sess: Any, doc_uuid: uuid.UUID) -> dict:
    """在给定 session 上把权威授予物化进 documents + document_objects（幂等）。"""
    rows = list(
        (await sess.scalars(
            select(DocumentObject).where(DocumentObject.document_id == doc_uuid)
        )).all()
    )
    grants = list(
        (await sess.scalars(
            select(AclGrant).where(AclGrant.document_id == doc_uuid)
        )).all()
    )
    mapping = compute_object_acl(rows, grants, doc_object_id=str(doc_uuid))

    for row in rows:
        subs, exp = mapping.get(str(row.object_id), ([], None))
        row.acl_allow = list(subs)
        row.acl_expires_at = exp
        row.acl_sync_state = ACL_SYNC_SYNCED

    doc = await sess.get(Document, doc_uuid)
    doc_allow: list[str] = []
    if doc is not None:
        doc_allow, doc_exp = mapping.get(str(doc_uuid), ([], None))
        doc.acl_allow = list(doc_allow)
        doc.acl_expires_at = doc_exp
        doc.acl_sync_state = ACL_SYNC_SYNCED

    await sess.flush()
    return {"objects": len(rows), "doc_allow": doc_allow}


async def _push_payload(document_id: Any) -> None:
    """把权威真值推给 Qdrant（best-effort；payload 是副本，推失败不阻断）。"""
    try:
        from app.services.security_cascade import _push_document_payload

        await _push_document_payload(document_id)
    except Exception:      # noqa: BLE001
        logger.exception("materialize_document_acl: payload push failed for %s", document_id)


async def materialize_document_acl(
    document_id: Any, *, session: Any = None, push: bool = True
) -> dict:
    """
    把某文档的 need-to-know 授予物化进 ``acl_allow`` / ``acl_expires_at``（幂等）.

    权威源是 ``acl_grants``；本函数从不"读 payload 来判定"，只把 PG 真值写成副本。
    """
    doc_uuid = _uuid(document_id)
    result = await _with_session(session, lambda s: _apply(s, doc_uuid))
    if push:
        await _push_payload(doc_uuid)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 授予：申请 → 审批 → 撤销
# ═══════════════════════════════════════════════════════════════════════════════


def _assert_doc_visible(actor: Any, doc: Document) -> None:
    """跨租户不可见（平台管理员除外）→ 404（不泄露存在性）。"""
    if is_platform_admin(actor):
        return
    if str(doc.tenant_id) != effective_tenant_id(actor):
        raise GrantError("文档不存在或无权访问", status_code=404)


async def request_grant(
    actor: Any,
    document_id: Any,
    subject: str,
    *,
    effect: str = GRANT_EFFECT_ALLOW,
    reason: str | None = None,
    expires_at: datetime | None,
    object_id: str | None = None,
    session: Any = None,
) -> AclGrant:
    """
    提交一条 need-to-know 授予申请（``pending``；pending 期间**不写** ``acl_allow``）.

    Raises:
        GrantError(403): 自我授予（``subject == user:<自己>``）—— **真正的代码拦截**。
        GrantError(400): 主体格式/效果非法、缺有效期、有效期已过、对象类型不可授予。
        GrantError(404): 文档/对象不可见或不存在。
    """
    normalized = normalize_subject(subject)
    actor_id = str(getattr(actor, "id", "") or "")
    if effect not in _VALID_EFFECTS:
        raise GrantError("effect 只能是 allow / deny", status_code=400)

    # ── 禁止自我授予（第一道，服务层真正拦截）──────────────────────────────────
    if is_self_grant(normalized, actor_id):
        await record_audit(
            "security.grant.request",
            user_id=getattr(actor, "id", None),
            username=getattr(actor, "username", None),
            resource_type="document",
            resource_id=str(document_id),
            detail=f"denied=self_grant; subject={normalized}",
        )
        raise GrantError("禁止自我授予：不能把 need-to-know 例外授予给自己", status_code=403)

    if expires_at is None:
        raise GrantError(
            "need-to-know 例外必须带有效期（acl_expires_at）", status_code=400
        )
    exp = _aware(expires_at)
    if exp <= _now():
        raise GrantError("有效期必须是未来时间", status_code=400)

    doc_uuid = _uuid(document_id)

    async def _do(sess: Any) -> AclGrant:
        doc = await sess.get(Document, doc_uuid)
        if doc is None:
            raise GrantError("文档不存在", status_code=404)
        _assert_doc_visible(actor, doc)

        target_object_id = str(object_id) if object_id else str(doc_uuid)
        obj = await sess.get(DocumentObject, target_object_id)
        if obj is None:
            raise GrantError(
                "对象不存在（need-to-know 需先物化 document_objects）", status_code=404
            )
        if str(obj.document_id) != str(doc_uuid):
            raise GrantError("对象与文档不匹配", status_code=400)
        if str(obj.object_type) not in GRANTABLE_OBJECT_TYPES:
            raise GrantError(
                "need-to-know 只能在 doc / image 对象上授予"
                "（派生对象不得通过 acl_allow 获得父之外的可见性）",
                status_code=400,
            )

        grant = AclGrant(
            id=uuid.uuid4(),
            object_id=target_object_id,
            document_id=doc_uuid,
            subject=normalized,
            effect=effect,
            status=GRANT_STATUS_PENDING,
            granted_by=getattr(actor, "id", None),
            reason=(reason or "").strip() or None,
            expires_at=exp,
        )
        sess.add(grant)
        await sess.flush()
        return grant

    grant = await _with_session(session, _do)
    await record_audit(
        "security.grant.request",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="document_object",
        resource_id=str(grant.object_id)[:64],
        detail=(
            f"document_id={doc_uuid}; subject={normalized}; effect={effect}; "
            f"expires_at={exp.isoformat()}"
        ),
    )
    logger.info(
        "need-to-know request: %s -> %s on %s (expires %s)",
        actor_id, normalized, grant.object_id, exp.isoformat(),
    )
    return grant


async def review_grant(
    reviewer: Any,
    grant_id: Any,
    *,
    approve: bool,
    comment: str | None = None,
    session: Any = None,
) -> AclGrant:
    """
    审批一条授予申请（批准即物化；拒绝不改可见性）.

    * 只有 ``pending`` 可审批（重复审批 → 409）；
    * **禁止自审自发**：审批人不得审批"授予给自己"的申请（→ 403）。
    """
    gid = _uuid(grant_id)
    reviewer_id = getattr(reviewer, "id", None)

    async def _do(sess: Any) -> AclGrant:
        grant = await sess.get(AclGrant, gid)
        if grant is None:
            raise GrantError("授予申请不存在", status_code=404)
        if grant.status != GRANT_STATUS_PENDING:
            raise GrantError(
                f"该授予已处理（当前状态：{grant.status}），无需重复操作",
                status_code=409,
            )
        if is_self_grant(grant.subject, reviewer_id):
            raise GrantError(
                "禁止自我授予：不能审批授予给自己的例外", status_code=403
            )
        if comment:
            grant.reason = (grant.reason or "")
            grant.reason = (grant.reason + f" | review={comment.strip()}").strip(" |")

        grant.status = (
            GRANT_STATUS_APPROVED if approve else GRANT_STATUS_REJECTED
        )
        grant.reviewer_id = reviewer_id
        grant.reviewed_at = _now()
        await sess.flush()

        if approve:
            await _apply(sess, grant.document_id)     # 批准即物化
        return grant

    grant = await _with_session(session, _do)
    if approve:
        await _push_payload(grant.document_id)
    await record_audit(
        "security.grant.review",
        user_id=reviewer_id,
        username=getattr(reviewer, "username", None),
        resource_type="document_object",
        resource_id=str(grant.object_id)[:64],
        detail=(
            f"{'approved' if approve else 'rejected'}; document_id={grant.document_id}; "
            f"subject={grant.subject}; comment={(comment or '-')}"
        ),
    )
    logger.info(
        "need-to-know %s: grant=%s subject=%s",
        "approved" if approve else "rejected", grant.id, grant.subject,
    )
    return grant


async def revoke_grant(
    actor: Any, grant_id: Any, *, session: Any = None
) -> AclGrant:
    """撤销一条 待审/已批准 的授予，并立即重算物化副本（到期回收的对偶操作）。"""
    gid = _uuid(grant_id)

    async def _do(sess: Any) -> AclGrant:
        grant = await sess.get(AclGrant, gid)
        if grant is None:
            raise GrantError("授予申请不存在", status_code=404)
        if grant.status not in (GRANT_STATUS_APPROVED, GRANT_STATUS_PENDING):
            raise GrantError(
                f"只能撤销 待审/已批准 的授予（当前状态：{grant.status}）",
                status_code=409,
            )
        grant.status = GRANT_STATUS_REVOKED
        grant.reviewer_id = getattr(actor, "id", None)
        grant.reviewed_at = _now()
        await sess.flush()
        await _apply(sess, grant.document_id)
        return grant

    grant = await _with_session(session, _do)
    await _push_payload(grant.document_id)
    await record_audit(
        "security.grant.review",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="document_object",
        resource_id=str(grant.object_id)[:64],
        detail=f"revoked; document_id={grant.document_id}; subject={grant.subject}",
    )
    return grant


async def list_grants(
    actor: Any,
    *,
    document_id: Any = None,
    status: str | None = None,
    session: Any = None,
) -> list[AclGrant]:
    """列出授予（可按文档 / 状态过滤）。跨公司不泄漏（非管理员锁在本租户）。"""

    async def _do(sess: Any) -> list[AclGrant]:
        stmt = select(AclGrant)
        if document_id:
            stmt = stmt.where(AclGrant.document_id == _uuid(document_id))
        if status:
            stmt = stmt.where(AclGrant.status == status)
        stmt = stmt.order_by(AclGrant.created_at.desc())
        rows = list((await sess.scalars(stmt)).all())
        if is_platform_admin(actor):
            return rows
        tenant = effective_tenant_id(actor)
        doc_ids = {r.document_id for r in rows}
        if not doc_ids:
            return rows
        docs = list(
            (await sess.scalars(
                select(Document).where(Document.id.in_(doc_ids))
            )).all()
        )
        visible = {d.id for d in docs if str(d.tenant_id) == tenant}
        return [r for r in rows if r.document_id in visible]

    return await _with_session(session, _do)


# ═══════════════════════════════════════════════════════════════════════════════
# 密级 / 可见性管理面（决策 7 / 8；变更触发 T4 级联）
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_security_level(value: Any) -> int | None:
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise GrantError("密级必须是整数", status_code=400) from exc
    if not (SECURITY_LEVEL_MIN <= ivalue <= SECURITY_LEVEL_MAX):
        raise GrantError(
            f"密级必须落在 {SECURITY_LEVEL_MIN}..{SECURITY_LEVEL_MAX}", status_code=400
        )
    return ivalue


async def _propagate_project_dimension(
    sess: Any, doc_uuid: uuid.UUID, visibility_mode: str, project_ids: Sequence[str]
) -> int:
    """
    把文档的**项目维度**（``visibility_mode`` + ``project_ids``）同步到全部对象行.

    为什么需要这一步：项目维度是**文档级**属性（入库物化时所有对象行都写同一份
    ``visibility_mode`` / ``project_ids``）。而 T4 的 ``sync_doc_row`` 只把
    ``visibility_mode`` 向下传染、**不更新** ``project_ids``，因此项目集合变更后
    派生对象会残留旧集合 —— 这里以文档真值为准补齐，保证"文档级变更 = 全对象一致"。
    """
    rows = list(
        (await sess.scalars(
            select(DocumentObject).where(DocumentObject.document_id == doc_uuid)
        )).all()
    )
    proj = sorted({str(p) for p in project_ids if str(p).strip()})
    for row in rows:
        row.visibility_mode = visibility_mode
        row.project_ids = proj
    await sess.flush()
    return len(rows)


async def set_document_security(
    actor: Any,
    document_id: Any,
    *,
    security_level: Any = None,
    visibility_mode: str | None = None,
    project_ids: Sequence[str] | None = None,
    session: Any = None,
) -> dict:
    """
    设置文档的密级 / 可见性模式 / 项目集合（**只改显式传入的字段**），并触发级联.

    流程（缺一不可）：

        1. ``security_cascade.sync_doc_row`` —— 用 T4 的**取严**算法把变更同步到
           派生对象（``effective_security_level = max(文档, 自身)``，deny/excluded 向下传染）；
        2. :func:`_propagate_project_dimension` —— 把项目维度同步到全部对象行；
        3. :func:`_apply` —— 重新物化 need-to-know 副本（``sync_doc_row`` 会把派生行
           的 ``acl_allow`` 清空，这里以 ``acl_grants`` 权威源为准恢复）。

    Returns ``{"document_id", "security_level", "visibility_mode", "project_ids",
    "object_rows"}``。
    """
    level = _validate_security_level(security_level)
    if visibility_mode is not None and visibility_mode not in _VALID_VISIBILITY:
        raise GrantError("visibility_mode 只能是 tier / project", status_code=400)
    proj_list = None
    if project_ids is not None:
        proj_list = [str(p).strip() for p in project_ids if str(p).strip()]

    doc_uuid = _uuid(document_id)

    async def _do(sess: Any) -> dict:
        doc = await sess.get(Document, doc_uuid)
        if doc is None:
            raise GrantError("文档不存在", status_code=404)
        _assert_doc_visible(actor, doc)

        # ① T4 的取严级联（security_level / visibility_mode / project_ids）
        from app.services.security_cascade import sync_doc_row

        await sync_doc_row(
            doc_uuid,
            session=sess,
            security_level=level,
            visibility_mode=visibility_mode,
            project_ids=proj_list,
        )

        # ② 项目维度同步到全部对象行（以文档真值为准）
        await sess.refresh(doc)
        new_vis = str(getattr(doc, "visibility_mode", None) or VISIBILITY_MODE_TIER)
        new_proj = [str(p) for p in (getattr(doc, "project_ids", None) or [])]
        object_rows = await _propagate_project_dimension(
            sess, doc_uuid, new_vis, new_proj
        )

        # ③ 重新物化 need-to-know 副本（权威源 acl_grants）
        await _apply(sess, doc_uuid)

        return {
            "document_id": str(doc_uuid),
            "security_level": int(getattr(doc, "security_level", 0) or 0),
            "visibility_mode": new_vis,
            "project_ids": new_proj,
            "object_rows": object_rows,
        }

    result = await _with_session(session, _do)
    await _push_payload(doc_uuid)
    await record_audit(
        "security.level.change",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="document",
        resource_id=str(doc_uuid),
        detail=(
            f"security_level={result['security_level']}; "
            f"visibility_mode={result['visibility_mode']}; "
            f"project_ids={result['project_ids']}"
        ),
    )
    logger.info(
        "security level change: doc=%s level=%s vis=%s projects=%s by=%s",
        doc_uuid, result["security_level"], result["visibility_mode"],
        result["project_ids"], getattr(actor, "username", "-"),
    )
    return result


async def escalate_object(
    actor: Any,
    document_id: Any,
    object_id: str,
    *,
    security_level: Any = None,
    excluded: bool | None = None,
    session: Any = None,
) -> dict:
    """
    对**单个对象**提级 / 剔除（图片对象触发 OCR 派生取严级联）.

    * ``image`` 对象 → ``security_cascade.cascade_image_derived``：其 OCR 派生的
      ``text`` / ``table`` / ``code`` 块同步取严（堵住"图看不了、字还能搜"的泄密面）；
    * 其它对象 → 就地更新该行的密级 / ``excluded``（派生块"只收紧不放宽"，
      ``effective_security_level`` 取 ``max``）。
    """
    level = _validate_security_level(security_level)
    if level is None and excluded is None:
        raise GrantError("至少要指定 security_level 或 excluded 之一", status_code=400)
    doc_uuid = _uuid(document_id)
    oid = str(object_id)

    async def _do(sess: Any) -> dict:
        obj = await sess.get(DocumentObject, oid)
        if obj is None:
            raise GrantError("对象不存在", status_code=404)
        if str(obj.document_id) != str(doc_uuid):
            raise GrantError("对象与文档不匹配", status_code=400)
        doc = await sess.get(Document, doc_uuid)
        if doc is None:
            raise GrantError("文档不存在", status_code=404)
        _assert_doc_visible(actor, doc)

        if str(obj.object_type) == OBJECT_TYPE_IMAGE:
            from app.services.security_cascade import cascade_image_derived

            res = await cascade_image_derived(
                doc_uuid, oid, session=sess,
                src_level=level, src_excluded=excluded,
            )
            await _apply(sess, doc_uuid)
            return {
                "document_id": str(doc_uuid),
                "object_id": oid,
                "object_type": OBJECT_TYPE_IMAGE,
                "cascaded": res,
            }

        if excluded is not None:
            obj.excluded = bool(excluded)
        if level is not None:
            obj.security_level = level
            obj.effective_security_level = max(
                level, int(obj.effective_security_level or 0)
            )
        obj.acl_sync_state = ACL_SYNC_PENDING
        await sess.flush()
        return {
            "document_id": str(doc_uuid),
            "object_id": oid,
            "object_type": str(obj.object_type),
            "cascaded": {},
        }

    result = await _with_session(session, _do)
    await _push_payload(doc_uuid)
    await record_audit(
        "image.escalate" if excluded is not True else "image.exclude",
        user_id=getattr(actor, "id", None),
        username=getattr(actor, "username", None),
        resource_type="document_object",
        resource_id=oid[:64],
        detail=(
            f"document_id={doc_uuid}; security_level={level}; excluded={excluded}"
        ),
    )
    return result


async def counted_null_effective_levels(*, session: Any = None) -> int:
    """
    ``document_objects.effective_security_level IS NULL`` 的行数.

    这是 ``ACL_SECURITY_PREFILTER_STRICT=true`` 的**前置条件**：回填执行完、
    该计数为 0 时，开启严格前置过滤不会把存量对象一夜之间排除掉。
    """
    from sqlalchemy import func

    async def _do(sess: Any) -> int:
        rows = (
            await sess.execute(
                select(func.count())
                .select_from(DocumentObject)
                .where(DocumentObject.effective_security_level.is_(None))
            )
        ).scalar()
        return int(rows or 0)

    try:
        return await _with_session(session, _do)
    except Exception:      # noqa: BLE001 — 计数失败不阻断设置页
        logger.exception("counted_null_effective_levels: query failed")
        return -1


__all__ = [
    "GRANTABLE_OBJECT_TYPES",
    "SUBJECT_KINDS",
    "GrantError",
    "allowed_subjects",
    "compute_object_acl",
    "counted_null_effective_levels",
    "escalate_object",
    "is_self_grant",
    "list_grants",
    "materialize_document_acl",
    "normalize_subject",
    "parse_subject",
    "request_grant",
    "review_grant",
    "revoke_grant",
    "set_document_security",
]
