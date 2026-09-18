"""
三层知识库的"发布 / 层级变更"服务.

    企业知识库
      ├─ 个人知识库  张三自己可访问          access_level=private（默认）
      ├─ 部门知识库  技术部所有人按权限访问   access_level=department + department_id
      └─ 公司知识库  全公司按权限访问         access_level=tenant

本模块是**层级变更的唯一实现点**：无论是"有权限的人直接发布"，还是"普通员工
申请共享被审核通过后自动发布"，都走同一个 ``set_document_access_level``，
因此 PG 行、向量载荷、审计日志三处永远同步。

能力判定也放在这里（而不是散落在路由里），前端拿到的按钮状态与后端真正执行的
校验来自同一份逻辑 —— 不会出现"按钮能点但接口 403"。
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select

from app.db.models import Document
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.services.tenancy import (
    ACCESS_DEPARTMENT,
    # `required_permission_message(ACCESS_PRIVATE)` 用到它。此前没导入，而该分支
    # 恰好被 document_management 的 `if required and ...` 短路挡住（private 的
    # publish_requirement 是 None），于是**一直没暴露**：谁写个"为什么不能收回
    # 个人库"的接口就会 NameError → 500。pyflakes 扫出来的，见 _audit_916。
    ACCESS_PRIVATE,
    ACCESS_TENANT,
    access_label,
    is_downgrade,
    is_upward_transition,
    normalize_access_level,
    normalize_tenant_id,
    publish_requirement,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TierError(Exception):
    """层级变更失败（携带用户可读中文信息与 HTTP 状态码）。"""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def resolve_department_for_level(
    level: str,
    *,
    user_department_id: str | None,
    doc_department_id: str | None,
) -> str | None:
    """
    目标层级的归属部门。

    部门库必须带部门：优先用文档原有部门，其次用操作者所在部门。
    公司库/个人库一律清空部门归属（避免旧部门残留导致日后误判）。
    """
    if level == ACCESS_DEPARTMENT:
        return (doc_department_id or "").strip() or user_department_id
    return None


def merge_department_rows(rows) -> list[dict]:
    """
    把 ``(department_id, department_name, member_count)`` 原始行合并成部门清单.

    纯函数（不碰数据库），单独抽出来是为了能被确定性单测覆盖 —— 这里的合并
    规则有两个容易写错的地方：

        1. 同一个 department_id 可能挂着不同的 ``department_name``（成员换过
           部门名、或历史数据里大小写/空格不一致）：必须按 **ID** 聚合，不能
           按名字 —— 否则同一部门在清单里出现两次，管理者选中"研发部"时系统
           收到的可能是另一个 ID，文档发过去谁也看不到。
        2. 名称缺失或等于 ID 时，要用后出现的非空名称回填，否则下拉里显示的是
           ``d3f2a1...`` 这种哈希，管理者根本认不出是哪个部门。

    空 ID / 空行直接丢弃（``department_id`` 为空的成员不属于任何部门）。
    """
    options: dict[str, dict] = {}
    for dept_id, dept_name, member_count in rows:
        key = str(dept_id).strip() if dept_id is not None else ""
        if not key:
            continue
        name = str(dept_name).strip() if dept_name else ""
        entry = options.setdefault(
            key,
            {"department_id": key, "department_name": name or key, "member_count": 0},
        )
        # 先出现的名称可能是 ID 兜底值（或空），后续行有真名就回填
        if name and entry["department_name"] == key:
            entry["department_name"] = name
        entry["member_count"] += int(member_count or 0)

    return sorted(options.values(), key=lambda item: item["department_name"])


async def list_department_options(tenant_id: str | None) -> list[dict]:
    """
    本公司"已有部门"清单（「转为部门文档」的候选目标）.

    数据源是**成员归属**（``users.department_id`` / ``department_name``）去重，
    而不是一张部门主数据表 —— 项目里的组织架构就是"成员挂在哪些部门"这件事
    本身：部门名称在入伙审核时由申请人填写、审核人当场修正（staff_service），
    没有第二份事实可以对齐。凭空造一张部门表反而会出现"表里有、没人属于它"
    的空部门，让管理者选到一个谁都不在里面、文档发过去等于谁都看不到的部门。

    因此这里按 ``tenant_id`` 圈定公司范围（公司隔离），只统计**在职**成员，
    避免把离职同事的历史部门继续当成可选目标 —— 那正是"选了个没人的部门"
    最现实的来源。

    Args:
        tenant_id: 目标公司；None/非法值按 ``DEFAULT_TENANT_ID`` 处理。

    Returns:
        按部门名排序的 ``[{"department_id", "department_name", "member_count"}]``
        （合并规则见 :func:`merge_department_rows`）。``member_count`` 给界面
        显示"研发部 · 12 人"，让管理者确认自己选中的确实是想要的那个部门。
    """
    tid = normalize_tenant_id(tenant_id)

    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(
                    User.department_id,
                    User.department_name,
                    func.count().label("member_count"),
                )
                .where(
                    User.tenant_id == tid,
                    User.is_active.is_(True),
                    User.department_id.is_not(None),
                    User.department_id != "",
                )
                .group_by(User.department_id, User.department_name)
            )
        ).all()

    return merge_department_rows(rows)


async def transfer_document_to_department(
    document_id: uuid.UUID,
    *,
    target_department_id: str,
    actor_id: uuid.UUID | None = None,
    actor_username: str | None = None,
    note: str = "",
) -> Document:
    """
    把文档**改归到指定部门**（公司知识库 → 该部门知识库；部门库 → 另一部门）.

    与"发布到部门知识库"的区别（这是它必须单独存在的原因）：

        发布到部门库  目标部门 = 操作者自己的部门。部门负责人用它把文档给
                      **本部门**同事看，不存在选择。
        转为部门文档  目标部门 = **显式指定的**某个部门。公司 HR / 知识库
                      管理员把一份已经全公司可见的文档下沉到具体某个部门，
                      例如"这份薪酬制度只该给人力资源部看"。

    它是一次**可见性收缩**（公司库 → 部门库之后，其他部门同事检索不到），
    因此与普通发布走不同闸门：调用方必须先确认操作者是公司级管理者
    （``document.read.all``），本函数只负责"目标部门是否真的存在、是否需要
    变更"这类事实校验，权限判定留在 API 层（与能力字段同源）。

    目标部门必须在**该公司已有部门清单**里：否则一次构造请求就能把文档塞进
    一个不存在的 department_id —— 那不是"发给某个部门"，而是"除平台管理员
    外谁都检索不到"，等于一次没有审计线索的软删除。

    Raises:
        TierError(404): 文档不存在。
        TierError(400): 目标部门不在本公司清单内 / 文档已在该部门。
    """
    target = (target_department_id or "").strip()
    if not target:
        raise TierError("请选择要转入的部门", status_code=400)

    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
    if doc is None:
        raise TierError("文档不存在或已被删除", status_code=404)

    options = await list_department_options(getattr(doc, "tenant_id", None))
    matched = next((o for o in options if o["department_id"] == target), None)
    if matched is None:
        raise TierError(
            "目标部门不在本公司已有部门中（可能已被撤销或尚未有成员归属），"
            "请重新选择",
            status_code=400,
        )

    current_dept = (getattr(doc, "department_id", None) or "").strip()
    if (
        normalize_access_level(getattr(doc, "access_level", None)) == ACCESS_DEPARTMENT
        and current_dept == target
    ):
        raise TierError(
            f"这份文档已经在「{matched['department_name']}」，无需重复转换",
            status_code=400,
        )

    detail_extra = (
        f"target={matched['department_name']}"
        f"({matched['department_id']}, {matched['member_count']}人)"
    )
    if note.strip():
        detail_extra = f"{detail_extra}; note={note.strip()[:200]}"

    return await set_document_access_level(
        document_id,
        level=ACCESS_DEPARTMENT,
        department_id=target,
        actor_id=actor_id,
        actor_username=actor_username,
        action="document.transfer.department",
        detail_extra=detail_extra,
    )


async def set_document_access_level(
    document_id: uuid.UUID,
    *,
    level: str,
    department_id: str | None,
    actor_id: uuid.UUID | None = None,
    actor_username: str | None = None,
    action: str = "document.publish",
    detail_extra: str = "",
) -> Document:
    """
    更新文档层级：PG 行 + 向量 ACL 载荷 + 审计，一次到位。

    Raises:
        TierError(404): 文档不存在。
    """
    normalized = normalize_access_level(level)

    async with get_db_session() as session:
        doc = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
        if doc is None:
            raise TierError("文档不存在或已被删除", status_code=404)

        previous = normalize_access_level(doc.access_level)
        doc.access_level = normalized
        doc.department_id = (
            (department_id or "").strip() or None
            if normalized == ACCESS_DEPARTMENT
            else None
        )
        await session.flush()
        await session.refresh(doc)

    # 向量载荷与 PG 同步（失败只告警，不阻断——PG 是判定权威）
    from app.services.vector_service import update_document_access_payload

    applied = await update_document_access_payload(
        str(document_id),
        access_level=normalized,
        department_id=doc.department_id,
    )
    if not applied:
        # 返回值 0 说明**向量点还不存在**（上传后立刻发布 —— 正常操作路径）或
        # 同步调用失败。此时 PG 已经改好、列表接口立刻可见，但 Qdrant 的 ACL
        # 副本还是上传时的 private，而检索的 ACL 前置过滤读的正是那一份 →
        # 除 owner 本人谁都检索不到。这不是要在这里补救（点还不存在，无从补），
        # 而是必须让入库收尾来追平（resync_document_acl_payload）。
        logger.warning(
            "Document %s set to %s but no vectors matched (ingestion still running?) "
            "— vector-side ACL will be aligned at ingestion completion",
            document_id, normalized,
        )

    detail = (
        f"{access_label(previous)}→{access_label(normalized)}"
        f"; department={doc.department_id or '-'}"
    )
    if detail_extra:
        detail = f"{detail}; {detail_extra}"

    await record_audit(
        action,
        user_id=actor_id,
        username=actor_username,
        resource_type="document",
        resource_id=str(document_id),
        detail=detail,
    )

    logger.info(
        "Document %s access level: %s → %s (dept=%s) by %s",
        document_id, previous, normalized, doc.department_id, actor_username,
    )
    return doc


async def resync_document_acl_payload(
    document_id: uuid.UUID | str,
    *,
    reason: str = "ingestion_completion",
    expected: tuple[str | None, str | None] | None = None,
) -> int:
    """
    按 PostgreSQL 真值重写该文档全部向量点的 ACL 载荷（幂等）.

    为什么必须有这个函数 —— **层级变更可能发生在向量点写入之前**：

        上传是异步的（``POST /upload`` 受理即返回 202，管线丢后台），上传那一刻
        ``access_level`` 就固定成 private；而"传完立刻点发布"正是前端与验收脚本
        的正常路径。发布走 :func:`set_document_access_level` → Qdrant
        ``set_payload``，按 ``document_id`` 过滤更新：**此时向量点往往还没写入**，
        于是更新匹配 0 个点并静默成功（旧实现不校验匹配数，照样记一条
        "Updated vector ACL payload" 的成功日志）。随后入库又用上传时刻的快照
        private 把点写进去，两份事实永久分叉：

            PostgreSQL  access_level=department    ← 列表接口、详情、删除都看这份
            Qdrant      access_level=private       ← 检索的 ACL 前置过滤看这份

        表现极具误导性：**列表里看得见、点得开、权限判断全对，但提问时谁都
        检索不到**（owner 除外，因为 private 对 owner 本来就放行）。排查时容易
        一路去怀疑检索、阈值、证据门控，而真正的分叉在发布那一刻。

    在**入库收尾**（所有 upsert 完成之后、标记 COMPLETED 之前）调用本函数，
    是唯一"必然正确"的时点：此前发生的任何发布/收回都会被追平，此后发生的
    发布必然能匹配到点。这是一个**读时对齐 + 写时兜底**的收口，不依赖时序运气。

    Args:
        document_id: 文档 UUID（str 也可以）。
        reason:      仅用于日志，标明本次对齐的触发场景。
        expected:    入库时使用的 ``(access_level, department_id)`` 快照。
            两者与 PG 真值一致时**直接跳过**（99% 的文档从未发布，省掉一次
            无意义的 count + set_payload）。传 ``None`` 表示强制对齐 ——
            存量数据修复脚本用它。

    Returns:
        实际被改写的向量点数（跳过时为 0）。
    """
    doc_uuid = (
        document_id if isinstance(document_id, uuid.UUID) else uuid.UUID(str(document_id))
    )

    async with get_db_session() as session:
        row = (
            await session.execute(
                select(Document.access_level, Document.department_id).where(
                    Document.id == doc_uuid
                )
            )
        ).first()

    if row is None:
        logger.warning("resync ACL payload (%s): document %s not found", reason, doc_uuid)
        return 0

    level = normalize_access_level(row[0])
    # 与 set_document_access_level 同一口径：只有部门库保留部门归属，
    # 个人库/公司库一律清空，避免旧部门值残留导致日后误判。
    dept = row[1] if level == ACCESS_DEPARTMENT else None

    if expected is not None:
        exp_level = normalize_access_level(expected[0]) if expected[0] else None
        exp_dept = expected[1] if exp_level == ACCESS_DEPARTMENT else None
        if (level, dept) == (exp_level, exp_dept):
            logger.debug(
                "resync ACL payload (%s): document=%s already aligned (level=%s) — skipped",
                reason, doc_uuid, level,
            )
            return 0

    from app.services.vector_service import update_document_access_payload

    applied = await update_document_access_payload(
        str(doc_uuid), access_level=level, department_id=dept
    )
    if not applied:
        logger.warning(
            "resync ACL payload (%s): document=%s level=%s but no vectors matched — "
            "PostgreSQL and the vector store still disagree (retrieval will miss it)",
            reason, doc_uuid, level,
        )
    else:
        logger.info(
            "resync ACL payload (%s): document=%s level=%s dept=%s points=%d",
            reason, doc_uuid, level, dept, applied,
        )
    return applied


def publish_capability(user, doc) -> dict:
    """
    描述 *user* 对 *doc* 的层级操作能力（前端按钮 + 后端校验共用）.

    返回：
        {
          "is_owner": bool,
          "current_level": "private|department|tenant",
          "current_label": "个人|部门|公司",
          "can_publish_department": bool,    # 是否可**直接**发布到部门库
          "can_publish_company": bool,       # 是否可**直接**发布到公司库
          "can_request_department": bool,    # 是否可**申请**发布到部门库
          "can_request_company": bool,       # 是否可**申请**发布到公司库
          "needs_share_request": bool,       # 是否有任何一层需要走「申请共享」
          "can_transfer_department": bool,   # 是否可把文档**改归到指定部门**
          "transfer_denied_reason": str,     # 不能改归时的中文原因
          "can_delete": bool,
          "delete_denied_reason": str,
          "publish_denied_reason": str,
        }

    ``can_publish_*`` 与 ``can_request_*`` 对**同一个层级**互斥：能直接发就不必
    申请，不能直接发才允许申请。两者对**不同层级**可以同时为真 —— 这正是
    "部门负责人"的形态（可直接发部门库、只能申请公司库）。
    """
    from app.services.permissions import has_permission
    from app.services.tenancy import (
        can_request_delete,
        delete_permission_for,
        normalize_access_level,
    )

    level = normalize_access_level(getattr(doc, "access_level", None))
    is_owner = bool(
        getattr(doc, "owner_id", None) is not None and user is not None
        and doc.owner_id == user.id
    )

    # 角色维度的发布权（与具体文档无关，纯看角色）
    can_dept_role = has_permission(user, "document.publish.department")
    can_company_role = has_permission(user, "document.publish.company")

    # ── 发布能力必须再按**文档当前层级**收窄 ───────────────────────────────
    #
    # 这里曾经是 `can_dept = has_permission(...)` —— 纯角色判定，文档在哪一层
    # 完全不参与。后果是一条**不需要审批的降级通道**：部门负责人面对一份已经
    # 发布到公司库的文档，can_publish_department 仍为 True，界面照常渲染
    # "发布到部门知识库"按钮，点下去 PATCH /documents/{id}/visibility 就把它真
    # 降成部门库，其他部门同事静默失去访问权 —— 审计日志里只是一条正常的
    # "层级变更"。申请链路上一轮已用 is_upward_transition 堵住向下，但"直接
    # 发布"这条路更短更隐蔽（无人审批），两处必须同一口径。
    #
    # 收回个人库不在此列：那是归属人的正当操作，由独立的「收回」按钮承载。
    can_dept = can_dept_role and not is_downgrade(level, ACCESS_DEPARTMENT)
    can_company = can_company_role and not is_downgrade(level, ACCESS_TENANT)

    # ── 「转为部门文档」：公司级管理者把已共享的文档**改归到指定部门** ─────────
    #
    # 与上面的 can_dept 是两件事，不要合并：`can_dept` 表达的是"能发布到
    # **我自己的**部门"（部门负责人对本部门），目标部门由服务端从操作者的
    # 归属推导，没有选择余地；本字段表达的是"能指定**任意一个**本公司部门"，
    # 是公司 HR / 知识库管理员整理组织资产时用的（薪酬制度只给人力资源部看）。
    #
    # 判定用 `document.read.all` 而不是 `document.publish.department`：前者
    # 恰好是"公司级管理者"的定义（企业管理员 / 知识库管理员 / 平台管理员），
    # 与 tenancy.TENANT_WIDE_READER_ROLES 同源；部门负责人虽有发布权，但他
    # 只能发本部门，让他把公司库文档改派到别的部门属于越权整理。
    #
    # 层级限定为 tenant / department：只有**已经共享出去的组织资产**才谈得上
    # "改归哪个部门"。个人库文档不在此列 —— 那是"要不要共享"，用现成的
    # 「发布到部门知识库」即可，不必在这里再开一条上行通道。
    can_manage_all = has_permission(user, "document.read.all")
    can_transfer = bool(can_manage_all and level in (ACCESS_TENANT, ACCESS_DEPARTMENT))

    # ── 申请能力必须**按目标层级**判定，不能合成一个总布尔 ────────────────────
    #
    # 这里曾经是 `needs_request = is_owner and not can_dept and not can_company`，
    # 前端据此把界面拆成互斥的两支（"有任一发布权 → 只渲染直接发布"）。后果：
    # **部门负责人拿不到「申请共享」入口** —— 他能直接发部门库，于是整个申请
    # 分支被隐藏，而"把文档提到公司库"这件事只能靠申请，他在界面上无路可走
    # （接口本身是放行的，所以是纯前端死角，日志里没有任何错误）。
    #
    # 规则改成逐层判断，与权限矩阵逐格对应：
    #   能直接发这一层  → 不需要申请（提交也会被 create_share_request 以 409 挡回）
    #   不能直接发这一层 → 可以申请（部门库 → 本部门负责人审；公司库 → 公司级审）
    #   目标不高于当前层 → 不能申请。申请只能**向上**：允许平级/向下的话，
    #     一份已发布到公司库的文档可以被申请"共享到部门库"，批准即降级，
    #     对正在引用它的同事是静默的可见性收缩。
    #
    # 注意 `not can_dept_role` 用的是**角色**而不是上面的 can_dept：后者已被
    # 层级收窄，"能不能直接发"是角色问题，"这一层要不要申请"才是层级问题。
    can_request_dept = bool(
        is_owner
        and not can_dept_role
        and is_upward_transition(level, ACCESS_DEPARTMENT)
    )
    can_request_company = bool(
        is_owner
        and not can_company_role
        and is_upward_transition(level, ACCESS_TENANT)
    )
    needs_request = can_request_dept or can_request_company

    allowed, reason = delete_permission_for(doc, user) if user is not None else (False, "未登录")
    # 自己没有删除权但看得见 → 走「申请删除」，由上级同意或拒绝
    request_delete = can_request_delete(doc, user) if user is not None else False

    if not (can_dept_role or can_company_role):
        publish_reason = (
            "你的角色暂无发布权限：可点击「申请共享」提交给部门负责人或知识库管理员审核"
        )
    elif not (can_dept or can_company):
        # 角色有发布权，但这篇文档已经在更高的层级上：按钮被层级收窄隐藏，
        # 这里补一句原因，避免用户以为按钮"丢了"。收回个人库不在此列，
        # 它由独立的「收回」入口承载，不受影响。
        publish_reason = (
            f"这份文档已在{access_label(level)}，不能再发布到更低的层级；"
            "如需仅自己可见，请使用「收回至个人知识库」"
        )
    else:
        publish_reason = ""

    if not can_manage_all:
        # 普通员工 / 部门负责人看不到这个入口，理由留空（不制造"按钮丢了"的疑问）
        transfer_reason = ""
    elif level == ACCESS_PRIVATE:
        transfer_reason = (
            "这份文档还在个人知识库，请先用「发布到部门知识库」共享出去，"
            "再调整它归属的部门"
        )
    else:
        transfer_reason = ""

    return {
        "is_owner": is_owner,
        "current_level": level,
        "current_label": access_label(level),
        "can_publish_department": can_dept,
        "can_publish_company": can_company,
        "can_request_department": can_request_dept,
        "can_request_company": can_request_company,
        "needs_share_request": needs_request,
        # 是否可把文档改归到**指定**部门（公司级管理者专用，见上方注释）
        "can_transfer_department": can_transfer,
        "transfer_denied_reason": transfer_reason,
        "can_delete": allowed,
        "delete_denied_reason": "" if allowed else reason,
        "can_request_delete": request_delete,
        "publish_denied_reason": publish_reason,
    }


def required_permission_message(level: str) -> str:
    """发布到 *level* 所需权限缺失时的中文提示。"""
    permission = publish_requirement(level)
    if level == ACCESS_TENANT:
        return "发布到公司知识库需要「知识库管理员」及以上权限，请提交「申请共享」"
    if level == ACCESS_DEPARTMENT:
        return "发布到部门知识库需要「部门负责人」及以上权限，请提交「申请共享」"
    if level == ACCESS_PRIVATE:
        return "仅文档所有者可将文档收回个人知识库"
    return f"缺少权限：{permission}"


__all__ = [
    "TierError",
    "set_document_access_level",
    "resync_document_acl_payload",
    "publish_capability",
    "required_permission_message",
    "resolve_department_for_level",
    "list_department_options",
    "merge_department_rows",
    "transfer_document_to_department",
]
