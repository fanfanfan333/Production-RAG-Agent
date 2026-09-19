"""
公司注册表服务（公司标识 / 展示名 / 归一化 / 自建集合的唯一实现点）.

核心约定（全项目唯一）
──────────────────────
* **公司标识 vs 展示名**：业务与权限一律用 ``tenant_id``（稳定标识）；展示名
  只从 ``companies.display_name`` 取。**禁止**用名称做绑定/比较/权限判定。
* **名称归一化规则**：``normalize_company_name_key = NFKC → 去全部空白 → casefold``。
  注册 / 建公司 / 改名 / 按名解析**必须**都调它；``companies.name_key`` 唯一约束
  是最后一道闸。
* **``owns_tenant_ids`` 的唯一来源**：``created_by == 当前 admin 的 user id`` 的
  公司集合（按 id 锁定，不是「任意 admin 的公司」）。多管理员场景下，A 建的测试
  公司不应自动对 B 可见/可检索。

错误统一用 :class:`CompanyError`（携带 HTTP 语义的 ``status_code``），由 API 层
映射成响应；服务层不依赖 FastAPI。
"""

from __future__ import annotations

import unicodedata
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.db.company_models import Company
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services.audit_service import record_audit
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 统一错误文案（全项目一致）
MSG_DUPLICATE = "公司已存在"
MSG_NOT_REGISTERED = "请先由管理员注册该公司"
MSG_NEED_ADMIN = "该操作需要平台管理员权限"
MSG_NOT_OWNS = "无权操作该公司"
MSG_EMPTY_NAME = "请填写公司名称"
MSG_NAME_TOO_LONG = "公司名称过长（最多 128 个字符）"

# 回填脚本 / 一次性迁移用：把历史租户裁定为注册表行。
# 表结构：tenant_id → (display_name, created_by_username 或 None, is_test)。
# 说明：A公司 / B公司 的 created_by = NULL（对 admin 恒不可见，满足 P0-6）；
#       bjld8 / 1z5pp 的 created_by = admin（admin 的自建测试公司）。
DEFAULT_BACKFILL_ASSIGNMENTS: tuple[dict, ...] = (
    {"tenant_id": "c8111de986583", "display_name": "测试公司1", "created_by": "admin", "is_test": True},
    {"tenant_id": "cfb08c53677c4", "display_name": "测试公司2", "created_by": "admin", "is_test": True},
    {"tenant_id": "c309a7cb9f496", "display_name": "A公司", "created_by": None, "is_test": False},
    {"tenant_id": "cf33b1db5679d", "display_name": "B公司", "created_by": None, "is_test": False},
)


class CompanyError(Exception):
    """公司注册表操作失败（携带 HTTP 状态码语义）。"""

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


# ── 纯函数：归一化 / 生成标识 ─────────────────────────────────────────────────


def normalize_company_name_key(name: str | None) -> str:
    """
    公司名的**唯一**归一化实现：NFKC → 去全部空白（含全角空格）→ casefold.

    为什么「去除**全部**空白」而不是「只 strip 首尾」：验收明确要求「重名
    （含空格/大小写变体）拒绝」，取「宁严勿松」的一侧（对公司名而言空白不承载
    语义）。NFKC 会把全角空格（U+3000）等折成普通空格，再用 ``str.isspace``
    一并抹掉。
    """
    raw = unicodedata.normalize("NFKC", name or "")
    without_space = "".join(ch for ch in raw if not ch.isspace())
    return without_space.casefold()


def generate_tenant_id() -> str:
    """
    生成新的、随机且安全的 ``tenant_id``（``"c" + uuid4().hex[:12]``）.

    满足 ``tenancy._TENANT_ID_RE``（它会拼进上传目录路径与 Qdrant payload，
    必须防穿越）。**不再**由名称派生 —— 名称派生会让「改名 = 换 id」，与 P0-4
    「改名零迁移」直接冲突。
    """
    return "c" + uuid.uuid4().hex[:12]


def _clean_name(display_name: str | None) -> tuple[str, str]:
    """校验并返回 ``(trimmed_name, name_key)``；空/超长抛 CompanyError。"""
    name = (display_name or "").strip()
    if not name:
        raise CompanyError(MSG_EMPTY_NAME)
    if len(name) > 128:
        raise CompanyError(MSG_NAME_TOO_LONG)
    key = normalize_company_name_key(name)
    if not key:
        raise CompanyError(MSG_EMPTY_NAME)
    return name, key


# ── 查询 ──────────────────────────────────────────────────────────────────────


async def get_company(tenant_id: str | None) -> Company | None:
    """按 tenant_id 取公司；不存在返回 None。"""
    if not tenant_id:
        return None
    async with get_db_session() as session:
        return await session.get(Company, tenant_id)


async def find_by_name(name: str | None) -> Company | None:
    """
    按公司名解析公司（走 ``name_key``，因此大小写/空格变体都能命中）.

    改名后**新名可解析、旧名失效**（P1-2）：因为查的是当前 ``name_key``。
    """
    key = normalize_company_name_key(name)
    if not key:
        return None
    async with get_db_session() as session:
        return await session.scalar(
            select(Company).where(Company.name_key == key).limit(1)
        )


async def list_registered_companies() -> list[Company]:
    """全部已注册公司（按展示名排序，供「身份验证」下拉候选）。"""
    async with get_db_session() as session:
        rows = (
            await session.execute(select(Company).order_by(Company.display_name))
        ).scalars().all()
    return list(rows)


async def company_display_names(
    tenant_ids: frozenset[str] | None,
) -> dict[str, str]:
    """
    批量取「租户集合 → 展示名」映射（列表页标注用，避免 N+1）.

    ``None`` = 取全部已注册公司；空集 = 空字典。
    """
    async with get_db_session() as session:
        stmt = select(Company.tenant_id, Company.display_name)
        if tenant_ids is not None:
            if not tenant_ids:
                return {}
            stmt = stmt.where(Company.tenant_id.in_(sorted(tenant_ids)))
        rows = (await session.execute(stmt)).all()
    return {tenant_id: display_name for tenant_id, display_name in rows}


async def tenant_ids_created_by(admin_id: uuid.UUID | None) -> frozenset[str]:
    """
    某平台管理员**自建**公司集合（= 它的 ``owns_tenant_ids``；其 ``tenant_ids``
    = 所属租户 ∪ 本集合，见 ``tenancy.scope_for``）.

    按 ``created_by == admin_id`` **按 id 锁定** —— 不是「任意 admin 的公司」，
    否则多管理员时 A 的测试公司会对 B 泄漏。
    """
    if admin_id is None:
        return frozenset()
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(Company.tenant_id).where(Company.created_by == admin_id)
            )
        ).scalars().all()
    return frozenset(rows)


async def test_tenant_ids() -> frozenset[str]:
    """
    全部「测试公司」（``companies.is_test = true``）的 ``tenant_id`` 集合.

    这是「测试公司集合」的**唯一来源**：内容消费排除
    （``tenancy.exclude_test_tenants``，被 ``tenancy.content_scope`` 与
    ``retrieval_service.retrieve_chunks`` 共用同一实现点）从这里取，禁止在别处再
    拼一遍 ``is_test`` 的 SQL —— 两处各写一份必然漂移。

    语义（用户**最终口径**，已覆盖此前"测试公司成员也看不到"的旧说法）：
        * **测试公司成员**在自己测试公司内**一切照常** —— 列表可见、检索**能命中**、
          可上传、可预览（用知识库测 bug 的前提）；测试账号不受任何排除影响。
        * **仅平台管理员 admin** 的**内容消费**（检索 / 摘要 / 文档关联 / 对话内
          文档列表）排除测试公司；admin 的**文档列表 / 管理端点**（``GET /documents``）
          **仍可见**测试公司文档，以便管理它们。
        * 排除只在 admin 身份下生效：判据是其 ``owns_tenant_ids`` 非空（本系统中只有
          平台管理员才可能拥有自建公司集合 → 非 admin 恒为空集，见 ``scope_for``）。
    ``is_test`` 是物化列（``created_by`` 被 ON DELETE SET NULL 清空后仍能辨类别），
    因此这里不需要回退到 ``created_by`` 推断。
    """
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(Company.tenant_id).where(Company.is_test.is_(True))
            )
        ).scalars().all()
    return frozenset(rows)


# ── 写：创建 / 改名 ───────────────────────────────────────────────────────────


async def create_company(actor: User | None, display_name: str) -> Company:
    """
    创建公司（平台管理员操作）.

    - 公司名唯一：重名（含大小写/空格变体）→ :class:`CompanyError` 409。
    - ``tenant_id`` 随机生成（与名称解耦）。
    - ``created_by = actor.id``，``is_test = actor.is_admin``。
    """
    name, key = _clean_name(display_name)
    is_admin = bool(actor is not None and getattr(actor, "is_admin", False))

    async with get_db_session() as session:
        existing = await session.scalar(
            select(Company).where(Company.name_key == key).limit(1)
        )
        if existing is not None:
            raise CompanyError(MSG_DUPLICATE, status_code=409)

        tenant_id = generate_tenant_id()
        # 随机 id 碰撞概率极低，但既然主键唯一，显式兜一下更稳。
        while await session.get(Company, tenant_id) is not None:
            tenant_id = generate_tenant_id()

        company = Company(
            tenant_id=tenant_id,
            display_name=name,
            name_key=key,
            created_by=(actor.id if actor is not None else None),
            is_test=is_admin,
        )
        session.add(company)
        try:
            await session.flush()
        except IntegrityError as exc:
            # 并发下两个同名请求可能同时通过前置查重；``name_key`` 唯一约束是
            # 最后一道闸 —— 把它翻译成与单线程一致的 409，而不是 500。
            raise CompanyError(MSG_DUPLICATE, status_code=409) from exc
        await session.refresh(company)

    logger.info(
        "company created tenant_id=%s name=%r is_test=%s by=%s",
        company.tenant_id, name, is_admin,
        getattr(actor, "username", None),
    )
    return company


async def assert_admin_owns(actor: User | None, tenant_id: str) -> None:
    """断言 actor 是平台管理员且 tenant_id ∈ 其自建集合；否则抛 CompanyError。"""
    if actor is None or not getattr(actor, "is_admin", False):
        raise CompanyError(MSG_NEED_ADMIN, status_code=403)
    owned = await tenant_ids_created_by(actor.id)
    if tenant_id not in owned:
        raise CompanyError(MSG_NOT_OWNS, status_code=403)


async def rename_company(actor: User | None, tenant_id: str, new_name: str) -> Company:
    """
    改公司展示名（平台管理员、且须自建）.

    - **``tenant_id`` 不变**：成员归属、文档归属、向量数据零迁移。
    - 只改 ``companies.display_name / name_key`` + 同步 ``users.company_name``
      （展示副本，供历史回溯与个人主页显示）。
    - 写 ``audit_logs(action="company.rename")``（旧名→新名 / 时间 / 操作者）。
    - 重名（与其他公司）→ 409。
    """
    name, key = _clean_name(new_name)
    await assert_admin_owns(actor, tenant_id)

    old_name = ""
    async with get_db_session() as session:
        company = await session.get(Company, tenant_id)
        if company is None:
            raise CompanyError("公司不存在", status_code=404)
        conflict = await session.scalar(
            select(Company)
            .where(Company.name_key == key, Company.tenant_id != tenant_id)
            .limit(1)
        )
        if conflict is not None:
            raise CompanyError(MSG_DUPLICATE, status_code=409)

        old_name = company.display_name
        if old_name != name:
            company.display_name = name
            company.name_key = key
            company.updated_at = func.now()
            # 展示副本同步：只同步本租户成员的公司名（唯一允许的用户字段写入）。
            await session.execute(
                update(User)
                .where(User.tenant_id == tenant_id)
                .values(company_name=name)
            )
            try:
                await session.flush()
            except IntegrityError as exc:
                # 改成一个已被占用的名字（并发下前置查重可能漏掉）→ 与单线程
                # 一致的 409，而不是 500。
                raise CompanyError(MSG_DUPLICATE, status_code=409) from exc
            await session.refresh(company)

    if old_name != name:
        await record_audit(
            "company.rename",
            user_id=(actor.id if actor is not None else None),
            username=(getattr(actor, "username", None) if actor is not None else None),
            resource_type="company",
            resource_id=tenant_id,
            detail=f"{old_name}→{name}; by {getattr(actor, 'username', '-')}",
        )
        logger.info(
            "company renamed tenant_id=%s %r → %r by=%s",
            tenant_id, old_name, name, getattr(actor, "username", None),
        )
    return company


# ── 一次性回填（幂等，支持 dry-run）──────────────────────────────────────────


async def backfill_from_existing(
    *,
    admin_username: str = "admin",
    assignments: tuple[dict, ...] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """
    把历史租户一次性裁定进注册表（幂等）.

    规则（``DEFAULT_BACKFILL_ASSIGNMENTS``）：
        c8111de986583 → 测试公司1（created_by=admin, is_test）
        cfb08c53677c4 → 测试公司2（created_by=admin, is_test）
        c309a7cb9f496 → A公司（created_by=NULL）
        cf33b1db5679d → B公司（created_by=NULL）
        default       → **不入表**

    - **幂等**：已存在的租户 → 更新（而非重复插入）；``name_key`` 冲突时
      跳过并记录。
    - 只写 ``companies`` 表 + 跟随改名的两家租户同步 ``users.company_name``；
      **不改** ``documents.tenant_id`` / Qdrant payload / 向量。
    - ``dry_run=True`` 时只计算计划、回滚不落库。

    Returns:
        每个裁定项的动作描述 ``[{tenant_id, display_name, action, reason}]``。
    """
    plans = assignments or DEFAULT_BACKFILL_ASSIGNMENTS
    report: list[dict] = []

    async with get_db_session() as session:
        admin_id: uuid.UUID | None = None
        if any(item.get("created_by") for item in plans):
            admin = await session.scalar(
                select(User).where(User.username == admin_username).limit(1)
            )
            if admin is None:
                raise CompanyError(
                    f"回填失败：找不到管理员账号 {admin_username!r}"
                )
            admin_id = admin.id

        for item in plans:
            tenant_id = str(item["tenant_id"])
            display_name = str(item["display_name"])
            key = normalize_company_name_key(display_name)
            wants_admin = bool(item.get("created_by"))
            created_by = admin_id if wants_admin else None
            is_test = bool(item.get("is_test", wants_admin))

            row = await session.get(Company, tenant_id)
            conflict = await session.scalar(
                select(Company)
                .where(Company.name_key == key, Company.tenant_id != tenant_id)
                .limit(1)
            )
            if conflict is not None:
                report.append({
                    "tenant_id": tenant_id, "display_name": display_name,
                    "action": "skip", "reason": f"name_key 被 {conflict.tenant_id} 占用",
                })
                continue

            if row is None:
                action = "insert"
                session.add(Company(
                    tenant_id=tenant_id, display_name=display_name, name_key=key,
                    created_by=created_by, is_test=is_test,
                ))
            else:
                changed = (
                    row.display_name != display_name
                    or row.name_key != key
                    or row.created_by != created_by
                    or bool(row.is_test) != is_test
                )
                action = "update" if changed else "noop"
                if changed:
                    row.display_name = display_name
                    row.name_key = key
                    row.created_by = created_by
                    row.is_test = is_test
                    row.updated_at = func.now()

            # 跟随改名的两家测试公司：同步展示副本（唯一允许的用户字段写入）。
            tenant_users_synced = 0
            if wants_admin:
                result = await session.execute(
                    update(User)
                    .where(User.tenant_id == tenant_id, User.company_name != display_name)
                    .values(company_name=display_name)
                )
                tenant_users_synced = int(result.rowcount or 0)

            report.append({
                "tenant_id": tenant_id, "display_name": display_name,
                "action": action, "created_by": str(created_by) if created_by else None,
                "is_test": is_test, "users_company_name_synced": tenant_users_synced,
            })

        if dry_run:
            await session.rollback()
        # 非 dry_run：交给 get_db_session 的退出提交

    logger.info("company backfill %s → %d 项", "dry-run" if dry_run else "applied", len(report))
    return report
