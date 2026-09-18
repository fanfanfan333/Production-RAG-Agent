"""
三层知识库（个人 / 部门 / 公司）权限矩阵与共享申请的确定性测试.

这是本次升级的核心业务规则，任何一条出错都会直接表现为"越权能看到 / 越权
能删 / 该能删的删不掉"，所以在这里把矩阵逐格钉死：

    角色            上传  个人库  发布到部门库  发布到公司库   删除他人文档
    普通员工        ✓     ✓      申请(需审核)    ✗             ✗
    部门负责人      ✓     ✓      ✓              申请(需审核)   本部门范围
    知识库管理员    ✓     ✓      ✓              ✓             全公司
    企业管理员      ✓     ✓      ✓              ✓             全公司

运行方式（容器内，宿主机缺依赖会 SKIP）：

    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python -m pytest tests/test_three_tier_kb.py -q -p no:cacheprovider"
"""

import sys
import uuid
from pathlib import Path

print("── 三层知识库权限矩阵测试 ──")

try:
    import pytest

    from app.db.user_models import User
    from app.services.permissions import (
        ROLE_LABELS,
        can_access_all_documents_platform,
        has_permission,
        permissions_of,
        role_label,
    )
    from app.services.tenancy import (
        ACCESS_DEPARTMENT,
        ACCESS_LABELS,
        ACCESS_PRIVATE,
        ACCESS_TENANT,
        DEFAULT_DOCUMENT_ACCESS_LEVEL,
        PLATFORM_SCOPE_LABEL,
        TENANT_WIDE_READER_ROLES,
        access_label,
        access_scope_name,
        can_access_document,
        can_request_delete,
        company_display_name,
        delete_permission_for,
        document_acl_clause,
        is_upward_transition,
        publish_requirement,
        scope_for,
    )
except ImportError as exc:      # pragma: no cover — 宿主机缺依赖 → 跳过
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    # 不能用 sys.exit()：pytest 在收集阶段导入本模块，抛 SystemExit 会让整个
    # 会话 INTERNALERROR，同目录其它用例全部跑不了。
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


_failures: list[str] = []


def check(label: str, condition: bool) -> None:
    if condition:
        print(f"  PASS {label}")
    else:
        print(f"  FAIL {label}")
        _failures.append(label)


# ── pytest 门禁：让 check() 失败**真正** fail ─────────────────────────────────
# 背景：check() 只 append 到 _failures，真正的失败判定在 __main__ 里 sys.exit(1)。
# pytest 收集的是 test_* 函数，函数体没有 raise ⇒ 无论 check 过不过都报 passed，
# 本文件全部 check 在 pytest 下等于"空气"（这些恰是本轮最核心的隔离断言）。
# 这里用 module-scope 的 autouse fixture：整份文件全部 test_* 跑完后（teardown）
# 统一判定 _failures —— 既让 pytest 能 fail，又保留脚本模式"先收集全部再汇总退出"
# 的诊断能力（不在第一条失败就中断）。
@pytest.fixture(autouse=True, scope="module")
def _fail_module_if_any_check_failed():
    yield
    if _failures:
        raise AssertionError(
            f"check() 失败 {len(_failures)} 项:\n  - " + "\n  - ".join(_failures)
        )


class _StubUser:
    def __init__(self, *, role, tenant_id="company_a", department_id=None, uid=None):
        self.id = uid or uuid.uuid4()
        self.role = role
        self.tenant_id = tenant_id
        self.department_id = department_id

    @property
    def is_admin(self):
        return self.role == "admin"


class _StubDoc:
    def __init__(self, *, owner_id, tenant_id="company_a",
                 access_level=ACCESS_PRIVATE, department_id=None):
        self.id = uuid.uuid4()
        self.owner_id = owner_id
        self.tenant_id = tenant_id
        self.access_level = access_level
        self.department_id = department_id


def _user(role, **kw):
    return _StubUser(role=role, **kw)


# ═════════════════════════════════════════════════════════════════════════════
# 1. 上传 / 发布能力：矩阵前三列
# ═════════════════════════════════════════════════════════════════════════════

def test_publish_matrix():
    employee = _user(User.ROLE_EMPLOYEE)
    dept_head = _user(User.ROLE_DEPT_MANAGER, department_id="tech")
    kb_admin = _user(User.ROLE_KB_ADMIN)
    company_admin = _user(User.ROLE_COMPANY_ADMIN)

    check("普通员工：可上传（document.write）", has_permission(employee, "document.write"))
    check("普通员工：不能直接发布到部门库",
          not has_permission(employee, "document.publish.department"))
    check("普通员工：不能直接发布到公司库",
          not has_permission(employee, "document.publish.company"))
    check("普通员工：可以提交共享申请",
          has_permission(employee, "share.request"))

    check("部门负责人：可直接发布到部门库",
          has_permission(dept_head, "document.publish.department"))
    check("部门负责人：不能直接发布到公司库",
          not has_permission(dept_head, "document.publish.company"))
    check("部门负责人：可审核部门级申请",
          has_permission(dept_head, "share.review.department"))
    check("部门负责人：不能审核公司级申请",
          not has_permission(dept_head, "share.review.company"))

    check("知识库管理员：可发布到公司库",
          has_permission(kb_admin, "document.publish.company"))
    check("知识库管理员：可审核公司级申请",
          has_permission(kb_admin, "share.review.company"))
    check("企业管理员：全公司业务权限齐备",
          all(has_permission(company_admin, p) for p in (
              "document.publish.department", "document.publish.company",
              "share.review.department", "share.review.company",
          )))

    check("平台管理员：通配符全权限", has_permission(_user(User.ROLE_ADMIN), "任意.权限"))


def test_legacy_roles_still_work():
    """历史角色必须保持升级前的行为，否则旧账号会突然失去能力。"""
    legacy_user = _user(User.ROLE_USER)
    legacy_editor = _user(User.ROLE_EDITOR)
    legacy_manager = _user(User.ROLE_MANAGER, department_id="tech")

    check("legacy user ≈ 普通员工：可上传、不可发布公司库",
          has_permission(legacy_user, "document.write")
          and not has_permission(legacy_user, "document.publish.company"))
    check("legacy editor ≈ 普通员工",
          has_permission(legacy_editor, "document.write")
          and not has_permission(legacy_editor, "document.publish.department"))
    check("legacy manager ≈ 部门负责人",
          has_permission(legacy_manager, "document.publish.department"))
    check("viewer 只读：不能上传",
          not has_permission(_user(User.ROLE_VIEWER), "document.write"))


def test_publish_requirement_mapping():
    check("个人库无需额外权限", publish_requirement(ACCESS_PRIVATE) is None)
    check("部门库 → publish.department",
          publish_requirement(ACCESS_DEPARTMENT) == "document.publish.department")
    check("公司库 → publish.company",
          publish_requirement(ACCESS_TENANT) == "document.publish.company")


# ═════════════════════════════════════════════════════════════════════════════
# 2. 层级中文标注
# ═════════════════════════════════════════════════════════════════════════════

def test_access_labels():
    check("private → 个人", access_label(ACCESS_PRIVATE) == "个人")
    check("department → 部门", access_label(ACCESS_DEPARTMENT) == "部门")
    check("tenant → 公司", access_label(ACCESS_TENANT) == "公司")
    check("NULL/未知 → 个人（老数据语义）", access_label(None) == "个人")
    check("未知值仍归一为个人", access_label("whatever") == "个人")
    check("中文全称：个人知识库", access_scope_name(ACCESS_PRIVATE) == "个人知识库")
    check("中文全称：公司知识库", access_scope_name(ACCESS_TENANT) == "公司知识库")
    check("默认上传层级 = 个人库（不默认外泄）",
          DEFAULT_DOCUMENT_ACCESS_LEVEL == ACCESS_PRIVATE)


def test_role_labels():
    check("employee → 普通员工", role_label("employee") == "普通员工")
    check("dept_manager → 部门负责人", role_label("dept_manager") == "部门负责人")
    check("kb_admin → 知识库管理员", role_label("kb_admin") == "知识库管理员")
    check("company_admin → 企业管理员", role_label("company_admin") == "企业管理员")
    check("每个有效角色都有中文名",
          all(r in ROLE_LABELS for r in User.VALID_ROLES))


# ═════════════════════════════════════════════════════════════════════════════
# 3. 删除权限：矩阵最后一列（问题2 的规则面）
# ═════════════════════════════════════════════════════════════════════════════

def test_delete_permission_matrix():
    owner = _user(User.ROLE_EMPLOYEE, uid=uuid.uuid4())
    employee = _user(User.ROLE_EMPLOYEE, department_id="tech")
    dept_head = _user(User.ROLE_DEPT_MANAGER, department_id="tech")
    other_dept_head = _user(User.ROLE_DEPT_MANAGER, department_id="sales")
    kb_admin = _user(User.ROLE_KB_ADMIN)
    company_admin = _user(User.ROLE_COMPANY_ADMIN)

    personal = _StubDoc(owner_id=owner.id, access_level=ACCESS_PRIVATE)
    dept_doc = _StubDoc(owner_id=owner.id, access_level=ACCESS_DEPARTMENT,
                        department_id="tech")
    company_doc = _StubDoc(owner_id=owner.id, access_level=ACCESS_TENANT)
    cross_company = _StubDoc(owner_id=owner.id, tenant_id="company_b",
                             access_level=ACCESS_TENANT)

    ok, _ = delete_permission_for(personal, owner)
    check("本人：可以删除自己的文档", ok)

    ok, reason = delete_permission_for(personal, employee)
    check("普通员工：不能删除他人个人库文档", not ok)
    check("普通员工：拒绝原因含中文说明", "个人知识库" in reason)

    ok, _ = delete_permission_for(dept_doc, dept_head)
    check("部门负责人：可删本部门范围的他人文档", ok)

    ok, _ = delete_permission_for(personal, dept_head)
    check("部门负责人：不能删他人个人库文档（非本部门范围）", not ok)

    ok, _ = delete_permission_for(dept_doc, other_dept_head)
    check("部门负责人：不能删其他部门的文档", not ok)

    ok, _ = delete_permission_for(personal, kb_admin)
    check("知识库管理员：**不能**删他人个人库文档（个人库对所有人私密）", not ok)

    ok, _ = delete_permission_for(company_doc, company_admin)
    check("企业管理员：可删全公司文档", ok)

    ok, reason = delete_permission_for(cross_company, kb_admin)
    check("跨公司：一律拒绝", not ok)
    check("跨公司：不暴露存在性（按'不存在'表述）", "不存在" in reason)

    platform_admin = _user(User.ROLE_ADMIN, tenant_id="company_c")
    # Rev2：admin 的删除范围同样收敛为**自建测试公司集合**（按 created_by 锁定）。
    ok, _ = delete_permission_for(
        company_doc, platform_admin,
        tenant_ids=frozenset({"company_a"}),
        owns_tenant_ids=frozenset({"company_a"}),
    )
    check("平台管理员：在**自建测试公司集合**内可删部门/公司库文档", ok)
    ok, _ = delete_permission_for(company_doc, platform_admin)
    check("平台管理员：对**非自建**公司不可删（收敛，不再跨全平台）", not ok)
    # 生产路径：admin 在**自建测试公司集合**（这是它真实的可见范围）内遇到他人私库 ——
    # 拒绝本身不能变，且要给出「个人库专属」文案（可读性/可解释性断言，强度不降）。
    ok, reason = delete_permission_for(
        personal, platform_admin,
        tenant_ids=frozenset({"company_a"}),
        owns_tenant_ids=frozenset({"company_a"}),
    )
    check("平台管理员：**不能**删他人个人库文档（全平台 ≠ 看穿个人库）", not ok)
    check("平台管理员：拒绝原因说明个人库只属于归属人",
          "个人知识库" in reason)

    # 保守兜底：**不传** tenant_ids 时 admin → 空集 fail-closed（防"漏传参数就回退
    # 全平台"）⇒ 拒绝，且文案退化为「不存在」（不泄漏存在性）。两条路径都要有断言
    # 保护：以后谁把 _default_tenant_ids_for 放宽或把 private 判定顺序改坏都会被抓。
    ok_no_scope, reason_no_scope = delete_permission_for(personal, platform_admin)
    check("平台管理员：漏传范围时按 fail-closed 拒绝（不回退全平台）", not ok_no_scope)
    check("平台管理员：漏传范围时不泄漏存在性（按'不存在'表述）",
          "不存在" in reason_no_scope)


def test_delete_follows_tier_not_ownership():
    """
    删除权跟着**文档所在层级**走，而不是"谁上传谁说了算".

    这是对旧行为的收紧：以前 `delete_permission_for` 一上来就 `owner == user`
    直接放行，于是文档一旦发布到部门库/公司库，作者仍能一键删除 —— 而同事的
    问答与报告正在引用它。现在高层的删除权收归上级，归属人走「申请删除」。
    """
    owner = _user(User.ROLE_EMPLOYEE, department_id="tech")
    kb_admin = _user(User.ROLE_KB_ADMIN)

    personal = _StubDoc(owner_id=owner.id, access_level=ACCESS_PRIVATE)
    dept_doc = _StubDoc(owner_id=owner.id, access_level=ACCESS_DEPARTMENT,
                        department_id="tech")
    company_doc = _StubDoc(owner_id=owner.id, access_level=ACCESS_TENANT)

    check("归属人：个人库文档可以直接删", delete_permission_for(personal, owner)[0])
    check("归属人：部门库文档不能直接删（已共享给部门）",
          not delete_permission_for(dept_doc, owner)[0])
    check("归属人：公司库文档不能直接删（已是组织资产）",
          not delete_permission_for(company_doc, owner)[0])

    ok, reason = delete_permission_for(company_doc, owner)
    check("被拒原因指向「申请删除」这条升级路径",
          "申请删除" in reason, )

    check("归属人：部门库文档可提交「申请删除」",
          can_request_delete(dept_doc, owner))
    check("归属人：公司库文档可提交「申请删除」",
          can_request_delete(company_doc, owner))
    check("归属人：个人库文档无需申请（本来就能删）",
          not can_request_delete(personal, owner))
    check("知识库管理员：无需申请（可直接删）",
          not can_request_delete(company_doc, kb_admin))

    # 旁观者：看得见的部门库文档可以申请删除，看不见的个人库文档不能（防探测）
    peer = _user(User.ROLE_EMPLOYEE, department_id="tech")
    check("同部门同事：可申请删除部门库文档", can_request_delete(dept_doc, peer))
    check("同部门同事：不能对他人个人库文档发起申请（不可见）",
          not can_request_delete(personal, peer))


# ═════════════════════════════════════════════════════════════════════════════
# 4. 读取可见性：三层 + read_all（管理员审核需要）
# ═════════════════════════════════════════════════════════════════════════════

def test_visibility_and_read_all():
    owner = _user(User.ROLE_EMPLOYEE, uid=uuid.uuid4())
    peer = _user(User.ROLE_EMPLOYEE, department_id="tech")
    kb_admin = _user(User.ROLE_KB_ADMIN)
    platform_admin = _user(User.ROLE_ADMIN, tenant_id="company_c")

    personal = _StubDoc(owner_id=owner.id, access_level=ACCESS_PRIVATE)
    company_doc = _StubDoc(owner_id=owner.id, access_level=ACCESS_TENANT)
    dept_doc = _StubDoc(owner_id=owner.id, access_level=ACCESS_DEPARTMENT,
                        department_id="tech")
    other_company_doc = _StubDoc(owner_id=owner.id, tenant_id="company_b",
                                 access_level=ACCESS_TENANT)

    check("个人库：同事不可见", not can_access_document(personal, peer))
    check("个人库：本人可见", can_access_document(personal, owner))
    check("个人库：知识库管理员**也不可见**（个人库对所有人私密）",
          not can_access_document(personal, kb_admin))
    check("个人库：平台管理员**也不可见**（全平台只覆盖部门库/公司库）",
          not can_access_document(personal, platform_admin))
    check("公司库：同公司可见", can_access_document(company_doc, peer))
    check("公司库：知识库管理员可见", can_access_document(company_doc, kb_admin))
    check("部门库：跨部门的知识库管理员可见（本公司内宽口径）",
          can_access_document(dept_doc, kb_admin))
    # Rev2：平台管理员的可见范围收敛为**自建测试公司集合**（按 created_by 锁定），
    # 不再"跨全平台"。
    check("公司库：平台管理员对**非自建**公司不可见（Rev2 收敛）",
          not can_access_document(other_company_doc, platform_admin))
    check("公司库：平台管理员在**自建测试公司集合**内可见",
          can_access_document(
              other_company_doc, platform_admin,
              tenant_ids=frozenset({"company_b"}),
              owns_tenant_ids=frozenset({"company_b"}),
          ))
    check("公司库：普通员工看不到其他公司",
          not can_access_document(other_company_doc, peer))
    check("read_all 角色集合 = 企业管理员 + 知识库管理员（**不含 admin**）",
          TENANT_WIDE_READER_ROLES == {
              User.ROLE_COMPANY_ADMIN, User.ROLE_KB_ADMIN
          })
    check("kb_admin 的 permissions 里含 document.read.all",
          "document.read.all" in permissions_of(kb_admin))
    check("can_access_all_documents_platform 只对平台管理员为真",
          can_access_all_documents_platform(platform_admin)
          and not can_access_all_documents_platform(kb_admin))


def test_scope_for_matches_acl():
    """
    ``scope_for`` 是权限入参的唯一组装点，它必须与 ACL 判定完全一致 ——
    任何"scope 说能看、can_access_document 说不能"的分歧都是越权或漏看。
    """
    employee = _user(User.ROLE_EMPLOYEE, tenant_id="company_a", department_id="tech")
    kb_admin = _user(User.ROLE_KB_ADMIN, tenant_id="company_a")
    platform_admin = _user(User.ROLE_ADMIN, tenant_id="default")

    s_emp = scope_for(employee)
    check("普通员工：锁本公司集合 + 本部门 + 自己的个人库",
          s_emp.tenant_ids == frozenset({"company_a"})
          and s_emp.department_id == "tech"
          and not s_emp.tenant_wide and not s_emp.owns_tenant_ids)
    s_kb = scope_for(kb_admin)
    check("知识库管理员：锁本公司集合 + 部门全通",
          s_kb.tenant_ids == frozenset({"company_a"}) and s_kb.tenant_wide
          and not s_kb.owns_tenant_ids)
    # Rev2：未传自建集合时，admin 的公司集合为空（fail-closed，不回退全平台）
    s_admin = scope_for(platform_admin)
    check("平台管理员：无自建公司时公司集合为空（fail-closed）",
          s_admin.tenant_ids == frozenset()
          and s_admin.owns_tenant_ids == frozenset())
    # 传入自建测试公司集合后：tenant_ids = owns_tenant_ids = 该集合
    owned = frozenset({"c8111de986583", "cfb08c53677c4"})
    s_admin_owned = scope_for(platform_admin, owned_tenant_ids=owned)
    check("平台管理员：公司集合 = 自建测试公司集合",
          s_admin_owned.tenant_ids == owned
          and s_admin_owned.owns_tenant_ids == owned
          and s_admin_owned.cross_tenant)
    check("平台管理员的个人库归属仍是自己（不会变成'看所有人个人库'）",
          s_admin.owner_id == platform_admin.id
          and s_admin_owned.owner_id == platform_admin.id)
    check("平台管理员展示身份 = 全平台",
          company_display_name(platform_admin) == PLATFORM_SCOPE_LABEL)
    check("普通用户展示身份仍是自己的公司",
          company_display_name(employee) == "company_a")


def test_admin_default_private_doc_still_visible():
    """
    Rev2 正确性锚点（P0 回归守卫）：admin 落在 ``default`` 租户的**自己的个人库**
    文档必须恒可见 —— ``private`` 只按 ``owner_id == 我`` 判定、**不参与租户过滤**。

    这正是"把个人库从公司边界的合取里拿出来做 or_"修掉的那处静默回归。
    """
    from app.services.tenancy import document_scope_clause
    from sqlalchemy.dialects import postgresql

    platform_admin = _user(User.ROLE_ADMIN, tenant_id="default",
                          uid=uuid.uuid4())
    default_private = _StubDoc(owner_id=platform_admin.id, tenant_id="default",
                               access_level=ACCESS_PRIVATE)

    # 单文档判定：admin 无自建公司（集合为空）→ 自己的 default 私库仍可见
    check("admin 的 default 私库：can_access_document 为真",
          can_access_document(default_private, platform_admin,
                              tenant_ids=frozenset(),
                              owns_tenant_ids=frozenset()))
    # 他人私库（同为空集合）→ 不可见
    check("admin 对他人 default 私库仍不可见",
          not can_access_document(
              _StubDoc(owner_id=uuid.uuid4(), tenant_id="default",
                       access_level=ACCESS_PRIVATE),
              platform_admin,
              tenant_ids=frozenset(), owns_tenant_ids=frozenset()))
    # list/search 的唯一 SQL 入口：空公司集合下 personal 分支必须仍在（private 出现）
    compiled = str(
        document_scope_clause(
            owner_id=platform_admin.id, department_id=None,
            tenant_ids=frozenset(), owns_tenant_ids=frozenset(), tenant_wide=True,
        ).compile(dialect=postgresql.dialect(),
                  compile_kwargs={"literal_binds": True})
    )
    check("空公司集合下 scope clause 仍保留 personal(private) 分支",
          "private" in compiled)


def test_acl_clause_never_opens_private():
    """
    ACL 子句在任何宽口径组合下都**不会**放开他人个人库.

    这条是产品红线"其他人的个人文档看不到"的编译期证据：把 owner_id 换掉，
    个人库条件就随之消失。
    """
    from sqlalchemy.dialects import postgresql

    def compiled(**kw):
        return str(
            document_acl_clause(**kw).compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        )

    wide = compiled(owner_id=None, department_id=None, tenant_wide=True)
    check("宽口径（部门全通）不含 private 分支",
          "private" not in wide)
    check("宽口径含部门库与公司库",
          "department" in wide and "tenant" in wide)

    mine = compiled(owner_id=uuid.uuid4(), department_id=None, tenant_wide=True)
    check("带上自己的 owner_id 后才出现 private 分支", "private" in mine)

    check("旧参数名 read_all 仍然等价于 tenant_wide",
          "department" in compiled(owner_id=None, department_id=None, read_all=True))


# ═════════════════════════════════════════════════════════════════════════════
# 4. 申请共享能力：**按目标层级**判定
# ═════════════════════════════════════════════════════════════════════════════

def test_request_capability_is_per_level():
    """
    回归：部门负责人曾经**拿不到「申请共享」入口**。

    旧实现把申请能力算成一个总布尔
    ``needs_request = is_owner and not can_dept and not can_company``，
    前端据此把弹窗拆成互斥的两支（"有任一发布权 → 只渲染直接发布"）。
    部门负责人 can_dept=True，于是整个申请分支被隐藏：界面里只剩"发布到部门
    知识库"，他要把文档提到公司库时**在界面上无路可走**（接口本身放行，所以
    日志里没有任何报错，纯前端死角）。

    现在逐层判定：某一层能直接发就不给申请入口，不能直接发就给。
    两者对**不同**层级可同时为真 —— 部门负责人正是这种形态。
    """
    from app.services.knowledge_tier_service import publish_capability

    owner_id = uuid.uuid4()
    dept_head = _user(User.ROLE_DEPT_MANAGER, department_id="tech", uid=owner_id)
    employee = _user(User.ROLE_EMPLOYEE, department_id="tech", uid=owner_id)
    kb_admin = _user(User.ROLE_KB_ADMIN, uid=owner_id)

    personal = _StubDoc(owner_id=owner_id, access_level=ACCESS_PRIVATE)
    dept_doc = _StubDoc(owner_id=owner_id, access_level=ACCESS_DEPARTMENT,
                        department_id="tech")
    company_doc = _StubDoc(owner_id=owner_id, access_level=ACCESS_TENANT)

    cap = publish_capability(dept_head, personal)
    check("部门负责人：可直接发布到部门库", cap["can_publish_department"])
    check("部门负责人：不能直接发布到公司库", not cap["can_publish_company"])
    check("部门负责人：部门库无需申请（本来就能直接发）",
          not cap["can_request_department"])
    check("部门负责人：可以申请公司库 ← 本次修复的核心",
          cap["can_request_company"])
    check("部门负责人：有可申请的层级 → needs_share_request 为真",
          cap["needs_share_request"])

    cap = publish_capability(dept_head, dept_doc)
    check("部门负责人：文档已在部门库 → 该层不再重复申请",
          not cap["can_request_department"])
    check("部门负责人：部门库文档仍可申请提到公司库", cap["can_request_company"])

    cap = publish_capability(employee, personal)
    check("普通员工：部门库要申请", cap["can_request_department"])
    check("普通员工：公司库要申请", cap["can_request_company"])
    check("普通员工：有可申请的层级", cap["needs_share_request"])

    cap = publish_capability(kb_admin, personal)
    check("知识库管理员：两级都能直接发布",
          cap["can_publish_department"] and cap["can_publish_company"])
    check("知识库管理员：没有任何需要申请的层级",
          not (cap["can_request_department"] or cap["can_request_company"]))
    check("知识库管理员：needs_share_request 为假", not cap["needs_share_request"])

    cap = publish_capability(employee, company_doc)
    check("文档已在公司库：不能再申请『部门库』（那是降级，不是共享）",
          not cap["can_request_department"])
    check("文档已在公司库：再无更高层级可申请", not cap["needs_share_request"])

    peer = _user(User.ROLE_EMPLOYEE, department_id="tech")
    cap = publish_capability(peer, personal)
    check("非归属人：没有任何申请入口（只有归属人能提申请）",
          not (cap["can_request_department"] or cap["can_request_company"]))

    cap = publish_capability(kb_admin, company_doc)
    check("公司库文档：知识库管理员没有任何需要申请的层级",
          not (cap["can_request_department"] or cap["can_request_company"]))
    check("发布能力**按文档层级收窄**：公司库文档不能再「发布」到部门库（那是降级）",
          not cap["can_publish_department"])
    check("同层级（公司→公司）不算降级，仍然允许",
          cap["can_publish_company"])


def test_direct_publish_cannot_downgrade():
    """
    回归：**直接发布**链路曾是一条不需要审批的降级通道。

    旧实现 ``can_publish_department = has_permission(user, "document.publish.department")``
    是纯角色判定，文档当前在哪一层完全不参与。后果：部门负责人面对一份**已经
    发布到公司库**的文档，能力字段仍为 True，界面照常渲染"发布到部门知识库"
    按钮，点下去 ``PATCH /documents/{id}/visibility`` 就把它真降成部门库 ——
    其他部门同事静默失去访问权，而审计日志里只是一条正常的"层级变更"。

    申请链路当时已用 is_upward_transition 堵住向下，但直接发布这条路更短、
    更隐蔽（无人审批）。现在两处同一口径：``can_publish_*`` 也按层级收窄。

    注意边界：**收回个人库不能被误伤** —— 那是归属人的正当操作，前端有独立
    的「收回」入口，文案也明示了后果。
    """
    from app.services.knowledge_tier_service import publish_capability
    from app.services.tenancy import is_downgrade

    # ── is_downgrade 本身：只回答"层级是否变低"，不管调用方拦不拦 ──────────
    check("公司 → 部门 = 降级", is_downgrade(ACCESS_TENANT, ACCESS_DEPARTMENT))
    check("公司 → 个人 = 降级（调用方需自行放行『收回』）",
          is_downgrade(ACCESS_TENANT, ACCESS_PRIVATE))
    check("部门 → 个人 = 降级", is_downgrade(ACCESS_DEPARTMENT, ACCESS_PRIVATE))
    check("个人 → 部门 = 不是降级", not is_downgrade(ACCESS_PRIVATE, ACCESS_DEPARTMENT))
    check("部门 → 公司 = 不是降级", not is_downgrade(ACCESS_DEPARTMENT, ACCESS_TENANT))
    check("同层级 = 不是降级", not is_downgrade(ACCESS_TENANT, ACCESS_TENANT))
    check("未知层级按个人库处理（None → 部门 不是降级）",
          not is_downgrade(None, ACCESS_DEPARTMENT))

    owner_id = uuid.uuid4()
    dept_head = _user(User.ROLE_DEPT_MANAGER, department_id="tech", uid=owner_id)
    kb_admin = _user(User.ROLE_KB_ADMIN, uid=owner_id)

    personal = _StubDoc(owner_id=owner_id, access_level=ACCESS_PRIVATE)
    dept_doc = _StubDoc(owner_id=owner_id, access_level=ACCESS_DEPARTMENT,
                        department_id="tech")
    company_doc = _StubDoc(owner_id=owner_id, access_level=ACCESS_TENANT)

    # ── 核心：公司库文档不能被"发布"到部门库 ──────────────────────────────
    cap = publish_capability(dept_head, company_doc)
    check("部门负责人对本公司库文档：不能再『发布到部门库』← 本次修复的核心",
          not cap["can_publish_department"])
    check("被拒时给出层级原因，而不是误报『你的角色暂无发布权限』"
          f"（reason={cap['publish_denied_reason']!r}）",
          "更低的层级" in cap["publish_denied_reason"])

    # ── 不能误伤：向上 / 同层 / 本人文档的发布能力保持原样 ────────────────
    cap = publish_capability(dept_head, personal)
    check("个人文档：部门负责人仍可直接发布到部门库（未被误伤）",
          cap["can_publish_department"])
    cap = publish_capability(dept_head, dept_doc)
    check("部门库文档：同层级不算降级，发布能力保留",
          cap["can_publish_department"])

    cap = publish_capability(kb_admin, dept_doc)
    check("部门库文档：知识库管理员仍可『发布到公司库』（向上，未被误伤）",
          cap["can_publish_company"])
    cap = publish_capability(kb_admin, personal)
    check("个人文档：知识库管理员两级都能发",
          cap["can_publish_department"] and cap["can_publish_company"])


def test_only_upward_transitions_are_requestable():
    """申请只能向上：平级/向下都要挡住（向下批准即文档降级）。"""
    check("个人库 → 部门库：向上，可申请",
          is_upward_transition(ACCESS_PRIVATE, ACCESS_DEPARTMENT))
    check("个人库 → 公司库：向上，可申请",
          is_upward_transition(ACCESS_PRIVATE, ACCESS_TENANT))
    check("部门库 → 公司库：向上，可申请",
          is_upward_transition(ACCESS_DEPARTMENT, ACCESS_TENANT))
    check("部门库 → 部门库：平级，不可申请",
          not is_upward_transition(ACCESS_DEPARTMENT, ACCESS_DEPARTMENT))
    check("公司库 → 部门库：**向下，不可申请**（批准会把文档降级）",
          not is_upward_transition(ACCESS_TENANT, ACCESS_DEPARTMENT))
    check("公司库 → 公司库：平级，不可申请",
          not is_upward_transition(ACCESS_TENANT, ACCESS_TENANT))
    check("未知层级按个人库处理（老数据语义）",
          is_upward_transition(None, ACCESS_DEPARTMENT)
          and is_upward_transition("whatever", ACCESS_DEPARTMENT))


def test_company_request_is_routed_to_company_reviewers():
    """
    部门负责人申请"公司库"时，审核人必须是**公司级**（知识库管理员 /
    企业管理员 / 平台管理员），落到部门负责人自己手里就是自审自批。

    同时校验跨公司隔离：B 公司的审核人看不到 A 公司的申请。
    """
    try:
        from app.services.share_service import can_review, review_scope
    except ImportError as exc:      # pragma: no cover — 宿主机缺依赖
        print(f"  SKIP share_service 不可导入（{exc}）")
        return

    class _StubRequest:
        def __init__(self, *, tenant_id, target_level, requester_id,
                     target_department_id=None):
            self.tenant_id = tenant_id
            self.target_level = target_level
            self.target_department_id = target_department_id
            self.requester_id = requester_id

    requester = _user(User.ROLE_DEPT_MANAGER, department_id="tech")
    same_dept_head = _user(User.ROLE_DEPT_MANAGER, department_id="tech")
    other_dept_head = _user(User.ROLE_DEPT_MANAGER, department_id="sales")
    kb_admin = _user(User.ROLE_KB_ADMIN)
    company_admin = _user(User.ROLE_COMPANY_ADMIN)
    other_company_kb = _user(User.ROLE_KB_ADMIN, tenant_id="company_b")
    platform_admin = _user(User.ROLE_ADMIN, tenant_id="company_c")

    to_company = _StubRequest(
        tenant_id="company_a", target_level=ACCESS_TENANT,
        requester_id=requester.id,
    )

    check("部门负责人：审核范围是部门级", review_scope(requester) == "department")
    check("部门负责人：审不了『公司库』申请（不能自审自批）",
          not can_review(to_company, same_dept_head))
    check("知识库管理员：可审本公司的公司库申请",
          can_review(to_company, kb_admin))
    check("企业管理员：可审本公司的公司库申请",
          can_review(to_company, company_admin))
    check("平台管理员：可审任意公司的申请",
          can_review(to_company, platform_admin))
    check("跨公司：B 公司的知识库管理员审不了 A 公司的申请（公司隔离）",
          not can_review(to_company, other_company_kb))
    check("申请人本人：不能审自己的申请",
          not can_review(to_company, requester))

    to_dept = _StubRequest(
        tenant_id="company_a", target_level=ACCESS_DEPARTMENT,
        requester_id=requester.id, target_department_id="tech",
    )
    check("部门库申请：本部门负责人可审", can_review(to_dept, same_dept_head))
    check("部门库申请：其他部门负责人审不了",
          not can_review(to_dept, other_dept_head))
    check("部门库申请：知识库管理员（公司级）也能审",
          can_review(to_dept, kb_admin))


# ═════════════════════════════════════════════════════════════════════════════
# 6. 「转为部门文档」：公司级管理者把已共享的文档改归到指定部门
# ═════════════════════════════════════════════════════════════════════════════

def test_transfer_department_capability():
    """
    公司 HR 把公司文档转为部门文档 —— 谁能做、在什么状态下能做.

    这条能力与「发布到部门知识库」必须分得清，它们是两个字段：

        can_publish_department  发到**自己的**部门（部门负责人），无选择余地
        can_transfer_department 指定**任意一个**本公司部门（公司 HR）

    前者按"角色 + 文档层级不得降级"判定；后者按"是不是公司级管理者"判定，
    而且**刻意包含降级**（公司库 → 部门库正是它的用途）。合并成一个布尔
    会立刻出事：要么部门负责人获得把公司库文档改派到别的部门的权力，要么
    HR 的这次操作被防降级补丁拦死。
    """
    from app.services.knowledge_tier_service import publish_capability

    owner_id = uuid.uuid4()
    employee = _user(User.ROLE_EMPLOYEE, department_id="tech", uid=owner_id)
    dept_head = _user(User.ROLE_DEPT_MANAGER, department_id="tech", uid=owner_id)
    kb_admin = _user(User.ROLE_KB_ADMIN, uid=owner_id)
    company_admin = _user(User.ROLE_COMPANY_ADMIN, uid=owner_id)
    platform_admin = _user(User.ROLE_ADMIN, tenant_id="company_c", uid=owner_id)

    personal = _StubDoc(owner_id=owner_id, access_level=ACCESS_PRIVATE)
    dept_doc = _StubDoc(owner_id=owner_id, access_level=ACCESS_DEPARTMENT,
                        department_id="tech")
    company_doc = _StubDoc(owner_id=owner_id, access_level=ACCESS_TENANT)

    # ── 公司库文档：只有公司级管理者能改归部门 ────────────────────────────────
    check("普通员工：不能把公司文档转为部门文档",
          not publish_capability(employee, company_doc)["can_transfer_department"])
    check("部门负责人：**不能**把公司文档改派到其他部门（越权整理组织资产）",
          not publish_capability(dept_head, company_doc)["can_transfer_department"])
    check("知识库管理员：可以把公司文档转为部门文档",
          publish_capability(kb_admin, company_doc)["can_transfer_department"])
    check("企业管理员（HR）：可以把公司文档转为部门文档",
          publish_capability(company_admin, company_doc)["can_transfer_department"])
    check("平台管理员：可跨公司把公司文档转为部门文档",
          publish_capability(platform_admin, company_doc)["can_transfer_department"])

    # ── 部门库文档：换部门归属（平级调动）也归这条能力 ────────────────────────
    check("知识库管理员：可把部门文档改归到另一个部门",
          publish_capability(kb_admin, dept_doc)["can_transfer_department"])
    check("部门负责人：不能把本部门文档改归到其他部门",
          not publish_capability(dept_head, dept_doc)["can_transfer_department"])

    # ── 个人库文档：先共享，再谈归属 ──────────────────────────────────────────
    cap = publish_capability(kb_admin, personal)
    check("个人文档：不能直接『转为部门文档』（应先用「发布到部门知识库」）",
          not cap["can_transfer_department"])
    check("个人文档被拒时给出可执行的中文原因",
          "发布到部门知识库" in cap["transfer_denied_reason"])

    # ── 关键回归：新能力不能把上一轮的防降级补丁撞开 ──────────────────────────
    cap = publish_capability(kb_admin, company_doc)
    check("公司库文档：知识库管理员的『发布到部门库』仍被防降级收窄（不被新功能绕过）",
          not cap["can_publish_department"])
    check("公司库文档：转为部门文档走的是一条独立能力，两者不互相污染",
          cap["can_transfer_department"] and not cap["can_publish_department"])


def test_merge_department_rows():
    """
    部门清单的合并规则（「转为部门文档」的可选目标）.

    清单来自成员归属的 group by 结果，两份数据最容易出问题的地方在这里：
    按 **ID** 聚合（不是按名字）、空名回填、空 ID 丢弃。写错的直接后果是
    "管理者选中的部门 ID 与显示的名字对不上"，文档被发进一个谁都不在的部门
    —— 对全公司静默不可见。
    """
    from app.services.knowledge_tier_service import merge_department_rows

    rows = [
        ("d_tech", "技术部", 3),
        ("d_sales", "销售部", 2),
        # 同一部门因成员改过名而出现两行：必须按 ID 合并、人数累加
        ("d_tech", "研发部", 4),
        # 名称缺失 → 后续用非空名回填，不让界面显示哈希 ID
        ("d_hr", None, 1),
        ("d_hr", "人力资源部", 2),
        # 空 ID / None → 不属于任何部门，直接丢弃
        ("", "无部门组", 5),
        (None, "无部门组", 5),
    ]
    merged = merge_department_rows(rows)
    by_id = {item["department_id"]: item for item in merged}

    check("按 ID 聚合：同一部门的重复行合为一条", len(merged) == 3)
    check("人数累加（技术部 3 + 4 = 7）", by_id["d_tech"]["member_count"] == 7)
    check("空 ID 的行被丢弃", "" not in by_id and "None" not in by_id)
    check("缺失的名称被非空名回填（不显示哈希 ID）",
          by_id["d_hr"]["department_name"] == "人力资源部")
    check("名称缺失时先以 ID 兜底，界面永远有字可显示",
          merge_department_rows([("d_x", None, 1)])[0]["department_name"] == "d_x")
    check("按部门名排序（人力资源部 / 技术部 / 销售部 的中文字典序）",
          [item["department_name"] for item in merged]
          == ["人力资源部", "技术部", "销售部"])


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for fn in (
        test_publish_matrix,
        test_legacy_roles_still_work,
        test_publish_requirement_mapping,
        test_access_labels,
        test_role_labels,
        test_delete_permission_matrix,
        test_visibility_and_read_all,
        test_scope_for_matches_acl,
        test_admin_default_private_doc_still_visible,
        test_acl_clause_never_opens_private,
        test_request_capability_is_per_level,
        test_direct_publish_cannot_downgrade,
        test_only_upward_transitions_are_requestable,
        test_company_request_is_routed_to_company_reviewers,
        test_transfer_department_capability,
        test_merge_department_rows,
    ):
        print(f"\n▶ {fn.__name__}")
        fn()

    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED: {len(_failures)} 项")
        for f in _failures:
            print("  -", f)
        sys.exit(1)
    print("全部通过：三层知识库权限矩阵符合预期")
