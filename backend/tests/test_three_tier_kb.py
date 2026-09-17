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

print("── 三层知识库权限矩阵测试 ──")

try:
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
    ok, _ = delete_permission_for(company_doc, platform_admin)
    check("平台管理员：跨公司可删部门/公司库文档", ok)
    ok, reason = delete_permission_for(personal, platform_admin)
    check("平台管理员：**不能**删他人个人库文档（全平台 ≠ 看穿个人库）", not ok)
    check("平台管理员：拒绝原因说明个人库只属于归属人",
          "个人知识库" in reason)


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
    check("公司库：平台管理员跨公司可见",
          can_access_document(other_company_doc, platform_admin))
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
    check("普通员工：锁本公司 + 本部门 + 自己的个人库",
          s_emp.tenant_id == "company_a" and s_emp.department_id == "tech"
          and not s_emp.tenant_wide and not s_emp.platform_wide)
    s_kb = scope_for(kb_admin)
    check("知识库管理员：锁本公司 + 部门全通",
          s_kb.tenant_id == "company_a" and s_kb.tenant_wide
          and not s_kb.platform_wide)
    s_admin = scope_for(platform_admin)
    check("平台管理员：无公司边界（跨公司）",
          s_admin.tenant_id is None and s_admin.platform_wide
          and s_admin.cross_tenant)
    check("平台管理员的个人库归属仍是自己（不会变成'看所有人个人库'）",
          s_admin.owner_id == platform_admin.id)
    check("平台管理员展示身份 = 全平台",
          company_display_name(platform_admin) == PLATFORM_SCOPE_LABEL)
    check("普通用户展示身份仍是自己的公司",
          company_display_name(employee) == "company_a")


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

    wide = compiled(owner_id=None, department_id=None, tenant_wide=True,
                    platform_wide=True)
    check("宽口径（部门全通 + 跨公司）不含 private 分支",
          "private" not in wide)
    check("宽口径含部门库与公司库",
          "department" in wide and "tenant" in wide)

    mine = compiled(owner_id=uuid.uuid4(), department_id=None, tenant_wide=True,
                    platform_wide=True)
    check("带上自己的 owner_id 后才出现 private 分支", "private" in mine)

    check("旧参数名 read_all 仍然等价于 tenant_wide",
          "department" in compiled(owner_id=None, department_id=None, read_all=True))


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
        test_acl_clause_never_opens_private,
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
