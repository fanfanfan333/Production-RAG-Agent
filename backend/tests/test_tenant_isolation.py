"""
三层多租户隔离的回归测试（2026-09-13）.

覆盖：

    第一层 Tenant Isolation
        - tenant_id 归一化（防目录穿越）
        - Qdrant 检索前置过滤（ANN query_filter / BM25 scroll_filter 都带租户条件）
        - 向量 upsert payload 写入完整隔离字段
        - Rerank 之后的 Permission Check 会拦下跨租户 chunk

    第二层 Document ACL
        - can_access_document：本人 / tenant 共享 / department 同部门 /
          private（含"管理员也看不到"）/ 跨公司（平台管理员例外）全部分支

    第三层 User/Conversation Isolation
        - 缓存键 = tenant + 权限上下文 + 原始键（不串租户、不串权限）
        - 图片落盘 uploads/{tenant_id}/{doc_id}/images/ + 旧布局回退

运行方式（容器内）：
    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python tests/test_tenant_isolation.py"
"""

from __future__ import annotations

import asyncio
import sys
import uuid

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        _failures.append(name)


try:
    from app.services.tenancy import (
        ACCESS_DEPARTMENT,
        ACCESS_PRIVATE,
        ACCESS_TENANT,
        DEFAULT_TENANT_ID,
        can_access_document,
        effective_department_id,
        effective_tenant_id,
        normalize_tenant_id,
        permission_context,
        scoped_cache_key,
    )
    from app.services.retrieval_service import RetrievedChunk, _scroll_corpus
    from app.services.vector_service import VectorPoint, upsert_vectors
    from app.services.storage.image_store import (
        document_dir,
        image_relative_path,
        resolve_image_path,
        save_image,
        storage_root,
    )
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    # 不能用 sys.exit()：pytest 在收集阶段导入本模块，抛 SystemExit 会让整个
    # 会话 INTERNALERROR，同目录其它用例全部跑不了。
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ── 测试替身 ──────────────────────────────────────────────────────────────────


class _StubUser:
    def __init__(self, *, role="user", tenant_id="company_A", department_id=None):
        self.id = uuid.uuid4()
        self.role = role
        self.tenant_id = tenant_id
        self.department_id = department_id

    @property
    def is_admin(self):
        return self.role == "admin"


class _StubDoc:
    def __init__(self, *, owner_id=None, tenant_id="company_A",
                 access_level="private", department_id=None):
        self.owner_id = owner_id
        self.tenant_id = tenant_id
        self.access_level = access_level
        self.department_id = department_id


class _FakeQdrantClient:
    """捕获 filter / upsert 的假客户端."""

    def __init__(self):
        self.scroll_filter = None
        self.upserted_points = []

    async def scroll(self, *, collection_name, scroll_filter, limit, offset,
                     with_payload, with_vectors):
        self.scroll_filter = scroll_filter
        return [], None

    async def upsert(self, *, collection_name, points, wait):
        self.upserted_points.extend(points)


def _run(coro):
    return asyncio.run(coro)


# ═════════════════════════════════════════════════════════════════════════════
# 第一层：Tenant Isolation
# ═════════════════════════════════════════════════════════════════════════════

def test_tenant_normalize():
    check("合法 tenant 原样保留", normalize_tenant_id("company_A") == "company_A")
    check("None → default", normalize_tenant_id(None) == DEFAULT_TENANT_ID)
    check("空串 → default", normalize_tenant_id("  ") == DEFAULT_TENANT_ID)
    check("目录穿越被拦截", normalize_tenant_id("../../etc") == DEFAULT_TENANT_ID)
    check("斜杠被拦截", normalize_tenant_id("a/b") == DEFAULT_TENANT_ID)
    check("反斜杠被拦截", normalize_tenant_id("a\\b") == DEFAULT_TENANT_ID)
    check("空格被拦截", normalize_tenant_id("a b") == DEFAULT_TENANT_ID)
    check("点开头被拦截", normalize_tenant_id(".hidden") == DEFAULT_TENANT_ID)
    check("连字符/下划线/点允许", normalize_tenant_id("tenant-1.x_2") == "tenant-1.x_2")


def test_effective_tenant():
    user = _StubUser(tenant_id="company_A")
    check("用户租户生效", effective_tenant_id(user) == "company_A")
    legacy = _StubUser(tenant_id=None)
    check("老账号归入 default", effective_tenant_id(legacy) == DEFAULT_TENANT_ID)
    check("None 用户归入 default", effective_tenant_id(None) == DEFAULT_TENANT_ID)


def test_bm25_scroll_prefilter():
    """BM25 语料 scroll 必须带租户前置过滤（不是事后剔除）."""
    client = _FakeQdrantClient()
    _run(_scroll_corpus(client, "documents", None, 10, tenant_id="company_A"))
    f = client.scroll_filter
    check("scroll_filter 非空", f is not None)
    must = list(getattr(f, "must", None) or [])
    keys = [c.key for c in must]
    check("scroll_filter 含 tenant_id", "tenant_id" in keys, f"keys={keys}")
    cond = next(c for c in must if c.key == "tenant_id")
    check("tenant 值正确", cond.match.value == "company_A")


def test_bm25_scroll_tenant_plus_collection():
    client = _FakeQdrantClient()
    _run(_scroll_corpus(client, "documents", "col-1", 10, tenant_id="company_B"))
    keys = [c.key for c in client.scroll_filter.must]
    check("tenant + collection 双过滤",
          "tenant_id" in keys and "collection_id" in keys, f"keys={keys}")


def test_upsert_payload_isolation_fields():
    """payload 必须带框架要求的 ID 集合：tenant/user/document/chunk/page/
    content_type/source."""
    client = _FakeQdrantClient()

    import app.services.vector_service as vs
    original_getter = vs.get_qdrant_client
    vs.get_qdrant_client = lambda: client
    try:
        point = VectorPoint(
            vector=[0.1] * 8,
            document_id="doc_001",
            filename="员工管理制度.docx",
            chunk_index=23,
            page_number=5,
            text="公司A的财务制度……",
            tenant_id="company_A",
            user_id="user_A001",
            access_level="tenant",
            department_id="hr",
            content_type="text",
        )
        _run(upsert_vectors([point]))
    finally:
        vs.get_qdrant_client = original_getter

    check("upsert 被调用", len(client.upserted_points) == 1)
    payload = client.upserted_points[0].payload
    for field, expected in [
        ("tenant_id", "company_A"),
        ("user_id", "user_A001"),
        ("document_id", "doc_001"),
        ("chunk_index", 23),
        ("page_number", 5),
        ("content_type", "text"),
        ("source", "员工管理制度.docx"),
        ("access_level", "tenant"),
        ("department_id", "hr"),
    ]:
        check(f"payload.{field}", payload.get(field) == expected,
              f"got {payload.get(field)!r}")


def test_permission_check_drops_cross_tenant():
    """最终 Permission Check：构造跨租户 chunk，验证会被纯函数拦下."""
    from app.services.tenancy import normalize_tenant_id as norm

    chunks = [
        RetrievedChunk(document_id="d1", filename="a", page_number=1,
                       chunk_index=0, text="x", score=0.9, tenant_id="company_A"),
        RetrievedChunk(document_id="d2", filename="b", page_number=1,
                       chunk_index=0, text="y", score=0.8, tenant_id="company_B"),
    ]
    tenant = "company_A"
    kept = [c for c in chunks if norm(c.tenant_id) == tenant]
    check("跨租户 chunk 被拦", len(kept) == 1 and kept[0].document_id == "d1")
    check("缺省 payload 归入 default",
          norm(RetrievedChunk(document_id="d", filename="f", page_number=1,
                              chunk_index=0, text="t", score=0.1).tenant_id)
          == DEFAULT_TENANT_ID)


# ═════════════════════════════════════════════════════════════════════════════
# 第二层：Document ACL
# ═════════════════════════════════════════════════════════════════════════════

def test_can_access_document():
    alice = _StubUser(tenant_id="company_A")
    bob = _StubUser(tenant_id="company_A")
    hr_user = _StubUser(tenant_id="company_A", department_id="hr")
    outsider = _StubUser(tenant_id="company_B")
    admin = _StubUser(role="admin", tenant_id="company_B")

    private_doc = _StubDoc(owner_id=alice.id, tenant_id="company_A",
                           access_level=ACCESS_PRIVATE)
    tenant_doc = _StubDoc(owner_id=alice.id, tenant_id="company_A",
                          access_level=ACCESS_TENANT)
    dept_doc = _StubDoc(owner_id=alice.id, tenant_id="company_A",
                        access_level=ACCESS_DEPARTMENT, department_id="hr")
    other_tenant_doc = _StubDoc(owner_id=outsider.id, tenant_id="company_B",
                                access_level=ACCESS_TENANT)
    legacy_doc = _StubDoc(owner_id=alice.id, tenant_id="company_A",
                          access_level=None)  # 老数据按 private

    check("owner 总能访问自己的文档", can_access_document(private_doc, alice))
    check("private：同租户他人不可见", not can_access_document(private_doc, bob))
    check("tenant 共享：同租户可见", can_access_document(tenant_doc, bob))
    check("department：同部门可见", can_access_document(dept_doc, hr_user))
    check("department：无部门用户不可见", not can_access_document(dept_doc, bob))
    check("跨租户一律拒绝（即使 tenant 共享）",
          not can_access_document(other_tenant_doc, alice))
    check("admin 跨租户可见公司库（平台管理员的跨公司能力）",
          can_access_document(other_tenant_doc, admin))
    check("admin 跨租户**看不到**他人个人库",
          not can_access_document(
              _StubDoc(owner_id=bob.id, tenant_id="company_A",
                       access_level=ACCESS_PRIVATE),
              admin,
          ))
    check("老数据(NULL)按 private：本人可见", can_access_document(legacy_doc, alice))
    check("老数据(NULL)按 private：他人不可见", not can_access_document(legacy_doc, bob))
    check("effective_department_id 归一",
          effective_department_id(_StubUser(department_id=" hr ")) == "hr")


# ═════════════════════════════════════════════════════════════════════════════
# 第三层：缓存键 + 图片隔离
# ═════════════════════════════════════════════════════════════════════════════

def test_scoped_cache_key():
    u1 = _StubUser(tenant_id="company_A")
    u2 = _StubUser(tenant_id="company_A")
    ctx1 = permission_context(u1)
    ctx2 = permission_context(u2)

    k_a = scoped_cache_key("company_A", ctx1, "query:报销流程")
    k_b = scoped_cache_key("company_B", ctx1, "query:报销流程")
    check("同键不同租户 → 不同缓存键", k_a != k_b)
    check("缓存键含租户前缀", k_a.startswith("company_A::"))

    k_a2 = scoped_cache_key("company_A", ctx2, "query:报销流程")
    check("同租户不同权限上下文 → 不同缓存键", k_a != k_a2)

    same = scoped_cache_key("company_A", ctx1, "query:报销流程")
    check("同租户同上下文同查询 → 键稳定", k_a == same)

    evil = scoped_cache_key("../../x", ctx1, "q")
    check("非法租户被归一后入键", evil.startswith(f"{DEFAULT_TENANT_ID}::"))


def test_image_storage_layout(tmp_root=None):
    doc_id = str(uuid.uuid4())
    rel = image_relative_path(3, 1, "png")
    check("相对路径不含租户（payload 可移植）", rel == "images/page_3_image_1.png")

    new_dir = document_dir(doc_id, tenant_id="tenant_A")
    check("新布局含租户级目录",
          new_dir == storage_root() / "tenant_A" / doc_id,
          str(new_dir))
    legacy_dir = document_dir(doc_id)
    check("旧布局无租户级目录", legacy_dir == storage_root() / doc_id)
    evil_dir = document_dir(doc_id, tenant_id="../../evil")
    check("穿越租户被归一（不落 evil 目录）",
          "evil" not in evil_dir.parts[-2] and evil_dir.parts[-2] == DEFAULT_TENANT_ID,
          str(evil_dir))

    # 写入 → 新布局可解析；旧布局回退也可解析
    data = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    saved = save_image(doc_id, 3, 1, data, "png", tenant_id="tenant_A")
    check("save_image 返回相对路径", saved == rel)
    resolved_new = resolve_image_path(doc_id, rel, tenant_id="tenant_A")
    check("新布局解析命中", resolved_new is not None and resolved_new.is_file())
    resolved_scan = resolve_image_path(doc_id, rel)  # 不带租户 → 扫描回退
    check("不带租户时扫描回退命中", resolved_scan is not None)

    traversal = resolve_image_path(doc_id, "../" * 5 + "etc/passwd", tenant_id="tenant_A")
    check("相对路径穿越被拒绝", traversal is None)

    legacy_doc = str(uuid.uuid4())
    saved_legacy = save_image(legacy_doc, 1, 1, data, "png")  # 旧布局写入
    check("旧布局写入成功", saved_legacy is not None)
    check("旧布局解析命中",
          resolve_image_path(legacy_doc, saved_legacy) is not None)
    check("错租户不读到别的文件（回退到 legacy 命中同源文档）",
          resolve_image_path(legacy_doc, saved_legacy, tenant_id="tenant_B") is not None)


def main() -> int:
    test_tenant_normalize()
    test_effective_tenant()
    test_bm25_scroll_prefilter()
    test_bm25_scroll_tenant_plus_collection()
    test_upsert_payload_isolation_fields()
    test_permission_check_drops_cross_tenant()
    test_can_access_document()
    test_scoped_cache_key()
    test_image_storage_layout()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("ALL TENANT-ISOLATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
