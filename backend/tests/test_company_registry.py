"""
公司注册表（T01）纯单元测试 —— 归一化 / 标识生成 / 校验 / 回填计划.

这些断言**不依赖数据库连接**，可在任何环境快速验证「公司名唯一性判定」与
「tenant_id 与名称解耦」两条最关键的规则：

  * 归一化：NFKC → 去全部空白（含全角空格）→ casefold，使大小写/空格变体
    在唯一约束前就已被判为同一家公司。
  * ``generate_tenant_id``：随机、安全（满足 ``tenancy._TENANT_ID_RE``），
    **不再**由名称派生 —— 这是「改名不改 tenant_id、零迁移」的前提。
  * ``_clean_name``：空 / 纯空白 / 超长一律抛 :class:`CompanyError`。
  * 回填计划：4 家租户、``default`` 不入表、A/B 的 ``created_by`` 为 None。

运行方式（容器内）：
    docker exec -w /app rag_backend python -m pytest tests/test_company_registry.py -q
"""

from __future__ import annotations

import sys

try:
    from app.services.company_registry import (
        DEFAULT_BACKFILL_ASSIGNMENTS,
        CompanyError,
        _clean_name,
        generate_tenant_id,
        normalize_company_name_key,
    )
    from app.services.tenancy import _TENANT_ID_RE, DEFAULT_TENANT_ID

    _IMPORT_OK = True
except ImportError as exc:  # pragma: no cover — 宿主机缺依赖 → 跳过
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")
    _IMPORT_OK = False


def test_normalize_name_key_collapses_case_and_space():
    """大小写 / 首尾空格 / 中间空格 / 全角空格 变体必须归一化到同一个 key。"""
    if not _IMPORT_OK:
        return
    base = normalize_company_name_key("测试公司1")
    assert base == normalize_company_name_key("  测试公司1  ")
    assert base == normalize_company_name_key("测 试 公 司 1")
    assert base == normalize_company_name_key("测　试　公　司　1")  # 全角空格
    # 英文大小写变体
    assert normalize_company_name_key("ACME") == normalize_company_name_key("acme")
    assert normalize_company_name_key("ACME") == normalize_company_name_key("  A C M E ")


def test_normalize_name_key_is_empty_for_blank():
    if not _IMPORT_OK:
        return
    assert normalize_company_name_key(None) == ""
    assert normalize_company_name_key("") == ""
    assert normalize_company_name_key("   ") == ""
    assert normalize_company_name_key("　　") == ""  # 全角空白


def test_generate_tenant_id_is_random_and_safe():
    """随机、安全、互不相同，且满足 tenant_id 字符集（防目录穿越）。"""
    if not _IMPORT_OK:
        return
    ids = {generate_tenant_id() for _ in range(50)}
    assert len(ids) == 50, "50 次生成应无碰撞"
    for tid in ids:
        assert _TENANT_ID_RE.match(tid), f"tenant_id 非法: {tid!r}"
    # 与名称无关：同一名称两次生成结果不同
    assert generate_tenant_id() != generate_tenant_id()


def test_clean_name_rejects_empty_and_overlong():
    if not _IMPORT_OK:
        return

    def _raises(value: str) -> bool:
        try:
            _clean_name(value)
            return False
        except CompanyError:
            return True

    assert _raises("")
    assert _raises("   ")
    assert _raises("x" * 129)
    # 合法边界
    name, key = _clean_name("  A公司  ")
    assert name == "A公司" and key == normalize_company_name_key("A公司")


def test_backfill_plan_shape():
    """回填计划：4 家租户、default 不入表、恰好两家归 admin、A/B 的 created_by 为 None。"""
    if not _IMPORT_OK:
        return
    tenant_ids = [item["tenant_id"] for item in DEFAULT_BACKFILL_ASSIGNMENTS]
    assert len(tenant_ids) == 4
    assert len(set(tenant_ids)) == 4, "回填租户不得重复"
    assert DEFAULT_TENANT_ID not in tenant_ids, "default 历史占位租户不得入表"

    admin_owned = [
        i for i in DEFAULT_BACKFILL_ASSIGNMENTS if i.get("created_by") == "admin"
    ]
    assert sorted(i["tenant_id"] for i in admin_owned) == [
        "c8111de986583",
        "cfb08c53677c4",
    ]
    assert all(i["is_test"] for i in admin_owned), "自建测试公司 is_test 必须为真"

    null_owned = [i for i in DEFAULT_BACKFILL_ASSIGNMENTS if i.get("created_by") is None]
    assert sorted(i["tenant_id"] for i in null_owned) == [
        "c309a7cb9f496",
        "cf33b1db5679d",
    ]
    assert all(not i["is_test"] for i in null_owned), "A/B 公司不是测试公司"


def test_create_company_maps_unique_violation_to_409(monkeypatch):
    """并发重名边界：name_key 唯一约束被触发 → 409「公司已存在」，而不是 500。"""
    if not _IMPORT_OK:
        return

    import asyncio

    import pytest
    from sqlalchemy.exc import IntegrityError

    from app.services import company_registry as cr

    class _StubSession:
        """前置查重不冲突（模拟并发窗口），flush 时唯一约束报错。"""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def scalar(self, *args, **kwargs):
            return None

        async def get(self, *args, **kwargs):
            return None

        def add(self, obj):
            return None

        async def flush(self):
            raise IntegrityError("INSERT companies", {}, Exception("dup key"))

        async def refresh(self, obj):
            return None

    monkeypatch.setattr(cr, "get_db_session", lambda: _StubSession())

    with pytest.raises(CompanyError) as err:
        asyncio.run(cr.create_company(None, "测试公司1"))
    assert err.value.status_code == 409
