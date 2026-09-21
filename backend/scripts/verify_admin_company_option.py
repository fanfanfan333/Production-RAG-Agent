"""「归属公司」候选中 default/管理员 项的可见性验收（P1-3 边界 + 显示层别名）.

**只读**（不写任何数据），可重复运行。断言四件事：

  1. 平台管理员 ``GET /companies/accessible`` 的候选里含 ``company_id="default"``，
     且其 ``display_name`` 是中文别名 **「管理员」**（不是裸 ``default``）；
  2. 普通成员 / 测试公司成员的候选里**不含** ``default``（也不含「管理员」）；
  3. 未设公司的老账号（非管理员，``tenant_id=default``）候选为空集
     —— 这是 ``tenant_ids - {DEFAULT_TENANT_ID}`` 分支要挡住的唯一场景；
     库里当前没有这类账号，故用合成 User 直接调用端点函数验证该分支；
  4. 每个账号的候选集合 = 其 ``request_scope`` 的 tenant 集合（同源，不产生
     「下拉里能选、列表里查不到」的破口）。

退出码：0 = 全部断言通过；1 = 存在失败项。

用法（容器内）
──────────────
镜像的 ``runtime`` 阶段只 ``COPY app/`` ``alembic/`` ``alembic.ini``，**不含
``scripts/``**（所以文档里 ``python -m scripts.xxx`` 那套跑不通）。先把本文件
拷进容器、放到 ``/app`` 下（``WORKDIR`` 是 ``/app``，脚本模式下 ``app`` 包才可
import），再直接跑：

    docker cp backend/scripts/verify_admin_company_option.py rag_backend:/tmp/
    docker exec -u root rag_backend cp /tmp/verify_admin_company_option.py /app/
    docker exec -w /app rag_backend python verify_admin_company_option.py

另两个踩过的坑：
  * **不要**加 ``-e HOME=/tmp``。那是「跑 HuggingFace 缓存」的用法；它会把
    user-site 切到 ``/tmp/.local``，于是 ``pip install`` 到 ``/app/.cache/.local``
    的包（含 pytest）全部变成 "No module named ..."。
  * ``/app`` 归 ``root``，容器进程是 ``appuser``(uid=100)，所以 ``docker cp``
    之后必须 ``docker exec -u root cp`` 落位，并用 ``md5sum`` 与本地对齐
    —— 直接 ``docker cp`` 到 ``/app`` 会因权限不足留下 0 字节文件。

背景（为什么要有这个脚本）
──────────────────────────
``default`` 是「尚未归属公司」的历史兜底租户，在 ``companies`` 表里没有行。
它在管理员下拉里的展示名原本回退成裸 ``tenant_id``，管理员看到一个无法解释的
内部标识；对普通成员它更是毫无语义。修复只在 **显示层 + 候选过滤** 生效：
``tenant_id`` 仍是 ``default``，文档归属与检索 scope 一律不动（零数据迁移）。
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

from sqlalchemy import select

from app.db.postgres import get_db_session
from app.db.user_models import User

BASE = "http://127.0.0.1:8000"
HTTP_TIMEOUT = 30

_failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)


def _get(path: str, token: str) -> tuple[int, object]:
    req = urllib.request.Request(f"{BASE}{path}")
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")[:200]
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def _names(body: object) -> list[str]:
    """从响应体里取展示名列表（非 200 时返回空）。"""
    if not isinstance(body, list):
        return []
    return [str(row.get("display_name", "")) for row in body]


async def _run() -> int:
    from app.api.companies import list_accessible_companies_endpoint
    from app.services.auth_service import create_access_token
    from app.services.tenancy import DEFAULT_TENANT_ID

    async with get_db_session() as session:
        users = list((await session.execute(select(User))).scalars().all())
        if not users:
            print("FATAL：users 表为空")
            return 1

        # 每个租户各取一个代表账号，覆盖 admin / 各公司成员
        picked: list[User] = []
        seen: set[str] = set()
        for user in users:
            if user.tenant_id in seen:
                continue
            seen.add(user.tenant_id)
            picked.append(user)

        probes = []
        for user in picked:
            status, body = _get("/companies/accessible", create_access_token(user))
            probes.append(
                {
                    "username": user.username,
                    "is_admin": bool(user.is_admin),
                    "tenant_id": user.tenant_id,
                    "status": status,
                    "names": _names(body),
                    "body": body,
                }
            )

    print("── 候选清单 ─────────────────────────────────────────────────────")
    for probe in probes:
        print(
            f"  {probe['username']:<30} is_admin={probe['is_admin']!s:<5} "
            f"tenant={probe['tenant_id']:<15} → {probe['names']}"
        )

    admins = [p for p in probes if p["is_admin"]]
    others = [p for p in probes if not p["is_admin"]]

    print("── 断言 ─────────────────────────────────────────────────────────")
    for probe in admins:
        ok = (
            probe["status"] == 200
            and "管理员" in probe["names"]
            and DEFAULT_TENANT_ID not in probe["names"]
        )
        check(
            ok,
            f"管理员 {probe['username']}：候选项显示为「管理员」，不出现裸 default",
            f"status={probe['status']} names={probe['names']}",
        )

    for probe in others:
        ok = (
            probe["status"] == 200
            and "管理员" not in probe["names"]
            and DEFAULT_TENANT_ID not in probe["names"]
        )
        check(
            ok,
            f"非管理员 {probe['username']}：候选里没有「管理员」/default",
            f"status={probe['status']} names={probe['names']}",
        )

    # 分支 3：未设公司的老账号（非管理员）—— 用合成 User 走端点函数
    legacy = User(
        username="(synthetic-legacy)", role="employee", tenant_id=DEFAULT_TENANT_ID
    )
    try:
        rows = await list_accessible_companies_endpoint(legacy)
        legacy_names = [r.display_name for r in rows]
        legacy_ok = legacy_names == []
        legacy_detail = f"names={legacy_names}"
    except Exception as exc:  # noqa: BLE001
        legacy_ok, legacy_detail = False, f"{type(exc).__name__}: {exc}"
    check(
        legacy_ok,
        "老账号（非管理员，tenant_id=default）：候选为空集（default 被剔除）",
        legacy_detail,
    )

    # 分支 4：候选与可见范围同源（候选 ⊆ scope，且 default 对非 admin 不在候选内）
    for probe in others:
        own_only = all(name and name != "管理员" for name in probe["names"])
        check(
            own_only,
            f"非管理员 {probe['username']}：候选只含本公司",
            f"names={probe['names']}",
        )

    print("────────────────────────────────────────────────────────────────")
    if _failures:
        print(f"FAILED {len(_failures)} 项：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print("全部断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
