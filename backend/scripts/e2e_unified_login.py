"""
企业统一身份登录（邮箱 + 已登记职责 + 密码）E2E 验收脚本.

覆盖产品链路：

    新账号注册（未登记职责）
      → 统一身份登录被拒（403 尚未登记职责）
      → 提交身份验证申请（含部门职责）
      → 上级审核通过，职责写入账号
      → 统一身份登录：
          职责一致        → 放行
          职责不一致      → 拒绝
          密码错          → 拒绝
          邮箱换大小写    → 放行（大小写不敏感）
          职责带首尾空格  → 放行（比对前 trim）

运行（容器内，需要 app 包与数据库）：

    docker exec -e HOME=/tmp rag_backend python /app/scripts/e2e_unified_login.py

脚本每次创建带随机后缀的临时账号与临时公司，跑完**自动清理**（``--keep`` 保留）。
"""

from __future__ import annotations

import argparse
import json
import random
import string
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE_DEFAULT = "http://127.0.0.1:8000"

# 不走系统代理：企业网里 http_proxy 常把 127.0.0.1 也劫持掉，会返回假的 502。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

_passed = 0
_failed = 0

DUTY = "嵌入式软件工程师"


def check(label: str, condition: bool, extra: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  PASS  {label}")
    else:
        _failed += 1
        print(f"  FAIL  {label} {extra}")


def call(
    method: str,
    path: str,
    *,
    base: str,
    token: str | None = None,
    payload: dict | None = None,
) -> tuple[int, dict | list | str]:
    url = f"{base}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with _OPENER.open(req, timeout=30) as resp:
            return resp.status, _decode(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read().decode("utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


def _decode(body: str):
    try:
        return json.loads(body)
    except Exception:  # noqa: BLE001
        return body


def detail(res: object) -> str:
    if isinstance(res, dict):
        return str(res.get("detail") or res)[:160]
    return str(res)[:160]


def unified_login(base: str, username: str, duty: str, password: str):
    return call(
        "POST",
        "/auth/unified-login",
        base=base,
        payload={"username": username, "duty": duty, "password": password},
    )


def login(base: str, username: str, password: str) -> str:
    status, res = call(
        "POST",
        "/auth/login",
        base=base,
        payload={"username": username, "password": password},
    )
    if status != 200 or not isinstance(res, dict):
        raise SystemExit(f"登录失败 {username}: {status} {detail(res)}")
    return res["access_token"]


def cleanup(user_name: str, company: str) -> str:
    """删除临时账号与其申请记录（容器内运行时生效）。"""
    try:
        import asyncio

        from sqlalchemy import delete

        from app.db.postgres import get_db_session
        from app.db.staff_models import StaffRequest
        from app.db.user_models import User
    except Exception as exc:  # noqa: BLE001
        return f"跳过（宿主机无 app 包：{type(exc).__name__}）"

    async def _run() -> tuple[int, int]:
        async with get_db_session() as session:
            r1 = await session.execute(
                delete(StaffRequest).where(
                    (StaffRequest.applicant_username == user_name)
                    | (StaffRequest.company_name == company)
                )
            )
            r2 = await session.execute(delete(User).where(User.username == user_name))
            await session.commit()
            return r1.rowcount or 0, r2.rowcount or 0

    try:
        requests_deleted, users_deleted = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        return f"失败（{type(exc).__name__}: {exc}）"
    return f"已删除 申请={requests_deleted} 账号={users_deleted}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_DEFAULT)
    parser.add_argument("--admin-user", default="admin")
    parser.add_argument("--admin-password", default="RagAdmin#2026")
    parser.add_argument("--keep", action="store_true", help="保留临时账号")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    company = f"统一身份公司{suffix}"
    department = f"研发部{suffix}"
    user_name = f"unified{suffix}@example.com"
    password = "Unified#2026"

    print("── 企业统一身份登录 E2E ──")
    print(f"   base={base}  临时账号={user_name}  职责={DUTY}")

    admin_token = login(base, args.admin_user, args.admin_password)
    check("管理员登录成功", bool(admin_token))

    # ── 1. 注册（尚未登记职责）──────────────────────────────────────────────
    status, res = call(
        "POST",
        "/auth/register",
        base=base,
        payload={"username": user_name, "password": password},
    )
    check("新账号注册成功（邮箱）", status == 201, f"status={status} {detail(res)}")

    # ── 2. 未登记职责 → 拒绝 ────────────────────────────────────────────────
    status, res = unified_login(base, user_name, DUTY, password)
    check("未登记职责时统一身份登录被拒（403）", status == 403, f"status={status} {detail(res)}")

    # ── 3. 提交身份验证申请 ────────────────────────────────────────────────
    user_token = login(base, user_name, password)
    status, res = call(
        "POST",
        "/staff/requests",
        base=base,
        token=user_token,
        payload={
            "company_name": company,
            "department_name": department,
            "duty": DUTY,
        },
    )
    check("提交身份验证申请成功", status == 201, f"status={status} {detail(res)}")
    request_id = res["request"]["id"] if status == 201 else None

    # ── 4. 审核通过（职责落库）──────────────────────────────────────────────
    status, inbox = call("GET", "/staff/requests/inbox", base=base, token=admin_token)
    items = inbox.get("items", []) if isinstance(inbox, dict) else []
    mine = next((i for i in items if i.get("applicant_username") == user_name), None)
    check("管理员在审核队列看到该申请", mine is not None)
    if request_id is None and mine:
        request_id = mine["id"]
    if not request_id:
        print("\n无法继续：申请未创建")
        print(f"\n  清理：{cleanup(user_name, company)}")
        print(f"\n结果：{_passed} 通过 / {_failed} 失败")
        return 1

    status, res = call(
        "POST",
        f"/staff/requests/{request_id}/review",
        base=base,
        token=admin_token,
        payload={
            "approve": True,
            "role": "employee",
            "department_name": department,
            "duty": DUTY,
            "reviewer_title": "研发部负责人",
            "reviewer_name": "李四",
        },
    )
    check("审核通过（写入职责）", status == 200, f"status={status} {detail(res)}")

    # ── 5. 统一身份登录：核心判定 ───────────────────────────────────────────
    status, res = unified_login(base, user_name, DUTY, password)
    check("职责一致 → 放行", status == 200, f"status={status} {detail(res)}")
    if status == 200:
        check("返回的令牌可用", bool(res.get("access_token")))
        check(
            "返回账号即为该企业名称",
            res.get("user", {}).get("username") == user_name,
            f"user={res.get('user', {}).get('username')}",
        )

    status, res = unified_login(base, user_name, "市场专员", password)
    check("职责不一致 → 拒绝（401）", status == 401, f"status={status} {detail(res)}")

    status, res = unified_login(base, user_name, DUTY, "WrongPass#2026")
    check("密码错误 → 拒绝（401）", status == 401, f"status={status} {detail(res)}")

    status, res = unified_login(base, user_name.upper(), DUTY, password)
    check("企业名称大小写不敏感 → 放行", status == 200, f"status={status} {detail(res)}")

    status, res = unified_login(base, user_name, f"  {DUTY}  ", password)
    check("职责首尾空格容忍 → 放行", status == 200, f"status={status} {detail(res)}")

    status, res = unified_login(base, "nobody@nowhere.com", DUTY, password)
    check("不存在的账号 → 拒绝（401）", status == 401, f"status={status} {detail(res)}")

    status, res = call(
        "POST",
        "/auth/unified-login",
        base=base,
        payload={"username": user_name, "duty": "  ", "password": password},
    )
    check("职责为空 → 422（表单校验）", status == 422, f"status={status} {detail(res)}")

    # ── 6. 普通登录仍然可用（两条链路并存）──────────────────────────────────
    status, _ = call(
        "POST",
        "/auth/login",
        base=base,
        payload={"username": user_name, "password": password},
    )
    check("普通邮箱密码登录仍可用", status == 200, f"status={status}")

    if args.keep:
        print(f"\n  （--keep）保留临时账号 {user_name} / 公司 {company}")
    else:
        print(f"\n  清理：{cleanup(user_name, company)}")

    print(f"\n结果：{_passed} 通过 / {_failed} 失败")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
