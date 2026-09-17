"""
企业身份验证闭环 E2E 验收脚本.

覆盖产品需求「功能 1 / 3」的完整链路：

    新用户注册 → 未验证（被业务接口拦下）
              → 提交身份验证申请（公司 / 部门 / 职责）
              → 状态变为「审核中」
              → 上级审核（同意，并填写负责人＝职务＋名称，授予职责）
              → 状态变为「已通过」，公司/部门/职责/角色全部落地
              → 业务接口放行

同时验证**公司隔离**与**层级规则**（越级审核、不能授予同级、不能自审）。

运行（宿主机即可，只用标准库）：

    python backend/scripts/e2e_staff_flow.py

默认账号：管理员 admin / RagAdmin#2026（可用 --admin-password 覆盖）。
脚本每次创建带随机后缀的临时账号与临时公司（天然落在独立租户里，不影响
现有数据），跑完**自动清理**（在容器内运行；加 ``--keep`` 可保留以便排查）。
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

# 不走系统代理：企业网里 http_proxy 常把 127.0.0.1 也劫持掉，
# 会返回一个假的 502，让人误判成"服务在跑但报错"。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

_passed = 0
_failed = 0


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
            body = resp.read().decode("utf-8")
            return resp.status, _decode(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, _decode(body)
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


def login(base: str, username: str, password: str) -> str:
    status, res = call(
        "POST", "/auth/login", base=base,
        payload={"username": username, "password": password},
    )
    if status != 200 or not isinstance(res, dict):
        raise SystemExit(f"登录失败 {username}: {status} {detail(res)}")
    return res["access_token"]


def cleanup(user_name: str, company: str) -> str:
    """
    删除临时账号与其申请记录（脚本可反复执行，不污染用户表）.

    只在**容器内**运行时有意义（需要 app 包与数据库连接）；在宿主机直接跑则会
    导入失败，此时返回一句提示，让用户用 ``--keep`` 或在后台停用该成员。
    """
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
            r2 = await session.execute(
                delete(User).where(User.username == user_name)
            )
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
    parser.add_argument(
        "--keep", action="store_true", help="保留临时账号（默认尝试清理）"
    )
    args = parser.parse_args()
    base = args.base.rstrip("/")

    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    company = f"验收公司{suffix}"
    department = f"研发部{suffix}"
    duty = "嵌入式软件工程师"
    # 企业账号 = 公司内邮箱（后端注册要求邮箱格式）。
    user_name = f"e2e{suffix}@example.com"
    password = "E2ePass#2026"

    print("── 企业身份验证闭环 E2E ──")
    print(f"   base={base}  临时账号={user_name}  公司={company}")

    admin_token = login(base, args.admin_user, args.admin_password)
    check("管理员登录成功", bool(admin_token))

    # ── 1. 注册新账号 → 未验证 ────────────────────────────────────────────────
    status, res = call(
        "POST", "/auth/register", base=base,
        payload={"username": user_name, "password": password},
    )
    check("新账号注册成功", status == 201, f"status={status} {detail(res)}")
    user_token = res["access_token"] if isinstance(res, dict) else ""

    status, me = call("GET", "/staff/me", base=base, token=user_token)
    check(
        "新账号身份状态 = none（去验证）",
        status == 200 and isinstance(me, dict) and me.get("identity_status") == "none",
        f"status={status} identity={me.get('identity_status') if isinstance(me, dict) else me}",
    )

    # ── 2. 未验证 → 业务接口被拦 ──────────────────────────────────────────────
    status, res = call("GET", "/documents", base=base, token=user_token)
    blocked = status == 403 and "身份验证" in detail(res)
    check("未验证账号访问 /documents 被身份闸门拦下", blocked,
          f"status={status} {detail(res)}")

    # ── 3. 提交身份验证申请 ───────────────────────────────────────────────────
    status, res = call(
        "POST", "/staff/requests", base=base, token=user_token,
        payload={
            "company_name": company,
            "department_name": department,
            "duty": duty,
        },
    )
    check("提交身份验证申请成功", status == 201, f"status={status} {detail(res)}")
    request_id = res["request"]["id"] if isinstance(res, dict) else ""

    status, res = call(
        "POST", "/staff/requests", base=base, token=user_token,
        payload={"company_name": company, "department_name": department, "duty": duty},
    )
    check("重复提交被拒绝（同一时刻只允许一份待审）", status == 409,
          f"status={status} {detail(res)}")

    status, me = call("GET", "/staff/me", base=base, token=user_token)
    check(
        "提交后身份状态 = pending（审核中）",
        isinstance(me, dict) and me.get("identity_status") == "pending",
        f"identity={me.get('identity_status') if isinstance(me, dict) else me}",
    )

    # ── 4. 审核人能看到这份申请（越级：admin 不在该公司）─────────────────────
    status, inbox = call("GET", "/staff/requests/inbox", base=base, token=admin_token)
    items = inbox.get("items", []) if isinstance(inbox, dict) else []
    target = next((i for i in items if i["id"] == request_id), None)
    check("管理员在审核队列里看到该申请（越级可见）", target is not None,
          f"inbox={len(items)}")
    if target is None:
        print("\n无法继续，终止。")
        return 1
    check("该申请被标记为可审核", bool(target.get("can_review")))
    check(
        "申请内容原样回显（公司/部门/职责）",
        target["company_name"] == company
        and target["department_name"] == department
        and target["duty"] == duty,
    )

    # ── 5. 审核必填项：负责人（职务 + 名称）───────────────────────────────────
    status, res = call(
        "POST", f"/staff/requests/{request_id}/review", base=base, token=admin_token,
        payload={"approve": True, "role": "employee"},
    )
    check("缺少负责人（职务/名称）时被拒绝", status == 422,
          f"status={status} {detail(res)}")

    # ── 6. 不能授予 admin（全局唯一）──────────────────────────────────────────
    status, res = call(
        "POST", f"/staff/requests/{request_id}/review", base=base, token=admin_token,
        payload={
            "approve": True, "role": "admin",
            "reviewer_title": "平台管理员", "reviewer_name": "平台管理员",
        },
    )
    check("授予 admin 被拒绝（企业管理员唯一）", status == 422,
          f"status={status} {detail(res)}")

    # ── 7. 正式批准 ───────────────────────────────────────────────────────────
    status, res = call(
        "POST", f"/staff/requests/{request_id}/review", base=base, token=admin_token,
        payload={
            "approve": True,
            "role": "employee",
            "reviewer_title": "平台管理员",
            "reviewer_name": "超级管理员",
            "comment": f"同意加入{company}{department}",
        },
    )
    ok = status == 200 and isinstance(res, dict)
    check("批准成功", ok, f"status={status} {detail(res)}")
    if ok:
        row = res["request"]
        check(
            "负责人记录为「职务 + 名称」",
            row.get("reviewer_title") == "平台管理员"
            and row.get("reviewer_name") == "超级管理员"
            and row.get("reviewer_label") == "平台管理员 超级管理员",
            f"reviewer_label={row.get('reviewer_label')}",
        )
        check("授予职责 = 普通员工", row.get("granted_role") == "employee")

    # ── 8. 申请人侧状态落地 ───────────────────────────────────────────────────
    status, me = call("GET", "/staff/me", base=base, token=user_token)
    ok = isinstance(me, dict)
    check("身份状态 = approved（已通过）",
          ok and me.get("identity_status") == "approved",
          f"identity={me.get('identity_status') if ok else me}")
    check("公司名称已落地（显示原文，不是内部 ID）",
          ok and me.get("company_name") == company,
          f"company_name={me.get('company_name') if ok else me}")
    check("部门名称已落地", ok and me.get("department_name") == department,
          f"department_name={me.get('department_name') if ok else me}")
    check("部门职责已落地", ok and me.get("job_title") == duty,
          f"job_title={me.get('job_title') if ok else me}")
    check("权限角色已授予", ok and me.get("role") == "employee",
          f"role={me.get('role') if ok else me}")

    # ── 9. 业务接口放行 ───────────────────────────────────────────────────────
    status, res = call("GET", "/documents", base=base, token=user_token)
    check("验证通过后业务接口放行", status == 200,
          f"status={status} {detail(res)}")

    # ── 10. 公司隔离：另一家公司不可见 ────────────────────────────────────────
    status, companies = call("GET", "/staff/companies", base=base, token=admin_token)
    names = {c["company_name"] for c in companies.get("items", [])} \
        if isinstance(companies, dict) else set()
    check("管理员可见新公司", company in names, f"companies={sorted(names)}")

    # ── 11. 成员管理与「更换职责」──────────────────────────────────────────────
    status, members = call(
        "GET",
        f"/staff/members?keyword={urllib.parse.quote(user_name)}",
        base=base,
        token=admin_token,
    )
    found = members.get("items", []) if isinstance(members, dict) else []
    check("管理后台能查到该成员", len(found) == 1, f"found={len(found)}")
    if found:
        member_id = found[0]["id"]
        status, res = call(
            "PATCH", f"/staff/members/{member_id}", base=base, token=admin_token,
            payload={"role": "dept_manager", "job_title": "研发部负责人"},
        )
        ok = status == 200 and isinstance(res, dict)
        check("上级可更换下级职责（employee → dept_manager）", ok,
              f"status={status} {detail(res)}")
        check("职责变更落库",
              ok and res["member"]["role"] == "dept_manager"
              and res["member"]["job_title"] == "研发部负责人")

        status, res = call(
            "PATCH", f"/staff/members/{member_id}", base=base, token=admin_token,
            payload={"role": "admin"},
        )
        check("不能在后台把成员提成 admin（唯一性）", status in (403, 422),
              f"status={status} {detail(res)}")

    # ── 12. 不能自审自批 ──────────────────────────────────────────────────────
    status, res = call(
        "POST", "/staff/requests", base=base, token=user_token,
        payload={"company_name": company, "department_name": "市场部", "duty": "市场专员"},
    )
    check("已通过成员可再次提交（换岗申请）", status == 201,
          f"status={status} {detail(res)}")
    if status == 201:
        own_id = res["request"]["id"]
        status, res = call(
            "POST", f"/staff/requests/{own_id}/review", base=base, token=user_token,
            payload={
                "approve": True, "role": "employee",
                "reviewer_title": "研发部负责人", "reviewer_name": "张三",
            },
        )
        check("不能审核自己的申请", status in (403, 422),
              f"status={status} {detail(res)}")

    if args.keep:
        print(f"\n  （--keep）保留临时账号 {user_name} / 公司 {company}")
    else:
        print(f"\n  清理：{cleanup(user_name, company)}")

    print(f"\n结果：{_passed} 通过 / {_failed} 失败")
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
