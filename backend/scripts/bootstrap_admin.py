"""
重建本地管理员账号（邮箱校验上线前的遗留用户名）。

为什么需要这个脚本
------------------
``POST /auth/register`` 现在的规则是「**第一个**注册用户自动成为 admin」，
且**强制邮箱格式**（见 ``auth_service._validate_registration``）。
而历史管理员用户名 ``admin`` 并不是邮箱 —— 一旦 Postgres 卷被清空，
就再也没有任何 API 路径能把它建回来，E2E 脚本（默认 admin / RagAdmin#2026）
会全部 401。

登录侧是明确保留兼容的：``authenticate_user`` 特意注明
「登录**不做**邮箱格式校验：存量的非邮箱账号（admin、lisi…）仍要能登」。
所以直接按既有格式写入一条 users 记录即可，与手工注册的账号完全等价。

用法
----
⚠️ 镜像里**没有** ``/app/scripts`` 目录（Dockerfile 只 COPY 了 app/、alembic/、
   alembic.ini），所以必须先复制到 /app 根下再执行 —— 放 scripts/ 会
   ``No such file or directory``::

    docker cp backend/scripts/bootstrap_admin.py rag_backend:/app/bootstrap_admin.py
    docker exec -u root -e HOME=/tmp rag_backend python /app/bootstrap_admin.py --dry-run
    docker exec -u root -e HOME=/tmp rag_backend python /app/bootstrap_admin.py
    # 可选参数：--username ops --password 'S3cret!' --reset-password --role kb_admin

脚本是幂等的：账号已存在时默认不动它，只报告。
"""

import argparse
import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import select  # noqa: E402

from app.db.postgres import get_db_session  # noqa: E402
from app.db.user_models import User  # noqa: E402
from app.services.auth_service import hash_password  # noqa: E402


async def _find(session, username: str):
    return await session.scalar(select(User).where(User.username == username).limit(1))


async def bootstrap(
    username: str,
    password: str,
    *,
    role: str = User.ROLE_ADMIN,
    tenant_id: str = "default",
    reset_password: bool = False,
    dry_run: bool = False,
) -> int:
    if role not in User.VALID_ROLES:
        print(f"❌ 非法角色 '{role}'；可选：{sorted(User.VALID_ROLES)}")
        return 2

    async with get_db_session() as session:
        existing = await _find(session, username)

        if existing is not None and not reset_password:
            print(
                f"ℹ️  账号已存在，未改动：username={existing.username} "
                f"role={existing.role} is_active={existing.is_active} id={existing.id}"
            )
            return 0

        if existing is not None:
            print(f"🔄 重置密码：username={existing.username} id={existing.id}")
            if not dry_run:
                existing.password_hash = hash_password(password)
            return 0

        if dry_run:
            print(f"🔍 [dry-run] 将创建：username={username} role={role} tenant_id={tenant_id}")
            return 0

        user = User(
            id=uuid.uuid4(),
            username=username,
            password_hash=hash_password(password),
            display_name=username,
            role=role,
            tenant_id=tenant_id,
        )
        session.add(user)
        await session.flush()
        print(f"✅ 已创建：username={user.username} role={user.role} id={user.id}")
        return 0


async def _main(args) -> int:
    return await bootstrap(
        args.username,
        args.password,
        role=args.role,
        tenant_id=args.tenant_id,
        reset_password=args.reset_password,
        dry_run=args.dry_run,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="重建本地管理员账号")
    p.add_argument("--username", default="admin", help="账号名（默认 admin）")
    p.add_argument("--password", default="RagAdmin#2026", help="密码")
    p.add_argument("--role", default=User.ROLE_ADMIN, help=f"角色（默认 {User.ROLE_ADMIN}）")
    p.add_argument("--tenant-id", default="default", help="租户 ID（默认 default）")
    p.add_argument("--reset-password", action="store_true", help="账号已存在时重置其密码")
    p.add_argument("--dry-run", action="store_true", help="只打印将要做的事，不写库")
    return asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
