"""签发一个开发用 JWT（容器内运行）.

用途：端到端联调时不想反复走登录页，直接给 admin/指定用户签一个 token。

    docker exec -e HOME=/tmp rag_backend python /app/scripts/mint_token.py admin

注意：只在受信任的开发环境使用；它绕过密码校验，属于后门脚本，不参与生产部署。
"""
import asyncio
import sys

USERNAME = sys.argv[1] if len(sys.argv) > 1 else "admin"


async def main() -> None:
    from sqlalchemy import select

    from app.db.postgres import get_db_session
    from app.db.user_models import User
    from app.services.auth_service import create_access_token

    async with get_db_session() as s:
        user = (
            await s.execute(select(User).where(User.username == USERNAME).limit(1))
        ).scalar_one_or_none()

    if user is None:
        print("NO_SUCH_USER")
        return

    print(create_access_token(user))


asyncio.run(main())
