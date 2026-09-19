"""一次性、幂等地修正 B公司 注册表行的 tenant_id 错别字（T06 Bug 修复）.

背景
────
回填常量曾把 B公司 的 tenant_id 误写为 ``cfb33b1db5679d``（多一个 'b'），
而 users / documents 里 B公司 的真实租户是 ``cf33b1db5679d``。后果：
B公司 的成员与文档在注册表里查不到公司，文档徽标的公司名退化为 tenant_id，
且「B公司」这个名字被一条空壳注册行占用（无法按名解析到真实租户）。

本脚本把注册行纠正为真实 tenant_id（主键更新，无其他表 FK 引用
``companies.tenant_id``，已全库 grep 确认）。**不触碰** users / documents /
向量数据。

幂等
────
* 错行存在且正确行不存在 → UPDATE 主键纠正；
* 错行与正确行同时存在（不应出现）→ 报错退出，人工处理；
* 只有正确行（已修过 / 新环境）→ noop。

用法（容器内）
──────────────
    docker exec -w /app rag_backend python -m scripts.fix_b_company_tenant_id --dry-run
    docker exec -w /app rag_backend python -m scripts.fix_b_company_tenant_id
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

WRONG_TENANT_ID = "cfb33b1db5679d"
RIGHT_TENANT_ID = "cf33b1db5679d"


async def _run(dry_run: bool) -> int:
    from sqlalchemy import func, select

    from app.db.company_models import Company
    from app.db.postgres import get_db_session
    # companies.created_by 外键指向 users.id：flush 排表时需要 users 进入 metadata
    from app.db.user_models import User as _User  # noqa: F401

    async with get_db_session() as session:
        wrong = await session.get(Company, WRONG_TENANT_ID)
        right = await session.get(Company, RIGHT_TENANT_ID)

        if right is not None and wrong is None:
            print(f"noop：注册行已是正确的 tenant_id={RIGHT_TENANT_ID}（{right.display_name}）")
            return 0
        if wrong is not None and right is not None:
            print(
                "ERROR：错行与正确行同时存在，需人工合并：\n"
                f"  wrong={wrong.tenant_id}({wrong.display_name})\n"
                f"  right={right.tenant_id}({right.display_name})"
            )
            return 1
        if wrong is None:
            print(f"noop：错行 {WRONG_TENANT_ID} 不存在")
            return 0

        print(
            f"计划：companies.tenant_id {WRONG_TENANT_ID} → {RIGHT_TENANT_ID} "
            f"（display_name={wrong.display_name}，其余列不变）"
        )
        if dry_run:
            print("（dry-run：未落库）")
            await session.rollback()
            return 0

        wrong.tenant_id = RIGHT_TENANT_ID
        wrong.updated_at = func.now()
        await session.flush()
        await session.refresh(wrong)
        print(f"已修正：{wrong.tenant_id} display_name={wrong.display_name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="修正 B公司注册行的 tenant_id 错别字（幂等）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不落库")
    args = ap.parse_args()
    return asyncio.run(_run(args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
