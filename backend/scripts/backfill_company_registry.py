"""一次性、幂等的公司注册表回填脚本（T01 决策 5）.

把「公司」升级为一等实体后，需要把**既有的 5 家租户**一次性裁定进注册表：

    c8111de986583 → 测试公司1   created_by=admin  is_test=true
    cfb08c53677c4 → 测试公司2   created_by=admin  is_test=true
    c309a7cb9f496 → A公司        created_by=NULL   is_test=false
    cfb33b1db5679d → B公司        created_by=NULL   is_test=false
    default       → 不入表（历史/占位租户，list_companies 本就排除 admin）

关键约束
────────
* **只写 ``companies`` 表 + 跟随改名的两家租户同步 ``users.company_name``**；
  **不改** ``documents.tenant_id`` / Qdrant payload / ``document_chunk_terms`` / 向量。
* **幂等**：已存在的租户 → 更新（而非重复插入）；重复运行结果一致。
* 支持 ``--dry-run``：只打印计划、回滚不落库。

用法（容器内）
──────────────
    docker exec -w /app rag_backend python -m scripts.backfill_company_registry --dry-run
    docker exec -w /app rag_backend python -m scripts.backfill_company_registry
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 允许「容器内直接 `python scripts/backfill_company_registry.py`」也能找到 app 包
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


async def _count_companies() -> int:
    from sqlalchemy import func, select

    from app.db.company_models import Company
    from app.db.postgres import get_db_session

    async with get_db_session() as session:
        return int((await session.execute(select(func.count()).select_from(Company))).scalar_one())


async def _run(args: argparse.Namespace) -> int:
    from app.services.company_registry import backfill_from_existing

    report = await backfill_from_existing(
        admin_username=args.admin_username,
        dry_run=args.dry_run,
    )
    print(json.dumps(
        {"dry_run": args.dry_run, "items": report},
        ensure_ascii=False, indent=2,
    ))

    if args.dry_run:
        print("（dry-run：未落库，重复运行结果一致）")
    else:
        total = await _count_companies()
        print(f"companies 表实际行数 = {total}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="回填公司注册表（幂等）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不落库")
    ap.add_argument("--admin-username", default="admin", help="把测试公司归属到该管理员账号")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
