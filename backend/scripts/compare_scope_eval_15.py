"""#15 逐例 before/after 对照：证明「测试公司隔离」只动了受影响的那一个用例.

⚠️ 为什么不能只换 scope
────────────────────
#13 之后 ``retrieve_chunks`` 内部**无条件**对平台管理员剔除测试公司
（``exclude_test_tenants``）。因此「传 request_scope」并不等于「改前行为」——
两次都会被内部剔除。要真正复现改前，必须把那次内部剔除也一并关掉。

对照口径（同一份金标、同一条检索链路）：

    before = 旧口径：``exclude_test_tenants`` 置为恒等（等价 #13 之前的代码）
             + admin 的 ``request_scope``（含测试公司）
    after  = 新口径：真实 ``exclude_test_tenants`` + admin 的 ``content_scope``

对 16 条正例逐例算 命中@10 与首个相关块的排名（rank），对 3 条负例算最高分
与是否拒答。期望：**只有牵扯测试公司文档的用例（紫罗兰计划）发生变化**，
其余逐例 rank 完全相同 —— 把「准确率下降」钉到具体某一例、并证明它不是回归。

容器内运行：
    docker cp backend/scripts/compare_scope_eval_15.py rag_backend:/app/scripts/
    docker exec -w /app rag_backend python scripts/compare_scope_eval_15.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import select, text

from app.config import get_settings
from app.db.postgres import get_db_session
from app.db.user_models import User
from app.services import retrieval_service as rs
from app.services.evaluation import item_from_chunk
from app.services.retrieval_service import retrieve_chunks
from app.services.tenancy import content_scope, request_scope

GOLDEN = Path("/app/eval/golden_v1.json")
ADMIN = "admin"


async def _resolve(golden: dict) -> dict[str, list[str]]:
    wanted = {it["filename"] for c in golden["cases"] for it in c["expect"]}
    async with get_db_session() as session:
        rows = (
            await session.execute(
                text("select filename, id::text from documents where filename = any(:n)"),
                {"n": list(wanted)},
            )
        ).all()
    by_name: dict[str, list[str]] = {}
    for filename, doc_id in rows:
        by_name.setdefault(filename, []).append(doc_id)
    return by_name


async def _rank(query: str, scope, rel: set[str]) -> tuple[int | None, bool]:
    chunks = await retrieve_chunks(query, top_k=10, **scope.acl_kwargs())
    for pos, c in enumerate(chunks, 1):
        if item_from_chunk(c).matches(rel):
            return pos, pos <= 10
    return None, False


async def _top(query: str, scope) -> float:
    chunks = await retrieve_chunks(query, top_k=10, **scope.acl_kwargs())
    return max((c.score for c in chunks), default=0.0)


async def main() -> None:
    settings = get_settings()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    by_name = await _resolve(golden)

    async with get_db_session() as session:
        user = (
            await session.execute(select(User).where(User.username == ADMIN))
        ).scalars().first()
    assert user is not None, "admin 账号不存在"

    before_scope = await request_scope(user)
    after_scope = await content_scope(user)
    print(f"before(request_scope).tenant_ids = {sorted(before_scope.tenant_ids or [])}")
    print(f"after (content_scope).tenant_ids = {sorted(after_scope.tenant_ids or [])}")
    print()

    cases = []
    for i, case in enumerate(golden["cases"], 1):
        rel = {
            f"{doc_id}::{it['chunk_index']}"
            for it in case["expect"]
            for doc_id in by_name.get(it["filename"], [])
        }
        cases.append((i, case["query"], rel))

    # ── before 态：关掉内部剔除（等价 #13 之前的代码）────────────────────────────
    async def _identity(tenant_ids, owns_tenant_ids):  # noqa: ANN001
        return tenant_ids, owns_tenant_ids

    original = rs.exclude_test_tenants
    before_rows: list[tuple[int | None, bool]] = []
    try:
        rs.exclude_test_tenants = _identity
        for _, query, rel in cases:
            before_rows.append(await _rank(query, before_scope, rel))
        neg_before = [
            await _top(c["query"], before_scope)
            for c in golden["negative_cases"]
        ]
    finally:
        rs.exclude_test_tenants = original

    # ── after 态：真实代码 + content_scope ──────────────────────────────────────
    after_rows = [await _rank(query, after_scope, rel) for _, query, rel in cases]
    neg_after = [await _top(c["query"], after_scope) for c in golden["negative_cases"]]

    print(f"{'#':>2}  {'case(query)':<32}  {'before rank/hit@10':>19}  {'after rank/hit@10':>18}  rel#  same")
    changed = 0
    for (i, query, rel), (rb, hb), (ra, ha) in zip(cases, before_rows, after_rows):
        same = (rb, hb) == (ra, ha)
        changed += 0 if same else 1
        print(
            f"{i:>2}  {query[:30]:<32}  {str(rb)+'/'+str(hb):>19}  "
            f"{str(ra)+'/'+str(ha):>18}  {len(rel):>4}  {same}"
        )

    print()
    print("── 负例（top_score / 是否拒答）──")
    thr = float(settings.EVIDENCE_GATE_MIN_TOP_SCORE)
    for i, case in enumerate(golden["negative_cases"], 1):
        tb, ta = neg_before[i - 1], neg_after[i - 1]
        print(
            f"{i:>2}  {case['query'][:32]:<32}  before={tb:.4f}(refuse={tb < thr})  "
            f"after={ta:.4f}(refuse={ta < thr})  same={(tb < thr) == (ta < thr)}"
        )

    # 汇总（与 run_eval_baseline 的 recall@10 口径一致）
    b_hit = sum(1 for _, h in before_rows if h) / len(before_rows)
    a_hit = sum(1 for _, h in after_rows if h) / len(after_rows)
    print()
    print(f"recall@10  before={b_hit:.4f}  after={a_hit:.4f}  (逐例变化 {changed}/{len(cases)})")


if __name__ == "__main__":
    asyncio.run(main())
