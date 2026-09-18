"""e2e 脚本共用的「删干净」助手：**Qdrant 向量 + PostgreSQL 行 + 落盘产物** 一起清.

为什么必须有这个模块
────────────────────
e2e 脚本收尾时把 ``Document`` 行直接 ``delete()`` 掉，行是没了，**Qdrant 里的向量
还在**。实测后果（本仓库生产库）：PG 只剩 9 份文档，Qdrant 却躺着 214 个点、
分布在 186 个 document_id 上 —— 约 95% 是已删文档的孤儿向量。

它们不是"占点存储"那么无害：

  * 检索的候选池是 **先 ANN 再按 PG 可见性过滤**。以平台管理员（``platform_wide``，
    不做租户过滤）为例，实测 ANN top-20 里只有 **4 条**属于现存文档，**16 条**
    是孤儿 —— 候选名额被已删文档吃掉 80%，用户自己的分片进不了候选池；
  * 项目自带的 ``POST /eval/run`` 正好以调用者身份检索，于是**评测报出的 Recall
    被这批脏数据压低**，"监控数字"跟着一起失真 —— 排查方向会被带偏到"检索算法
    不行"，而真正的问题在清理脚本里；
  * 集合规模、段数、索引开销都按 214 点算，实际有效数据只有 14 点。

因此删除入口只有一处，且顺序不可颠倒（与 ``document_query_service.delete_document``
同一取向：**先删向量，失败就不删行**，宁可留下可重试的完整状态，也不留孤儿）。

用法
────
    from _e2e_purge import purge_documents

    # 按任意 SQLAlchemy 条件删（文件名前缀 / owner 等）
    await purge_documents(Document.filename.like("purge_mdel%"))
    await purge_documents(Document.owner_id.in_(uids))

    # 已知 id 时更直接
    await purge_documents_by_ids(doc_ids)

返回 ``{"documents": n, "vectors_batch": m, "images": k}``，便于脚本打印与断言。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

# 与本目录其它脚本同一约定（见 tests/_module_skip.py）：直接 `python scripts/xxx.py`
# 时脚本目录已在 sys.path[0]；但被拷到 /tmp 再跑、或从别处 import 时不一定在，
# 这里显式兜一次，避免"同一份脚本换个调用方式就 ImportError"。
import sys

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


async def purge_documents_by_ids(document_ids: Iterable[str]) -> dict:
    """
    按文档 id 清理：**先 Qdrant，后 PG，最后磁盘**。

    顺序为什么不能反：先删 PG 行再删向量，一旦中间失败就永久留下孤儿向量
    （这正是本仓库 186 个孤儿 id 的成因）；反过来的失败方向是"向量删了但行还在"
    —— 下一次重试仍能删掉，是**可收敛**的坏状态。
    """
    from sqlalchemy import delete, select

    from app.db.models import Document
    from app.db.postgres import get_db_session
    from app.services.vector_service import delete_by_document_ids

    ids = [str(i) for i in document_ids if i]
    if not ids:
        return {"documents": 0, "vectors_batch": 0, "images": 0}

    # 取一份 tenant 以便清理磁盘上 uploads/{tenant}/{document_id} 的归档
    tenants: dict[str, str | None] = {}
    async with get_db_session() as session:
        rows = (
            await session.execute(
                select(Document.id, Document.tenant_id).where(
                    Document.id.in_(ids)
                )
            )
        ).all()
        tenants = {str(r[0]): r[1] for r in rows}

    # 1) Qdrant（一次批量请求；向量不存在时也不报错）
    n_vec = await delete_by_document_ids(ids)

    # 2) PG 行
    async with get_db_session() as session:
        result = await session.execute(delete(Document).where(Document.id.in_(ids)))
        n_docs = result.rowcount or 0

    # 3) 落盘产物（best-effort：磁盘清理失败不该让清理脚本整体失败）
    n_imgs = 0
    try:
        from app.services.storage import delete_document_images

        for doc_id, tenant_id in tenants.items():
            try:
                delete_document_images(doc_id, tenant_id=tenant_id)
                n_imgs += 1
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    return {"documents": n_docs, "vectors_batch": n_vec, "images": n_imgs}


async def purge_documents(*criteria) -> dict:
    """按条件先查出 id，再交给 :func:`purge_documents_by_ids`。"""
    from sqlalchemy import select

    from app.db.models import Document
    from app.db.postgres import get_db_session

    if not criteria:
        # 无条件 = 清空全库。这几乎总是调用方写错了（漏传条件），
        # 与 retrieval 的 fail-closed 同一取向：不做破坏性默认动作。
        raise ValueError("purge_documents() 需要至少一个条件，拒绝无条件清空")

    async with get_db_session() as session:
        ids = [
            str(r[0])
            for r in (await session.execute(select(Document.id).where(*criteria))).all()
        ]
    return await purge_documents_by_ids(ids)


async def delete_vectors_for_documents(document_ids: Iterable[str]) -> int:
    """
    只清 Qdrant 向量，不动 PG —— 给"自己管事务"的调用方用.

    有些 e2e 脚本在一个 ``async with get_db_session()`` 里同时删多张表并统一
    commit；此时再走 :func:`purge_documents_by_ids`（它自己开会话、自己删 PG 行）
    会和外层事务争同一批行。给它们一个**只做向量侧**的入口，就能在同一个
    事务块里"先清向量、再删行"，既不留孤儿，也不引入跨会话竞争。
    """
    from app.services.vector_service import delete_by_document_ids

    ids = [str(i) for i in document_ids if i]
    if not ids:
        return 0
    return await delete_by_document_ids(ids)


__all__ = [
    "delete_vectors_for_documents",
    "purge_documents",
    "purge_documents_by_ids",
]
