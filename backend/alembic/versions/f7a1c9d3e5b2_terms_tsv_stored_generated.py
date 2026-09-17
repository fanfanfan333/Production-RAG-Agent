"""chunk terms: 表达式索引 → 存储生成列 (tsvector)

Revision ID: f7a1c9d3e5b2
Revises: l7g8h9i0j1k2
Create Date: 2026-09-16

背景（为什么这不是纯粹的"换个写法"）
────────────────────────────────────
原实现用的是 GIN **表达式索引**：

    CREATE INDEX ix_chunk_terms_fts
        ON document_chunk_terms USING gin (to_tsvector('simple', terms));

它能加速 ``WHERE to_tsvector('simple', terms) @@ q``，但**不能**加速排序：
``ts_rank_cd(to_tsvector('simple', terms), q)`` 是一个普通表达式，PG 会对
每一条命中行重新分词。命中集越大越慢，而且不报错、只是"检索越来越慢"。

实测（20 万条词项行，每行约 250 个中文 bigram，命中 20 万行）：
**单次关键词查询 178 秒**，其中 99% 花在重复分词上。

修复：把 tsvector 物化成**存储生成列**（PG 12+），GIN 索引建在列上。
写侧只在 INSERT 时分词一次；读侧 ``@@`` 与 ``ts_rank_cd`` 都直接读该列。

幂等写法
────────
与 ``Base.metadata.create_all``（应用启动时执行）兼容：全部用 IF NOT EXISTS /
IF EXISTS，且先探测列是否已存在。两处都跑过也不会报错。

锁策略（生产必读）
──────────────────
本迁移的三条 DDL 都需要 ACCESS EXCLUSIVE 锁，其中 ADD COLUMN ... STORED 与
DROP COLUMN 还会重写整表。因此 upgrade()/downgrade() 开头都设了
``SET LOCAL lock_timeout = '5s'``：拿不到锁就快速失败，而不是无限排队把
线上所有读写堵在锁队列后面。失败后重跑即可（DDL 全是幂等的）。

已有大表上执行请安排在低峰期；若 5 秒拿不到锁，迁移会报
``canceling statement due to lock timeout`` 并整体回滚 —— 这是预期行为，
不是数据损坏。
"""
from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "f7a1c9d3e5b2"
down_revision = "l7g8h9i0j1k2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 0. 限时取锁（生产安全阀）。
    #
    #    下面三条 DDL 全都要 ACCESS EXCLUSIVE 锁，而 ``ADD COLUMN ...
    #    GENERATED ... STORED`` 还会**重写整张表**。不加限制的话，这条语句
    #    会排在所有在途读写后面无限等待 —— 一旦等不到，它自己不会超时，
    #    反而把后面新来的查询全堵在锁队列里，表现为整站"卡住"。真到那一步
    #    只能 kill 掉迁移进程，而表已经被部分重写。
    #
    #    设了 lock_timeout 之后语义变成"拿不到锁就 5 秒后失败退出"：
    #    迁移失败可以重跑，业务被拖垮不能重来。宁可失败，不可阻塞。
    #
    #    SET LOCAL 的作用域是当前事务 —— alembic 默认把整个迁移跑在一个
    #    事务里（env.py 的 ``context.begin_transaction()``），事务结束自动
    #    失效，不会泄漏到后续迁移或连接池里的其他会话。
    #
    #    运维提示：本迁移在已有大表上执行时，请安排在低峰期。
    op.execute("SET LOCAL lock_timeout = '5s'")

    # 1. 加存储生成列。GENERATED ... STORED 会**重写整表**并把所有历史行
    #    的 tsvector 一次算好 —— 这一步是 O(表大小)，但只发生一次。
    op.execute(
        """
        ALTER TABLE document_chunk_terms
        ADD COLUMN IF NOT EXISTS terms_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', terms)) STORED
        """
    )

    # 2. 在列上建 GIN 索引（先删旧的表达式索引，避免同一份数据被索引两遍：
    #    旧索引实测占到 197 MB，而它对新查询已经没有任何价值）。
    op.execute("DROP INDEX IF EXISTS ix_chunk_terms_fts")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunk_terms_fts_tsv
        ON document_chunk_terms USING gin (terms_tsv)
        """
    )

    # 3. 让统计信息跟上（不然 planner 还会按旧的选择率估行）。
    op.execute("ANALYZE document_chunk_terms")


def downgrade() -> None:
    # 同样限时取锁：DROP COLUMN 需要 ACCESS EXCLUSIVE，大表上一样会排队。
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("DROP INDEX IF EXISTS ix_chunk_terms_fts_tsv")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunk_terms_fts
        ON document_chunk_terms USING gin (to_tsvector('simple', terms))
        """
    )
    op.execute("ALTER TABLE document_chunk_terms DROP COLUMN IF EXISTS terms_tsv")
