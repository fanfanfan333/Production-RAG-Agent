"""
SQLAlchemy ORM models.

All models inherit from Base (defined in app.db.postgres).
Tables are created on startup via Base.metadata.create_all.
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.postgres import Base


class DocumentStatus(str, PyEnum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    ALREADY_EXISTS = "already_exists"


class Document(Base):
    """
    Persists metadata for every uploaded PDF.

    Lifecycle:
        PENDING → PARSING → CHUNKING → EMBEDDING → INDEXING → COMPLETED
                                                            ↘ FAILED
    """

    __tablename__ = "documents"
    # 判重按用户范围：同一文件不同用户各自独立索引（全局唯一会误伤多用户场景）
    __table_args__ = (
        UniqueConstraint("owner_id", "file_hash", name="uq_documents_owner_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="documentstatus", create_type=True),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Progress Tracking
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    embedded_chunks: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    failed_chunks: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    current_stage: Mapped[str | None] = mapped_column(String(64), default="pending", server_default="pending", nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Multi-user ownership & business collection (企业落地第一阶段)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,   # legacy rows uploaded before auth have no owner
        index=True,
    )
    collection_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="SET NULL"),
        nullable=True,   # null = 未分配
        index=True,
    )

    # ── Multi-Tenant 隔离（第一、二层）──────────────────────────────────────
    # tenant_id：文档归属的租户；Qdrant payload 同步写入，检索前置过滤。
    # access_level：private(仅本人) / department(同部门) / tenant(全租户)。
    # department_id：access_level=department 时的归属部门。
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default",
        nullable=False, index=True,
    )
    access_level: Mapped[str] = mapped_column(
        String(20), default="private", server_default="private",
        nullable=False, index=True,
    )
    department_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True,
    )

    # New Multi-Format & OCR Metadata
    file_type: Mapped[str] = mapped_column(String(32), default="pdf", server_default="pdf", nullable=False)
    parser_used: Mapped[str] = mapped_column(String(64), default="PyMuPDF", server_default="PyMuPDF", nullable=False)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    ocr_engine: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extraction_method: Mapped[str] = mapped_column(String(64), default="native", server_default="native", nullable=False)

    # 图片对象统计（部分1+2）：识别到的图片总数 / 建成独立检索对象的图片数。
    # 图片本体落盘在 uploads/{document_id}/images/，Qdrant payload 记录 image_path。
    image_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    image_object_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<Document id={self.id} filename={self.filename!r} "
            f"status={self.status.value} chunks={self.chunk_count}>"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata 系统 / Parent-Child / 关键词腿 —— 规模化的三张表
# ═══════════════════════════════════════════════════════════════════════════════
#
# 三者都是**独立表**而不是 documents 上的新列，理由很实际：本服务启动时用
# ``Base.metadata.create_all`` 建表，它只会创建"不存在的表"，**不会给已存在的表
# 加列**。加列的方案在既有部署上会静默失效（代码写入一个数据库里不存在的列 →
# 运行时报错），而独立表会被 create_all 自动建出来，升级路径零操作。
# （alembic 迁移里同样用 IF NOT EXISTS，两种机制任一先跑都不会冲突。）


class DocumentMetadataRow(Base):
    """
    文档级元数据（与 Document 一对一）.

    为什么放 PG 而不是 Qdrant payload：``outline``（章节大纲）与 ``keywords``
    是**文档级**信息，放进向量 payload 就会随每个 chunk 复制一遍。1000 份文档
    × 每份 200 chunk = 20 万份大纲副本，既撑大向量库、又让 scroll/scroll 回来的
    语料体积翻几倍。payload 只保留"过滤要用"的扁平字段（见 metadata.to_payload），
    完整信息留在这里，进上下文时按 document_id 一次取回。
    """

    __tablename__ = "document_metadata"
    __table_args__ = (
        UniqueConstraint("document_id", name="uq_document_metadata_document"),
        Index("ix_document_metadata_tenant", "tenant_id", "doc_type", "doc_year"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default", nullable=False,
    )

    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    author: Mapped[str | None] = mapped_column(String(256), nullable=True)
    doc_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    doc_number: Mapped[str | None] = mapped_column(String(128), nullable=True)
    doc_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    doc_year: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    language: Mapped[str] = mapped_column(
        String(16), default="zh", server_default="zh", nullable=False,
    )

    # 完整元数据（keywords / business_tags / outline / extra）—— JSONB 便于
    # 后续加字段而不改表结构；过滤走上面几个显式列（JSONB 上的范围过滤不建
    # 表达式索引会退化成顺序扫描）。
    payload: Mapped[dict] = mapped_column(
        JSONB, default=dict, server_default=sa_text("'{}'::jsonb"), nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentMetadataRow doc={self.document_id} type={self.doc_type!r} "
            f"year={self.doc_year} lang={self.language}>"
        )


class ChunkParent(Base):
    """
    Parent-Child 的父块正文（小节级 ``parent`` / 章节级 ``section``）.

    存在的唯一理由：**把父块正文从子块 payload 里挪出来**。

    改造前 ``TextChunk.parent_text`` 会把父块完整正文写进**每一个**子块：
    1000 份文档 × 200 子块 × 5 KB 父块 ≈ 1 GB 的纯冗余，而且它会随
    Qdrant scroll 一起被读回内存（关键词腿建索引时），把"上千文档"从
    "慢"变成"起不来"。

    改造后子块只带 ``parent_id``，命中后在检索层按 id **批量 IN 查询**
    （见 retrieval_service._hydrate_parents）——一次 SQL 换回整批父块正文。
    """

    __tablename__ = "chunk_parents"
    __table_args__ = (
        UniqueConstraint("parent_id", name="uq_chunk_parents_parent_id"),
        Index("ix_chunk_parents_document", "document_id", "level", "idx"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    # 形如 "{document_id}:p:{index}" / "{document_id}:s:{index}"，
    # 与 Qdrant payload 里的 parent_id 逐字一致（跨库 join 的唯一凭据）
    parent_id: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default", nullable=False,
    )
    # 冗余 ACL 载荷：hydration 时可能发生在"已通过 PG 文档级 ACL 校验"之后，
    # 但保留一份便于审计与将来把过滤下推到这里。
    access_level: Mapped[str] = mapped_column(
        String(20), default="private", server_default="private", nullable=False,
    )
    department_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    level: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True,   # parent | section
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    char_start: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    section_path: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=sa_text("'[]'::jsonb"), nullable=False,
    )
    child_indexes: Mapped[list] = mapped_column(
        JSONB, default=list, server_default=sa_text("'[]'::jsonb"), nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<ChunkParent {self.parent_id} level={self.level} "
            f"pages={self.page_start}-{self.page_end} chars={len(self.text)}>"
        )


class DocumentChunkTerm(Base):
    """
    关键词腿（BM25/全文检索）的词项表 —— 上千文档的规模关键.

    历史实现把整个 Qdrant 语料 scroll 进内存、现场建 BM25 倒排表，并**静默截断**
    到 ``HYBRID_MAX_CORPUS_POINTS``（默认 10000 条）。1000 份文档 × 每份 ~200
    chunk ≈ 20 万条 chunk，也就是说关键词腿**只看得到前 5% 的语料**，其余永远
    召不回来 —— 而且不报错、不告警。查询里恰好只有后面那 95% 才有的型号/编号时，
    用户得到"知识库没有这个"的答复，实际文档就在库里。

    这里把词项落库：入库时按 chunk 生成"字符 bigram + ASCII 词"的空白分隔串，
    并用一个**存储生成列** ``terms_tsv`` 持有它的 tsvector，GIN 索引建在该列上，
    ``ts_rank_cd`` 排序。全部在数据库侧完成：

      * 召回线性可控 —— 走索引而不是全表打分，语料规模只影响索引深度；
      * 内存零放大 —— 不再需要把 20 万条 payload 拉进进程；
      * 权限可下推 —— join documents 做租户/ACL 过滤，与向量腿语义一致。

    ⚠️ **存储生成列不是风格选择，是性能必需**：改成表达式索引
    ``USING gin (to_tsvector('simple', terms))`` 之后，``@@`` 过滤仍走索引，
    但排序里的 ``ts_rank_cd(to_tsvector('simple', terms), q)`` 会被 PG 当作
    普通表达式，对每一条命中行**重新分词**。命中集一大就退化成分钟级
    （实测 20 万条命中行 = 178 秒/次）。详见 ``terms_tsv`` 字段的注释。

    为什么用 bigram 而不是 PG 自带的 ``zhparser``/``pg_jieba``：那些需要额外
    安装扩展（生产环境未必有权装、版本还要对上），而本项目的 ``tokenize()``
    本来就是"ASCII 词 + 中文 bigram"，两者切分口径一致才能让 RRF 融合有意义。
    """

    __tablename__ = "document_chunk_terms"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "chunk_index", name="uq_chunk_terms_document_chunk",
        ),
        Index("ix_chunk_terms_document", "document_id"),
        Index("ix_chunk_terms_tenant_coll", "tenant_id", "collection_id"),
        # GIN 索引，**建在 ``terms_tsv`` 这一列上**（而不是表达式上）。
        # 为什么必须是这样：见 terms_tsv 字段的注释 —— 表达式索引只能加速过滤，
        # 排序那一半会退化成逐行重新分词，20 万行上是分钟级。
        Index("ix_chunk_terms_fts_tsv", "terms_tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64), default="default", server_default="default", nullable=False,
    )
    access_level: Mapped[str] = mapped_column(
        String(20), default="private", server_default="private", nullable=False,
    )
    department_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    collection_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # 该 chunk 在 Qdrant 里的点 id（= vector_service.generate_point_id 的产物）。
    # 存它是为了让关键词腿**自洽**：PG 只负责回答"哪些点命中了、排在多前"，
    # 拿到点 id 后按 id 批量 retrieve 即可取回完整 payload。否则就得为了拿
    # payload 再把整个语料 scroll 一遍 —— 那正是本模块要消灭的内存放大。
    point_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # 空格分隔的词项串（bigram + ASCII 词）。上限由 KEYWORD_BIGRAM_MAX_CHARS 控制
    terms: Mapped[str] = mapped_column(Text, nullable=False)
    # ``terms`` 的 tsvector 形态，**存储生成列**（PG 12+，``STORED``）。
    #
    # 为什么不用表达式索引 ``USING gin (to_tsvector('simple', terms))``：
    # 表达式索引只能服务 WHERE 里的 ``@@`` 过滤；一旦排序要算
    # ``ts_rank_cd(to_tsvector('simple', terms), q)``，PG 会把它当成一个普通
    # 表达式，对**每一条命中行重新分词一次**。GIN 命中集越大越致命 ——
    # 实测 20 万条命中行（每行约 250 个中文 bigram）时，单次关键词查询
    # **178 秒**，其中 99% 花在这次重复分词上。
    #
    # 换成存储生成列后：写侧只在 INSERT 时分词一次，读侧 ``@@`` 与
    # ``ts_rank_cd`` 都直接读该列，检索耗时与命中集大小解耦。
    #
    # 代价：插入时多一份 tsvector 存储（约与 terms 同量级），换取读侧不再重算。
    # 关键词腿是**读多写少**的路径（入库一次、被检索无数次），这个交换是划算的。
    terms_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', terms)", persisted=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunkTerm doc={self.document_id} chunk={self.chunk_index} "
            f"terms={len(self.terms)}>"
        )
