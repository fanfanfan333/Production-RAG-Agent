"""
Document upload pipeline orchestrator.

This service wires together every step of the ingestion pipeline:

    ┌──────────┐   bytes   ┌─────────────┐   pages   ┌──────────┐
    │  Raw file│ ────────▶ │  Parser     │ ─────────▶ │  Chunker │
    └──────────┘           └─────────────┘            └──────────┘
                                                            │ chunks
                                                            ▼
    ┌──────────┐  vectors  ┌──────────────────┐  batch  ┌───────────────┐
    │  Qdrant  │ ◀──────── │ embedding_service │ ◀────── │ this service  │
    └──────────┘   upsert  └──────────────────┘         └───────────────┘
         ▲                                                      │
         │ checkpoint                                           │
    ┌──────────┐                                                │
    │PostgreSQL│ ◀──────────────────────────────────────────────┘
    └──────────┘

Features: Chunk-level checkpointing, idempotent resume, dynamic batch fallback.
"""

import asyncio
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, select, update

from app.config import get_settings
from app.db.models import (
    ChunkParent,
    Document,
    DocumentMetadataRow,
    DocumentStatus,
)
from app.db.postgres import get_db_session
from app.schemas.document import DocumentResult, UploadResponse
from app.services.audit_service import record_audit
from app.services.chunker import (
    build_chunk_hierarchy,
    build_image_chunks,
)
from app.services.embedding_service import embed_batch_with_retry
from app.services.metadata import extract_metadata
from app.services.parsers import get_parser_for_file
from app.services.pg_keyword_search import replace_document_terms, terms_for_text
from app.services.storage import save_original
from app.services.structure import parse_structure
from app.services.text_cleaning import clean_extraction, clean_image_texts
from app.services.vector_service import VectorPoint, upsert_vectors, generate_point_id, get_existing_point_ids
from app.utils.logging import get_logger

logger = get_logger(__name__)


# Global concurrency limiter for embeddings
_embedding_semaphore = None

def get_embedding_semaphore():
    global _embedding_semaphore
    if _embedding_semaphore is None:
        _embedding_semaphore = asyncio.Semaphore(get_settings().MAX_CONCURRENT_EMBEDDINGS)
    return _embedding_semaphore


async def _create_document_record(
    filename: str,
    file_size: int,
    file_hash: str,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    tenant_id: str = "default",
    department_id: str | None = None,
    access_level: str = "private",
) -> Document:
    from sqlalchemy.exc import IntegrityError

    try:
        async with get_db_session() as session:
            doc = Document(
                id=uuid.uuid4(),
                filename=filename,
                file_size=file_size,
                file_hash=file_hash,
                status=DocumentStatus.PENDING,
                current_stage="pending",
                owner_id=owner_id,
                collection_id=collection_id,
                tenant_id=tenant_id,
                department_id=department_id,
                access_level=access_level,
            )
            session.add(doc)
            await session.flush()
            await session.refresh(doc)
            return doc
    except IntegrityError:
        # 复合唯一约束 (owner_id, file_hash) 冲突：同一用户并发上传同一文件。
        # 回退到按"该用户 + 哈希"查询（不同用户的同内容文件互不影响）。
        async with get_db_session() as session:
            from sqlalchemy import select
            query = select(Document).where(Document.file_hash == file_hash)
            if owner_id is not None:
                query = query.where(Document.owner_id == owner_id)
            doc = await session.scalar(query.limit(1))
            if not doc:
                raise RuntimeError("IntegrityError caught but document not found on fallback.")
            return doc


async def _update_document(
    doc_id: uuid.UUID,
    **kwargs,
) -> None:
    """Patch a document row with arbitrary fields + bump updated_at."""
    async with get_db_session() as session:
        kwargs["updated_at"] = datetime.now(tz=timezone.utc)
        await session.execute(
            update(Document).where(Document.id == doc_id).values(**kwargs)
        )


@dataclass
class PreparedUpload:
    """一次上传「快速前段」的产物。

    要么给出一个待处理的文档行（`result is None`），要么直接给出终态结果
    （`result` 非空，例如"已存在"或"正在处理中"）。判重与建行都是毫秒级操作，
    留在请求线程里同步完成；真正的重活（解析 / 逐图 OCR / 向量化）交给后台任务，
    请求据此可以立即返回 —— 一份 13 MB 文档的入库要 7 分钟以上，让浏览器为它
    挂着一个 HTTP 长连接是不可接受的（用户一关页面就"什么都没发生"）。
    """

    doc: Document | None = None
    result: DocumentResult | None = None
    is_resume: bool = False
    tenant_id: str = "default"
    department_id: str | None = None
    access_level: str = "private"
    file_size: int = 0
    file_hash: str = ""

    @property
    def done(self) -> bool:
        """True 表示无需再跑管线，`result` 已是终态。"""
        return self.result is not None


# ── 后台入库任务注册表 ────────────────────────────────────────────────────────
# 必须持有 Task 的强引用：asyncio 对 create_task 的返回值只保留弱引用，若没有
# 别处引用它，任务可能在运行途中被 GC 回收并静默取消（CPython 的已知陷阱）。
_ingestion_tasks: dict[uuid.UUID, asyncio.Task] = {}


def _build_term_rows(items: list, max_chars: int) -> list[tuple[int, str, str]]:
    """
    为关键词腿准备 ``[(chunk_index, point_id, terms_string), ...]``.

    抽成模块级函数是为了能丢进线程池（``asyncio.to_thread``）—— 词项化是纯 CPU
    的字符扫描，一份 200 页文档会跑几百万次正则匹配，在事件循环里做会把
    /health 拖到超时（编排器据此判 unhealthy 并重启容器，把入库中的进程
    SIGTERM 掉）。

    *items* 是 ``[(point_id, chunk), ...]``。**point_id 必须由调用方传入**而不是
    在这里重算：它是关键词腿与向量腿之间唯一的连接键（PG 回答"哪些点命中"，
    Qdrant 按点 id 回 payload），两处各算一次只要参数有一个分叉，就会出现
    "检索命中但取不回正文"—— 而且只在部分文档上复现。调用方已经在做幂等检查时
    算好了这份映射，直接复用。
    """
    rows: list[tuple[int, str, str]] = []
    for point_id, chunk in items:
        try:
            terms = terms_for_text(chunk.text, max_chars)
        except Exception:            # noqa: BLE001 — 单块失败不影响其余块
            logger.exception("terms_for_text failed for chunk %s", chunk.chunk_index)
            continue
        if terms and point_id:
            rows.append((int(chunk.chunk_index), str(point_id), terms))
    return rows


async def _persist_metadata(
    *,
    doc_id: uuid.UUID,
    tenant_id: str,
    meta,
) -> None:
    """
    写文档级元数据（**独立事务**）.

    独立事务是刻意的：元数据、父块、词项三者在逻辑上互不依赖，任何一个失败都不
    该把另两个一起回滚。共用一个 session 时，只要其中一个语句抛异常，整个事务
    进入 aborted 状态，后续语句全部失败并一起回滚 —— 现场表现是
    "向量 payload 里有 doc_type/年份，PG 表里却什么都没有"，而日志只留一条
    "persistence failed"，看不出是哪一块坏了。

    先删后插而不是 upsert：文档重新入库时元数据字段可能**变少**
    （改名、换分类），残留的旧值会让"按类型过滤"命中一份已经不再属于该类别的
    文档 —— 一个不会报错、只会悄悄多召回的错误。
    """
    async with get_db_session() as session:
        await session.execute(
            delete(DocumentMetadataRow).where(
                DocumentMetadataRow.document_id == doc_id
            )
        )
        session.add(DocumentMetadataRow(
            document_id=doc_id,
            tenant_id=tenant_id,
            title=meta.title[:512] if meta.title else None,
            author=meta.author,
            doc_type=meta.doc_type,
            doc_number=meta.doc_number,
            doc_date=meta.doc_date,
            doc_year=meta.doc_year,
            language=meta.language,
            payload=meta.to_db_dict(),
        ))


async def _persist_parents(
    *,
    doc_id: uuid.UUID,
    tenant_id: str,
    access_level: str,
    department_id: str | None,
    parents: list,
) -> None:
    """
    写父块正文（**独立事务**，小节级 + 章节级）.

    父块正文留在 PG 而不是子块 payload 里，是本次改造里省掉最多空间的单点决策：
    改造前 ``parent_text`` 会被复制进**每一个**子块的向量 payload，
    1000 份文档 × 200 子块 × 5 KB ≈ 1 GB 纯冗余，而且会随关键词腿的语料读取
    被整体拉回内存。现在子块只带 ``parent_id``，命中后由检索层按 id 批量 IN 查询。
    """
    async with get_db_session() as session:
        await session.execute(
            delete(ChunkParent).where(ChunkParent.document_id == doc_id)
        )
        session.add_all([
            ChunkParent(
                parent_id=p.parent_id,
                document_id=doc_id,
                tenant_id=tenant_id,
                access_level=access_level,
                department_id=department_id,
                level=p.level,
                idx=p.index,
                text=p.text,
                char_start=p.char_start,
                char_end=p.char_end,
                page_start=p.page_start,
                page_end=p.page_end,
                line_start=p.line_start,
                line_end=p.line_end,
                heading=p.heading[:512] if p.heading else None,
                section_path=list(p.section_path),
                child_indexes=list(p.child_indexes),
            )
            for p in parents
        ])


async def _persist_terms(
    *,
    doc_id: uuid.UUID,
    tenant_id: str,
    access_level: str,
    department_id: str | None,
    collection_id: uuid.UUID | None,
    items: list,
    max_chars: int,
) -> int:
    """
    写关键词腿词项（**独立事务**），返回写入行数.

    词项化（字符扫描 + bigram）是纯 CPU 的活儿，必须丢进线程池：一份 200 页
    文档会跑几百万次匹配，在事件循环里做会把 /health 拖到超时 —— 而编排器
    据此判 unhealthy 并重启容器，直接把正在入库的进程 SIGTERM 掉。
    """
    rows = await asyncio.to_thread(_build_term_rows, items, max_chars)
    async with get_db_session() as session:
        return await replace_document_terms(
            session,
            document_id=doc_id,
            tenant_id=tenant_id,
            access_level=access_level,
            department_id=department_id,
            collection_id=str(collection_id) if collection_id else None,
            rows=rows,
        )


def is_ingesting(doc_id: uuid.UUID) -> bool:
    """该文档是否已有后台入库任务在跑（同文件并发上传时据此幂等复用）。"""
    task = _ingestion_tasks.get(doc_id)
    return task is not None and not task.done()


def active_ingestion_count() -> int:
    """当前在跑的入库任务数（健康检查 / 监控用）。"""
    return sum(1 for t in _ingestion_tasks.values() if not t.done())


async def prepare_upload(
    filename: str,
    content: bytes,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
    department_id: str | None = None,
    access_level: str | None = None,
) -> PreparedUpload:
    """上传的「快速前段」：判重 + 建文档行（PENDING）。不解析、不向量化。"""
    from app.services.tenancy import (
        DEFAULT_DOCUMENT_ACCESS_LEVEL,
        VALID_ACCESS_LEVELS,
        normalize_tenant_id,
    )

    tenant_id = normalize_tenant_id(tenant_id)
    if access_level not in VALID_ACCESS_LEVELS:
        access_level = DEFAULT_DOCUMENT_ACCESS_LEVEL
    file_size = len(content)
    file_hash = hashlib.sha256(content).hexdigest()

    # ── 0. 判重（按用户范围）────────────────────────────────────────────────
    # 只与"当前用户自己"的文档判重：不同用户上传同一文件应各自独立索引，
    # 旧的无主数据（owner 为 NULL）不应阻塞任何用户的首次上传。
    async with get_db_session() as session:
        dedup_query = select(Document).where(Document.file_hash == file_hash)
        if owner_id is not None:
            dedup_query = dedup_query.where(Document.owner_id == owner_id)
        existing_doc = await session.scalar(dedup_query.limit(1))

    if existing_doc is not None:
        if existing_doc.status == DocumentStatus.COMPLETED:
            logger.info("Duplicate document detected: filename='%s' owner=%s", filename, owner_id)
            return PreparedUpload(
                result=DocumentResult(
                    document_id=existing_doc.id,
                    filename=filename,
                    status=DocumentStatus.ALREADY_EXISTS,
                    message="该文档你已上传并索引过，无需重复上传。",
                    existing_document_id=existing_doc.id,
                    uploaded_at=existing_doc.created_at,
                    page_count=existing_doc.page_count,
                    chunk_count=existing_doc.chunk_count,
                    image_count=existing_doc.image_count,
                    image_object_count=existing_doc.image_object_count,
                    file_size_bytes=existing_doc.file_size,
                    created_at=existing_doc.created_at,
                )
            )

        if is_ingesting(existing_doc.id):
            # 同一文件被并发重复提交：复用已在跑的任务，不再起第二个 —— 否则两份
            # 解析 / OCR 抢同一份 CPU，还可能把同一批向量写两遍。
            logger.info(
                "Document id=%s is already being ingested — reusing the running job.",
                existing_doc.id,
            )
            return PreparedUpload(
                result=DocumentResult(
                    document_id=existing_doc.id,
                    filename=filename,
                    status=existing_doc.status,
                    message="该文档正在处理中，无需重复提交。",
                    page_count=existing_doc.page_count,
                    chunk_count=existing_doc.chunk_count,
                    file_size_bytes=existing_doc.file_size,
                    created_at=existing_doc.created_at,
                )
            )

        # 上次中断留下的残档（FAILED）→ 幂等续传：复用原行，不新建
        logger.info(
            "Resuming partial document id=%s from state %s",
            existing_doc.id, existing_doc.status.value,
        )
        # 立刻把行翻回 PENDING（而不是等后台任务真正开跑）。
        #
        # 为什么必须在**请求线程**里就改：上传接口已经在响应里承诺"已受理，
        # 正在后台解析并建立索引"。如果这一行还挂着上一次的 status=failed /
        # error_message，用户传完刷新列表看到的就是"失败 + 旧报错"，与刚拿到的
        # 202 自相矛盾；后台任务如果被排在队列后面，这个窗口能持续好几秒。
        # 顺带清零计数，避免"新一轮的进度"里混着上一轮的失败块数。
        await _update_document(
            existing_doc.id,
            status=DocumentStatus.PENDING,
            current_stage="pending",
            total_chunks=0,
            embedded_chunks=0,
            failed_chunks=0,
            error_message=None,
        )
        return PreparedUpload(
            doc=existing_doc,
            is_resume=True,
            # 续传的老文档保持原有隔离属性（不随本次上传者变更，防越权接管）
            tenant_id=existing_doc.tenant_id,
            department_id=existing_doc.department_id,
            access_level=existing_doc.access_level,
            file_size=file_size,
            file_hash=file_hash,
        )

    # ── 1. 落元数据行 ───────────────────────────────────────────────────────
    doc = await _create_document_record(
        filename, file_size, file_hash, owner_id, collection_id,
        tenant_id=tenant_id, department_id=department_id,
        access_level=access_level,
    )
    return PreparedUpload(
        doc=doc,
        tenant_id=tenant_id,
        department_id=department_id,
        access_level=access_level,
        file_size=file_size,
        file_hash=file_hash,
    )


async def _run_ingestion(
    prep: PreparedUpload,
    filename: str,
    content: bytes,
    settings,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
) -> DocumentResult:
    """执行状态机第 2–6 阶段：解析 → 分块 → 向量化 → 入库。

    `prep` 必须已经持有文档行（即 `prep.done is False`）。
    """
    doc = prep.doc
    assert doc is not None, "_run_ingestion 需要 prepare_upload 产出的文档行"
    tenant_id = prep.tenant_id
    department_id = prep.department_id
    access_level = prep.access_level
    file_size = prep.file_size

    try:
        doc_id_str = str(doc.id)

        # ── 2. Mark PARSING ───────────────────────────────────────────────────
        # 这里顺手把**上一轮的计数与错误**清干净。续传（FAILED 复用原行）时
        # 这些字段会留着上一次的值：曾出现"文档已 COMPLETED、向量 13/13"，
        # 但 failed_chunks 还是 13，监控面板于是报假故障；error_message 同理，
        # 重试成功的文档仍挂着上一次的报错。计数是"本轮"的，就该按轮清零。
        await _update_document(
            doc.id,
            status=DocumentStatus.PARSING,
            current_stage="parsing",
            total_chunks=0,
            embedded_chunks=0,
            failed_chunks=0,
            error_message=None,
        )
        parser = get_parser_for_file(filename)
        # document_id 传入解析器：内嵌图片据此落盘到
        # uploads/{tenant_id}/{doc_id}/images/（第三层：图片按租户隔离），
        # 并生成稳定的 image_id（部分2）。
        # 解析是**同步阻塞**的重活（PDF/DOCX 版面解析 + 逐图 OCR/表格还原/
        # Vision 调用）。直接在事件循环里 await 一个同步函数会把整个 loop 占死：
        # 实测上传 13 MB 文档期间 /health 连续超时，容器被健康检查判成
        # unhealthy（编排器会据此重启容器，把入库中的进程直接 SIGTERM 掉），
        # 且其它用户的提问/上传全部挂起。丢进线程池即可，行为完全不变。
        extraction = await asyncio.to_thread(
            parser.parse,
            content,
            filename=filename,
            document_id=doc_id_str,
            tenant_id=tenant_id,
        )

        # ── 2.1 归档原始文件（部分2：uploads/{tenant_id}/{document_id}/）───────
        if settings.ARCHIVE_ORIGINAL_DOCUMENT:
            # 写盘同样是阻塞 I/O，同理下沉到线程池。
            await asyncio.to_thread(
                save_original, doc_id_str, filename, content, tenant_id=tenant_id,
            )

        # ── 2.2 文档结构解析（MinerU / Marker / Docling → 原生兜底）──────────────
        # 这一层的产出有三个用途，都在下游：
        #   ① 更准的章节结构 → 父子分块的**父块边界**（小节完整，而不是被字符
        #      窗口劈开）+ 元数据大纲 + 引用卡片上的"出自 3.2 节"；
        #   ② 更好的表格/公式还原（MinerU/Marker 的版面模型）；
        #   ③ 更干净的阅读顺序（多栏 PDF 的原生抽取常常串行错乱）。
        #
        # 硬约束：**page_marks 必须可信**。拿不到真实逐页偏移的 provider 结果会被
        # parse_structure 直接丢弃（见 structure/model.py 的论证）—— 宁可退回原生
        # （页码正确、结构稍弱），也不要一份页码全错的漂亮 Markdown，因为细粒度
        # 引用（"第 3 页第 12 行"）正是建立在页码之上。
        #
        # 解析器全不可用时 provider == "native"，此时 markdown/pages 与解析器产出
        # 完全一致，对既有部署是**零回归**。
        structured = await asyncio.to_thread(
            parse_structure,
            content,
            filename,
            extraction=extraction,
            settings=settings,
        )
        if structured.provider != "native":
            # 换掉正文与页表：后续的清洗、分块、页码解析全部基于新结构。
            # page_spans 给出每页在新 markdown 中的字符区间，重建 ExtractedPage
            # 才能让 page_for_offset / 分块器的 page_resolver 继续工作。
            extraction.full_text = structured.markdown
            spans = structured.page_spans()
            if spans:
                from app.services.parsers.base import ExtractedPage

                # ⚠️ ``text`` 必须**填上该页在 markdown 中的真实片段**，不能留空。
                #
                # 这里曾经留空（text=""），后果是整篇正文被静默清空：下游
                # ``clean_extraction`` 是**按页重拼**正文的
                # （``join(gap + clean(page.text))``）——页正文为空时重拼结果只剩
                # 页与页之间的分隔符。单页文档直接变成空串，于是分块产出 0 个
                # chunk，入库以"Extraction produced zero chunks"失败；多页文档
                # 更糟：不报错，只是正文塌缩成几十个换行符，向量库与关键词索引里
                # 留下一条"有 chunk 但没内容"的记录。
                #
                # 之所以当初会漏掉：``page_spans`` 的语义是"偏移"，看起来只需要
                # 偏移就够了；而 ``page_for_offset`` 确实只用偏移。但 ``page.text``
                # 另有消费者（清洗），两处需求不一致时"少填一个字段"不会报错，
                # 只会在一个完全不相关的模块里表现为数据丢失。
                extraction.pages = [
                    ExtractedPage(
                        page_number=pno,
                        text=structured.markdown[cs:ce],
                        char_start=cs,
                        char_end=ce,
                    )
                    for pno, cs, ce in spans
                ]
                extraction.page_count = len(spans)
            extraction.parser_used = f"structure:{structured.provider}"
            logger.info(
                "Document '%s' (id=%s): structure parser '%s' — %d page(s), %d node(s)",
                filename, doc_id_str, structured.provider,
                structured.page_count, len(structured.nodes),
            )

        # ── 2.5 入库清洗 + 注入检测（问题3 文档防护）────────────────────────────
        # 上传/解析后、切块入库前处理两件事：
        #
        #   1) 正文清洗 —— 剥 BOM / 零宽字符 / C0-C1 控制符 / 软连字符，统一换行、
        #      去掉行尾空白、把各类 Unicode 空格折成半角空格。
        #      ⚠️ 旧实现只在"命中注入"时回写清洗结果（`if not scan.clean: ...`），
        #      于是干净文档的 BOM 与零宽字符一路进了向量库与 BM25 语料：查询
        #      走的是清洗后的版本、文档却是脏的，"同形不同码"对不上；引用卡片
        #      把 chunk 原文回显给用户时还会凭空多出空格/方块。
        #
        #   2) 注入屏蔽 —— 文档里嵌入的"忽略规则/泄露系统提示词"等模型控制指令
        #      段落就地屏蔽，使中毒内容不进入向量库与 BM25 语料。检索时
        #      context_builder 还会再清洗一次 —— 纵深防御。
        #
        # 两件事都**按页**做并重算页偏移（见 text_cleaning 模块文档）：不能
        # 整体改写 full_text —— page_for_offset 依赖解析时算出的
        # char_start/char_end，整体改写会让其后所有偏移平移、页码整体漂移，
        # 引用卡片上的"第 N 页"就变成错的。
        cleaned = await asyncio.to_thread(
            clean_extraction, extraction.pages, extraction.full_text,
        )
        if not cleaned.pagemap_intact:
            logger.info(
                "Document '%s' (id=%s): page map not trustworthy — kept the "
                "original text as-is (page numbers stay correct)",
                filename, doc_id_str,
            )
        if cleaned.changed:
            extraction.full_text = cleaned.full_text
            for page, (start, end) in zip(extraction.pages, cleaned.page_spans):
                page.char_start, page.char_end = start, end

        # 图片文本走的是**独立通道**（image chunk，不拼进 full_text），
        # 因此必须单独清洗，否则"图片提取"的产物照样带着控制符入库。
        #
        # 这里与正文通道共用 `clean_and_mask`：注入屏蔽对图片一视同仁。此前
        # 只剥控制符，导致"把指令画进图里"成了绕过入库扫描的现成路径 ——
        # 正文被扫、图片不被扫，攻击者只需要选没设防的那条。
        images_result = await asyncio.to_thread(
            clean_image_texts, extraction.images,
        )
        if images_result.touched:
            logger.info(
                "Document '%s' (id=%s): cleaned %d image text payload(s)",
                filename, doc_id_str, images_result.touched,
            )
        if images_result.masked_paragraphs:
            logger.warning(
                "Document '%s' (id=%s): masked %d injection-like paragraph(s) "
                "in image text before indexing",
                filename, doc_id_str, images_result.masked_paragraphs,
            )
            await record_audit(
                "security.image_injection.masked",
                user_id=owner_id,
                resource_type="document",
                resource_id=doc_id_str,
                detail=(
                    f"channel=image; paragraphs={images_result.masked_paragraphs}; "
                    f"images={images_result.touched}; filename={filename}"
                ),
            )

        if cleaned.masked_paragraphs:
            logger.warning(
                "Document '%s' (id=%s): masked %d injection-like paragraph(s) "
                "before indexing", filename, doc_id_str, cleaned.masked_paragraphs,
            )
            await record_audit(
                "security.document_injection.masked",
                user_id=owner_id,
                resource_type="document",
                resource_id=doc_id_str,
                detail=f"paragraphs={cleaned.masked_paragraphs}; filename={filename}",
            )

        # ── 2.6 Metadata 抽取（检索范围收敛 + 引用溯源）──────────────────────────
        # 上千文档时"检索范围"本身就是准确率的一部分：不做元数据预过滤时，
        # 同主题的几十份不同年份/部门的文档会互相稀释候选池 —— 真正那一段可能
        # 排在 Top50 之外，根本没进精排，而链路上每一环都"正常"。
        #
        # 纯确定性抽取（不调 LLM）：元数据会参与**过滤**，一个编造出来的年份会
        # 静默把正确文档排除掉。确定性抽取可能漏（漏了只是少一个过滤维度，安全），
        # LLM 抽取可能错（错了是静默错杀）。宁漏不错。
        doc_meta = await asyncio.to_thread(
            extract_metadata,
            text=extraction.full_text,
            filename=filename,
            outline=structured.outline(),
            settings=settings,
        )
        logger.info(
            "Document '%s' (id=%s): metadata title=%r type=%s year=%s lang=%s "
            "keywords=%d sections=%d",
            filename, doc_id_str, doc_meta.title[:40], doc_meta.doc_type,
            doc_meta.doc_year, doc_meta.language, len(doc_meta.keywords),
            doc_meta.section_count,
        )

        # ── 3. Mark CHUNKING ──────────────────────────────────────────────────
        await _update_document(
            doc.id, 
            status=DocumentStatus.CHUNKING, 
            current_stage="chunking",
            file_type=extraction.file_type,
            parser_used=extraction.parser_used,
            ocr_used=extraction.ocr_used,
            ocr_engine=extraction.ocr_engine,
            extraction_method=extraction.extraction_method,
            page_count=extraction.page_count,
            image_count=len(extraction.images),
        )
        
        # 分块是纯 CPU（正则、游标推进、父块元数据），大文档上同样会占住
        # 事件循环数百毫秒到数秒，一并下沉。
        #
        # 走 build_chunk_hierarchy（而不是 build_chunks）：父块关系的建立与分块
        # **必须用同一批参数、同一次调用**。分开调用时"忘记建父子"或"父子用了
        # 另一套 min/max"都不会报错，只会在检索时表现为"父块回填静默失效" ——
        # 一个没有任何日志、只能靠对比回答质量发现的退化。
        #
        # nodes 传入结构树：父块边界因此按**章节**切（小节完整），而不是按固定
        # 字符窗口切（会把一个完整小节劈成两半，回填给模型的上下文缺了开头的
        # 口径定义 —— 这正是"看起来有依据、其实答错"的典型来源）。
        hierarchy = await asyncio.to_thread(
            build_chunk_hierarchy,
            text=extraction.full_text,
            document_id=doc_id_str,
            # 结构解析链尾（native）时 nodes 来自正则抽标题，同样可用；
            # PARENT_CHILD_ENABLED=False 时退回旧的字符窗口父块。
            nodes=structured.nodes if settings.PARENT_CHILD_ENABLED else None,
            sections_enabled=settings.PARENT_CHILD_SECTION_ENABLED,
            parent_min_chars=settings.PARENT_MIN_CHARS,
            parent_max_chars=settings.PARENT_MAX_CHARS,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
            max_chunk_size=settings.MAX_CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            page_resolver=extraction.page_for_offset,
            enable_small_to_big=(
                settings.HIERARCHICAL_RAG_ENABLED and not settings.PARENT_CHILD_ENABLED
            ),
            # 表格类文档（csv/xlsx）整体标记为 table；Markdown 表格块由
            # chunker 逐块识别 —— 两者都服务于"表格检索"（部分3）。
            default_content_type="table" if extraction.is_tabular else "text",
        )
        chunks = hierarchy.children

        # ── 3.1 图片作为独立检索对象（部分1）──────────────────────────────────
        # 图片不再混进正文：每张有可检索文本（OCR 或 vision caption）的图片
        # 建成独立的 image chunk，chunk_index 紧接文本块之后，避免图文撞号。
        image_chunks = await asyncio.to_thread(
            build_image_chunks,
            extraction.images,
            start_index=len(chunks),
        )

        all_chunks = chunks + image_chunks
        if not all_chunks:
            raise ValueError(
                "Extraction produced zero chunks — document may be empty "
                "(or images without OCR text / vision caption)."
            )

        await _update_document(
            doc.id,
            total_chunks=len(all_chunks),
            chunk_count=len(all_chunks),
            image_object_count=len(image_chunks),
        )

        # 三层图片处理的分布（table/chart/diagram/screenshot/photo）——
        # 日志里能直接看出"分流真的发生了"，而不是所有图片都被当成一张图。
        type_counts: dict[str, int] = {}
        for img in extraction.images:
            kind = getattr(img, "image_type", None) or "photo"
            type_counts[kind] = type_counts.get(kind, 0) + 1
        engine_counts: dict[str, int] = {}
        for img in extraction.images:
            engine = getattr(img, "analyze_engine", None) or "ocr"
            engine_counts[engine] = engine_counts.get(engine, 0) + 1

        logger.info(
            "id=%s: %d text chunk(s) + %d image chunk(s) from %d extracted image(s); "
            "image_types=%s analyze=%s",
            doc_id_str, len(chunks), len(image_chunks), len(extraction.images),
            type_counts or "{}", engine_counts or "{}",
        )

        # ── 3.2 元数据 / 父块 / 关键词词项落库 ─────────────────────────────────
        # 三张表都是**独立表**（不是 documents 上的新列）：本服务启动用
        # `Base.metadata.create_all`，它只会创建不存在的表，**不会给已存在的表加列**
        # ——加列的方案在既有部署上会静默失效。独立表的升级路径是零操作。
        #
        # 为什么父块正文不留在子块 payload 里：改造前 `parent_text` 会把父块完整
        # 正文复制进**每一个**子块。1000 份文档 × 200 子块 × 5 KB ≈ 1 GB 纯冗余，
        # 而且这些 payload 会被关键词腿的语料读取整体拉回内存。现在子块只带
        # parent_id，命中后由检索层按 id 批量 IN 查询（见 retrieval_service）。
        # 点 id 映射：chunk → 它在 Qdrant 里的点 id。
        #
        # 必须在词项落库**之前**算出来：词项行要带 point_id —— 关键词腿只从 PG
        # 拿"哪些点命中"，再按点 id 批量 retrieve 取回 payload。少了它，PG 就得
        # 为了拿正文再 scroll 整个语料（正是本次要消灭的内存放大）。
        #
        # 算一次、两处复用（这里给词项行，第 4 步幂等检查用同一份），避免
        # generate_point_id 的参数在两处悄悄分叉 —— 分叉的表现是"部分文档检索
        # 命中却取不回正文"，一个只在数据规模上来之后才暴露的问题。
        chunk_map: dict[str, object] = {}
        for c in all_chunks:
            pid = generate_point_id(
                doc_id_str, filename, c.page_number, c.section, c.heading,
                c.chunk_index, c.text,
                content_type=c.content_type,
                image_id=c.image_id,
            )
            chunk_map[pid] = c

        # 三块落库**各自独立**（独立事务 + 独立 try）。它们彼此没有依赖，任何一块
        # 失败都不该连带丢掉另两块。历史实现是三者共用一个 try + 一个 session：
        # 关键词词项的行形状写错（unpack 异常）时，元数据与父块**也一起回滚**，
        # 而日志只说 "persistence failed" —— 现场表现是"向量 payload 里有
        # doc_type/年份、PG 表里却是空的"，一个从日志看不出因果的不一致。
        try:
            await _persist_metadata(doc_id=doc.id, tenant_id=tenant_id, meta=doc_meta)
        except Exception:
            logger.exception(
                "Document '%s' (id=%s): metadata persistence failed — ingestion "
                "continues (no metadata pre-filter / provenance for this doc)",
                filename, doc_id_str,
            )

        try:
            await _persist_parents(
                doc_id=doc.id,
                tenant_id=tenant_id,
                access_level=access_level,
                department_id=department_id,
                parents=hierarchy.parents,
            )
        except Exception:
            logger.exception(
                "Document '%s' (id=%s): parent-block persistence failed — ingestion "
                "continues (small-to-big context unavailable for this doc)",
                filename, doc_id_str,
            )

        keyword_backend = str(
            getattr(settings, "HYBRID_KEYWORD_BACKEND", "postgres")
        ).lower()
        if keyword_backend == "postgres":
            try:
                written = await _persist_terms(
                    doc_id=doc.id,
                    tenant_id=tenant_id,
                    access_level=access_level,
                    department_id=department_id,
                    collection_id=collection_id,
                    items=list(chunk_map.items()),
                    max_chars=settings.KEYWORD_BIGRAM_MAX_CHARS,
                )
                logger.info(
                    "id=%s: wrote %d keyword term row(s) for the PG keyword leg",
                    doc_id_str, written,
                )
            except Exception:
                # 关键词腿缺失时向量腿仍然可用（少一路召回，不会崩）。但这条日志
                # 必须显式说明"这份文档没进关键词索引" —— 否则它退化成
                # "用户搜不到、而系统一切正常"，那是最难定位的一类问题。
                logger.exception(
                    "Document '%s' (id=%s): keyword-term persistence failed — this "
                    "document is NOT covered by the PG keyword leg (vector-only recall)",
                    filename, doc_id_str,
                )

        # ── 4. Deterministic Idempotency Check ────────────────────────────────
        # chunk_map（chunk → 点 id）已在上一步算好，这里直接复用 ——
        # 幂等判定的键必须与写入用的键**逐字一致**，算两次就是在赌
        # generate_point_id 的参数列表不会在两处漂移。
        await _update_document(doc.id, status=DocumentStatus.EMBEDDING, current_stage="checking_existing_vectors")

        existing_ids = await get_existing_point_ids(list(chunk_map.keys()))
        missing_ids = set(chunk_map.keys()) - existing_ids
        missing_chunks = [chunk_map[pid] for pid in missing_ids]
        
        embedded_count = len(existing_ids)
        await _update_document(doc.id, embedded_chunks=embedded_count)
        
        logger.info(
            "id=%s → %d total chunks. %d already exist in Qdrant, %d missing.",
            doc_id_str, len(all_chunks), embedded_count, len(missing_chunks)
        )

        # ── 5. Embed and Index Missing Chunks (Batch Streaming) ────────────────
        await _update_document(doc.id, current_stage="embedding_and_indexing")
        
        # Dynamic batch sizing loop
        current_batch_size = settings.EMBEDDING_BATCH_SIZE
        chunk_idx = 0
        failed_chunks_count = 0
        
        semaphore = get_embedding_semaphore()
        
        while chunk_idx < len(missing_chunks):
            batch = missing_chunks[chunk_idx : chunk_idx + current_batch_size]
            batch_texts = [c.text for c in batch]
            
            try:
                # Concurrency limit applies specifically to the embedding API call
                async with semaphore:
                    vectors = await embed_batch_with_retry(batch_texts, task_type="RETRIEVAL_DOCUMENT")
                
                # Checkpointing Qdrant + Postgres atomically per batch
                points = [
                    VectorPoint(
                        vector=vectors[i],
                        document_id=doc_id_str,
                        filename=filename,
                        chunk_index=c.chunk_index,
                        page_number=c.page_number,
                        text=c.text,
                        heading=c.heading,
                        section=c.section,
                        # ── Multi-Tenant 隔离载荷（与 Document 行同源）────────
                        tenant_id=tenant_id,
                        user_id=str(owner_id) if owner_id else None,
                        access_level=access_level,
                        department_id=department_id,
                        # ── 位置信息：细粒度引用溯源（行号随 payload 落库）──────
                        line_start=c.line_start,
                        line_end=c.line_end,
                        collection_id=str(collection_id) if collection_id else None,
                        # ── 部分3：payload 写入内容类型与图片信息 ──────────────
                        content_type=c.content_type,
                        image_id=c.image_id,
                        image_path=c.image_path,
                        image_caption=c.image_caption,
                        image_type=c.image_type,
                        # ── 多引擎图片理解：产出引擎 + 置信度 + 人工复核标记 ──
                        analyze_engine=c.analyze_engine,
                        analyze_confidence=c.analyze_confidence,
                        manual_review=c.manual_review,
                        # ── 产出质检 + 双通道融合（可验证的事实）──────────────
                        analyze_quality=dict(c.analyze_quality or {}),
                        analyze_fusion=dict(c.analyze_fusion or {}),
                        # ── 图片位置：文档内序号 + 页面边界框（细粒度引用）──────
                        position=c.position,
                        bbox=c.bbox,
                        # ── Parent-Child：只写 parent_id / section_id ──────────
                        # **不写 parent_text**（已弃用）：父块正文复制进每个子块
                        # 是 GB 级冗余，正文由 chunk_parents 表单独持有，
                        # 命中后由检索层批量回填。
                        parent_id=c.parent_id,
                        parent_index=c.parent_index,
                        section_id=c.section_id,
                        section_path=c.section_path,
                        # ── Metadata 可过滤维度（扁平、已建 payload 索引）──────
                        doc_meta=doc_meta.to_payload(),
                    )
                    for i, c in enumerate(batch)
                ]
                await upsert_vectors(points)
                
                # Checkpoint progress
                embedded_count += len(batch)
                await _update_document(doc.id, embedded_chunks=embedded_count)
                
                chunk_idx += len(batch)
                
                # Slowly recover batch size if we had previously shrunk it
                if current_batch_size < settings.EMBEDDING_BATCH_SIZE:
                    current_batch_size = min(settings.EMBEDDING_BATCH_SIZE, current_batch_size + 4)
                    
            except Exception as batch_exc:
                # Dynamic fallback: if embedding failed (e.g. 429 too big), shrink batch and try again
                logger.warning("Batch of %d failed: %s", len(batch), batch_exc)
                if current_batch_size > settings.EMBEDDING_BATCH_SIZE_FLOOR:
                    current_batch_size = max(settings.EMBEDDING_BATCH_SIZE_FLOOR, current_batch_size // 2)
                    logger.info("Falling back to smaller batch size: %d", current_batch_size)
                    # Do not increment chunk_idx, loop will retry with smaller batch
                else:
                    logger.error("Batch failed at minimum size of 1. Chunk is unrecoverable.")
                    failed_chunks_count += 1
                    chunk_idx += 1
                    await _update_document(doc.id, failed_chunks=failed_chunks_count)
                    # We continue to the next chunk so one bad chunk doesn't poison the whole doc
        
        if failed_chunks_count > 0:
            raise RuntimeError(f"{failed_chunks_count} chunks failed to embed permanently.")

        # ── 5.5 ACL 载荷收尾对齐（必须在标记 COMPLETED 之前）──────────────────
        # 发布操作可能发生在向量点写入**之前**（上传异步 + "传完立刻发布"是正常
        # 路径）：那时 Qdrant 的 set_payload 匹配 0 个点并静默成功，随后入库把
        # 上传时刻的 private 写进 payload，PG 与向量库永久分叉 —— 列表里看得见、
        # 检索却查不到。这里是唯一"所有点必定已 upsert 完成"的时点，能追平此前
        # 发生的任何层级变更。详见 resync_document_acl_payload 的详细说明。
        #
        # 传 expected=入库快照：绝大多数文档从未发布，PG 真值与快照一致时直接
        # 跳过，省掉一次 count + set_payload。
        # 失败只告警：ACL 载荷是 PG 的副本，对齐不该让整份文档判失败。
        try:
            from app.services.knowledge_tier_service import (
                resync_document_acl_payload,
            )

            await resync_document_acl_payload(
                doc.id,
                reason="ingestion_completion",
                expected=(access_level, department_id),
            )
        except Exception:      # noqa: BLE001 — 载荷对齐失败不阻断入库收尾
            logger.exception(
                "Document '%s' (id=%s): ACL payload resync failed — PostgreSQL stays "
                "authoritative, but the vector-side ACL copy may be stale",
                filename, doc_id_str,
            )

        # ── 5.6【T4】对象级权限物化（必须在标记 COMPLETED 之前）───────────────────
        # 为这份文档的五种对象（doc / text_chunk / table / code / image）写
        # `document_objects` 权威行，并把权限字段冗余推给 Qdrant payload。
        # 位置与 resync_document_acl_payload 同理：此刻"所有向量点必定已 upsert
        # 完成"，是唯一能保证派生对象与源文档权限一致的时点（§7.1 写入时机）。
        # OCR 派生块（image_id 非空）的 parent_object_id 指向**源图片对象**，
        # 有效密级取 max(文档, 源图) —— 堵住"图看不了但字还能搜到"的破口（决策 12）。
        # 失败只告警：权限副本可异步追平，不该让整份文档判失败（PG 仍是权威源）。
        try:
            from sqlalchemy import select as _select

            from app.services.security_cascade import materialize_document_objects

            async with get_db_session() as _sess:
                fresh_doc = (
                    await _sess.execute(
                        _select(Document).where(Document.id == doc.id)
                    )
                ).scalar_one_or_none()
            acl_points = [
                {
                    "id": pid,
                    "payload": {
                        "image_id": c.image_id,
                        "image_path": c.image_path,
                        "content_type": c.content_type,
                        "chunk_index": c.chunk_index,
                        "page_number": c.page_number,
                    },
                }
                for pid, c in chunk_map.items()
            ]
            acl_stats = await materialize_document_objects(fresh_doc or doc, acl_points)
            logger.info(
                "Document '%s' (id=%s): materialized %s object ACL rows",
                filename, doc_id_str, acl_stats,
            )
        except Exception:      # noqa: BLE001 — 物化失败不阻断入库收尾
            logger.exception(
                "Document '%s' (id=%s): document_objects materialization failed — "
                "PostgreSQL stays authoritative, object-level ACL may be incomplete",
                filename, doc_id_str,
            )

        # ── 6. Mark COMPLETED ─────────────────────────────────────────────────
        await _update_document(
            doc.id,
            status=DocumentStatus.COMPLETED,
            current_stage="completed"
        )

        return DocumentResult(
            document_id=doc.id,
            filename=filename,
            status=DocumentStatus.COMPLETED,
            page_count=extraction.page_count,
            # 必须用 all_chunks（文本块 + 独立图片块），与落库的 chunk_count 一致
            chunk_count=len(all_chunks),
            # ── 部分1/2：图片统计（识别到的图片总数 / 建成独立检索对象的图片数）──
            image_count=len(extraction.images),
            image_object_count=len(image_chunks),
            file_size_bytes=file_size,
            created_at=doc.created_at,
            # 段落级注入屏蔽数（清洗与屏蔽现在同属 text_cleaning.CleanedExtraction）。
            # ⚠️ 这里曾残留 `scan.hit_count if not scan.clean else 0` —— 清洗接入后
            # `scan` 已不存在，NameError 会在**整条管线跑完的最后一刻**抛出，于是
            # chunk / 向量 / 图片全都写好了，文档状态却是 FAILED（前端显示"失败"，
            # 检索却查得到）。这类"只在收尾处炸"的错最难从单测发现，必须靠真实
            # 入库链路回归（见 _audit_916/audit_iso.py）。
            injection_masked=cleaned.masked_paragraphs,
        )

    except Exception as exc:
        logger.exception("Failed to process '%s': %s", filename, exc)
        # 建行已在 prepare_upload 中完成（且它自行处理建行失败），走到这里 doc
        # 必然存在，于是失败一律落在**这一行**上 —— 前端才能指着具体文档说
        # "这份失败了、原因是 X"，而不是只抛一个孤零零的 500。
        await _update_document(
            doc.id,
            status=DocumentStatus.FAILED,
            current_stage="failed",
            error_message=str(exc)[:2000],
        )
        return DocumentResult(
            document_id=doc.id,
            filename=filename,
            status=DocumentStatus.FAILED,
            file_size_bytes=file_size,
            error=str(exc),
            created_at=doc.created_at,
        )


async def _process_single_file(
    filename: str,
    content: bytes,
    settings,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
    department_id: str | None = None,
    access_level: str | None = None,
) -> DocumentResult:
    """同步版单文件入库：判重 + 建行 + 跑完整管线，等全部结束才返回。

    保留给脚本、批处理与测试使用（它们要的就是"调完即已入库"）。
    面向浏览器的 `POST /upload` 走 `prepare_upload` +
    `schedule_ingestion`，不等管线跑完。
    """
    prep = await prepare_upload(
        filename, content,
        owner_id=owner_id, collection_id=collection_id,
        tenant_id=tenant_id, department_id=department_id,
        access_level=access_level,
    )
    if prep.done:
        return prep.result  # type: ignore[return-value]
    return await _run_ingestion(
        prep, filename, content, settings,
        owner_id=owner_id, collection_id=collection_id,
    )


async def _ingestion_runner(
    doc_id: uuid.UUID,
    prep: PreparedUpload,
    filename: str,
    content: bytes,
    settings,
    owner_id: uuid.UUID | None,
    collection_id: uuid.UUID | None,
) -> None:
    """后台任务入口：跑完管线后把任务从注册表摘掉。"""
    try:
        await _run_ingestion(
            prep, filename, content, settings,
            owner_id=owner_id, collection_id=collection_id,
        )
    finally:
        _ingestion_tasks.pop(doc_id, None)


def schedule_ingestion(
    prep: PreparedUpload,
    filename: str,
    content: bytes,
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
) -> DocumentResult:
    """把入库排进后台任务并立即返回「已受理」结果。

    进程重启会带走未完成的后台任务 —— 这正是 `recover_stuck_documents()`
    存在的意义：下次启动把这些行标成 FAILED，用户重传同一文件即幂等续传。
    """
    assert prep.doc is not None, "schedule_ingestion 需要 prepare_upload 产出的文档行"
    doc = prep.doc
    task = asyncio.create_task(
        _ingestion_runner(
            doc.id, prep, filename, content, get_settings(), owner_id, collection_id,
        )
    )
    _ingestion_tasks[doc.id] = task
    task.add_done_callback(lambda _t, _id=doc.id: _ingestion_tasks.pop(_id, None))
    logger.info(
        "Ingestion scheduled in background — id=%s filename='%s' owner=%s (active=%d)",
        doc.id, filename, owner_id, active_ingestion_count(),
    )
    return DocumentResult(
        document_id=doc.id,
        filename=filename,
        status=DocumentStatus.PENDING,
        message="已受理，正在后台解析并建立索引。可以离开此页面，稍后在文档页查看进度。",
        file_size_bytes=prep.file_size,
        created_at=doc.created_at,
    )


async def process_uploads(
    files: list[tuple[str, bytes]],
    owner_id: uuid.UUID | None = None,
    collection_id: uuid.UUID | None = None,
    tenant_id: str | None = None,
    department_id: str | None = None,
    access_level: str | None = None,
) -> UploadResponse:
    """
    Process multiple uploads sequentially, attributing them to *owner_id*
    and grouping them into *collection_id* when provided.

    *tenant_id* / *department_id* / *access_level* 来自上传者的身份上下文
    （三层隔离）：文档行与 Qdrant payload 同源写入，检索层据此做
    租户前置过滤 + Document ACL。
    """
    settings = get_settings()
    results: list[DocumentResult] = []

    for filename, content in files:
        result = await _process_single_file(
            filename, content, settings,
            owner_id=owner_id, collection_id=collection_id,
            tenant_id=tenant_id, department_id=department_id,
            access_level=access_level,
        )
        results.append(result)

    succeeded = sum(1 for r in results if r.status == DocumentStatus.COMPLETED)
    failed = sum(1 for r in results if r.status == DocumentStatus.FAILED)

    return UploadResponse(
        total=len(results),
        succeeded=succeeded,
        failed=failed,
        documents=results,
    )


async def recover_stuck_documents() -> None:
    """
    On startup, find any documents that were left in a processing state due
    to a server crash. Mark them as FAILED so they can be resumed cleanly
    by the idempotent resume logic on next upload.
    """
    async with get_db_session() as session:
        result = await session.execute(
            update(Document)
            .where(Document.status.in_([
                DocumentStatus.PENDING,
                DocumentStatus.PARSING,
                DocumentStatus.CHUNKING,
                DocumentStatus.EMBEDDING,
                DocumentStatus.INDEXING,
            ]))
            .values(
                status=DocumentStatus.FAILED,
                current_stage="stuck_recovered",
                error_message="Server restarted during processing. Re-upload the identical file to resume."
            )
        )
        if result.rowcount > 0:
            logger.info("Recovered %d stuck documents from previous crash. Marked as FAILED to allow resume.", result.rowcount)

