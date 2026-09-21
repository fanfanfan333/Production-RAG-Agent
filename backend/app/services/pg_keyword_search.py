"""
关键词腿（全文检索）—— 上千文档时的规模关键.

历史实现的三个问题
──────────────────
1. **静默截断**：`retrieval_service` 把整个 Qdrant 语料 scroll 进内存建 BM25，
   但受 ``HYBRID_MAX_CORPUS_POINTS``（默认 10000）限制。1000 份文档 × 每份
   约 200 chunk ≈ 20 万条，关键词腿只看得到**前 5%**。查询里含只有靠后文档
   才有的型号/编号时，用户得到"知识库没有"——文档其实就在库里，而且链路不报错。
2. **内存放大**：BM25 的倒排表是 Python dict/Counter，20 万条中文 chunk 的
   bigram 词项表轻松上 GB。与同机的 Ollama 抢内存，冷启动时直接 OOM。
3. **缓存泄漏**：``_bm25_cache`` 按权限上下文分键且**永不淘汰**，
   租户/部门组合一多，缓存条目无限增长。

本模块把关键词腿移到 PostgreSQL：词项在入库时落库，检索走 GIN 索引 + 
``ts_rank_cd`` 排序。收益是"召回线性可控 + 内存零放大 + 权限可下推"。

与 BM25 的关系（诚实说明）
──────────────────────────
``ts_rank_cd`` **不是** BM25 —— 它没有 k1/b 的长度归一，也没有 IDF 的
saturation 项。所以：

  * 排序**质量**略逊于真正的 BM25（尤其是"长文档天然占优"这一点）；
  * 但它是**带索引的**，而内存 BM25 在本项目里连完整语料都覆盖不到。

在 1000+ 文档的真实规模下，"覆盖 100% 语料的中等排序"远胜"覆盖 5% 语料的
好排序"—— 后者会把漏召回伪装成"没有相关资料"。这也是默认后端选 postgres 的
唯一理由。小语料（< ``HYBRID_MAX_CORPUS_POINTS``）时用内存 BM25 仍然更好，
把 ``HYBRID_KEYWORD_BACKEND=memory`` 切回去即可。
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.logging import get_logger

logger = get_logger(__name__)

# tsquery 里只允许字母/数字/中日韩汉字作为词元；其余（- . _ 与操作符 & | ! ( ) :）
# 一律剥掉。不剥的话，一个含 "-" 的型号（"abx-300"）会被 tsquery 解析成
# "NOT" 表达式，静默变成完全不同的检索意图。
_TSQUERY_SAFE_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff\u3040-\u30ff]+")

# 单次查询最多拼多少个 OR 词元。太多会让 tsquery 解析与打分变慢，
# 而超过这个数量的词元对召回几乎没有增量贡献（它们已经是长尾）。
_MAX_TSQUERY_TERMS = 64


def terms_for_text(text: str, max_chars: int = 4000) -> str:
    """
    把 chunk 正文转成"空格分隔的词项串"，供 ``to_tsvector('simple', ...)`` 索引.

    复用 ``hybrid_search.tokenize`` —— 与查询侧**必须是同一套切分**，
    否则索引里的词元和查询里的词元永远对不上（一个最隐蔽的"检索突然查不到"
    成因：两边分词口径不同，但各自都跑得好好的）。

    截断到 *max_chars*：超长 chunk（表格/代码）的词项数量爆炸，而尾部词项对
    召回的贡献极低。截断发生在**字符**层面而不是词项层面，保持函数便宜（不需要
    先把十万字符全 tokenize 再丢）。
    """
    from app.services.hybrid_search import tokenize

    if not text:
        return ""
    body = text[:max_chars] if max_chars > 0 else text
    tokens = tokenize(body)
    if not tokens:
        return ""
    # 去重但不改变顺序意义：词项串只用于建 GIN 索引，重复词元不会改变
    # ts_rank_cd 的打分（它按覆盖度而非词频），去掉能显著缩小索引体积。
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " ".join(unique)


def build_tsquery(query: str) -> str:
    """
    把自然语言查询转成 tsquery 字符串（OR 连接）.

    为什么是 OR 而不是 ``plainto_tsquery`` 的 AND：AND 要求文档包含**全部**
    查询词元，中文 bigram 一多就必然零结果（"2024年主营业务收入" 会产生
    "年主""主营""营业""务收""收入" 等十余个 bigram，几乎没有文档全含）。
    那样关键词腿会长期返回空 —— 看起来"没有精确匹配"，其实是查询写错了语义。

    OR + ``ts_rank_cd`` 则按"覆盖了多少个查询词元"排序，正是我们想要的：
    覆盖越多排越前，覆盖少的也有机会进候选池。
    """
    from app.services.hybrid_search import tokenize

    if not query:
        return ""
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in tokenize(query):
        clean = _TSQUERY_SAFE_RE.sub("", raw)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        tokens.append(clean)
        if len(tokens) >= _MAX_TSQUERY_TERMS:
            break
    return " | ".join(tokens)


def is_query_usable(tsquery: str) -> bool:
    return bool(tsquery.strip())


async def replace_document_terms(
    session: AsyncSession,
    *,
    document_id: uuid.UUID | str,
    tenant_id: str,
    access_level: str,
    department_id: str | None,
    collection_id: str | None,
    rows: list[tuple[int, str, str]],
) -> int:
    """
    用新的词项**整体替换**一份文档的词项行（幂等）.

    先删后插而不是 upsert：文档重新入库时 chunk 数量可能变少（分块参数改了、
    正文清洗后变短），留下来的旧行会指向已不存在的 chunk_index，检索命中后
    在 Qdrant 里找不到对应点，表现为"召回了一条空证据"，进一步把 evidence gate
    的覆盖率判定带偏。整体替换没有这个尾巴。

    *rows* 是 ``[(chunk_index, point_id, terms_string), ...]``，
    空 terms 的行不会被写入。
    """
    from sqlalchemy import delete
    from app.db.models import DocumentChunkTerm

    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id

    await session.execute(
        delete(DocumentChunkTerm).where(DocumentChunkTerm.document_id == doc_uuid)
    )

    payload = [
        {
            "document_id": doc_uuid,
            "tenant_id": tenant_id,
            "access_level": access_level,
            "department_id": department_id,
            "collection_id": collection_id,
            "chunk_index": int(chunk_index),
            "point_id": str(point_id),
            "terms": terms,
        }
        for chunk_index, point_id, terms in rows
        if terms and terms.strip() and point_id
    ]
    if payload:
        session.add_all([DocumentChunkTerm(**item) for item in payload])
    await session.flush()
    return len(payload)


async def delete_document_terms(
    session: AsyncSession,
    document_id: uuid.UUID | str,
) -> None:
    """删除一份文档的全部词项行（文档被删时调用）."""
    from sqlalchemy import delete
    from app.db.models import DocumentChunkTerm

    doc_uuid = uuid.UUID(str(document_id)) if not isinstance(document_id, uuid.UUID) else document_id
    await session.execute(
        delete(DocumentChunkTerm).where(DocumentChunkTerm.document_id == doc_uuid)
    )


def _apply_metadata_filter(stmt, metadata_filter, term_model):
    """
    把 ``MetadataFilter`` 下推成 SQL 谓词（strict：条件必须满足，缺失即排除）.

    三类条件的下推方式刻意不同：

      · ``document_ids`` / ``exclude_document_ids`` —— 纯 id，直接 on 点表的
        ``document_id``，**不需要 join**。这是"指定文档范围"的最常见用法，
        走这条路的查询不会被 metadata 表的 join 拖慢。
      · ``doc_types`` / ``years`` / ``languages`` —— 走 ``document_metadata``
        的**显式列**（都有索引）。刻意不查 ``payload`` JSONB：JSONB 上的条件
        没有表达式索引会退化成顺序扫描，而这三类恰好是过滤频率最高的。
      · ``tags`` —— 只能查 JSONB（业务标签是变长数组），用 ``?|`` 数组包含
        操作符而不是 ``@>``：用户问"财务部的材料"时命中文档的标签集里**任意
        一个**匹配即算命中（``@>`` 要求全含，会把只打了"财务"但没打"审计"的
        文档排除掉，而用户并没有要求同时属于两个部门）。

    条件为空则完全不加谓词 —— 与 ``MetadataFilter.is_empty`` 的语义对齐，
    避免"没指定年份"被实现成"年份 IS NULL"。
    """
    if metadata_filter is None:
        return stmt

    from sqlalchemy import cast
    from sqlalchemy.dialects.postgresql import array as pg_array
    from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
    from sqlalchemy import Text as SAText

    from app.db.models import DocumentMetadataRow

    ids = _uuids_or_none(metadata_filter.document_ids)
    if metadata_filter.document_ids:
        # 显式给了 id 但全部解析失败 → 空集（不能悄悄退化成"不过滤"）
        stmt = (
            stmt.where(term_model.document_id.in_(ids))
            if ids else stmt.where(term_model.document_id.is_(None))
        )
    excluded = _uuids_or_none(metadata_filter.exclude_document_ids)
    if excluded:
        stmt = stmt.where(~term_model.document_id.in_(excluded))

    if not (metadata_filter.doc_types or metadata_filter.years
            or metadata_filter.languages or metadata_filter.tags):
        return stmt

    stmt = stmt.join(
        DocumentMetadataRow,
        DocumentMetadataRow.document_id == term_model.document_id,
    )
    if metadata_filter.doc_types:
        stmt = stmt.where(DocumentMetadataRow.doc_type.in_(list(metadata_filter.doc_types)))
    if metadata_filter.years:
        stmt = stmt.where(DocumentMetadataRow.doc_year.in_(list(metadata_filter.years)))
    if metadata_filter.languages:
        stmt = stmt.where(DocumentMetadataRow.language.in_(list(metadata_filter.languages)))
    if metadata_filter.tags:
        tags_expr = cast(
            pg_array([_lit(t) for t in metadata_filter.tags]),
            PG_ARRAY(SAText),
        )
        stmt = stmt.where(
            DocumentMetadataRow.payload["business_tags"].op("?|")(tags_expr)
        )
    return stmt


def _lit(value):
    """一个带类型标注的字符串字面量（供 ``pg_array`` 组合，避免隐式类型推断）."""
    from sqlalchemy import literal, String as SAString
    return literal(str(value), type_=SAString)


def _uuids_or_none(values) -> list:
    out = []
    for v in values or []:
        try:
            out.append(uuid.UUID(str(v)))
        except (TypeError, ValueError):
            continue
    return out


async def keyword_search(
    session: AsyncSession,
    *,
    query: str,
    limit: int,
    tenant_ids: frozenset[str] | None,
    owner_id: str | None,
    user_department_id: str | None,
    tenant_wide: bool = False,
    owns_tenant_ids: frozenset[str] = frozenset(),
    collection_id: str | None = None,
    unrestricted: bool = False,
    metadata_filter=None,
    scope=None,
    pred=None,
) -> list[tuple[str, int, str, float]]:
    """
    关键词检索，返回 ``[(document_id, chunk_index, point_id, rank), ...]``.

    ``point_id`` 是 Qdrant 的点 id：调用方据此批量 retrieve 取回 payload，
    无需为了拿 payload 再 scroll 整个语料（那正是本模块要消灭的内存放大）。

    权限过滤与向量腿**用同一个规则**（``tenancy.document_acl_clause``）：
    两条腿的可见集合不一致时，RRF 融合出的结果会包含某一条腿看不到的文档，
    表现为"检索能召回、点开引用却说无权限"这种自相矛盾。

    ``metadata_filter``（``metadata.MetadataFilter``）在 SQL 侧下推，与向量腿
    的 Qdrant 前置过滤**同一语义（strict）**：命中的每类条件都必须满足，字段
    缺失的行按不匹配处理（join 不上就是排除）。两条腿的搜索空间必须一致 ——
    向量腿被年份收窄而关键词腿没有的话，关键词腿会把别的年份的精确词命中送进
    RRF，恰好把被排除的段落又拉回候选池，"过滤"就形同虚设。

    【T3 双路同源】``scope``（``UserScope``）给了时，权限条件改用
    ``to_sql(scope.predicate(), Document)``：它与向量腿的 ``to_qdrant(pred)``
    由**同一个** ``ScopePredicate`` 编译，并在此之上追加密级 / 项目 / deny /
    excluded（决策 6.1「同一 pred 只构造一次」）。``scope`` 为 ``None`` 时退回
    既有 ``document_scope_clause`` 三维行为（旧调用点零变化）。

    全部过滤在 SQL 侧完成（索引 + join），不把候选拉进进程再筛。
    """
    from app.db.models import Document, DocumentChunkTerm, DocumentStatus
    from app.services.tenancy import document_scope_clause

    tsquery = build_tsquery(query)
    if not is_query_usable(tsquery) or limit <= 0:
        return []
    if not unrestricted and scope is None and pred is None and not (
        owner_id is not None or tenant_ids is not None
        or tenant_wide or bool(owns_tenant_ids)
    ):
        # 与 retrieve_chunks 的 fail-closed 一致：没有权限上下文就不检索
        logger.error("keyword_search called without permission scope — refused (fail-closed)")
        return []

    # 直接读**存储生成列** terms_tsv，而不是在查询里现算
    # ``to_tsvector('simple', terms)``。
    #
    # 这个区别是数量级的：表达式写法下 GIN 只加速 WHERE 的 @@，排序那一半
    # ``ts_rank_cd(to_tsvector('simple', terms), q)`` 会被 PG 当成普通表达式，
    # 对**每一条命中行重新分词一次**。实测 20 万条命中行（每行约 250 个中文
    # bigram）单次查询 **178 秒**，99% 耗在重复分词上 —— 而且完全不报错，
    # 只是"检索越来越慢"，规模越大越慢，根因藏在 SQL 里。
    # 读存储列后，写侧只在入库时分词一次，检索耗时与命中集大小解耦。
    tsv = DocumentChunkTerm.terms_tsv
    tq = func.to_tsquery("simple", tsquery)
    rank = func.ts_rank_cd(tsv, tq).label("rank")

    stmt = (
        select(
            DocumentChunkTerm.document_id,
            DocumentChunkTerm.chunk_index,
            DocumentChunkTerm.point_id,
            rank,
        )
        .join(Document, Document.id == DocumentChunkTerm.document_id)
        .where(tsv.op("@@")(tq))
        .where(Document.status == DocumentStatus.COMPLETED)
    )

    if collection_id:
        stmt = stmt.where(DocumentChunkTerm.collection_id == collection_id)

    if not unrestricted and pred is not None:
        # 【T3 双路同源（决策 6.1）】入口（retrieve_chunks）已编译好的**同一个**
        # ScopePredicate 直接下推：与向量腿 ``to_qdrant(pred)`` 用的是同一个实例，
        # 本腿不再 ``scope.predicate()`` 现编（否则会得到另一个实例，破坏同源）。
        # to_sql 内部**复用**既有 document_scope_clause（三维一行不改），并追加
        # 密级/项目/deny/excluded。编译失败必须 fail-closed（返回空）。
        from app.services.security_policy import ScopeCompileError, to_sql

        try:
            stmt = stmt.where(to_sql(pred, Document))
        except ScopeCompileError:
            logger.error("keyword_search: to_sql(pred) 编译失败 — fail-closed 返回空")
            return []
        except Exception:      # noqa: BLE001
            logger.exception("keyword_search: to_sql(pred) 意外失败 — fail-closed 返回空")
            return []
    elif not unrestricted and scope is not None:
        # 兼容旧调用点：只给了 UserScope 时，本腿自行现编一个 pred（这会得到与
        # 向量腿**不同**的实例）—— 仅用于不在 T3 主链路上的历史调用；主链路
        # 一律经 ``pred=`` 传入同一实例。
        from app.services.security_policy import ScopeCompileError, to_sql

        try:
            compiled = scope.predicate()
            stmt = stmt.where(to_sql(compiled, Document))
        except ScopeCompileError:
            logger.error("keyword_search: to_sql(pred) 编译失败 — fail-closed 返回空")
            return []
        except Exception:      # noqa: BLE001
            logger.exception("keyword_search: 编译 ScopePredicate 失败 — fail-closed 返回空")
            return []
    elif not unrestricted:
        # 三层隔离的唯一 SQL 组装点：公司边界 ∪ 自己个人库（private 与租户无关）。
        # 与向量腿用**同一个** `document_scope_clause` —— 两条腿可见集合不一致时，
        # RRF 融合出的结果会包含某一条腿看不到的文档。
        try:
            owner_uuid = uuid.UUID(owner_id) if owner_id else None
        except (TypeError, ValueError):
            owner_uuid = None
        stmt = stmt.where(
            document_scope_clause(
                owner_id=owner_uuid,
                department_id=user_department_id,
                tenant_ids=tenant_ids,
                owns_tenant_ids=owns_tenant_ids,
                tenant_wide=tenant_wide,
            )
        )

    stmt = _apply_metadata_filter(stmt, metadata_filter, DocumentChunkTerm)

    stmt = stmt.order_by(rank.desc()).limit(max(1, int(limit)))

    try:
        result = await session.execute(stmt)
    except Exception:
        # 索引缺失/迁移没跑（document_chunk_terms 不存在）时会抛在这里。
        # 关键词腿是增益而非依赖 —— 记日志后返回空，让向量腿独立完成检索，
        # 而不是让整个提问 500。
        logger.exception("keyword_search failed — degrading to vector-only")
        return []

    out: list[tuple[str, int, str, float]] = []
    for row in result.all():
        try:
            out.append((str(row[0]), int(row[1]), str(row[2]), float(row[3] or 0.0)))
        except (TypeError, ValueError):
            continue
    logger.debug("keyword_search: %d hit(s) for tsquery=%r", len(out), tsquery[:120])
    return out


async def count_terms(session: AsyncSession) -> int:
    """词项行总数（诊断/健康检查用：确认关键词腿真的有数据）."""
    from app.db.models import DocumentChunkTerm

    try:
        result = await session.execute(select(func.count()).select_from(DocumentChunkTerm))
        return int(result.scalar() or 0)
    except Exception:      # noqa: BLE001
        return 0
