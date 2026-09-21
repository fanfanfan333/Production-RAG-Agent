"""
Qdrant vector store operations.

Responsibilities:
  - Ensure the collection exists with the correct vector configuration.
  - Upsert document chunk vectors with rich payload for retrieval.
  - Provide a clean delete-by-document-id helper for future use.
"""

import uuid
from dataclasses import dataclass, field

from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.utils.logging import get_logger

logger = get_logger(__name__)

NAMESPACE_RAG = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# 允许进 Qdrant payload 的元数据键白名单 —— 与 metadata.DocumentMetadata.to_payload
# 一一对应，也对应 `_ensure_payload_indexes` 里建了索引的字段。白名单而不是
# `**meta`：payload 会被复制到每个 chunk，塞进任何非预期字段都会放大存储，
# 且会制造"库里字段名悄悄变了但索引没跟上"的静默退化。
_DOC_META_KEYS = (
    "title", "doc_type", "doc_year", "language", "doc_number",
    "author", "keywords", "business_tags",
)


def _flatten_doc_meta(meta: dict | None) -> dict:
    """把元数据拍平成可直接进 payload 的标量/短数组（白名单 + 截断）."""
    if not isinstance(meta, dict):
        return {}
    out: dict = {}
    for key in _DOC_META_KEYS:
        if key not in meta:
            continue
        value = meta[key]
        if value is None or value == "" or value == []:
            continue
        if isinstance(value, (list, tuple)):
            out[key] = [str(v)[:64] for v in list(value)[:12] if str(v).strip()]
        elif isinstance(value, bool):
            out[key] = value
        elif isinstance(value, int):
            out[key] = value
        else:
            out[key] = str(value)[:200]
    return out


def generate_point_id(
    document_id: str, filename: str, page_number: int, section: str | None, heading: str | None, chunk_index: int, text: str,
    content_type: str = "text",
    image_id: str | None = None,
) -> str:
    """
    Generate a deterministic UUID for a chunk based on its content and metadata.

    document_id participates in the key: the same file uploaded by different
    users (different Document rows) gets different vectors — otherwise the
    second upload would OVERWRITE the first user's points and steal their
    vectors (isolation bug).

    content_type / image_id participate too: an image chunk and a text chunk
    that happen to share a chunk_index (should not happen, but defensive) still
    get distinct ids, so the image/table对象永远不会覆盖正文向量。
    """
    stable_name = (
        f"{document_id}::{filename}::{page_number}::{section}::{heading}::{chunk_index}::{content_type}::{image_id or ''}::{text}"
    )
    return str(uuid.uuid5(NAMESPACE_RAG, stable_name))



@dataclass
class VectorPoint:
    """A single vector + payload to be upserted into Qdrant."""

    vector: list[float]
    document_id: str           # UUID string of the parent Document row
    filename: str
    chunk_index: int
    page_number: int
    text: str
    heading: str | None = None
    section: str | None = None
    collection_id: str | None = None   # KB collection UUID (business grouping)

    # ── Multi-Tenant 隔离（第一、二层）──────────────────────────────────────
    # tenant_id 进 payload 并建 keyword 索引：检索的**前置**过滤（向量 ANN 与
    # BM25 语料 scroll 都带 tenant filter），跨租户 chunk 在检索阶段不可见。
    # user_id 是上传者；access_level/department_id 是 Document ACL 的载荷，
    # PG valid_docs 校验与它们保持一致（双写同源：都来自 Document 行）。
    tenant_id: str = "default"
    user_id: str | None = None
    access_level: str = "private"
    department_id: str | None = None

    # ── 位置信息（细粒度引用溯源）────────────────────────────────────────────
    # 1-based 闭区间行号；旧 chunk 为 None，检索/引用层自动降级为只显示页码。
    line_start: int | None = None
    line_end: int | None = None

    # ── 内容类型与图片信息（部分3：Qdrant payload 增加图片信息）─────────────────
    # content_type: text | table | image
    # image_*: 仅 content_type="image" 时非空，检索命中后可回显原始图片
    content_type: str = "text"
    image_id: str | None = None
    image_path: str | None = None       # 文档相对路径 images/page_3_image_1.png
    image_caption: str | None = None
    # 图片分类结论（三层图片处理）：table | formula | code | chart | diagram |
    # screenshot | photo
    image_type: str | None = None
    # 产出该图片内容的引擎（docling / table-transformer / paddleocr / vision …）
    analyze_engine: str | None = None
    # 置信度门控：最终置信度 + 是否需要人工复核
    analyze_confidence: float = 0.0
    manual_review: bool = False
    # ── 产出质检 + 双通道融合（可验证的事实）────────────────────────────────
    # analyze_quality：代码语法 / OCR 行置信度 / VLM 幻觉检查 / 结构校验；
    # analyze_fusion：双通道策略与最终选中的通道。整份报告入库（不扁平化），
    # 因为前端要按 reason 逐条展示警示；旧索引缺这两个字段时检索层自动降级。
    analyze_quality: dict = field(default_factory=dict)
    analyze_fusion: dict = field(default_factory=dict)

    # ── 图片位置（保留图片位置 → 细粒度引用）──────────────────────────────────
    # position：图片在整篇文档中的序号（1-based，跨页累计）。文本块为 None。
    # bbox：图片在所在页面上的边界框 (x1, y1, x2, y2)，点坐标、原点左上、y 向下。
    #   DOCX 无页面几何 → None，检索/引用层自动降级为只显示页码 + 序号。
    position: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    # ── small-to-big / Hierarchical RAG metadata（全部 optional 向后兼容） ──
    # ⚠️ parent_text 已**弃用**：它把父块正文复制进每一个子块，1000 份文档时
    # 是 GB 级冗余。新链路只写 parent_id，正文由 ChunkParent 表持有
    # （见 metadata / chunker 模块注释）。保留字段是为了让旧调用点不炸，
    # 新代码不要再填它。
    parent_id: str | None = None
    parent_text: str | None = None
    parent_char_start: int | None = None
    parent_char_end: int | None = None

    # ── 结构感知父子（新）──────────────────────────────────────────────────────
    parent_index: int | None = None
    section_id: str | None = None
    section_path: list[str] | None = None

    # ── Metadata 系统的可过滤字段（扁平、已建 payload 索引）────────────────────
    # 只放"过滤要用"的：完整元数据（outline / keywords 全量）在 PG 的
    # document_metadata 表里，不在这里随每个 chunk 复制。
    doc_meta: dict = field(default_factory=dict)


async def ensure_collection() -> None:
    """
    Create the Qdrant collection if it does not already exist.
    Safe to call on every startup — idempotent.
    """
    settings = get_settings()
    client = get_qdrant_client()
    collection_name = settings.QDRANT_COLLECTION

    existing = await client.get_collections()
    names = {c.name for c in existing.collections}

    if collection_name in names:
        logger.debug("Qdrant collection '%s' already exists", collection_name)
        await _ensure_payload_indexes(collection_name)
        await _ensure_collection_tuning(collection_name)
        return

    await client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=settings.EMBEDDING_DIMENSION,
            distance=qmodels.Distance.COSINE,
        ),
        # Optimiser settings tuned for read-heavy RAG workloads.
        # memmap_threshold：大语料下向量走内存映射，不常驻内存（见 _ensure_collection_tuning）
        optimizers_config=qmodels.OptimizersConfigDiff(
            indexing_threshold=20_000,
            memmap_threshold=50_000,
        ),
        # Payload index for fast filtered retrieval by document_id
        on_disk_payload=True,
    )

    await _ensure_payload_indexes(collection_name)
    await _ensure_collection_tuning(collection_name)

    logger.info(
        "Created Qdrant collection '%s' (dim=%d, distance=COSINE)",
        collection_name,
        settings.EMBEDDING_DIMENSION,
    )


async def _ensure_payload_indexes(collection_name: str) -> None:
    """Idempotently create the payload indexes used by filtered retrieval."""
    client = get_qdrant_client()

    indexes = [
        ("document_id", qmodels.PayloadSchemaType.KEYWORD),
        ("collection_id", qmodels.PayloadSchemaType.KEYWORD),  # KB filtering (第一阶段)
        ("tenant_id", qmodels.PayloadSchemaType.KEYWORD),      # 租户检索前置过滤（三层隔离）
        ("access_level", qmodels.PayloadSchemaType.KEYWORD),   # Document ACL 过滤
        ("content_type", qmodels.PayloadSchemaType.KEYWORD),   # 图文分流 / 表格检索 (部分3)
        ("image_type", qmodels.PayloadSchemaType.KEYWORD),     # 按图片类型过滤（三层图片处理）
        ("analyze_engine", qmodels.PayloadSchemaType.KEYWORD), # 按产出引擎过滤（多引擎管线）
        ("manual_review", qmodels.PayloadSchemaType.BOOL),      # 捞出"需人工复核"的图片块
        # ── 规模与隔离补充（1000+ 文档时缺一不可）────────────────────────────
        # department_id / user_id：Qdrant 的过滤是"有索引才走索引"，没有索引的
        # 过滤条件会退化成**全量扫描后再筛**。它们在 Document ACL 里是核心维度
        # （部门库同部门可见、个人库仅本人），不建索引等于每次检索都在扫全库。
        ("department_id", qmodels.PayloadSchemaType.KEYWORD),
        ("user_id", qmodels.PayloadSchemaType.KEYWORD),
        # ── Metadata 系统的可过滤维度（缩小候选池，提升上千文档时的准确率）──
        ("doc_type", qmodels.PayloadSchemaType.KEYWORD),
        ("doc_year", qmodels.PayloadSchemaType.INTEGER),
        ("language", qmodels.PayloadSchemaType.KEYWORD),
        ("business_tags", qmodels.PayloadSchemaType.KEYWORD),
        ("keywords", qmodels.PayloadSchemaType.KEYWORD),
        # ── Parent-Child：按父块/章节聚合（同父去重、章节级回填）──────────────
        ("parent_id", qmodels.PayloadSchemaType.KEYWORD),
        ("section_id", qmodels.PayloadSchemaType.KEYWORD),
        # ── FIX-E（T5 预发布）：五维隔离过滤字段索引（Deny-4/5/6 前置剪枝用）────
        # 这 9 个字段是 Qdrant 侧密级 / 项目 / deny / 剔除过滤的输入。不建索引时
        # 过滤退化成全量扫描（性能问题 + 可被放大量候选池拖垮的 DoS 面）。
        # 类型：数值用 INTEGER/FLOAT，字符串用 KEYWORD，数组用 KEYWORD（Qdrant
        # 自动按数组处理）。保持本函数**幂等**（已存在的索引异常被忽略）。
        ("security_level", qmodels.PayloadSchemaType.INTEGER),
        ("parent_security_level", qmodels.PayloadSchemaType.INTEGER),
        ("effective_security_level", qmodels.PayloadSchemaType.INTEGER),
        ("visibility_mode", qmodels.PayloadSchemaType.KEYWORD),
        ("project_ids", qmodels.PayloadSchemaType.KEYWORD),
        ("acl_allow", qmodels.PayloadSchemaType.KEYWORD),
        ("acl_deny", qmodels.PayloadSchemaType.KEYWORD),
        ("excluded", qmodels.PayloadSchemaType.BOOL),
        # acl_expires_at_ts：Qdrant 无法对 keyword 时间字段做比较，过期判定必须
        # 走数值（与 security_policy.P_ACL_EXPIRES_AT_TS 同义）。
        ("acl_expires_at_ts", qmodels.PayloadSchemaType.FLOAT),
    ]
    for field_name, schema in indexes:
        try:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )
        except Exception as exc:
            # Usually "index already exists" — safe to ignore on steady state
            logger.debug("Payload index on '%s': %s", field_name, exc)


async def _ensure_collection_tuning(collection_name: str) -> None:
    """
    幂等地把"面向大语料"的优化器/存储参数推到已有集合上.

    只在创建时设 ``optimizers_config`` 的话，**已存在的集合永远拿不到新参数**
    —— 升级到上千文档规模时，老集合会继续用建库时的默认值。这里显式 update，
    让存量部署也能受益。

    - ``memmap_threshold``：超过该点数的向量段改走内存映射文件而不是常驻内存。
      向量本体（1024 维 float32 = 4 KB/条）在 20 万条时约 800 MB，全常驻会和
      Ollama 抢内存；memmap 之后由 OS 页缓存按需换入，冷查询略慢但不会 OOM。
    - ``indexing_threshold``：达到即构建 HNSW。低于它会做暴力检索（准确但不
      可扩展）；建图之后是近似检索（快且可控）。
    """
    client = get_qdrant_client()
    try:
        await client.update_collection(
            collection_name=collection_name,
            optimizers_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=20_000,
                memmap_threshold=50_000,
            ),
        )
        logger.debug("Applied collection tuning to '%s'", collection_name)
    except Exception as exc:      # noqa: BLE001 — 调优失败不应阻止启动
        logger.warning("Collection tuning on '%s' skipped: %s", collection_name, exc)


async def upsert_vectors(points: list[VectorPoint]) -> int:
    """
    Upsert a batch of vectors into the Qdrant collection.

    Each point receives a random UUID as its Qdrant point ID.
    The ``document_id`` field in the payload is the PostgreSQL row UUID
    and is the join key for cross-store lookups.

    Returns:
        Number of points upserted.
    """
    if not points:
        return 0

    settings = get_settings()
    client = get_qdrant_client()

    qdrant_points = []

    for p in points:
        deterministic_id = generate_point_id(
            p.document_id, p.filename, p.page_number, p.section, p.heading, p.chunk_index, p.text,
            content_type=p.content_type,
            image_id=p.image_id,
        )

        qdrant_points.append(
            qmodels.PointStruct(
                id=deterministic_id,
                vector=p.vector,
                payload={
                    "document_id": p.document_id,
                    "filename": p.filename,
                    "chunk_index": p.chunk_index,
                    "page_number": p.page_number,
                    "text": p.text,
                    "char_count": len(p.text),
                    "heading": p.heading,
                    "section": p.section,
                    # ── Multi-Tenant 隔离载荷（框架要求的 ID 集合）────────────
                    # tenant_id + user_id + document_id + chunk_id(=本点 id) +
                    # page + content_type + source(=filename) 全部随 payload
                    # 落库；检索前置过滤用 tenant_id，审计/回显用其余字段。
                    "tenant_id": p.tenant_id,
                    "user_id": p.user_id,
                    "access_level": p.access_level,
                    "department_id": p.department_id,
                    "source": p.filename,
                    # ── 位置信息（细粒度引用溯源）──────────────────────────
                    "line_start": p.line_start,
                    "line_end": p.line_end,
                    "collection_id": p.collection_id,
                    # ── 内容类型与图片信息（部分3）────────────────────────────
                    "content_type": p.content_type,
                    "image_id": p.image_id,
                    "image_path": p.image_path,
                    "image_caption": p.image_caption,
                    "image_type": p.image_type,
                    # ── 置信度门控（图片理解引擎 + 人工复核标记）──────────────
                    "analyze_engine": p.analyze_engine,
                    "analyze_confidence": p.analyze_confidence,
                    "manual_review": p.manual_review,
                    # ── 产出质检 + 双通道融合（图片理解的可验证事实）────────
                    # 存整份报告：前端要逐条展示"代码语法未通过""OCR 置信度
                    # 偏低"这类警示，扁平成一个分数就丢失了可解释性。
                    "analyze_quality": p.analyze_quality,
                    "analyze_fusion": p.analyze_fusion,
                    # ── 图片位置（细粒度引用：第几页的第几处）──────────────────
                    # position 是文档内序号；bbox 是页面坐标。两者都为 None 时
                    # 前端只显示页码 —— 不假装有位置信息。
                    "position": p.position,
                    "bbox": list(p.bbox) if p.bbox else None,
                    # Hierarchical RAG metadata — 仅当 chunker 实际生成了 parent
                    # 字段时才会有值；旧 chunk 留空，不影响检索行为。
                    "parent_id": p.parent_id,
                    "parent_text": p.parent_text,
                    "parent_char_start": p.parent_char_start,
                    "parent_char_end": p.parent_char_end,
                    # ── 结构感知父子：检索层据此按父块/章节聚合与回填 ──────────
                    "parent_index": p.parent_index,
                    "section_id": p.section_id,
                    "section_path": p.section_path,
                    # ── Metadata 可过滤维度（值均为短标量/短数组，便于建索引）──
                    # 展开写而不是嵌套一层 dict：Qdrant 的 payload 过滤按
                    # **点号路径**取字段，"doc_meta.doc_type" 这类嵌套路径
                    # 建索引与查询都更啰嗦，扁平化更稳。
                    **_flatten_doc_meta(p.doc_meta),
                },
            )
        )

    await client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=qdrant_points,
        wait=True,          # wait for WAL flush — guarantees durability
    )

    logger.info(
        "Upserted %d vectors into '%s'",
        len(qdrant_points),
        settings.QDRANT_COLLECTION,
    )
    return len(qdrant_points)


async def delete_by_document_id(document_id: str) -> None:
    """
    Remove all vectors whose payload.document_id matches the given UUID.
    Useful for re-processing or deleting a document.
    """
    settings = get_settings()
    client = get_qdrant_client()

    await client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )
    logger.info("Deleted vectors for document_id=%s", document_id)


async def delete_by_document_ids(document_ids: "list[str] | set[str] | tuple[str, ...]") -> int:
    """
    Batch variant of :func:`delete_by_document_id` — one round trip for many docs.

    为什么必须有批量版本（不是"循环调用单条版"）：清理类调用（e2e 收尾、孤儿
    向量回收）动辄涉及上百份文档，逐份一次 HTTP 往返在千份规模下是几百次 RTT，
    而且**部分失败会留下"删了一半"的中间态**，比全有或全无更难排查。Qdrant 的
    ``MatchAny`` 把整个 id 集合压进一次 delete。

    返回的是 Qdrant 的 ``operation_id``（不是删除点数）—— 本函数只保证"请求已被
    接受"；需要精确点数时由调用方用 ``count`` 前后对比，或直接以 PG 行数为准。

    ``document_ids`` 为空时直接返回，不发请求 —— 空 ``MatchAny(any=[])`` 的语义
    在各版本间并不一致，别把"什么都没指定"赌在服务端行为上。
    """
    settings = get_settings()
    ids = [str(i) for i in document_ids if i]
    if not ids:
        return 0

    client = get_qdrant_client()
    await client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchAny(any=ids),
                    )
                ]
            )
        ),
    )
    logger.info("Deleted vectors for %d document_id(s) in one request", len(ids))
    return len(ids)


async def update_document_access_payload(
    document_id: str,
    *,
    access_level: str,
    department_id: str | None = None,
) -> int:
    """
    把一份文档所有向量点的 ACL 载荷改成新的层级（发布共享时调用）。

    ``access_level`` / ``department_id`` 是"第二层 Document ACL"在向量库里的
    副本。PG 侧改了但向量侧不改，两份事实就会分叉 —— 一旦将来 ANN 直接按
    payload 过滤，被发布的文档会**看起来发布成功但检索不到**。所以发布/
    收回层级必须同时更新两边。

    Returns:
        实际被改写的向量点数。**返回 0 有两种含义**，调用方必须区别对待：
          * 该文档在 Qdrant 里还没有向量点（入库尚未走到 upsert）；
          * 调用失败（异常已吞掉，只留 warning 日志）。
        两种情况都不中断发布流程（PG 是判定权威），但都意味着**向量侧的 ACL
        副本尚未生效** —— 必须由入库收尾的
        ``knowledge_tier_service.resync_document_acl_payload`` 追平，
        否则该文档在检索链路上对非 owner 永久不可见。
    """
    settings = get_settings()
    client = get_qdrant_client()

    payload: dict = {"access_level": access_level}
    # department_id 显式置 None 才能清掉旧的部门归属（个人库/公司库不需要它）
    payload["department_id"] = department_id

    selector = qmodels.FilterSelector(
        filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=document_id),
                )
            ]
        )
    )

    try:
        # 先确认**真的有向量点被覆盖**。``set_payload`` 在匹配 0 个点时同样返回
        # 成功，所以"调用没抛异常"完全不能证明载荷改了 —— 而这正是发布/入库
        # 竞态长期静默的原因：上传后立刻发布时向量点往往还没写入，更新落空并
        # 记一条 "Updated vector ACL payload" 的成功日志，随后入库又把上传时刻
        # 的 private 写进 payload，两份事实永久分叉。
        # ``document_id`` 建了 payload 索引（见 _ensure_payload_indexes），
        # 因此 exact count 走索引，不会退化成全量扫描。
        counted = await client.count(
            collection_name=settings.QDRANT_COLLECTION,
            count_filter=selector.filter,
            exact=True,
        )
        matched = int(getattr(counted, "count", 0) or 0)
        if matched == 0:
            logger.warning(
                "update_document_access_payload: document_id=%s has no vectors yet — "
                "access_level=%s NOT applied to the vector store. Ingestion must "
                "re-sync at completion (resync_document_acl_payload), otherwise the "
                "document stays invisible to retrieval while looking published in PG.",
                document_id, access_level,
            )
            return 0

        await client.set_payload(
            collection_name=settings.QDRANT_COLLECTION,
            payload=payload,
            points=selector,
            wait=True,
        )
    except Exception as exc:      # noqa: BLE001 — 载荷同步失败不应中断发布流程
        logger.warning(
            "update_document_access_payload failed for %s: %s", document_id, exc
        )
        return 0

    logger.info(
        "Updated vector ACL payload for document_id=%s → access_level=%s dept=%s points=%d",
        document_id, access_level, department_id, matched,
    )
    return matched


async def get_existing_point_ids(point_ids: list[str]) -> set[str]:
    """
    Given a list of Qdrant point IDs, returns a set of the ones that already exist
    in the collection. Useful for idempotent resumes to avoid re-embedding.
    """
    if not point_ids:
        return set()

    settings = get_settings()
    client = get_qdrant_client()

    # retrieve only the IDs without payload/vectors to be fast
    response = await client.retrieve(
        collection_name=settings.QDRANT_COLLECTION,
        ids=point_ids,
        with_payload=False,
        with_vectors=False
    )
    
    return {str(point.id) for point in response}

