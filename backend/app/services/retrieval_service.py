"""
Semantic retrieval service (Phase 4 / 混合检索优化).

Embeds the query, performs an ANN search against Qdrant, and — when hybrid
search is enabled — fuses the vector ranking with a BM25 keyword ranking via
Reciprocal Rank Fusion (RRF).  The keyword leg catches exact-term matches
(product codes, proper nouns) that pure semantic similarity tends to miss.

Reuses embed_texts() from Phase 2 — task_type is the only difference between
document ingestion and query embedding.
"""

import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace

from app.config import get_settings
from app.db.qdrant import get_qdrant_client
from app.services.embedding_service import embed_batch_with_retry
from app.services.evidence_trust import apply_trust_weighting, compute_trust
from app.services.hybrid_search import BM25Index, rrf_fuse
from app.services.metadata import MetadataFilter, matches_payload
from app.services.pg_keyword_search import keyword_search
from app.services.security_policy import (
    Decision,
    ObjectACLView,
    ScopeCompileError,
    ScopePredicate,
    allows,
    to_qdrant,
    to_sql,
)
from app.services.security_scope import UserScope, cache_key_for_scope
from app.services.tenancy import (
    DEFAULT_TENANT_ID,
    document_scope_clause,
    exclude_test_tenants,
    normalize_tenant_id,
    scoped_cache_key,
    tenant_scope_fingerprint,
)
from app.utils.logging import get_logger
from app.utils.timing import timed_stage

logger = get_logger(__name__)


def image_position_label(content_type: str, position: int | None) -> str | None:
    """
    图片的一句话位置：``第 2 张图`` / ``第 3 个表格``；无序号时 None.

    这是"图片位置怎么说"的**单一实现点** —— 上下文头部、SSE sources、
    前端引用卡片都复用它。图片没有行号，用文档内序号定位是唯一既准确
    又符合直觉的说法。
    """
    if not position:
        return None
    noun = "个表格" if content_type == "table" else "张图"
    return f"第 {position} {noun}"


@dataclass
class RetrievedChunk:
    """A single chunk returned by the vector search."""

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float   # cosine similarity (0 – 1)

    # ── 内容类型与图片信息（部分3）──────────────────────────────────────────
    # content_type: text | table | image —— 下游据此做图文分流：
    #   text/table → 直接进上下文
    #   image      → image_path 取原图 → Vision 看图 → 结论进上下文并回显原图
    content_type: str = "text"
    image_id: str | None = None
    image_path: str | None = None      # 文档相对路径 images/page_3_image_1.png
    image_caption: str | None = None   # 入库期 vision caption（若有）
    # 图片分类结论（三层图片处理）：table | chart | diagram | screenshot | photo。
    # 前端据此渲染"表格/图表/流程图"徽标；注意 content_type 与 image_type 是
    # 两个维度 —— 表格图片的 content_type="table" 但 image_type 仍是 "table"，
    # 图表则是 content_type="image" + image_type="chart"。
    image_type: str | None = None
    # ── 置信度门控（多引擎图片理解）─────────────────────────────────────────
    # analyze_engine: 产出该图片内容的引擎；analyze_confidence: 最终置信度；
    # manual_review: 兜底后仍未通过校验 → 前端打"待复核"标记。
    analyze_engine: str | None = None
    analyze_confidence: float = 0.0
    manual_review: bool = False

    # ── Multi-Tenant 隔离（第一层）──────────────────────────────────────────
    # 该 chunk 归属的租户（来自 Qdrant payload；旧索引缺失时为 default）。
    tenant_id: str = DEFAULT_TENANT_ID

    # ── small-to-big / Hierarchical RAG metadata（全部默认 None，向后兼容）──
    # parent_id 决定回填哪一个父块；parent_text 由检索层**回填**（见
    # _hydrate_parents），不再来自向量 payload —— 那会把父块正文复制进每个
    # 子块（1000 份文档 ≈ GB 级冗余）。
    parent_id: str | None = None        # document_id + ":p:" + parent_index
    parent_text: str | None = None      # 命中后由 chunk_parents 表回填
    parent_char_start: int | None = None
    parent_char_end: int | None = None
    heading: str | None = None          # 复制 chunker 写出的 heading
    section: str | None = None          # 复制 chunker 写出的 section

    # ── 结构感知父子（新）──────────────────────────────────────────────────────
    parent_index: int | None = None
    section_id: str | None = None
    section_path: list[str] | None = None

    # ── Metadata（自动抽取，进引用展示与可解释过滤）────────────────────────────
    doc_type: str | None = None
    doc_year: int | None = None
    language: str | None = None
    title: str | None = None
    doc_number: str | None = None
    author: str | None = None
    keywords: list[str] = field(default_factory=list)
    business_tags: list[str] = field(default_factory=list)

    # ── 证据可信度（反幻觉：解析期已知的"不可信"接到排序上）────────────────────
    # trust_score：0~1，1 = 解析期未发现任何问题；trust_reasons：为什么被降权
    # （逐条可解释，前端可直接展示）。见 evidence_trust 模块。
    trust_score: float = 1.0
    trust_reasons: list[str] = field(default_factory=list)

    # ── 位置信息（细粒度引用溯源）────────────────────────────────────────────
    # 1-based 闭区间行号：引用卡片据此告诉用户"这是文档的第几行"。
    # 旧索引没有该字段时为 None，引用层降级为只显示页码。
    line_start: int | None = None
    line_end: int | None = None
    # 图片位置：position = 文档内序号（第几张图）；bbox = 页面边界框
    # (x1, y1, x2, y2)，点坐标、原点左上。文本块两者皆为 None。
    # DOCX 无页面几何 → bbox 为 None，位置标签降级为"第 N 张图"。
    position: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    # ── 产出质检 + 双通道融合（图片理解的可验证事实）──────────────────────────
    # analyze_quality：代码语法是否通过 / OCR 行置信度 / VLM 幻觉检查 / 结构校验；
    # analyze_fusion：双通道策略与最终选中的通道。旧索引缺这两个字段时为 {}，
    # 下面几个 property 会自动降级为"无警示"。
    analyze_quality: dict = field(default_factory=dict)
    analyze_fusion: dict = field(default_factory=dict)

    # ── 第 11 环对象级复核所需的**原始 payload**（内部字段，不进 API 出参）─────────
    # 只有内存 BM25 腿需要它：向量腿与 PG 腿各自手里就有 payload 字典，而内存腿
    # 的候选是按 ``(document_id, chunk_index)`` 从语料列表里取出来的 —— 原实现
    # 丢掉 payload 后，第 11 环 ``_object_level_filter`` 拿到 ``None`` 就"宁缺勿错"
    # 地放行，于是该腿**整条跳过** ``allows()``：同一条腿在 ``HYBRID_KEYWORD_BACKEND
    # =memory`` 下与另两条腿判定不同源，且剔除不进审计。带上一份引用即可同源。
    # ``repr=False``：日志里不打印整块 payload；``compare=False``：不影响相等语义。
    _acl_payload: dict | None = field(default=None, repr=False, compare=False)

    @property
    def is_image(self) -> bool:
        """该 chunk 是否为图片对象（部分1：图片作为独立检索对象）."""
        return self.content_type == "image"

    @property
    def quality_score(self) -> float:
        """质检总分（0~1）；无质检信息时返回 1.0（不误报警示）."""
        try:
            return float(self.analyze_quality.get("score", 1.0))
        except (TypeError, ValueError):
            return 1.0

    @property
    def quality_ok(self) -> bool:
        return bool(self.analyze_quality.get("ok", True))

    @property
    def quality_reasons(self) -> list[str]:
        """质检未通过的原因（前端引用卡片逐条展示）."""
        reasons = self.analyze_quality.get("reasons") or []
        return [str(r) for r in reasons if str(r).strip()]

    @property
    def ocr_confidence(self) -> float:
        """这次读字的平均行置信度；引擎未上报时返回 0.0."""
        ocr = self.analyze_quality.get("ocr") or {}
        try:
            return float(ocr.get("mean", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def fusion_strategy(self) -> str | None:
        """双通道融合策略：complementary / vision-first / ocr-first；无则 None."""
        return self.analyze_fusion.get("strategy") or None

    @property
    def fusion_chosen(self) -> str | None:
        """最终采用的通道：ocr / vlm；无则 None."""
        return self.analyze_fusion.get("chosen") or None

    @property
    def is_image_derived(self) -> bool:
        """该 chunk 是否由图片产出（含"图片表格"，它们 content_type="table"）."""
        return bool(self.image_id)

    @property
    def line_span(self) -> str | None:
        """人类可读的行号区间，如 ``"12-28"``；无位置信息时返回 None."""
        if self.line_start is None:
            return None
        if self.line_end is None or self.line_end == self.line_start:
            return str(self.line_start)
        return f"{self.line_start}-{self.line_end}"

    @property
    def bbox_span(self) -> str | None:
        """人类可读的图片坐标，如 ``"x 120-460, y 80-320"``；无坐标时 None."""
        if not self.bbox:
            return None
        x1, y1, x2, y2 = self.bbox
        return f"x {x1:.0f}-{x2:.0f}, y {y1:.0f}-{y2:.0f}"

    @property
    def position_span(self) -> str | None:
        """图片的一句话位置（``"第 2 张图"`` / ``"第 3 个表格"``）；文本块为 None."""
        if not self.is_image_derived:
            return None
        return image_position_label(self.content_type, self.position)

    @property
    def section_label(self) -> str | None:
        """
        章节路径的一句话（``"第 3 章 财务情况 · 3.2 核算方法"``）；无则 None.

        有了它，引用卡片就能说到"出自哪一节"，而不仅仅是"第几页第几行" ——
        页行定位回答"在哪"，章节定位回答"属于什么"，后者对核实结论更关键
        （同一行可能属于不同小节的论述）。
        """
        if not self.section_path:
            return None
        parts = [str(p).strip() for p in self.section_path if str(p).strip()]
        return " · ".join(parts) or None

    def location_label(self) -> str:
        """
        一句话定位：``《年报.pdf》第 3 页，第 12-28 行``.

        这是"溯源时用一句话标明这是检索文档的哪几行"的单一实现点 ——
        上下文头部、SSE sources、前端引用卡片都复用它，避免三处各写一遍
        导致文案漂移。

        文本块用**行号**定位；图片没有行号，改用**文档内序号**
        （``第 2 张图`` / ``第 3 个表格``）—— 位置标签宁可简短准确，
        也不要把 bbox 这类机器坐标塞进自然语言句子（坐标由前端按需渲染）。

        末尾追加**年份与章节**（有则显示）：千份文档的库里"哪一年、哪一节"
        往往比"第几页"更能让用户确认引用是否正确。
        """
        parts = [self.filename or "未知文档"]
        if self.doc_year:
            parts.append(f"{self.doc_year} 年")
        section = self.section_label
        if section:
            parts.append(section)
        if self.page_number:
            parts.append(f"第 {self.page_number} 页")
        span = self.line_span
        if span:
            parts.append(f"第 {span} 行")
        else:
            pos = self.position_span
            if pos:
                parts.append(pos)
        return f"《{parts[0]}》" + ("，" + "，".join(parts[1:]) if len(parts) > 1 else "")


def _line_fields(payload: dict) -> dict:
    """从 Qdrant payload 读取位置信息（行号），缺失时安全返回 None."""
    def _as_int(value):
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "line_start": _as_int(payload.get("line_start")),
        "line_end": _as_int(payload.get("line_end")),
    }


def _position_fields(payload: dict) -> dict:
    """
    从 Qdrant payload 读取图片位置（文档内序号 + 页面边界框）.

    旧索引没有这两个字段 → 返回 None，位置标签自动降级为只显示页码。
    坐标做了防御性校验：必须是 4 个可转 float 的分量，否则整体丢弃 ——
    半截坐标画出来的框比没有框更容易误导人。
    """
    position = payload.get("position")
    try:
        position = int(position) if position is not None else None
    except (TypeError, ValueError):
        position = None

    raw = payload.get("bbox")
    bbox = None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            bbox = tuple(float(v) for v in raw)
        except (TypeError, ValueError):
            bbox = None

    return {"position": position, "bbox": bbox}


def _analysis_fields(payload: dict) -> dict:
    """
    从 Qdrant payload 读取产出质检与双通道融合结论.

    旧索引没有这两个字段 → 返回空 dict，上层 property 自动降级为
    "无警示"（``quality_score`` 默认 1.0），不会因为索引没升级就误报。
    """
    def _as_dict(value) -> dict:
        return dict(value) if isinstance(value, dict) else {}

    return {
        "analyze_quality": _as_dict(payload.get("analyze_quality")),
        "analyze_fusion": _as_dict(payload.get("analyze_fusion")),
    }


def split_by_modality(
    chunks: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """
    把检索结果按模态拆成 (文本/表格块, 图片块) —— 对应设计稿
    "Top 5 Context → text chunk / image chunk" 的分流（部分5）。
    """
    text_chunks = [c for c in chunks if c.content_type != "image"]
    image_chunks = [c for c in chunks if c.content_type == "image"]
    return text_chunks, image_chunks


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int]:
    """Stable identity of a chunk across the vector and keyword legs."""
    return (chunk.document_id, chunk.chunk_index)


def _media_fields(payload: dict) -> dict:
    """从 Qdrant payload 提取内容类型与图片信息（部分3）."""
    return {
        "content_type": str(payload.get("content_type") or "text"),
        "image_id": payload.get("image_id"),
        "image_path": payload.get("image_path"),
        "image_caption": payload.get("image_caption"),
        # 三层图片处理：把分类结论带到检索结果（前端徽标 / 按类型分流）
        "image_type": payload.get("image_type"),
        # 多引擎管线的产出引擎与置信度（前端展示"由谁解析、多可信"）
        "analyze_engine": payload.get("analyze_engine"),
        "analyze_confidence": float(payload.get("analyze_confidence") or 0.0),
        "manual_review": bool(payload.get("manual_review") or False),
        # Multi-Tenant：chunk 归属租户（旧索引缺失 → default，不视为跨租户）
        "tenant_id": normalize_tenant_id(payload.get("tenant_id")),
        # ── 结构感知父子（旧索引为空 → 回填逻辑自动跳过，不报错）──────────────
        "parent_index": _as_int(payload.get("parent_index")),
        "section_id": payload.get("section_id"),
        "section_path": _as_str_list(payload.get("section_path")),
        # ── Metadata（可过滤维度 + 引用展示）──────────────────────────────────
        "doc_type": payload.get("doc_type"),
        "doc_year": _as_int(payload.get("doc_year")),
        "language": payload.get("language"),
        "title": payload.get("title"),
        "doc_number": payload.get("doc_number"),
        "author": payload.get("author"),
        "keywords": _as_str_list(payload.get("keywords")),
        "business_tags": _as_str_list(payload.get("business_tags")),
    }


def _as_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_str_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


# ── BM25 corpus cache ─────────────────────────────────────────────────────────
# 缓存键 = tenant_id + 权限上下文 + (collection 范围) —— 三层隔离的缓存规则：
# 跨租户/跨权限视图绝不共享同一个 BM25 语料索引。TTL 到期后惰性重建，
# 新上传的文档在一个 TTL 窗口内进入 BM25 召回。
#
# ⚠️ 必须**有上限**。旧实现是一个永不淘汰的普通 dict：缓存键里含 owner_id 与
# department_id，也就是说**每个用户、每个部门组合**都会在进程里留一份完整的
# 语料副本（连同 Python 对象开销）。企业里几百号人跑一天，内存就是几 GB 起 ——
# 而且不报错、不告警，直到 OOM 才被发现。
# 现在用 OrderedDict 做 LRU，条目数由 HYBRID_BM25_CACHE_MAX_ENTRIES 控制。

_bm25_cache: "OrderedDict[str, tuple[float, list[RetrievedChunk], BM25Index]]" = OrderedDict()


def _bm25_cache_get(key: str, ttl: float):
    hit = _bm25_cache.get(key)
    if hit is None:
        return None
    if time.monotonic() - hit[0] >= ttl:
        _bm25_cache.pop(key, None)
        return None
    _bm25_cache.move_to_end(key)
    return hit


def _bm25_cache_put(key: str, value, max_entries: int) -> None:
    _bm25_cache[key] = value
    _bm25_cache.move_to_end(key)
    while len(_bm25_cache) > max(1, max_entries):
        _bm25_cache.popitem(last=False)


async def _scroll_corpus(
    client,
    collection_name: str,
    collection_id: str | None,
    max_points: int,
    tenant_ids: frozenset[str] | None = None,
    pred: ScopePredicate | None = None,
) -> list[RetrievedChunk]:
    """
    Scroll the whole Qdrant collection (payload only, no vectors).

    *tenant_ids* 是**检索前置过滤**（第一层隔离）：BM25 语料在向量库侧
    就按租户集合裁剪，跨租户 chunk 连语料都进不来，而不是事后剔除。

    ``tenant_ids`` 语义与 ``tenant_clause`` 三分支一致：空集 → fail-closed
    返回空语料；``None``（诊断）→ 不加租户条件；非空 → ``MatchAny``。

    *pred*（【T3】）：给了五维 ``ScopePredicate`` 时，语料过滤改用**与向量腿
    同一份** ``to_qdrant(pred)``（含 Deny-4/5/6），``tenant_ids`` 参数随之忽略。
    这样内存 BM25 与向量腿的可见集合同源，不会出现"一条腿看得见、另一条看不见"。
    """
    from qdrant_client.http import models as qmodels

    if pred is not None:
        # 五维同源：语料在向量库侧就按 clearance / 项目 / deny 裁剪。
        # collection_id 仍是业务知识库范围（与向量腿的 must 叠加口径一致）。
        base = to_qdrant(pred)
        must = []
        if collection_id:
            must.append(
                qmodels.FieldCondition(
                    key="collection_id",
                    match=qmodels.MatchValue(value=collection_id),
                )
            )
        must_not = list(getattr(base, "must_not", None) or [])
        scroll_filter = qmodels.Filter(
            must=must or None, must_not=must_not or None,
        )
        return await _scroll_pages(client, collection_name, scroll_filter, max_points)

    if tenant_ids is not None and not tenant_ids:
        # fail-closed：空集没有任何租户，语料为空（绝不退化成"不限制"）
        return []

    must = []
    if tenant_ids:
        values = sorted(tenant_ids)
        if len(values) == 1:
            must.append(
                qmodels.FieldCondition(
                    key="tenant_id",
                    match=qmodels.MatchValue(value=normalize_tenant_id(values[0])),
                )
            )
        else:
            must.append(
                qmodels.FieldCondition(
                    key="tenant_id",
                    match=qmodels.MatchAny(any=[normalize_tenant_id(v) for v in values]),
                )
            )
    if collection_id:
        must.append(
            qmodels.FieldCondition(
                key="collection_id",
                match=qmodels.MatchValue(value=collection_id),
            )
        )
    scroll_filter = qmodels.Filter(must=must) if must else None

    return await _scroll_pages(client, collection_name, scroll_filter, max_points)


async def _scroll_pages(
    client,
    collection_name: str,
    scroll_filter,
    max_points: int,
) -> list[RetrievedChunk]:
    """实际的分页 scroll（含 payload → RetrievedChunk 的字段映射）."""
    corpus: list[RetrievedChunk] = []
    offset = None
    while len(corpus) < max_points:
        points, next_offset = await client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            try:
                page_number = int(payload.get("page_number") or 1)
            except (TypeError, ValueError):
                page_number = 1
            try:
                chunk_index = int(payload.get("chunk_index") or 0)
            except (TypeError, ValueError):
                chunk_index = 0
            corpus.append(
                RetrievedChunk(
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "unknown")),
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=str(payload.get("text", "")),
                    score=0.0,
                    # 带上原始 payload：内存 BM25 腿要拿它跑第 11 环 allows()
                    # （见 RetrievedChunk._acl_payload 的说明）。同一份字典被缓存
                    # 里的语料共享，不额外复制。
                    _acl_payload=payload,
                    **_media_fields(payload),
                    **_line_fields(payload),
                    **_position_fields(payload),
                    **_analysis_fields(payload),
                )
            )
        if next_offset is None:
            break
        offset = next_offset

    return corpus[:max_points]


async def _bm25_candidates(
    query: str,
    fetch_n: int,
    valid_docs: set[str],
    collection_name: str,
    collection_id: str | None,
    owner_id: str | None,
    client,
    tenant_ids: frozenset[str] | None = None,
    perm_context: str = "anonymous",
    scope: "UserScope | None" = None,
    pred: "ScopePredicate | None" = None,
) -> list[RetrievedChunk]:
    """
    Keyword (BM25) leg of the hybrid search.

    Returns chunks ranked best-first, restricted to *valid_docs* (documents
    that exist in PostgreSQL, are COMPLETED, and pass the tenant+ACL filter).

    *fetch_n* 与向量腿的候选池对齐（框架图 Vector Top20 + BM25 Top20）：
    两腿取同样深的候选，RRF 融合时哪一路都不会被浅列表系统性压低。

    缓存键 = tenant_id + 权限上下文 + collection 范围（第三层：缓存隔离），
    租户/权限视图不同的请求不会命中同一份语料索引。

    【T3】*scope* 给了五维 ``UserScope`` 时：
      · 缓存键经 ``cache_key_for_scope(scope, raw)``（决策 9：按指纹分区）；
      · 语料 scroll 用**同一份** ``to_qdrant(pred)``（与向量腿同源）。
    """
    settings = get_settings()
    if scope is not None:
        cache_key = cache_key_for_scope(
            scope, f"bm25::{collection_id or '__all__'}::{owner_id or '__admin__'}"
        )
    else:
        cache_key = scoped_cache_key(
            tenant_ids,
            perm_context,
            f"bm25::{collection_id or '__all__'}::{owner_id or '__admin__'}",
        )

    now = time.monotonic()
    cached = _bm25_cache_get(cache_key, settings.HYBRID_CACHE_TTL_SECONDS)
    if cached is not None:
        _, visible, index = cached
    else:
        # 【T3 同一 pred 只构造一次（决策 6.1）】入口已编译好的 ``pred`` 直接复用
        # —— 与向量腿 ``to_qdrant(pred)``、PG 关键词腿 ``to_sql(pred)`` 共用**同一个**
        # ScopePredicate 实例。仅当未提供 pred 时才从 scope 现编（兼容旧调用点）。
        if pred is None and scope is not None:
            pred = scope.predicate()
        corpus = await _scroll_corpus(
            client,
            collection_name,
            collection_id,
            settings.HYBRID_MAX_CORPUS_POINTS,
            tenant_ids=tenant_ids,
            pred=pred,
        )
        # 截断**必须告警**。旧实现静默截断到 10000 条 —— 1000 份文档时关键词腿
        # 只看得到前 5% 的语料，其余永远召不回来，而且没有任何日志。用户得到
        # "知识库没有这个"的结论时，文档其实就在库里。这里把"没覆盖全"变成一条
        # 显式、可搜索的 ERROR，并把后端切到 postgres 的决策有据可依。
        if len(corpus) >= settings.HYBRID_MAX_CORPUS_POINTS:
            logger.error(
                "BM25 corpus truncated at %d points (scope=%s) — keyword leg is "
                "NOT covering the full corpus. Switch HYBRID_KEYWORD_BACKEND to "
                "'postgres' or raise HYBRID_MAX_CORPUS_POINTS.",
                settings.HYBRID_MAX_CORPUS_POINTS, cache_key,
            )
        visible = [c for c in corpus if c.document_id in valid_docs]
        index = BM25Index([c.text for c in visible])
        _bm25_cache_put(
            cache_key,
            (now, visible, index),
            settings.HYBRID_BM25_CACHE_MAX_ENTRIES,
        )
        logger.info(
            "BM25 index built: scope=%s corpus=%d visible=%d",
            cache_key,
            len(corpus),
            len(visible),
        )

    if len(visible) < 2:
        return []

    hits = index.search(query, top_n=max(fetch_n, 1))
    return [visible[h.index] for h in hits]


async def _pg_keyword_candidates(
    query: str,
    fetch_n: int,
    collection_id: str | None,
    tenant_ids: frozenset[str] | None,
    owner_id: str | None,
    user_department_id: str | None,
    tenant_wide: bool,
    owns_tenant_ids: frozenset[str],
    unrestricted: bool,
    metadata_filter: MetadataFilter | None = None,
    scope: "UserScope | None" = None,
    pred: "ScopePredicate | None" = None,
) -> list[tuple[str, int, str]]:
    """
    关键词腿的 PostgreSQL 实现（上千文档时的默认后端）.

    与内存 BM25 的区别**不是性能调优，而是覆盖范围**：

        memory   —— 把 Qdrant 语料 scroll 进内存建索引，受
                    HYBRID_MAX_CORPUS_POINTS 限制。1000 份文档（约 20 万 chunk）
                    时只看得到前 5%，其余静默漏召回；建索引本身要十几秒并占用
                    数百 MB（实测 4 万条合成语料：12.9 秒）。
        postgres —— 词项在入库时落库，检索走 GIN 索引 + ts_rank_cd。
                    召回覆盖 100% 语料，内存零放大，权限在 SQL 侧下推。

    返回 ``[(document_id, chunk_index, point_id), ...]``（按 ts_rank_cd 降序）。
    point_id 用于把命中点的 payload 取回来（``_fetch_by_point_ids``）——
    关键词腿否则无法给出"只有它命中的 chunk"的正文。

    【T3 双路同源】*scope* 给了五维 ``UserScope`` 时透传给 ``keyword_search``，
    由它用 ``to_sql(scope.predicate(), Document)`` 下推 —— 与向量腿的
    ``to_qdrant(pred)`` 由**同一个** ``ScopePredicate`` 编译（决策 6.1）。
    """
    from app.db.postgres import get_db_session

    try:
        async with get_db_session() as session:
            rows = await keyword_search(
                session,
                query=query,
                limit=max(fetch_n, 1),
                tenant_ids=tenant_ids,
                owner_id=owner_id,
                user_department_id=user_department_id,
                tenant_wide=tenant_wide,
                owns_tenant_ids=owns_tenant_ids,
                collection_id=collection_id,
                unrestricted=unrestricted,
                metadata_filter=metadata_filter,
                scope=scope,
                # 【T3 双路同源】把入口编译好的**同一个** pred 透传给 keyword_search，
                # 由它 to_sql(pred, Document) 下推 —— 与向量腿 to_qdrant(pred)
                # 用的是同一个 ScopePredicate 实例（决策 6.1，不再是各腿各自现编）。
                pred=pred,
            )
    except Exception:
        logger.exception("PG keyword leg failed — falling back to vector-only")
        return []
    return [(doc_id, chunk_index, point_id) for doc_id, chunk_index, point_id, _rank in rows]


def _summarize_metadata_filter(flt: MetadataFilter | None) -> str:
    """把过滤器渲染成一行人类可读的说明（日志 + 引用面板展示"本次检索范围"）."""
    if flt is None or flt.is_empty():
        return "无"
    parts = []
    if flt.doc_types:
        parts.append(f"类型={','.join(flt.doc_types)}")
    if flt.years:
        parts.append(f"年份={','.join(str(y) for y in flt.years)}")
    if flt.languages:
        parts.append(f"语言={','.join(flt.languages)}")
    if flt.tags:
        parts.append(f"标签={','.join(flt.tags)}")
    if flt.document_ids:
        parts.append(f"文档={len(flt.document_ids)}份")
    if flt.exclude_document_ids:
        parts.append(f"排除={len(flt.exclude_document_ids)}份")
    return " ".join(parts) or "无"


def _visibility_conditions(
    owner_id: str | None,
    department_id: str | None,
    tenant_wide: bool,
    tenant_ids: frozenset[str] | None,
    owns_tenant_ids: frozenset[str],
) -> list:
    """
    把 PG 侧 ``document_scope_clause`` 的语义**下推到向量层**（ANN 之前）.

    为什么必须下推：向量腿原先只过滤 ``tenant_id``，于是**同一公司内别人的个人库
    向量也会被召回**，占满候选名额后再在 PG 层被 ACL 丢掉。正确性没问题，但候选池
    被稀释 —— 文档一多（比如每人私库里几百份），一个用户查询的 20 个候选可能全部
    来自别人的私库，自己一份都进不来，表现为「库里明明有却查不到」。

    为什么用 ``must_not`` + 嵌套 ``Filter``，而不是前置白名单：这是刻意的
    **fail-open**。Qdrant 的 ``FieldCondition`` 在字段缺失时**不匹配**，所以
    「迁移前写入、payload 里没有 access_level / user_id 的老向量」不会被任何一条
    排除条件命中 → 原样保留 → 仍由 PG 的 ``document_scope_clause`` 做最终判定。

    **Rev2 形态（Deny-list，与 PG 的 ``or_(公司边界, 自己个人库)`` 逐条同构）**：

        Deny-1 个人库：排除「private 且 user_id ≠ 我 且 tenant ∉ 自建集合」
                      —— 命中「我本人」或「我自建的测试公司」即**不排除**。
        Deny-2 公司边界：非个人库（department / tenant）必须 tenant ∈ 集合；
                      空集 fail-closed 排除全部非个人库；None 不加边界（诊断）。
        Deny-3 部门库：同部门（``tenant_wide`` 放开）—— 语义不变。

    ⚠️ 字段名必须与 ``vector_service`` 落库时一致：Qdrant payload 里的所有者字段叫
    **``user_id``**（上传者），**不是 ``owner_id``**。写错名字不会报错 —— 排除条件
    永远不匹配 → 所有 private 向量被整体排除 → 用户连自己的文档都搜不到。
    改动本函数后必须跑 ``_audit_916/probe_acl_prefilter_916.py`` 的 A/B 对照。
    """
    from qdrant_client.http import models as qmodels

    def _value(key: str, value):
        return qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))

    def _any(key: str, values):
        vals = [str(v) for v in values]
        # 单值用 MatchValue、多值用 MatchAny：单值走 MatchValue 更利于选索引。
        if len(vals) == 1:
            return qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=vals[0]))
        return qmodels.FieldCondition(key=key, match=qmodels.MatchAny(any=vals))

    conds: list = []

    # ── Deny-1 个人库：private 与租户无关（除非「我本人」或「我自建公司」）──────
    if owner_id:
        allow = [_value("user_id", str(owner_id))]
        if owns_tenant_ids:
            allow.append(_any("tenant_id", sorted(owns_tenant_ids)))
        conds.append(
            qmodels.Filter(must=[_value("access_level", "private")], must_not=allow)
        )
    else:
        # 无 owner：排除全部 private
        conds.append(qmodels.Filter(must=[_value("access_level", "private")]))

    # ── Deny-2 公司边界：非个人库文档必须 tenant ∈ 集合 ────────────────────────
    non_personal = qmodels.Filter(
        should=[
            _value("access_level", "department"),
            _value("access_level", "tenant"),
        ]
    )
    if tenant_ids is None:
        pass                                             # 诊断：不加租户边界
    elif not tenant_ids:                                 # 空集 → fail-closed
        conds.append(non_personal)
    else:
        conds.append(
            qmodels.Filter(must=[non_personal],
                           must_not=[_any("tenant_id", sorted(tenant_ids))])
        )

    # ── Deny-3 部门库：同部门（tenant_wide 放开）——语义不变 ────────────────────
    if not tenant_wide:
        conds.append(
            qmodels.Filter(
                must=[_value("access_level", "department")],
                must_not=([_value("department_id", str(department_id))]
                          if department_id else None),
            )
        )
    return conds


def _metadata_conditions(flt: MetadataFilter | None) -> list:
    """
    把 ``MetadataFilter`` 翻成 Qdrant 的 ``FieldCondition`` 列表（ANN 前置过滤）.

    为什么必须放在 ANN **之前**而不是"召回后再剔"：候选池深度是固定的
    （``candidate_pool``）。千份文档时"2024年营收"会在十几个年份的报告里都命中
    语义相近的段落，Top-N 里挤满了 2021/2022/2023 的段落；真正那一份 2024 的
    可能排在第 N+1 位，**根本没进候选池**，精排再强也救不回来。前置过滤是把
    搜索空间本身缩小，而不是事后打捞。

    语义与 PG 关键词腿的解析版一致（strict）：每类条件都是 must（AND）；
    ``document_ids`` / ``exclude_document_ids`` 映射成 match any / must_not。
    """
    if flt is None or flt.is_empty():
        return []

    from qdrant_client.http import models as qmodels

    conds = []

    def _or_condition(key: str, values: list):
        if not values:
            return None
        # 单值用 MatchValue、多值用 MatchAny：MatchAny 在单值下也能工作，但
        # 单值走 MatchValue 更利于 Qdrant 选索引，且日志/调试更可读。
        if len(values) == 1:
            return qmodels.FieldCondition(
                key=key, match=qmodels.MatchValue(value=values[0]),
            )
        return qmodels.FieldCondition(
            key=key, match=qmodels.MatchAny(any=list(values)),
        )

    for key, values in (
        ("doc_type", flt.doc_types),
        ("doc_year", flt.years),
        ("language", flt.languages),
    ):
        cond = _or_condition(key, list(values) if values else [])
        if cond is not None:
            conds.append(cond)

    # 标签：business_tags / keywords 任一命中即可（OR 两个字段），
    # 与 PG 侧 `?|` 的"任一标签命中"语义保持一致。
    if flt.tags:
        tag_conds = [
            c for c in (
                _or_condition("business_tags", list(flt.tags)),
                _or_condition("keywords", list(flt.tags)),
            ) if c is not None
        ]
        if len(tag_conds) == 1:
            conds.append(tag_conds[0])
        elif tag_conds:
            conds.append(qmodels.Filter(should=tag_conds))

    return conds



async def _fetch_by_point_ids(
    point_ids: list[str],
    collection_name: str,
) -> dict[str, dict]:
    """
    按点 id 批量取回 payload（``{point_id: payload}``）.

    关键词腿只从 PG 拿到"哪些点命中"，正文仍必须回到 Qdrant 取 —— 这样两条腿
    给出的 chunk **是同一份数据**（同一次 upsert 写入），不会出现"一条腿有
    image_path 另一条没有"这种融合时的字段不一致。
    """
    if not point_ids:
        return {}
    client = get_qdrant_client()
    out: dict[str, dict] = {}
    # 分批：Qdrant 的 retrieve 单次请求体有大小限制，几千个 id 直接发会被拒。
    batch = 256
    for i in range(0, len(point_ids), batch):
        chunk = point_ids[i:i + batch]
        try:
            records = await client.retrieve(
                collection_name=collection_name,
                ids=chunk,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            logger.exception("retrieve by point ids failed (%d ids) — skipping batch", len(chunk))
            continue
        for record in records:
            if record.payload:
                out[str(record.id)] = record.payload
    return out


async def _hydrate_parents(chunks: list[RetrievedChunk]) -> int:
    """
    批量回填父块正文（small-to-big 的第二步：**一次 IN 查询换回整批父块**）.

    为什么是这个形状（而不是让子块自带 parent_text）
    ────────────────────────────────────────────────
    改造前 ``parent_text`` 会被复制进**每一个**子块的向量 payload：

        1000 份文档 × 每份 ~200 个子块 × 父块 ~5 KB ≈ 1 GB

    这 1 GB 不只是存储浪费 —— 关键词腿（内存后端）会把整份 payload 拉回进程
    建索引，于是"上千文档"从"慢"变成"起不来"。现在子块只带 ``parent_id``
    （形如 ``{doc_id}:p:{idx}``），命中后在这里按 id 批量取回。

    为什么按 parent_id 而不是按 (document_id, parent_index) 查：
    ``parent_id`` 上有唯一索引，且与 Qdrant payload 逐字一致 —— 用同一个键
    跨库 join，就不会出现"两边算法对 index 的理解差一位"这种只在部分文档上
    复现的错位。回填失败的 chunk 保留 parent_text=None，调用方照常可用
    （降级为"只有子块"），绝不因为回填失败让整个提问失败。

    Returns:
        成功回填的 chunk 数（供日志/诊断）。
    """
    parents = [c for c in chunks if c.parent_id and not c.parent_text]
    if not parents:
        return 0

    from sqlalchemy import select
    from app.db.postgres import get_db_session
    from app.db.models import ChunkParent
    import uuid

    # 安全约束：父块必须与**子块自己所属的文档**绑定。
    # parent_id 是 "{document_id}:p:{index}"，正常情况天然同文档；但 Qdrant payload
    # 一旦被污染（parent_id 被改写成别的文档/租户），仅按 parent_id 回填就会把**他人
    # 文档的父块正文**灌进上下文。因此这里同时用子块的 document_id 集合做谓词，
    # 并在回填时逐条核对父块 document_id == 子块 document_id —— 父块跨文档/跨租户泄漏
    # 从根上关死（fail-closed：无法核对的子块不回填）。
    wanted_docs: set[uuid.UUID] = set()
    for c in parents:
        if not c.document_id:
            continue
        try:
            wanted_docs.add(uuid.UUID(str(c.document_id)))
        except (ValueError, TypeError):
            continue

    wanted = list({c.parent_id for c in parents})
    rows: dict[str, ChunkParent] = {}
    # 分批：一条 SQL 里的 IN 列表过长会撞上 PG 的参数上限（32767 个 bind），
    # 而 top_k 放大后的候选池在诊断路径下可能上千。
    batch = 500
    try:
        async with get_db_session() as session:
            for i in range(0, len(wanted), batch):
                res = await session.execute(
                    select(ChunkParent).where(
                        ChunkParent.parent_id.in_(wanted[i:i + batch]),
                        ChunkParent.document_id.in_(list(wanted_docs)),
                    )
                )
                for row in res.scalars():
                    rows[row.parent_id] = row
    except Exception:
        # 表不存在（迁移没跑）/ 数据库抖动：父块回填是**增益**，不是依赖。
        # 降级为"只有子块"仍然能回答，只是上下文少了一层。
        logger.exception("parent hydration failed — continuing with child chunks only")
        return 0

    filled = 0
    for c in parents:
        row = rows.get(c.parent_id)
        if row is None:
            continue
        # 双保险：父块必须与子块同属一份文档（防止 parent_id 与 document_id 不一致的污染）
        if str(row.document_id) != str(c.document_id):
            logger.warning(
                "parent hydration: document mismatch for parent_id=%s "
                "(child doc=%s parent doc=%s) — skipped",
                c.parent_id, c.document_id, row.document_id,
            )
            continue
        c.parent_text = row.text
        c.parent_char_start = row.char_start
        c.parent_char_end = row.char_end
        if c.parent_index is None:
            c.parent_index = row.idx
        if not c.section_path and row.section_path:
            c.section_path = list(row.section_path)
        if not c.heading and row.heading:
            # 父块的标题比子块的更可能是"小节名"（子块可能一开始就在正文中段，
            # 拿不到标题行）。只在子块没有标题时补，不覆盖。
            c.heading = row.heading
        filled += 1

    logger.debug("parent hydration: %d/%d chunk(s) filled from %d parent row(s)",
                 filled, len(parents), len(rows))
    return filled


def _dedup_by_parent(
    chunks: list[RetrievedChunk],
    *,
    max_per_parent: int = 2,
) -> list[RetrievedChunk]:
    """
    同父去重：同一个父块最多保留 *max_per_parent* 个子块（按当前顺序 = 相关度降序）.

    为什么必须有这一步（不是"可选优化"）：千份文档时，同一份文档的相邻子块
    语义高度重叠，精排给它们的打分也接近。Top-5 里出现 4 条来自同一节的子块是
    常态 —— 用户拿到的"5 条证据"其实只覆盖 1 个出处，而真正互补的第 2 个出处
    被挤掉了。答案因此看起来"有 5 条引用支撑"，实际上单点依据。

    保留 2 而不是 1：同一节里"表格 + 说明文字"或"结论 + 数据"是**互补**的，
    全砍成一个反而丢信息。上限 2 是"覆盖广度"与"局部完整度"的折中。

    没有 parent_id 的 chunk（旧索引 / 未启用父子）不受影响 —— 它们各自独立，
    按原有的 (document_id, chunk_index) 已经天然唯一。
    """
    if max_per_parent <= 0:
        return chunks

    seen: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for c in chunks:
        if not c.parent_id:
            kept.append(c)
            continue
        used = seen.get(c.parent_id, 0)
        if used >= max_per_parent:
            continue
        seen[c.parent_id] = used + 1
        kept.append(c)
    return kept


def _apply_parent_score_decay(
    chunks: list[RetrievedChunk],
    decay: float,
) -> list[RetrievedChunk]:
    """
    同一父块内第 2 条及以后的证据按 ``decay ** (k-1)`` 递减，并按新分数重排.

    为什么需要（而不是"同父去重已经够了"）
    ────────────────────────────────────────
    ``_dedup_by_parent`` 只做**数量**截断（同一父块最多留 2 条），它不动分数。
    于是同一个父块的第 2 条子块仍然带着精排给它的高分待在原位：这两条证据共享
    同一段父块上下文，第 2 条的**新增信息量**远小于第 1 条，却和"另一个出处的
    第 1 条证据"平起平坐。结果就是 top_k 里出处数被冗余证据吃掉 —— 用户看到
    "5 条引用"，实际只覆盖 2 个出处。

    衰减之后，同父第 2 条要挤掉别处第 1 条，必须比它高出约 ``1/decay - 1``
    （decay=0.85 时约 18%）—— 这才是"互补出处优先"的正确取舍。

    两个边界：
      · 只在**父块回填成功**（``parent_text`` 有值）的 chunk 上生效。这正是配置项
        ``PARENT_SCORE_DECAY`` 的语义 —— "父块回填**后**"的打分衰减；回填失败的
        子块没有父块上下文可共享，谈不上冗余。
      · 必须**重排**。只降分不重排等于"没降"（下游按顺序截断）—— 与
        ``_apply_trust`` 同一个坑，本文件已经踩过两次。
    """
    if not chunks or not 0.0 < decay < 1.0:
        return chunks

    seen: dict[str, int] = {}
    decayed = 0
    for c in chunks:
        if not c.parent_id or not c.parent_text:
            continue
        k = seen.get(c.parent_id, 0)
        seen[c.parent_id] = k + 1
        if k > 0:
            c.score = c.score * (decay ** k)
            decayed += 1

    if not decayed:
        return chunks
    logger.debug(
        "parent score decay: %d redundant sibling(s) decayed (decay=%.2f)",
        decayed, decay,
    )
    # 降分之后必须重排：排序决定后面的截断，不重排等于宽容了冗余证据。
    return sorted(chunks, key=lambda c: c.score, reverse=True)


def _dedup_identical_content(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    跨文档相同内容去重：同一段正文被重复上传的文档各存了一份时只保留一条.

    真实场景（截图问题）：用户把同一份文件重复上传了 3 次（3 个 document_id、
    内容逐字相同）。向量腿与 BM25 腿会把 3 份拷贝的**同一段**都召回到前列，
    精排分数也几乎相同 —— top_k 名额被重复证据占满，真正不同的文档被挤掉；
    引用列表里挂着 N 条内容一模一样的来源，"引用力度"看起来很足实际是单点依据，
    模型也容易把重复内容当成"多篇文档都这么说"而过度自信。

    判据是**归一化正文**（去全部空白后取前 512 字符），不按 document_id：
    重复上传意味着 document_id 不同、chunk_index 却相同，正文才是相同性的
    唯一可靠依据。保留当前顺序里的第一条（调用方已按相关度/可信度降序），
    其余丢弃。正文为空的 chunk（纯图片块）不参与去重。
    """
    seen: set[str] = set()
    kept: list[RetrievedChunk] = []
    dropped = 0
    for c in chunks:
        text = (c.text or "").strip()
        if not text:
            kept.append(c)
            continue
        key = re.sub(r"\s+", "", text)[:512]
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(c)
    if dropped:
        logger.info(
            "retrieval: dropped %d duplicated-content chunk(s) from repeated uploads",
            dropped,
        )
    return kept


def _apply_trust(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    把解析期已知的可信度信号接进最终排序（反幻觉的最后一道自动闸）.

    两件事，顺序不能反：

      1. **先算分**：``compute_trust`` 只看解析期已经记录在案的事实（产出引擎、
         置信度、人工复核标记、质检结论、正文长度），不引入任何新猜测。
      2. **再乘法混入**：``score × (1 - w + w × trust)``（w 默认 0.15）。用乘法
         是因为精排分表达"相关"、可信度表达"可信"，两者相乘才是"既相关又可信"；
         相加会让一条高分低可信的证据压过一条中分高可信的证据，恰好把最危险的
         （读错的 OCR、跑偏的 VLM 转写）抬上来。

    低于 ``EVIDENCE_TRUST_MIN`` 的直接丢弃：那是"解析期已判定需要人工复核"
    这一档，把它留在上下文里等于主动请模型基于错证据作答。

    重排后**重新按分排序** —— 只降分不重排会让"降权"变成"没降"（顺序没变，
    而下游是按顺序截断的）。
    """
    settings = get_settings()
    if not chunks or not getattr(settings, "EVIDENCE_TRUST_ENABLED", True):
        return chunks

    floor = float(getattr(settings, "EVIDENCE_TRUST_MIN", 0.30))
    weight = float(getattr(settings, "EVIDENCE_TRUST_RERANK_WEIGHT", 0.15))

    kept: list[RetrievedChunk] = []
    for c in chunks:
        report = compute_trust(
            text=c.text,
            content_type=c.content_type,
            image_id=c.image_id,
            analyze_engine=c.analyze_engine,
            analyze_confidence=c.analyze_confidence,
            manual_review=c.manual_review,
            analyze_quality=c.analyze_quality,
        )
        c.trust_score = report.score
        c.trust_reasons = report.reasons
        if report.score < floor:
            logger.info(
                "evidence trust: dropping chunk '%s' idx=%d (trust=%.2f < %.2f) reasons=%s",
                c.filename, c.chunk_index, report.score, floor, report.reasons,
            )
            continue
        if report.reasons:
            c.score = apply_trust_weighting(c.score, report.score, weight)
            logger.debug(
                "evidence trust: chunk '%s' idx=%d trust=%.2f → score=%.4f reasons=%s",
                c.filename, c.chunk_index, report.score, c.score, report.reasons,
            )
        kept.append(c)

    kept.sort(key=lambda x: x.score, reverse=True)
    return kept


def _apply_score_filters(
    candidates: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Pure-vector filtering (original Phase 4 semantics):
    absolute RETRIEVAL_MIN_SCORE floor + relative RETRIEVAL_MAX_GAP from the
    top score.  *candidates* must already be sorted by score descending.
    """
    settings = get_settings()
    if not candidates:
        return []

    top_score = candidates[0].score
    gap_cutoff = top_score - settings.RETRIEVAL_MAX_GAP

    kept: list[RetrievedChunk] = []
    for c in candidates:
        if c.score < settings.RETRIEVAL_MIN_SCORE:
            logger.info(
                "Filtering out chunk filename='%s' index=%d (score=%.4f < floor=%.2f)",
                c.filename, c.chunk_index, c.score, settings.RETRIEVAL_MIN_SCORE,
            )
            continue
        if c.score < gap_cutoff:
            logger.info(
                "Filtering out chunk filename='%s' index=%d (score=%.4f < gap_cutoff=%.4f)",
                c.filename, c.chunk_index, c.score, gap_cutoff,
            )
            continue
        kept.append(c)
    return kept[:top_k]


def _search_query_sets(
    query: str,
    extra_queries: list[str] | None,
    extra_vector_queries: list[str] | None,
) -> tuple[list[str], list[str]]:
    """
    把三路查询入参整理成两条通道，各自去重保序（纯函数，便于单测）.

    返回 ``(both_legs, vector_only)``：

      both_legs     主查询 + *extra_queries*（子问题 / 改写变体）——
                    向量腿与关键词腿**都**走。它们是"提问"，BM25 的词项
                    精确匹配在这里是有价值的。
      vector_only   *extra_vector_queries*（HyDE 假设答案段落）——
                    **只**走向量腿。它是一段"答案"，其价值在于把查询向量
                    拉到答案分布上；喂给 BM25 只会得到一份词项宽泛杂乱的
                    排名，在 RRF 里稀释真正的精确词命中（编号/型号类问题
                    受害最明显）。

    为什么必须是两条通道而不是一条入参：合并之后"哪一路走了哪条腿"这个
    决策就藏进了调用方，关键词腿会无条件遍历全部查询 —— 这是这个函数
    存在的唯一理由。
    """
    both: list[str] = [query] if query else []
    for q in extra_queries or []:
        if q and q not in both:
            both.append(q)

    vector_only: list[str] = []
    for q in extra_vector_queries or []:
        if q and q not in both and q not in vector_only:
            vector_only.append(q)

    return both, vector_only


# ═══════════════════════════════════════════════════════════════════════════════
# Query ⇄ Scope 绑定体（决策 15-1：文本可变、Scope 不可变）
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScopedQuery:
    """一条"待检索查询"与其 Scope 的**绑定体**（决策 15-1）.

    ``text`` 可换（改写 / 变体 / 子问题 / HyDE），``scope`` **恒为同一实例**：
    :meth:`with_text` 用 ``dataclasses.replace(self, text=...)`` 只替换文本，
    ``scope`` 字段**不出现在参数表里** —— 想换也换不了。

    这是"改写只能改 query 文本、不能改 Scope"的**结构保证**（不靠约定、也不靠
    code review）。本类刻意放在检索层（而非权限模块）：它是"检索入口"的输入
    契约，与 :func:`retrieve_chunks_scoped` 同处一层。
    """

    text: str
    scope: UserScope                 # frozen；同一请求内是**同一个对象**
    kind: str = "main"               # main | variant | subquery | hyde（日志用）

    def with_text(self, new_text: str) -> "ScopedQuery":
        """**唯一**允许改写入口：只换 ``text``，``scope`` 原样透传。"""
        return replace(self, text=new_text)


# ═══════════════════════════════════════════════════════════════════════════════
# 第 11 环：对象级复核 + 越权审计（决策 10 的第 11 环）
# ═══════════════════════════════════════════════════════════════════════════════


def _security_strict_mode() -> bool:
    """``SECURITY_STRICT_MODE``（读配置失败按**严格**，与 ``security_scope._strict_mode`` 同口径）."""
    try:
        return bool(get_settings().SECURITY_STRICT_MODE)
    except Exception:      # noqa: BLE001 — 读不到配置时不能静默变成"放行"
        logger.exception("SECURITY_STRICT_MODE 读取失败 —— 按严格模式处理")
        return True


async def _record_acl_drops(
    drops: list[tuple[Decision, str, str]],
    *,
    stage: str,
    scope_fingerprint: str | None,
) -> None:
    """
    批量记录"因权限被剔除"的审计（共享知识 13：``acl.drop.<stage>``）.

    为什么不逐条 ``await``：一次检索可能剔除几十个对象（权限刚收紧时最多，而那
    正是最该快的时候），逐条 = 几十次串行事务，全在用户请求路径上。改走
    :func:`audit_service.record_acl_drops` 后**每批一次事务**；每条剔除仍单独成行
    （``resource_id`` = 对象 id，可逐对象追溯），要求不变。

    ``drops`` 每项 ``(decision, object_id, document_id)``。

    审计是旁路：坏掉不能影响检索（fail-open），只记日志。
    """
    if not drops:
        return
    try:
        from app.services.audit_service import record_acl_drops

        await record_acl_drops(
            stage,
            [
                {
                    "object_id": object_id,
                    "document_id": document_id,
                    "reason": decision.reason,
                    "gate": decision.gate,
                }
                for decision, object_id, document_id in drops
            ],
            scope_fingerprint=scope_fingerprint,
        )
    except Exception:      # noqa: BLE001 — 审计是旁路，坏掉不能影响检索
        logger.exception(
            "acl.drop batch audit failed (stage=%s, n=%d)", stage, len(drops)
        )


async def _object_level_filter(
    chunks: list[RetrievedChunk],
    payloads: dict[tuple[str, int], dict],
    pred: ScopePredicate,
    *,
    stage: str,
    scope_fingerprint: str | None,
) -> tuple[list[RetrievedChunk], int]:
    """
    出参前的**对象级**复核（第 11 环）::

        对每个候选：``allows(pred, ObjectACLView.from_payload(payload))``。
        被剔除的对象 → 写审计（``acl.drop.<stage>``）并丢弃。

    为什么必须是**对象级**而不是文档级：文档级只能判"这份文档我能不能看"，
    无法表达"这张图 / 这个表格被单独提级或剔除"。图片/表格/代码块的对象级
    收紧只有在这一层才生效（PRD P0-6）。

    取不到 payload 的候选（例如老向量点根本没有安全字段）：默认**保留**，交由
    文档级 ``valid_docs`` 与前置过滤共同兜底 —— 不因为"拿不到对象视图"就凭空
    丢弃证据（那会把可用召回也一起杀掉）。``SECURITY_STRICT_MODE=true`` 时改为
    **丢弃**（fail-closed）：严格模式的口径就是"安全字段没回填完就别放行"，
    与 ``to_qdrant`` 的 ``ACL_SECURITY_PREFILTER_STRICT`` 同向；读配置失败按严格
    （与 ``security_scope._strict_mode`` 同口径），否则"配置读不到"会静默变成放行。

    Returns:
        ``(保留的 chunks, 被剔除计数)``
    """
    strict = _security_strict_mode()
    kept: list[RetrievedChunk] = []
    drops: list[tuple[Decision, str, str]] = []
    for chunk in chunks:
        payload = payloads.get(_chunk_key(chunk))
        if payload is None:
            if strict:
                drops.append((
                    Decision(False, "missing_object_view", "tenant"),
                    str(chunk.document_id),
                    chunk.document_id,
                ))
                continue
            kept.append(chunk)
            continue
        decision = allows(pred, ObjectACLView.from_payload(payload))
        if decision.allowed:
            kept.append(chunk)
            continue
        drops.append((decision, str(payload.get("object_id") or ""), chunk.document_id))
    # 一次批量落库（写库次数与剔除数量解耦）；每条剔除仍是独立审计行。
    await _record_acl_drops(drops, stage=stage, scope_fingerprint=scope_fingerprint)
    return kept, len(drops)


async def retrieve_chunks_scoped(
    query: str,
    *,
    scope: UserScope,
    top_k: int = 5,
    score_threshold: float = 0.0,
    collection_name: str | None = None,
    collection_id: str | None = None,
    extra_queries: list[str] | None = None,
    extra_vector_queries: list[str] | None = None,
    enable_hierarchical: bool | None = None,
    metadata_filter: MetadataFilter | None = None,
) -> list[RetrievedChunk]:
    """
    **唯一推荐**的用户请求检索入口（决策 10-①：签名强制）.

    四层"Query 必须绑定 Scope"的结构保证，本函数是第①层的落点：

      ① 签名强制 —— ``scope`` 是**必填关键字参数**；入口只有一个，任何用户请求
         都必须把 ``UserScope`` 带进来（由 ``app/api/query.py`` 一次性签发）。
      ② 运行时不可变 —— ``UserScope`` / ``ScopePredicate`` 均 ``frozen=True``，
         集合字段一律 ``frozenset``；链路内任何"补权限"的写法都会
         ``FrozenInstanceError``，而不是悄悄生效。
      ③ CI 静态门禁（AST）—— ``tests/test_no_unscoped_retrieval.py`` 扫描所有
         ``retrieve_chunks`` / ``keyword_search`` / ``client.search`` /
         ``client.scroll`` 调用点，断言必须带 ``scope=``（或 ``unrestricted=``）
         或位于显式白名单 —— "漏传"在合并前就被拦下，而不是等线上发现。
      ④ 禁止链路内重新签发 —— 这里**只消费**传入的 ``scope``，绝不调用
         ``scope_for`` / ``request_security_scope``（同一门禁强制）。

    兼容性：既有 :func:`retrieve_chunks` **保留不动**。它的既有 fail-closed
    （无任何权限上下文时 ``return []``）继续生效，因此"漏传 scope"的后果是
    **空结果**，不是全库。

    Args:
        scope: 请求级五维权限（``UserScope``，frozen）。
        query / top_k / ...: 见 :func:`retrieve_chunks`。

    Returns:
        ``list[RetrievedChunk]``，best first；``scope`` 为 ``None`` 时 fail-closed 返回 ``[]``。
    """
    if scope is None:
        # fail-closed 是**行为**，但**静默**不是可接受的可观测性：用户看到的是
        # "知识库没有相关内容"——与"真的没有"逐字相同，运维在日志里也只会看到
        # 一条与请求无关的 error。这里把"有人没带 Scope 就进来了"落成一条审计，
        # 事后可查（谁、什么时候、多少次），同时保留 ERROR 日志。
        # 【发现问题 #15】审计是旁路，失败不改变 fail-closed 的返回。
        logger.error(
            "retrieve_chunks_scoped called without a UserScope — refused (fail-closed)"
        )
        try:
            from app.services.audit_service import record_audit

            await record_audit(
                "retrieval.unscoped_refused",
                resource_type="retrieval",
                detail=f"query={query[:120]!r}",
            )
        except Exception:      # noqa: BLE001 — 审计失败不影响 fail-closed 语义
            logger.exception("unscoped-refusal audit failed")
        return []
    return await retrieve_chunks(
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        collection_name=collection_name,
        owner_id=str(scope.base.owner_id) if scope.base.owner_id else None,
        collection_id=collection_id,
        extra_queries=extra_queries,
        extra_vector_queries=extra_vector_queries,
        enable_hierarchical=enable_hierarchical,
        tenant_ids=scope.base.tenant_ids,
        user_department_id=scope.base.department_id,
        tenant_wide=scope.base.tenant_wide,
        owns_tenant_ids=scope.base.owns_tenant_ids,
        metadata_filter=metadata_filter,
        scope=scope,
    )


@timed_stage("retrieval")
async def retrieve_chunks(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.0,
    collection_name: str | None = None,
    owner_id: str | None = None,
    collection_id: str | None = None,
    extra_queries: list[str] | None = None,
    extra_vector_queries: list[str] | None = None,
    enable_hierarchical: bool | None = None,
    tenant_ids: frozenset[str] | None = None,
    user_department_id: str | None = None,
    tenant_wide: bool = False,
    owns_tenant_ids: frozenset[str] = frozenset(),
    unrestricted: bool = False,
    metadata_filter: MetadataFilter | None = None,
    scope: "UserScope | None" = None,
) -> list[RetrievedChunk]:
    """
    两阶段检索：粗排（召回+融合）→ 精排（cross-encoder rerank）.

    粗排阶段（HYBRID_SEARCH_ENABLED 时）：
      1. 向量腿 —— 原查询 + *extra_queries*（多查询扩展变体/子问题）
         + *extra_vector_queries*（HyDE 段落）逐路 ANN，
         同一 chunk 取最高向量分，各路排名进入 RRF；
      2. 关键词腿 —— BM25 对 query + *extra_queries* 逐路检索，排名同样进入 RRF；
         *extra_vector_queries* **不进**关键词腿（见 QUERY_HYDE_VECTOR_ONLY）；
      3. RRF 融合排名 + 向量 floor/gap 过滤 → 粗排候选池。

    精排阶段（RERANKER_ENABLED 时）：
      对粗排候选池（截断到 RERANKER_MAX_CANDIDATES 条）逐对 cross-encoder
      打分重排，取 top_k。chunk.score 被替换为归一化精排分（[0,1]）。
      精排只对 query + *extra_queries* 打分 —— 不带上 HyDE 段落：HyDE 与
      文档分布天然相似，拿它当打分基准会把一批"像答案但不对题"的候选抬上来，
      与它"只负责把召回拉宽"的定位矛盾。

    权限入参（由 ``tenancy.request_scope(user).acl_kwargs()`` 组装，成套出现）：

        owner_id          个人库归属人 —— **恒为本人 id**。None = 看不到任何个人库。
        tenant_ids        第一层公司过滤**集合**。None = 不限制（仅 unrestricted
                          诊断）；空集 = fail-closed（返回空）；非空 = IN (...)。
        user_department_id 第二层部门条件。
        tenant_wide       本租户内部门库全通（企业管理员 / 知识库管理员 / admin）。
        owns_tenant_ids   「可见他人 private」的租户集合（仅 admin = 自建测试公司）。
        unrestricted      **仅供后端诊断脚本 / 系统内部调用**：完全不做权限过滤。
                          任何用户请求路径都不允许传它。
        scope             请求级五维 ``UserScope``（**推荐入口**
                          :func:`retrieve_chunks_scoped` 透传）。给了它时：
                          ① 五维条件由**同一份** ``to_qdrant(pred)`` 下推（第 7 环
                          向量腿 + 内存 BM25 scroll），PG 关键词腿走 ``to_sql(pred)``；
                          ② 出参前做**对象级** ``allows(pred, obj)`` 复核（第 11 环）；
                          ③ 权限相关缓存键按 ``scope_fingerprint`` 分区。
                          ``None``（兼容旧调用点）时退回既有三维标量行为。

    注意：``private / NULL`` 层级只按 ``owner_id == 我`` 判定、**不参与租户过滤**；
    ``department / tenant`` 层级才受 ``tenant_ids`` 约束。``tenant_wide`` 只放宽
    部门维度，``owns_tenant_ids`` 只放开 private（P0-8 唯一例外）。

    Args:
        query:            Natural-language question from the user.
        top_k:            Maximum number of chunks to return.
        score_threshold:  Unused placeholder (kept for API compat).
        collection_name:  Override the default collection from settings.
        collection_id:    Restrict results to one knowledge-base collection
                          via the Qdrant payload filter (None = all).
        extra_queries:    检索变体 / 子问题，每路独立向量 + 关键词召回后
                          RRF 融合，提升召回率。
        extra_vector_queries:
                          只走向量腿的额外查询（HyDE 假设答案段落）。它不参与
                          关键词腿，也不参与精排打分 —— 只负责把语义召回的
                          覆盖面拉宽。数量不占 MULTI_QUERY_MAX_EXTRA 额度。
        metadata_filter:  元数据预过滤（doc_type / year / language / tags /
                          document_ids）。**在 ANN 之前**生效，因此它不是
                          "召回后再剔"，而是真的缩小了搜索空间 —— 千份文档时
                          这一点直接决定正确段落能否进 Top-K。空过滤器不过滤。

    Returns:
        List of RetrievedChunk, best first.
    """
    settings = get_settings()
    client = get_qdrant_client()
    coll = collection_name or settings.QDRANT_COLLECTION

    # ── 五维 Scope 编译（唯一 IR 输入；同一次请求只编译一次）─────────────────
    # ``scope`` 给了 → 三个编译器（to_qdrant / to_sql / allows）共用这**同一个**
    # frozen ``ScopePredicate`` 实例（决策 6.1「同一 pred 只构造一次」）。
    # 任何编译失败都**不得降级为不过滤** —— fail-closed 返回空（共享知识 7）。
    pred: ScopePredicate | None = None
    scope_fingerprint: str | None = None
    if scope is not None and not unrestricted:
        try:
            pred = scope.predicate()
        except ScopeCompileError:
            logger.exception("retrieve_chunks: scope.predicate() 编译失败 — fail-closed 返回空")
            return []
        except Exception:      # noqa: BLE001
            logger.exception("retrieve_chunks: 无法编译 ScopePredicate — fail-closed 返回空")
            return []
        scope_fingerprint = scope.scope_fingerprint

    if not unrestricted and pred is None and not (
        owner_id is not None
        or tenant_ids is not None          # 空集也算"给了范围"（fail-closed）
        or tenant_wide
        or bool(owns_tenant_ids)
    ):
        # fail-closed：调用方没有给出任何权限上下文时**不检索**，而不是"默认全开"。
        # 历史实现里"什么都不传 = admin 全览"，任何一处漏传权限参数都会静默越权。
        # 传了 ``scope`` 即视为"给了权限上下文"（哪怕五维全空 —— 那也由编译器
        # 的 fail-closed 分支兜住，而不是在这里退化成"不限制"）。
        logger.error(
            "retrieve_chunks called without any permission scope — refused (fail-closed). "
            "Pass owner_id/tenant_ids from tenancy.request_scope(), or unrestricted=True "
            "for diagnostic scripts only."
        )
        return []

    # ── 测试公司文档：**平台管理员 admin 不可检索**（用户新规则）─────────────────
    # 规则实现收敛在 ``tenancy.exclude_test_tenants``（**唯一实现点**）：本函数
    # 与 ``tenancy.content_scope`` 共用同一段逻辑，杜绝「两处各排除一遍」。
    # 触发条件 = 调用者是平台管理员（``owns_tenant_ids`` 非空 ⟺ admin，见
    # ``tenancy.scope_for``）；非 admin（含测试公司成员）原样返回，保证测试账号
    # 在自己公司内可正常检索。剔除后 admin 的 tenant_ids 变空集，但**不影响**
    # admin 自己落在 default 的个人库（个人库分支只看 ``owner_id == 我``）。
    # 三条检索腿（① Qdrant 向量 ② PG 关键词/内存 BM25 ③ DB 兜底校验）全部消费
    # 下面这两个局部变量，因此一处剔除、三腿同源生效。
    # ``unrestricted=True``（后端诊断脚本专用，非用户请求路径）跳过本排除。
    if not unrestricted:
        tenant_ids, owns_tenant_ids = await exclude_test_tenants(
            tenant_ids, owns_tenant_ids
        )

    # 检索查询集合：原查询（改写后）+ 去重变体
    queries, vector_only = _search_query_sets(
        query, extra_queries, extra_vector_queries
    )

    # Embed all queries with RETRIEVAL_QUERY task type (one batch call)
    # 向量腿跑 queries + vector_only（embed_batch 保序，下面 ann_results 与之对齐）
    vectors = await embed_batch_with_retry(
        queries + vector_only, task_type="RETRIEVAL_QUERY"
    )

    # 检索前置过滤（框架图：Permission Filter 在 Vector/BM25 之前）──────────
    # Rev2：**从 must 中移除单值 tenant_id**。租户边界改由 _visibility_conditions
    # 的 Deny-2 承接 —— 否则「自己 private 但在别的租户」的向量会被 must 一起滤掉
    # （= admin 看不到自己 default 私库的向量侧镜像）。collection_id 仍是业务知识库范围。
    search_filter = None
    if collection_id:
        from qdrant_client.http import models as qmodels

        search_filter = qmodels.Filter(must=[
            qmodels.FieldCondition(
                key="collection_id",
                match=qmodels.MatchValue(value=collection_id),
            )
        ])

    # 元数据前置过滤：与租户过滤**同一位置**（ANN 之前，见 _metadata_conditions
    # 的说明）。放在这里而不是"融合后剔除"是千份文档准确率的关键差别。
    meta_conds = []
    if metadata_filter is not None and not metadata_filter.is_empty():
        if not getattr(settings, "METADATA_FILTER_ENABLED", True):
            logger.warning("metadata_filter supplied but METADATA_FILTER_ENABLED=false — ignored")
        else:
            meta_conds = _metadata_conditions(metadata_filter)
            if meta_conds:
                from qdrant_client.http import models as qmodels

                if search_filter is None:
                    search_filter = qmodels.Filter(must=list(meta_conds))
                else:
                    search_filter.must = list(search_filter.must or []) + list(meta_conds)
                logger.info(
                    "metadata pre-filter active: %s",
                    _summarize_metadata_filter(metadata_filter),
                )

    # 可见性前置过滤（个人库 / 部门库 / 密级 / 项目 / deny）—— 与 tenant / metadata
    # 过滤**同层**，都在 ANN 之前。不下推的话，同公司内别人的私库向量会占满候选
    # 名额，再被 PG 侧 ACL 丢掉，用户自己的文档可能一份都进不了候选池（见
    # _visibility_conditions 的详细说明）。
    #
    # 【T3 双路同源下推（决策 6）】
    #   * 有 ``pred``（推荐入口）→ 用**同一份** ``to_qdrant(pred)``：它内部已复用
    #     既有 Deny-1/2/3，并追加 Deny-4 密级 / Deny-5 项目 / Deny-6 剔除·deny。
    #     此时**不再叠加** ``_visibility_conditions`` —— 后者与 to_qdrant 的 Deny-3
    #     有已知分歧（前者不豁免 owner/project），叠加会把已对齐的例外又排除掉。
    #   * 无 ``pred``（旧调用点）→ 保持既有三维行为，一行不改。
    if not unrestricted and pred is not None:
        try:
            scope_filter = to_qdrant(pred)
        except ScopeCompileError:
            logger.exception("retrieve_chunks: to_qdrant(pred) 编译失败 — fail-closed 返回空")
            return []
        acl_conds = list(getattr(scope_filter, "must_not", None) or [])
        if acl_conds:
            from qdrant_client.http import models as qmodels

            if search_filter is None:
                search_filter = qmodels.Filter(must_not=list(acl_conds))
            else:
                search_filter.must_not = (
                    list(search_filter.must_not or []) + list(acl_conds)
                )
        logger.info(
            "scope pre-filter active: fp=%s (to_qdrant, 五维同源)",
            scope_fingerprint,
        )
    elif not unrestricted and getattr(settings, "ACL_PREFILTER_ENABLED", True):
        acl_conds = _visibility_conditions(
            owner_id=owner_id,
            department_id=user_department_id,
            tenant_wide=tenant_wide,
            tenant_ids=tenant_ids,
            owns_tenant_ids=owns_tenant_ids,
        )
        if acl_conds:
            from qdrant_client.http import models as qmodels

            if search_filter is None:
                search_filter = qmodels.Filter(must_not=list(acl_conds))
            else:
                search_filter.must_not = (
                    list(search_filter.must_not or []) + list(acl_conds)
                )
            logger.info(
                "acl pre-filter active: owner=%s department_scope=%s tenants=%s owns=%s",
                bool(owner_id), not tenant_wide,
                tenant_scope_fingerprint(tenant_ids),
                tenant_scope_fingerprint(owns_tenant_ids),
            )

    # Wider vector candidate pool so RRF fusion has material to work with
    candidate_pool = max(top_k * 4, top_k + 8, settings.RERANKER_MAX_CANDIDATES)

    # ── 过取（召回率防御，见 config.ANN_OVERFETCH_FACTOR 的说明）─────────────
    # candidate_pool 是"**可用**候选"的目标条数；下面 Qdrant 侧要多取几倍，因为在
    # PG 可见性/状态校验之后会有一部分被丢掉（老向量 fail-open、已删文档的孤儿点、
    # 元数据冲突）。不过取时 ann_limit == candidate_pool，行为与升级前完全一致。
    overfetch_factor = max(1, int(getattr(settings, "ANN_OVERFETCH_FACTOR", 4) or 1))
    overfetched = overfetch_factor > 1
    ann_limit = candidate_pool
    if overfetched:
        ann_limit = min(
            candidate_pool * overfetch_factor,
            max(candidate_pool, int(getattr(settings, "ANN_OVERFETCH_MAX", 400) or 400)),
        )

    # ── 向量腿：每路查询独立 ANN，结果按 chunk 去重合并 ─────────────────────
    # 多路查询用 gather 并发发出（框架图 Vector Search 与 BM25 Search 并行），
    # 每路一次 RTT 而不是串行 N 次 RTT；单路失败会随 gather 整体抛出，
    # 与原先串行时同一查询失败即失败的行为一致。
    async def _ann_one(query_vector: list[float]):
        return await client.search(
            collection_name=coll,
            query_vector=query_vector,
            limit=ann_limit,
            with_payload=True,
            query_filter=search_filter,
        )

    ann_results = await asyncio.gather(*(_ann_one(qv) for qv in vectors))

    all_results: dict[tuple[str, int], tuple[float, dict]] = {}
    vector_rank_lists: list[list[tuple[str, int]]] = []
    for results in ann_results:
        rank_list: list[tuple[str, int]] = []
        for rank, hit in enumerate(results):
            payload = hit.payload or {}
            key = (
                str(payload.get("document_id", "")),
                int(payload.get("chunk_index", 0) or 0),
            )
            if key not in all_results or hit.score > all_results[key][0]:
                all_results[key] = (float(hit.score), payload)
            rank_list.append(key)
        vector_rank_lists.append(rank_list)

    search_results_count = len(all_results)
    if search_results_count == 0:
        logger.info("No raw candidates returned from Qdrant search.")
        return []

    chunks: list[RetrievedChunk] = []

    import uuid
    from sqlalchemy import select
    from app.db.postgres import get_db_session
    from app.db.models import Document

    doc_ids = set()
    for _, payload in all_results.values():
        doc_id_str = payload.get("document_id")
        if doc_id_str:
            try:
                doc_ids.add(uuid.UUID(doc_id_str))
            except ValueError:
                pass

    valid_docs = set()
    if doc_ids:
        async with get_db_session() as session:
            from app.db.models import DocumentStatus
            q = select(Document.id).where(
                Document.id.in_(doc_ids),
                Document.status == DocumentStatus.COMPLETED,
            )
            if unrestricted:
                # 仅供诊断脚本：不做任何权限过滤（与升级前的"admin 全览"等价）
                pass
            elif pred is not None:
                # 【T3 双路同源（决策 6.1 / 第 11 环）】文档级兜底校验与两条检索腿
                # 同源：**同一个** ``pred`` → ``to_sql(pred, Document)``。它内部复用
                # 既有 ``document_scope_clause`` 并追加密级 / 项目 / deny / excluded，
                # 可见集合与向量腿 ``to_qdrant(pred)``、PG 腿 ``to_sql(pred)`` 完全
                # 一致。编译失败必须 fail-closed（返回空），绝不降级为"不过滤"。
                try:
                    q = q.where(to_sql(pred, Document))
                except ScopeCompileError:
                    logger.exception(
                        "retrieve_chunks: to_sql(pred, Document) 编译失败 — fail-closed 返回空"
                    )
                    return []
            else:
                # 三层隔离的唯一 SQL 组装点：公司边界 ∪ 自己个人库。
                # 向量层已做前置过滤，这里是纵深防御 —— Qdrant 里可能残留
                # 迁移前没有 tenant_id / access_level payload 的旧向量。
                #
                # ⚠️ 不再在这里手拼 tenant/ACL：`document_scope_clause` 已把
                # 「private 与租户无关」的口径封装好，手拼必然与列表/关键词腿分叉。
                q = q.where(
                    document_scope_clause(
                        owner_id=uuid.UUID(str(owner_id)) if owner_id else None,
                        department_id=user_department_id,
                        tenant_ids=tenant_ids,
                        owns_tenant_ids=owns_tenant_ids,
                        tenant_wide=tenant_wide,
                    )
                )
            res = await session.execute(q)
            valid_docs = {str(r[0]) for r in res}

    if not valid_docs:
        logger.warning("No valid (COMPLETED + tenant/ACL-visible) documents in candidates.")
        return []

    # First gather all valid chunks (excluding orphans), keyed for fusion
    merged: dict[tuple[str, int], RetrievedChunk] = {}
    # 第 11 环所需的"对象视图"来源：payload 里才有密级 / 项目 / ACL / excluded。
    object_payloads: dict[tuple[str, int], dict] = {}
    meta_skipped = 0
    acl_dropped: set[str] = set()
    # 第 11 环的剔除审计先攒后写（见 _record_acl_drops）：一次检索可能剔除几十个
    # 候选，逐条 await 就是几十次串行事务压在请求路径上。
    postcheck_drops: list[tuple[Decision, str, str]] = []
    for (doc_id_str, _ci), (score, payload) in all_results.items():
        if doc_id_str not in valid_docs:
            # 措辞要准确：文档**通常是存在的**，只是「未 COMPLETED 或在本人的
            # ACL 范围内不可见」——早期写成 "non-existent document_id" 害得排查
            # 的人（包括我自己）去追一个并不存在的孤儿数据问题。逐条 warn 一次
            # 会刷 8 行噪音，这里收敛成按文档去重、循环后汇总一条。
            acl_dropped.add(doc_id_str)
            continue
        # ── 第 11 环：**对象级**复核（决策 10 / PRD P0-6）─────────────────────
        # 文档级通过（valid_docs）≠ 对象级通过：图片 / 表格 / 代码块可以被单独
        # 提级或剔除，只有逐对象 ``allows(pred, obj)`` 才拦得住。pred 为 None
        # （旧调用点）时跳过，保持既有行为零变化。
        if pred is not None:
            decision = allows(pred, ObjectACLView.from_payload(payload))
            if not decision.allowed:
                postcheck_drops.append((
                    decision, str(payload.get("object_id") or ""), doc_id_str,
                ))
                acl_dropped.add(doc_id_str)
                continue
        # 元数据纵深防御：前置过滤在 Qdrant 侧已生效，这里只拦"字段存在且冲突"
        # 的漏网项（旧向量缺字段时放行 —— 见 metadata.matches_payload 的说明）。
        if meta_conds and not matches_payload(metadata_filter, payload):
            meta_skipped += 1
            continue
        key = (doc_id_str, _ci)
        object_payloads[key] = payload
        merged[key] = RetrievedChunk(
                document_id=doc_id_str,
                filename=payload.get("filename", "unknown"),
                page_number=int(payload.get("page_number", 1)),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=payload.get("text", ""),
                score=score,
                parent_id=payload.get("parent_id"),
                parent_text=payload.get("parent_text"),
                parent_char_start=payload.get("parent_char_start"),
                parent_char_end=payload.get("parent_char_end"),
                heading=payload.get("heading"),
                section=payload.get("section"),
                **_media_fields(payload),
                **_line_fields(payload),
                **_position_fields(payload),
                **_analysis_fields(payload),
            )

    # Ensure descending order by best vector score
    valid_candidates = sorted(merged.values(), key=lambda c: c.score, reverse=True)

    # ── 过取收敛：在"**可用**候选"里取前 candidate_pool 条 ──────────────────
    # 走到这里，merged 里的每个 key 都已经过了 valid_docs（COMPLETED + 租户/ACL）
    # 与元数据校验 —— 所以这里的截断才配得上"候选池"这个说法。不过取时
    # valid_candidates 本来就 ≤ ann_limit == candidate_pool，这段不生效。
    if overfetched and len(valid_candidates) > candidate_pool:
        keep = {_chunk_key(c) for c in valid_candidates[:candidate_pool]}
        dropped = len(merged) - len(keep)
        merged = {k: v for k, v in merged.items() if k in keep}
        valid_candidates = valid_candidates[:candidate_pool]
        logger.info(
            "ANN over-fetch: %d raw ANN hit(s) → %d usable candidate(s); "
            "dropped %d beyond the pool of %d",
            ann_limit, len(keep), dropped, candidate_pool,
        )

    # Rebuild the primary (original-query) rank list restricted to merged keys
    primary_keys = set(merged.keys())
    vector_rank_lists = [
        [k for k in rl if k in primary_keys] for rl in vector_rank_lists
    ]

    # ── 关键词腿：每路查询各一条 ───────────────────────────────────────────
    # 权限上下文指纹（第三层缓存隔离，决策 9）：owner + department + 宽口径标志
    # 任一不同即不同键；绝不与非本人/非本部门/非同权限视图共享语料缓存。
    #
    # 【T3】有 ``scope`` 时，取值来源换成 ``scope.scope_fingerprint``（五维指纹，
    # 含 clearance / project_ids / principals）—— 比旧三维串更细，天然按 Scope
    # 分区。变量 ``perm_ctx`` 本身**保留**：它被 ``_bm25_candidates`` 的形参消费。
    if scope_fingerprint is not None:
        perm_ctx = scope_fingerprint
    else:
        perm_ctx = (
            f"{owner_id or '__none__'}:{user_department_id or '-'}"
            f":{'TW' if tenant_wide else '-'}"
            f":{tenant_scope_fingerprint(tenant_ids)}"
            f":{tenant_scope_fingerprint(owns_tenant_ids)}"
        )
    backend = str(getattr(settings, "HYBRID_KEYWORD_BACKEND", "postgres") or "postgres").strip().lower()
    keyword_rank_lists: list[list[tuple[str, int]]] = []
    floor = settings.RETRIEVAL_MIN_SCORE
    # 关键词腿命中的 chunk 里，只有它命中（向量腿没召回）的那些，向量分是未知的。
    # 给它们 floor 分（0.30 起）作为**诚实的占位**：它们靠精确词项命中赢得位置，
    # 不是靠语义相似；用 0 分会让它们在 RRF 融合后的 floor/gap 过滤里被误杀。
    keyword_only_score = max(floor, 0.30)

    if settings.HYBRID_SEARCH_ENABLED and backend == "postgres":
        # ── PG 关键词腿（默认）：覆盖 100% 语料，内存零放大 ──────────────────
        # 多路查询并发发出（与向量腿的 gather 对齐）：串行 N 次会让多查询扩展
        # 把延迟乘 N，而扩展本该是"用延迟换召回"的温和交换。
        pg_lists = await asyncio.gather(*(
            _pg_keyword_candidates(
                query=q,
                fetch_n=candidate_pool,          # 与向量腿同深
                collection_id=collection_id,
                tenant_ids=tenant_ids,
                owner_id=owner_id,
                user_department_id=user_department_id,
                tenant_wide=tenant_wide,
                owns_tenant_ids=owns_tenant_ids,
                unrestricted=unrestricted,
                metadata_filter=metadata_filter,
                # 【T3 双路同源】同一个 UserScope → 关键词腿 SQL 走 to_sql(pred)，
                # 与向量腿的 to_qdrant(pred) 由**同一个** ScopePredicate 编译。
                scope=scope,
                # 入口已编译的 pred 直接透传，PG 腿不再 scope.predicate() 现编。
                pred=pred,
            )
            for q in queries
        ))

        # 先把所有命中点的 id 汇总（按点 id 去重），**一次**批量取回 payload ——
        # 而不是每路查询各取一次。多查询扩展时这是 N 次 RTT → 分批后的 1~2 次。
        #
        # 注：向量腿已经拿到 payload 的点会被重复取一次。这是**故意**的取舍 ——
        # 要跳过得把 point_id 写进向量 payload（每 chunk 多一个 36 字节字段，
        # 1000 份文档 × 200 chunk ≈ 10 MB 永久存储），换来的只是省一次批量
        # retrieve（走 id 索引，毫秒级）。存储换延迟不划算。
        wanted: list[str] = []
        seen_points: set[str] = set()
        for rows in pg_lists:
            for _doc, _ci, point_id in rows:
                if point_id and point_id not in seen_points:
                    seen_points.add(point_id)
                    wanted.append(point_id)
        payloads = await _fetch_by_point_ids(wanted, coll) if wanted else {}

        for rows in pg_lists:
            rank_list: list[tuple[str, int]] = []
            for doc_id_str, _ci, point_id in rows:
                if doc_id_str not in valid_docs:
                    acl_dropped.add(doc_id_str)
                    continue
                key = (doc_id_str, int(_ci or 0))
                payload = payloads.get(point_id or "")
                if payload is None:
                    # 取不回 payload 就不能构造证据（正文/文件名/位置全在 payload 里）。
                    # 宁缺：一条没有正文的"命中"进上下文只会污染答案。
                    continue
                # ── 第 11 环：对象级复核（与向量腿同一份 allows(pred, obj)）─────
                if pred is not None:
                    decision = allows(pred, ObjectACLView.from_payload(payload))
                    if not decision.allowed:
                        postcheck_drops.append((
                            decision, str(payload.get("object_id") or ""), doc_id_str,
                        ))
                        acl_dropped.add(doc_id_str)
                        continue
                if meta_conds and not matches_payload(metadata_filter, payload):
                    meta_skipped += 1
                    continue
                rank_list.append(key)
                object_payloads.setdefault(key, payload)
                if key not in merged:
                    merged[key] = RetrievedChunk(
                        document_id=doc_id_str,
                        filename=payload.get("filename", "unknown"),
                        page_number=int(payload.get("page_number", 1) or 1),
                        chunk_index=key[1],
                        text=payload.get("text", ""),
                        score=keyword_only_score,
                        parent_id=payload.get("parent_id"),
                        parent_text=payload.get("parent_text"),
                        parent_char_start=payload.get("parent_char_start"),
                        parent_char_end=payload.get("parent_char_end"),
                        heading=payload.get("heading"),
                        section=payload.get("section"),
                        **_media_fields(payload),
                        **_line_fields(payload),
                        **_position_fields(payload),
                        **_analysis_fields(payload),
                    )
            keyword_rank_lists.append(rank_list)

    elif settings.HYBRID_SEARCH_ENABLED:
        # ── 内存 BM25 腿（小语料可选后端；超限会显式告警而非静默截断）────────
        for q in queries:
            try:
                bm25_ranked = await _bm25_candidates(
                    query=q,
                    # 与向量腿同深（Vector TopN + BM25 TopN 对齐，见框架图）
                    fetch_n=candidate_pool,
                    valid_docs=valid_docs,
                    collection_name=coll,
                    collection_id=collection_id,
                    owner_id=owner_id,
                    client=client,
                    tenant_ids=tenant_ids,
                    perm_context=perm_ctx,
                    # 【T3 双路同源】语料 scroll 用**同一份** to_qdrant(pred)；
                    # user_scope 用于 cache_key_for_scope 缓存分区（决策 9），
                    # pred 为入口编译好的**同一个**实例（决策 6.1）。
                    scope=scope,
                    pred=pred,
                )
                rows: list[tuple[str, int]] = []
                for c in bm25_ranked:
                    key = _chunk_key(c)
                    # 内存后端的语料来自 Qdrant payload，元数据过滤只能后置做
                    # （前置无法下推到 scroll）。这里补齐，保持与 PG 后端同语义。
                    if meta_conds and not matches_payload(metadata_filter, {
                        "document_id": c.document_id,
                        "doc_type": c.doc_type,
                        "doc_year": c.doc_year,
                        "language": c.language,
                        "business_tags": c.business_tags,
                        "keywords": c.keywords,
                    }):
                        meta_skipped += 1
                        continue
                    rows.append(key)
                    # 【第 11 环同源】把语料里带回来的原始 payload 交给收尾的对象级
                    # 复核：缺这一步，该腿的候选在 ``_object_level_filter`` 里没有
                    # 对象视图 → 只能放行，等于整条腿**跳过** ``allows()``，且剔除
                    # 不进审计。空 payload（老点）不注册，保持"宁缺勿错"的旧口径。
                    if c._acl_payload:
                        object_payloads.setdefault(key, c._acl_payload)
                    if key not in merged:
                        # 以 BM25 候选 chunk 为基底、仅覆盖 score —— 这样**全部**
                        # 字段（含图片分类 image_type / 位置 position·bbox / 质检
                        # analyze_quality·analyze_fusion / 引擎 analyze_engine·
                        # analyze_confidence·manual_review）都随命中带入，与 PG
                        # 关键词腿（**_media_fields + _position_fields +
                        # _analysis_fields）契约一致。此前手写逐字段构造漏掉上述
                        # 8 个，导致内存后端下前端徽标/原图/位置标签失效。
                        merged[key] = replace(c, score=keyword_only_score)
                keyword_rank_lists.append(rows)
            except Exception:
                logger.exception(
                    "Hybrid BM25 leg failed — falling back to vector-only retrieval"
                )

    # 两条关键词腿的剔除审计在这里一次写库（向量腿在循环前已攒好，共用一个列表）。
    await _record_acl_drops(
        postcheck_drops, stage="postcheck", scope_fingerprint=scope_fingerprint
    )

    if meta_skipped:
        logger.info(
            "metadata post-filter dropped %d chunk(s) from keyword/vector legs: %s",
            meta_skipped, _summarize_metadata_filter(metadata_filter),
        )

    if acl_dropped:
        # 正常情况下这一条应该很少出现：可见性已经下推到 Qdrant（个人库/部门库）
        # 与 PG SQL（关键词腿）。剩下的多是"未 COMPLETED"或迁移前的老向量。
        # 数量明显偏高就说明 ACL 预过滤没生效（ACL_PREFILTER_ENABLED=false？）。
        logger.info(
            "acl filter dropped candidates from %d document(s) not visible/COMPLETED "
            "for this scope: %s", len(acl_dropped), sorted(acl_dropped)[:5],
        )

    # ── 粗排融合（RRF over vector legs + keyword legs） ──────────────────────
    rank_lists = vector_rank_lists + keyword_rank_lists
    strong_keyword = set()
    for rl in keyword_rank_lists:
        strong_keyword.update(rl[:top_k])

    use_hier = (
        settings.HIERARCHICAL_RAG_ENABLED
        if enable_hierarchical is None
        else enable_hierarchical
    )
    trust_on = bool(getattr(settings, "EVIDENCE_TRUST_ENABLED", True))

    # 精排/截断请求条数 = top_k + 回补余量。
    #
    # 为什么需要余量：出参前还有两道**淘汰**（同父去重、可信度过低丢弃）。如果
    # 精排只给 top_k 条，淘汰之后返回的证据就少于调用方要求的条数 —— 用户要 5 条
    # 支撑却只拿到 3 条，而且没有任何提示，看起来像"知识库只有这些"。
    # 多要几条的代价只是 cross-encoder 多算几次（≤ top_k/2），换来的是"淘汰之后
    # 仍然凑得满"。
    headroom = max(0, top_k // 2) if (trust_on or use_hier) else 0
    fetch_final = min(
        top_k + headroom,
        max(top_k, int(settings.RERANKER_MAX_CANDIDATES)),
    )

    async def _finalize(picks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        统一收尾：父块回填（small-to-big）→ 同父去重 → 同父冗余衰减 →
        可信度加权 → 跨文档去重 → 截到 top_k.

        顺序是设计出来的，不能随意调换：

          回填 → 去重：去重的判据是 parent_id 对应的**父块**，必须先确认哪些
            子块真的能回填到父块（回填失败的子块没有父块可共享，去重它们
            等于凭空丢证据）。
          去重 → 衰减：衰减要作用在"最终保留的那两条"上。反过来先衰减再
            去重，"保留哪两条"的判据会被衰减结果污染 —— 那是另一个决策。
          去重 → 可信度：去重按当前相关度顺序保留每个父块最好的两条；若先做
            可信度加权，一条低可信但高分、与高可信同父的 chunk 会先把名额占掉，
            加权后才沉下去 —— 名额已经浪费了。
          衰减 → 截断：衰减会**改变顺序**，截断必须发生在重排之后。
          可信度 → 截断：同理，加权也会**改变顺序**（两次踩同一个坑，见
            _apply_trust 与 _apply_parent_score_decay 的注释）。
        """
        if not picks:
            return picks

        if use_hier:
            # 父块只作**上下文**，不参与打分：它更长，参与打分会让"大段落天然
            # 占优"，那就退回成"按章节检索"了（子块精确定位的意义全丢）。
            await _hydrate_parents(picks)
            picks = _dedup_by_parent(
                picks,
                max_per_parent=int(getattr(settings, "PARENT_CHILD_MAX_PER_PARENT", 2)),
            )
            # 去重只截数量、不动分数：同父第 2 条证据的新增信息量远小于第 1 条，
            # 不衰减就等于让冗余证据挤掉别人的唯一出处。顺序必须在去重**之后**
            # （先衰减会改变"保留哪两条"，那是另一个决策）。
            picks = _apply_parent_score_decay(
                picks,
                float(getattr(settings, "PARENT_SCORE_DECAY", 0.85)),
            )

        picks = _apply_trust(picks)
        if bool(getattr(settings, "RETRIEVAL_CONTENT_DEDUP_ENABLED", True)):
            # 跨文档相同内容去重放在可信度加权**之后**：保留的是加权后
            # 排名最靠前的那份拷贝；去重释放的名额由 headroom 回补。
            picks = _dedup_identical_content(picks)
        return picks[:top_k]

    # ── 精排（cross-encoder rerank）：用原始（改写后）查询打分 ─────────────
    # 向量腿与关键词腿都无排名材料时（HYBRID_SEARCH_ENABLED=false 且只有一路
    # 向量召回失败等），退化为纯向量 floor/gap 过滤 —— 这条路径同样要收尾，
    # 否则"只有这条路径"的场景下父块/可信度全部失效（最隐蔽的一类不一致）。
    fallback_path = not rank_lists or not any(rank_lists)

    if fallback_path:
        candidates = _apply_score_filters(valid_candidates, fetch_final)
        # ⚠️ 只要**有**候选就必须进精排，不能因为"只有 1 条"就跳过 ——
        # 跳过会让 chunk.score 停留在**粗排分**（向量余弦 / 关键词占位 0.30），
        # 而下游（rag_graph 幻觉守卫、evidence_gate、retrieval_grader、
        # 前端"相关度 %"）统一按**精排 sigmoid 分**口径解读。
        # 实测后果：一条语义上毫不相关的 chunk 被关键词腿提为唯一候选时，
        # 带着 0.1~0.3 的余弦分被当作"精排判它相关"，直接绕过拒答回答
        # "库里根本没有"的问题（余弦基线在小语料里普遍偏高，而交叉编码器
        # 对同一对文本会给出接近 0 的分）。代价仅一次 forward pass。
        if settings.RERANKER_ENABLED and candidates:
            from app.services.reranker import rerank_chunks

            chunks = await rerank_chunks(
                query,
                candidates,
                fetch_final,
                min_score=(
                    settings.RERANK_MIN_SCORE
                    if settings.RERANK_MIN_SCORE_FILTER
                    else None
                ),
            )
        else:
            chunks = candidates
    else:
        # 加权 RRF：向量腿与关键词腿可分别配置权重（默认 1.0/1.0 =
        # 经典 unweighted RRF）。调 HYBRID_BM25_WEIGHT > HYBRID_VECTOR_WEIGHT
        # 可让"型号/编号等精确词命中"更占优，反之更偏语义相似。
        fused_scores = rrf_fuse(
            rank_lists,
            k=settings.HYBRID_RRF_K,
            weights=(
                [settings.HYBRID_VECTOR_WEIGHT] * len(vector_rank_lists)
                + [settings.HYBRID_BM25_WEIGHT] * len(keyword_rank_lists)
            ),
        )

        # Filter policy: vector floor+gap OR strong keyword hit
        top_vector_score = valid_candidates[0].score if valid_candidates else 0.0
        gap_cutoff = top_vector_score - settings.RETRIEVAL_MAX_GAP

        survivors = [
            (key, c)
            for key, c in merged.items()
            if (c.score >= settings.RETRIEVAL_MIN_SCORE and c.score >= gap_cutoff)
            or key in strong_keyword
        ]
        survivors.sort(
            key=lambda kc: fused_scores.get(kc[0], 0.0), reverse=True
        )

        # 粗排候选池：截断到精排上限，控制 cross-encoder 推理量
        candidates = [c for _, c in survivors[: settings.RERANKER_MAX_CANDIDATES]]

        # 同 fallback 分支：只要有候选就要打分，避免粗排分冒充精排分（量纲泄漏）。
        if settings.RERANKER_ENABLED and candidates:
            from app.services.reranker import rerank_chunks

            chunks = await rerank_chunks(
                query,
                candidates,
                fetch_final,
                queries=queries,
                min_score=(
                    settings.RERANK_MIN_SCORE
                    if settings.RERANK_MIN_SCORE_FILTER
                    else None
                ),
            )
        else:
            chunks = candidates

    # ── 收尾：父块回填 → 同父去重 → 可信度加权 → 截断 ───────────────────────
    chunks = await _finalize(chunks)

    # ── Permission Check（框架图：Rerank 之后、进 Context 之前的最终闸）──────
    # 前面的 ANN 前置过滤 + PG valid_docs 已经保证了租户/ACL 可见性，这里是
    # 纯函数的最后兜底：任何环节出 bug 漏进来的、DB 侧不可见的 chunk 都在出参前拦下。
    #
    # ⚠️ 这里**不能**简单按「tenant ∈ tenant_ids」过滤：`private` 层级与租户无关
    # （admin 自己落在 default 的私库必须保留，否则就是 Rev2 修复的那个回归）。
    # 改用 PG 侧权威可见集合 `valid_docs`（由 `document_scope_clause` 算出）。
    if valid_docs:
        before = len(chunks)
        chunks = [c for c in chunks if c.document_id in valid_docs]
        if len(chunks) != before:
            logger.error(
                "Permission Check dropped %d chunk(s) outside the DB-visible set — "
                "前置过滤存在漏洞，请排查", before - len(chunks),
            )

    # ── 【T3 追加】对象级最终复核（第 11 环的收尾一跳）──────────────────────
    # 上面是**文档级**权威集合；这里再补一道**对象级** ``allows(pred, obj)``：
    # 文档级可见 ≠ 对象级可见（图片 / 表格 / 代码块可被单独提级或剔除）。
    # 保留上面的文档级检查（不删），只在其后**追加**对象级 —— 纵深防御。
    if pred is not None and chunks:
        chunks, obj_dropped = await _object_level_filter(
            chunks,
            object_payloads,
            pred,
            stage="postcheck",
            scope_fingerprint=scope_fingerprint,
        )
        if obj_dropped:
            logger.error(
                "Object-level Permission Check dropped %d chunk(s) that passed the "
                "document-level set — 前置对象级过滤存在漏洞，请排查", obj_dropped,
            )

    logger.info(
        "Retrieved %d chunks (floor=%.3f gap=%.3f hybrid=%s backend=%s rerank=%s "
        "queries=%d vector_only=%d top_k=%d hierarchical=%s meta=%s tenant=%s scope_fp=%s)",
        len(chunks),
        settings.RETRIEVAL_MIN_SCORE,
        settings.RETRIEVAL_MAX_GAP,
        settings.HYBRID_SEARCH_ENABLED,
        backend if settings.HYBRID_SEARCH_ENABLED else "-",
        settings.RERANKER_ENABLED,
        len(queries),
        len(vector_only),
        top_k,
        use_hier,
        _summarize_metadata_filter(metadata_filter),
        tenant_scope_fingerprint(tenant_ids),
        scope_fingerprint or "-",
    )
    return chunks
