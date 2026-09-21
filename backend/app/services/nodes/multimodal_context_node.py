"""
Multimodal Context builder（部分5 + 部分6）.

对应设计稿的两处要求：

    部分5 —— Hybrid Retrieval 优化：
        Reranker → Top 5 Context → { text chunk , image chunk }
        image chunk 携带 image_path → Vision Model → LLM

    部分6 —— LangGraph 新增 multimodal_context 节点：
        route → rewrite → retrieve → grade → **multimodal_context** → generate

本模块做的事就是那个 ``for chunk in retrieved_chunks`` 循环的自然扩展：

    for chunk in chunks:
        if chunk.content_type in ("text", "table"):
            context.append(chunk.text)                    # 直接进上下文
        elif chunk.content_type == "image":
            image = load_image(chunk.image_path)          # 从落盘路径取原图
            vision = await analyze_image(image, query)    # 带着问题看图
            context.append({text, image_path, vision})    # 结论进上下文

设计取舍
────────
- **图文分流后统一编号**：[Source N] 编号跨文本/表格/图片连续，引用不会错位。
- **Vision 缺失不是错误**：没有多模态模型时，图片块退化为"它的可检索文本"
  （OCR + 入库期 caption），并在上下文中明确标注"视觉分析不可用"，
  让 LLM 知道这不是"图中没有相关信息"，而是"没能看图"。
- **图片始终回显**：无论 Vision 是否可用，image_path 都会随 sources 下发，
  前端引用卡片因此能展示原始图片（部分2"返回原始图片"）。
- **预算约束**：图片的视觉结论有独立字符预算，避免多张图片把 num_ctx 挤爆。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from app.config import get_settings
from app.services.image_understanding import IMAGE_TYPE_LABELS
from app.services.nodes.context_builder import compress_text
from app.services.prompt_security import sanitize_document_context
from app.services.retrieval_service import (
    RetrievedChunk,
    image_position_label,
    split_by_modality,
)
from app.services.storage import resolve_image_path
from app.services.vision import get_vision_service
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 引用的文本预览长度（与 context_builder 保持一致）
SNIPPET_CHARS = 300


@dataclass
class MultimodalBlock:
    """一个进入最终上下文的内容块（文本 / 表格 / 图片）."""

    index: int                     # 1-based → [Source N]
    content_type: str              # text | table | image
    filename: str
    page_number: int
    chunk_index: int
    score: float
    text: str                      # 实际进入上下文的正文
    # ── 位置信息（细粒度引用溯源）：1-based 闭区间行号 ─────────────────────
    line_start: int | None = None
    line_end: int | None = None
    image_path: str | None = None
    image_id: str | None = None
    vision: str | None = None      # 检索期 Vision 对图片的问答结论
    # 入库期的图片分类结论（table/formula/code/chart/diagram/screenshot/photo）
    # 让上下文里的图片块自带"这是什么图"的信息，LLM 与前端都能用
    image_type: str | None = None
    # 产出引擎 + 置信度 + 待人工复核（多引擎图片理解管线）
    analyze_engine: str | None = None
    analyze_confidence: float = 0.0
    manual_review: bool = False
    # ── 图片位置（细粒度引用）：文档内序号 + 页面边界框 ──────────────────────
    # 图片没有行号，用 position 定位（"第 2 张图"）；bbox 供前端按需画框。
    # 文本块两者皆为 None。
    position: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    # ── 产出质检 + 双通道融合（可验证的事实）─────────────────────────────────
    # analyze_quality：代码语法是否通过 / OCR 行置信度 / VLM 幻觉检查；
    # analyze_fusion：双通道策略与最终选中的通道。前端引用卡片据此显示
    # "代码语法未通过""OCR 置信度偏低"这类**有依据**的警示。
    analyze_quality: dict = field(default_factory=dict)
    analyze_fusion: dict = field(default_factory=dict)

    @property
    def quality_warning(self) -> str | None:
        """质检未通过时的一句话警示（无问题时 None）."""
        if not self.analyze_quality or self.analyze_quality.get("ok", True):
            return None
        reasons = [str(r) for r in (self.analyze_quality.get("reasons") or []) if str(r).strip()]
        return "；".join(reasons[:2]) or "产出未通过质检"

    @property
    def line_span(self) -> str | None:
        """人类可读行号区间（如 ``"12-28"``）；无位置信息时 None."""
        if self.line_start is None:
            return None
        if self.line_end is None or self.line_end == self.line_start:
            return str(self.line_start)
        return f"{self.line_start}-{self.line_end}"

    @property
    def is_image_derived(self) -> bool:
        """该块是否由图片产出（含"图片表格"，其 content_type 是 table）."""
        return bool(self.image_id)

    @property
    def position_span(self) -> str | None:
        """图片的一句话位置（``"第 2 张图"`` / ``"第 3 个表格"``）；文本块为 None."""
        if not self.is_image_derived:
            return None
        return image_position_label(self.content_type, self.position)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "content_type": self.content_type,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "text": self.text,
            # 位置信息：前端引用卡片据此显示"第几行"
            "line_start": self.line_start,
            "line_end": self.line_end,
            "line_span": self.line_span,
            # 图片位置：文档内序号 + 页面边界框
            "position": self.position,
            "bbox": list(self.bbox) if self.bbox else None,
            "position_span": self.position_span,
            "image_path": self.image_path,
            "image_id": self.image_id,
            "vision": self.vision,
            "image_type": self.image_type,
            "analyze_engine": self.analyze_engine,
            "analyze_confidence": self.analyze_confidence,
            "manual_review": self.manual_review,
            # 产出质检 + 双通道融合（图片理解的可验证事实）
            "analyze_quality": self.analyze_quality,
            "analyze_fusion": self.analyze_fusion,
            "quality_warning": self.quality_warning,
        }


@dataclass
class MultimodalContext:
    """multimodal_context 节点的产出."""

    context: str                                   # 可直接插入 system prompt
    sources: list[dict] = field(default_factory=list)     # SSE sources 事件
    blocks: list[MultimodalBlock] = field(default_factory=list)
    text_count: int = 0
    image_count: int = 0
    vision_used: int = 0
    vision_available: bool = False
    masked_count: int = 0
    # ── 【T4】第 12 环对象级校验的产物（未传 pred 时恒为「无剔除」）────────────
    dropped_count: int = 0
    acl_status: str = "clean"

    @property
    def has_evidence(self) -> bool:
        return bool(self.blocks)

    def blocks_as_dicts(self) -> list[dict]:
        return [b.to_dict() for b in self.blocks]


def _load_image_bytes(document_id: str, image_path: str | None) -> bytes | None:
    """从落盘路径读取原始图片字节（部分2）."""
    if not image_path:
        return None
    resolved = resolve_image_path(document_id, image_path)
    if resolved is None:
        return None
    try:
        return resolved.read_bytes()
    except Exception as exc:      # noqa: BLE001
        logger.warning("Failed to read image %s/%s: %s", document_id, image_path, exc)
        return None


def _image_url(document_id: str, image_path: str | None) -> str | None:
    """前端可访问的图片 URL（相对后端根路径）."""
    if not image_path:
        return None
    name = image_path.split("/")[-1]
    return f"/documents/{document_id}/images/{name}"


# 公开别名：master_graph 等外部模块以 `image_url_for` 引用它（见 __all__）
image_url_for = _image_url


def _snippet(text: str) -> str:
    safe, _ = sanitize_document_context(text[:SNIPPET_CHARS])
    return safe


async def build_multimodal_context(
    chunks: list[RetrievedChunk],
    query: str,
    *,
    enable_images: bool | None = None,
    pred=None,
    view_index=None,
    materialized: bool | Mapping[str, bool] = True,
    user_id: str | None = None,
    username: str | None = None,
) -> MultimodalContext:
    """
    把精排后的 chunks 组装成"图文并茂"的上下文.

    Args:
        chunks:        精排后的检索结果（Top-K）
        query:         用户问题（改写后的自包含问题最佳）—— Vision 看图时带着它
        enable_images: 是否启用图片视觉分析；None 跟随
                       settings.MULTIMODAL_CONTEXT_ENABLED
        pred:          【T4】**可选**的 :class:`ScopePredicate`。给出时**在 Vision 之前**
                       先逐块走 :func:`allows`（图片级 ACL），并给每条引用附
                       ``permission_snapshot``；不给出时行为与改动前**逐字一致**。
        view_index:    ``{document_id: {chunk_key: ObjectACLView}}``（仅在给出 pred 时生效）。
        materialized:  对象权限行是否已物化；``False`` 时缺行回退允许。亦可传
                       ``{document_id: bool}``（``security_cascade.load_view_indexes``
                       的逐文档结论）—— 缺失的文档按 ``False`` 处理。
    """
    settings = get_settings()

    # ── 【T4】第 12 环对象级最终校验：**必须先于 Vision**（图不能先被识别再被挡）──
    acl_status = "clean"
    dropped_count = 0
    if pred is not None and view_index is not None:
        from app.services.nodes.final_check_node import (
            audit_acl_drops,
            filter_chunks_by_acl,
        )

        doc_ids = {str(getattr(c, "document_id", "") or "") for c in chunks}
        # 【T4-批量】``materialized`` 既接受 ``bool``（广播，向后兼容）也接受
        # ``{document_id: bool}``（``load_view_indexes`` 的逐文档结论）；Mapping 中
        # 缺失的文档按 ``False``（未物化 ⇒ 缺行回退允许）处理。
        if isinstance(materialized, Mapping):
            mat_map = {
                doc_id: bool(materialized.get(doc_id, False))
                for doc_id in doc_ids if doc_id
            }
        else:
            mat_map = {doc_id: bool(materialized) for doc_id in doc_ids if doc_id}
        outcome = filter_chunks_by_acl(chunks, pred, view_index, materialized=mat_map)
        # 【T5 上线前修复】第 12 环剔除**必须留痕**：否则事后无法回答
        # "这个用户为什么看不到这张图"（PRD P0-10）。audit_acl_drops 自带
        # best-effort 保护，不会打断回答路径。
        if outcome.dropped:
            await audit_acl_drops(
                outcome, pred, user_id=user_id, username=username
            )
        chunks = list(outcome.allowed)
        dropped_count = outcome.dropped_count
        acl_status = outcome.status

    if not chunks:
        return MultimodalContext(
            context="No relevant documents were found in the knowledge base for this query.",
            dropped_count=dropped_count,
            acl_status=acl_status,
        )

    use_images = (
        settings.MULTIMODAL_CONTEXT_ENABLED if enable_images is None else enable_images
    )

    text_chunks, image_chunks = split_by_modality(chunks)

    # Vision 可用性只探测一次；不可用时优雅降级（部分4）
    vision_available = False
    if use_images and image_chunks:
        try:
            vision_available = await get_vision_service().is_available()
        except Exception as exc:      # noqa: BLE001
            logger.warning("Vision availability probe failed: %s", exc)
            vision_available = False

    max_vision = max(int(settings.MAX_VISION_IMAGES_PER_QUERY), 0)
    image_text_budget = max(int(settings.MULTIMODAL_IMAGE_TEXT_BUDGET), 0)

    blocks: list[MultimodalBlock] = []
    sources: list[dict] = []
    masked = 0
    vision_used = 0
    index = 0

    # 文本预算（与 context_builder 同一套机制：单源上限 + 总量软上限）
    per_source_budget = max(int(settings.CONTEXT_MAX_CHARS_PER_SOURCE), 0)
    remaining_total = max(int(settings.CONTEXT_MAX_TOTAL_CHARS), 0)
    compression_on = bool(settings.CONTEXT_COMPRESSION_ENABLED)
    expand_parent = bool(settings.HIERARCHICAL_RAG_ENABLED)

    # 统一遍历原始顺序，保证 [Source N] 与 sources 数组一一对应
    for chunk in chunks:
        index += 1

        if chunk.is_image:
            vision_text: str | None = None
            image_bytes = _load_image_bytes(chunk.document_id, chunk.image_path)

            if (
                use_images
                and vision_available
                and image_bytes
                and vision_used < max_vision
            ):
                try:
                    vision_text = await get_vision_service().analyze_image(
                        image_bytes, query
                    )
                    if vision_text:
                        vision_used += 1
                except Exception as exc:      # noqa: BLE001
                    logger.warning(
                        "Vision analyze failed for %s: %s", chunk.image_path, exc
                    )

            body_parts: list[str] = []
            # 图片自身的可检索文本（OCR + 入库期 caption）
            if chunk.text.strip():
                body_parts.append(chunk.text.strip())
            # 检索期看图得到的针对性结论
            if vision_text:
                body_parts.append(f"视觉分析: {vision_text}")
            if not body_parts:
                body_parts.append(
                    "（该图片没有可提取文字，且当前未启用/未安装视觉模型，无法解读图片内容）"
                    if not vision_available
                    else "（该图片没有可提取文字）"
                )

            body_raw = "\n".join(body_parts)
            safe_text, was_masked = sanitize_document_context(body_raw)
            if was_masked:
                masked += 1

            if image_text_budget > 0 and len(safe_text) > image_text_budget:
                safe_text = safe_text[:image_text_budget]

            block = MultimodalBlock(
                index=index,
                content_type="image",
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=round(float(chunk.score), 4),
                text=safe_text,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                # 图片位置：文档内序号 + 页面边界框（细粒度引用）
                position=chunk.position,
                bbox=chunk.bbox,
                image_path=chunk.image_path,
                image_id=chunk.image_id,
                vision=vision_text,
                # 入库期的分类结论与解析引擎 —— 之前这里漏传，导致图片块在
                # 上下文/SSE 里丢了"这是什么图、由谁解析、多可信"的信息，
                # 而表格图片分支却带着 —— 两条支路现在对齐。
                image_type=chunk.image_type,
                analyze_engine=chunk.analyze_engine,
                analyze_confidence=float(chunk.analyze_confidence or 0.0),
                manual_review=bool(chunk.manual_review),
                analyze_quality=dict(chunk.analyze_quality or {}),
                analyze_fusion=dict(chunk.analyze_fusion or {}),
            )
        else:
            # ── small-to-big：命中子块后回填父块完整语义 ──────────────────────
            body = chunk.text
            if expand_parent and chunk.parent_text:
                body = chunk.parent_text

            safe_text, was_masked = sanitize_document_context(body)
            if was_masked:
                masked += 1

            # ── Context Compression：查询感知的句子级压缩 ────────────────────
            if compression_on and per_source_budget > 0:
                allowance = per_source_budget
                if remaining_total > 0:
                    allowance = min(
                        allowance,
                        max(remaining_total, settings.CONTEXT_MIN_SOURCE_CHARS),
                    )
                if allowance < len(safe_text):
                    safe_text = compress_text(safe_text, query, allowance)
                if remaining_total > 0:
                    remaining_total -= len(safe_text)

            block = MultimodalBlock(
                index=index,
                content_type=chunk.content_type or "text",
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=round(float(chunk.score), 4),
                text=safe_text,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                # 图片位置：图片表格走的是这条支路（content_type="table"），
                # 位置信息同样要带上，否则"第几个表格"就丢了。
                position=chunk.position,
                bbox=chunk.bbox,
                image_type=chunk.image_type,
                analyze_engine=chunk.analyze_engine,
                analyze_confidence=float(chunk.analyze_confidence or 0.0),
                manual_review=bool(chunk.manual_review),
                analyze_quality=dict(chunk.analyze_quality or {}),
                analyze_fusion=dict(chunk.analyze_fusion or {}),
            )

        blocks.append(block)
        sources.append(
            {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text_snippet": _snippet(chunk.text) or block.text[:SNIPPET_CHARS],
                "score": round(float(chunk.score), 4),
                # ── 位置信息（细粒度引用）：行号 + 一句话溯源 ─────────────────
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "location": chunk.location_label(),
                # ── 图片位置：文档内序号 + 页面边界框 ────────────────────────
                "position": chunk.position,
                "bbox": list(chunk.bbox) if chunk.bbox else None,
                # ── 部分3：内容类型与图片信息随引用下发 ─────────────────────
                "content_type": block.content_type,
                "image_id": chunk.image_id,
                "image_path": chunk.image_path,
                "image_url": _image_url(chunk.document_id, chunk.image_path),
                "image_caption": chunk.image_caption,
                "vision": block.vision,
                "image_type": chunk.image_type,
                # 多引擎管线的产出引擎 / 置信度 / 待复核（前端展示与筛选）
                "analyze_engine": chunk.analyze_engine,
                "analyze_confidence": float(chunk.analyze_confidence or 0.0),
                "manual_review": bool(chunk.manual_review),
                # 产出质检 + 双通道融合（前端引用卡片据此显示警示与"怎么来的"）
                "analyze_quality": dict(chunk.analyze_quality or {}),
                "analyze_fusion": dict(chunk.analyze_fusion or {}),
                "quality_score": chunk.quality_score,
                "quality_warning": block.quality_warning,
                "fusion_strategy": chunk.fusion_strategy,
            }
        )

        # ── 【T4】引用权限快照（决策 16）：只存指纹，不存明文权限属性 ──────────
        if pred is not None:
            from app.services.nodes.final_check_node import (
                SNAPSHOT_KEY,
                build_permission_snapshot,
                chunk_key as _chunk_key,
                derive_parent_object_id,
                object_type_of_chunk,
            )

            doc_key = str(chunk.document_id)
            _view = (view_index or {}).get(doc_key, {}).get(_chunk_key(chunk))
            sources[-1][SNAPSHOT_KEY] = build_permission_snapshot(
                _view,
                pred,
                object_id=str(_view.object_id) if _view is not None else _chunk_key(chunk),
                object_type=object_type_of_chunk(chunk.content_type, chunk.image_id),
                parent_object_id=derive_parent_object_id(
                    doc_key, chunk.content_type, chunk.image_id
                ),
            )

    # ── 组装最终上下文文本 ────────────────────────────────────────────────────
    # kind 会把图片类型带给 LLM（"图表"/"流程图"…）—— 让模型知道这段证据
    # 是一张柱状图而不是一张照片，对"图里的数据是多少"这类问题很关键。
    parts: list[str] = []
    for block in blocks:
        if block.content_type == "image":
            label = IMAGE_TYPE_LABELS.get(block.image_type or "", "图片")
            kind = label if block.vision else f"{label}（视觉分析不可用）"
        elif block.content_type == "table":
            kind = "表格"
        else:
            kind = None

        # 位置提示：文本块给"哪几行"，图片块给"第几张图" —— 模型据此才能
        # 写出可核对的引用（如 "[Source 2]（第 3 页，第 2 张图）"）。
        # 两者都没有时就不写，不编造位置。
        if block.line_span:
            loc = f", lines {block.line_span}"
        elif block.position_span:
            loc = f", {block.position_span}"
        else:
            loc = ""

        header = (
            f"[Source {block.index}] {block.filename}, page {block.page_number}"
            + loc
            + f" (relevance: {block.score:.2f})"
            + (f" — {kind}" if kind else "")
        )
        parts.append(f"{header}\n{block.text}")

    context = "\n\n---\n\n".join(parts)

    if masked:
        logger.warning(
            "Multimodal context masked %d block(s) before generation", masked
        )
    if image_chunks:
        logger.info(
            "Multimodal context: %d text/table + %d image chunk(s); "
            "vision_available=%s vision_used=%d",
            len(text_chunks), len(image_chunks), vision_available, vision_used,
        )

    return MultimodalContext(
        context=context,
        sources=sources,
        blocks=blocks,
        text_count=len(text_chunks),
        image_count=len(image_chunks),
        vision_used=vision_used,
        vision_available=vision_available,
        masked_count=masked,
        dropped_count=dropped_count,
        acl_status=acl_status,
    )


__all__ = [
    "MultimodalBlock",
    "MultimodalContext",
    "build_multimodal_context",
    "image_url_for",
]
