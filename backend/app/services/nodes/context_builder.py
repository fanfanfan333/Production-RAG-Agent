"""
Context Builder（架构图 Context Builder 节点）.

把检索到的 chunks 组装成喂给 LLM 的上下文块。抽出独立模块的理由：

- generate / document_summary / doc_relations 三条分支都要"组装上下文"，
  以前这段逻辑散落在各 node 里各写一遍，编号规则和安全过滤容易走偏。
- 统一在这里处理四件事，之后所有分支行为一致：
    1. 提示注入清洗（prompt_security.sanitize_document_context）
       —— 检索到的文档是不可信数据，永远不能当成指令执行
    2. 统一 [Source N] 编号（引用标记靠这个编号才能对上）
    3. small-to-big 父块回填（Hierarchical RAG）
    4. **Context Compression**（架构图 Reranker → Threshold → Compression → LLM）
       —— 查询感知的句子级抽取式压缩：与问题无关的句子被裁掉，只留
       支撑回答的证据句。父块回填把 1000 字子块放大成几千字父块，
       不压缩的话 5 个 source 就能把 8k 的 num_ctx 挤爆，把真正相关
       的证据稀释成"长尾噪声"。压缩在注入清洗**之后**执行，被屏蔽
       的注入段落不会作为候选句重新入选。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import get_settings
from app.services.hybrid_search import tokenize
from app.services.prompt_security import sanitize_document_context
from app.services.retrieval_service import RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── Context Compression（句子级抽取式压缩）────────────────────────────────────

_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def _split_sentences(text: str) -> list[str]:
    """中英文通用的轻量分句（。！？!?；; 与换行都是句子边界）."""
    return [m.group(0).strip() for m in _SENT_RE.finditer(text) if m.group(0).strip()]


def compress_text(body: str, query: str | None, budget: int) -> str:
    """
    把 *body* 压缩到 *budget* 字符以内，尽量保留与 *query* 相关的句子.

    算法（确定性、零依赖、零 LLM 调用）：
      1. 分句后按「查询词覆盖率」给每个句子打分（hybrid_search.tokenize
         同一套分词，保证与 BM25 腿对"关键词"的理解一致）；
      2. 按分数从高到低贪心装入预算，直到装不下为止；
      3. 选中的句子按**原始顺序**重新拼接 —— 证据的可读性不因压缩被打乱；
      4. 全是超长单句装不进时，保底取首句截断（首句通常是定义句）。

    query 为空 / 无查询词时退化为"按原始顺序保句子截断"，仍比硬砍安全。
    """
    if budget <= 0 or len(body) <= budget:
        return body

    sentences = _split_sentences(body)
    if not sentences:
        return body[:budget]

    q_tokens = set(tokenize(query)) if query else set()

    def _score(sentence: str) -> float:
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens & set(tokenize(sentence)))
        return overlap / len(q_tokens)

    # 覆盖率降序，同分保持原始顺序（稳定排序）
    order = sorted(range(len(sentences)), key=lambda i: (-_score(sentences[i]), i))

    selected: list[int] = []
    used = 0
    for i in order:
        length = len(sentences[i])
        if used + length <= budget:
            selected.append(i)
            used += length

    if not selected:
        # 每个句子都超预算 —— 取首句截断，绝不返回空
        return sentences[0][:budget]

    selected.sort()
    return "".join(sentences[i] for i in selected)


@dataclass
class BuiltContext:
    """上下文组装结果."""

    context: str              # 可直接插入 system prompt 的上下文块
    sources: list[dict]       # 供前端引用展示的元数据（SSE sources 事件）
    masked_count: int         # 被安全过滤命中的片段数
    expanded_count: int       # 被 small-to-big 父块回填的片段数
    compressed_count: int = 0  # 被 Context Compression 裁剪的片段数


def build_context(
    chunks: list[RetrievedChunk],
    *,
    snippet_chars: int = 300,
    expand_parent: bool | None = None,
    query: str | None = None,
) -> BuiltContext:
    """
    把 *chunks* 组装成编号上下文.

    Args:
        chunks:          检索结果（已精排，按分数降序）
        snippet_chars:   前端引用预览截断长度
        expand_parent:   是否启用 small-to-big 父块回填；None = 跟随
                         settings.HIERARCHICAL_RAG_ENABLED
        query:           用户问题（改写后的自包含问题最佳）。提供且
                         CONTEXT_COMPRESSION_ENABLED 时启用查询感知压缩；
                         None 时退化为按句子边界截断，行为向后兼容。

    Returns:
        BuiltContext —— context 为空字符串表示没有任何证据
    """
    settings = get_settings()
    # 图片 URL 口径与 multimodal 分支共用同一函数，避免两处各拼一遍（漂移）。
    # 惰性导入：仅在本函数被调用时解析，规避模块级循环导入。
    from app.services.nodes.multimodal_context_node import image_url_for
    use_parent = (
        settings.HIERARCHICAL_RAG_ENABLED if expand_parent is None else expand_parent
    )
    compression = settings.CONTEXT_COMPRESSION_ENABLED
    per_source_budget = max(settings.CONTEXT_MAX_CHARS_PER_SOURCE, 0)
    # 总预算是软上限：后面的 source 至少保住 CONTEXT_MIN_SOURCE_CHARS，
    # 避免低排位证据被一刀切掉导致 [Source N] 编号与 grader 数量对不上
    remaining_total = max(settings.CONTEXT_MAX_TOTAL_CHARS, 0)

    if not chunks:
        return BuiltContext(
            context="No relevant documents were found in the knowledge base for this query.",
            sources=[],
            masked_count=0,
            expanded_count=0,
            compressed_count=0,
        )

    parts: list[str] = []
    sources: list[dict] = []
    masked = 0
    expanded = 0
    compressed = 0

    for chunk in chunks:
        # ── small-to-big 回填：命中子块后，把父块完整文本交给 LLM ──────────
        # 检索打分用的是子块（短、准），但生成答案需要更完整的上下文，
        # 所以这里换成父块文本。父块不存在就退回子块，不影响功能。
        body = chunk.text
        if use_parent and chunk.parent_text:
            body = chunk.parent_text
            expanded += 1

        # 检索到的文本是数据，绝不能当成指令 —— 注入清洗（压缩之前做，
        # 被屏蔽的注入段落就不会作为"相关句"被压缩器重新选中）
        safe_text, was_masked = sanitize_document_context(body)
        if was_masked:
            masked += 1

        # ── Context Compression：单源预算 + 总预算双重约束 ─────────────────
        if compression and per_source_budget > 0:
            allowance = per_source_budget
            if remaining_total > 0:
                allowance = min(
                    allowance,
                    max(remaining_total, settings.CONTEXT_MIN_SOURCE_CHARS),
                )
            if allowance < len(safe_text):
                compressed_body = compress_text(safe_text, query, allowance)
                if len(compressed_body) < len(safe_text):
                    compressed += 1
                safe_text = compressed_body
            if remaining_total > 0:
                remaining_total -= len(safe_text)

        # 上下文头部带上"哪几行"：模型据此可以把引用写细，
        # 也让"这条结论出自原文第几行"在 prompt 里就有据可依。
        # 图片块没有行号 → 退化为"第几张图"，同样给模型一个可核对的定位。
        span = chunk.line_span
        if span:
            loc = f", lines {span}"
        elif chunk.position_span:
            loc = f", {chunk.position_span}"
        else:
            loc = ""
        header = (
            f"[Source {len(parts) + 1}] {chunk.filename}, page {chunk.page_number}"
            + loc
            + f" (relevance: {chunk.score:.2f})"
        )
        parts.append(f"{header}\n{safe_text}")

        sources.append({
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            # 引用预览用子块原文（更贴切用户看到的高亮），不用父块；
            # 但它仍是不可信文档数据，屏蔽注入段落后再展示（问题3 文档防护）。
            "text_snippet": sanitize_document_context(
                chunk.text[:snippet_chars]
            )[0],
            "score": round(chunk.score, 4),
            "expanded_to_parent": bool(use_parent and chunk.parent_text),
            # ── 位置信息（细粒度引用）：行号 + 一句话溯源 ──────────────────
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "location": chunk.location_label(),
            # ── 图片位置：文档内序号 + 页面边界框 ───────────────────────────
            "position": chunk.position,
            "bbox": list(chunk.bbox) if chunk.bbox else None,
            # 产出质检 + 双通道融合（图片理解的可验证事实）
            "analyze_quality": dict(chunk.analyze_quality or {}),
            "analyze_fusion": dict(chunk.analyze_fusion or {}),
            "quality_score": chunk.quality_score,
            # ── 内容类型与图片信息（与 multimodal 分支 sources **契约一致**）────
            # 关闭 MULTIMODAL_CONTEXT_ENABLED 走本函数时，若缺这几个键，前端的
            # isImageSource 判定失效 → 图片引用不显示原图、无徽标。此处补齐，保证
            # 降级路径的 sources 与多模态路径同构。
            "content_type": chunk.content_type,
            "image_id": chunk.image_id,
            "image_path": chunk.image_path,
            "image_url": image_url_for(chunk.document_id, chunk.image_path),
            "image_caption": chunk.image_caption,
            "image_type": chunk.image_type,
            "analyze_engine": chunk.analyze_engine,
            "analyze_confidence": float(chunk.analyze_confidence or 0.0),
            "manual_review": bool(chunk.manual_review),
        })

    if masked:
        logger.warning(
            "Context builder masked %d document chunk(s) before generation", masked
        )
    if expanded:
        logger.info(
            "Context builder expanded %d/%d chunk(s) to parent context (small-to-big)",
            expanded, len(chunks),
        )
    if compressed:
        logger.info(
            "Context compression: %d/%d chunk(s) condensed (query=%r)",
            compressed, len(chunks), (query or "")[:60],
        )

    return BuiltContext(
        context="\n\n---\n\n".join(parts),
        sources=sources,
        masked_count=masked,
        expanded_count=expanded,
        compressed_count=compressed,
    )


def build_digest_context_from_digests(digests: list[dict]) -> str:
    """
    把跨文档摘要（relation_service 产出）组装成上下文.

    与 build_context 分开是因为 digest 的结构不同（按文档而非按 chunk），
    但仍复用同一套安全清洗。
    """
    if not digests:
        return "The knowledge base currently contains no indexed documents."

    parts: list[str] = []
    for i, d in enumerate(digests, start=1):
        filename = d.get("filename", "unknown")
        summary = d.get("summary") or d.get("digest") or d.get("text_snippet") or ""
        safe_text, _ = sanitize_document_context(str(summary))
        parts.append(f"[{i}] {filename}\n{safe_text}")

    return "\n\n---\n\n".join(parts)
