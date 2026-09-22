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
from collections.abc import Mapping
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
    # ── 【T4】第 12 环对象级校验的产物（未传 pred 时恒为「无剔除」）────────────
    dropped_count: int = 0     # 被对象级 ACL 剔除的片段数
    acl_status: str = "clean"  # clean | partial | all_dropped


def _schedule_acl_drop_audit(
    outcome,
    pred,
    *,
    user_id: str | None = None,
    username: str | None = None,
) -> None:
    """
    同步装配路径的第 12 环审计调度（**best-effort**）.

    ``build_context`` 是同步函数，而 ``audit_acl_drops`` 是协程。两种做法：
    (a) 把 ``build_context`` 改成 async —— 会波及 master_graph 两处调用点与
    现有单测；(b) 在当前事件循环上派一个任务。选 (b)：审计是旁路副作用，
    不该改变装配函数的同步契约。

    无运行中的事件循环（同步脚本 / 单测环境）时**静默放弃**审计：审计永远
    不能成为回答路径的失败点（与 ``audit_acl_drops`` 的 best-effort 口径一致）。
    """
    try:
        import asyncio

        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        from app.services.nodes.final_check_node import audit_acl_drops

        loop.create_task(
            audit_acl_drops(outcome, pred, user_id=user_id, username=username)
        )
    except Exception:      # noqa: BLE001 — 调度失败同样不打断装配
        logger.warning("build_context: acl drop audit scheduling failed", exc_info=True)


def build_context(
    chunks: list[RetrievedChunk],
    *,
    snippet_chars: int = 300,
    expand_parent: bool | None = None,
    query: str | None = None,
    pred=None,
    view_index=None,
    materialized: bool | Mapping[str, bool] = True,
    user_id: str | None = None,
    username: str | None = None,
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
        pred:            【T4】**可选**的 :class:`ScopePredicate`。给出时逐块走
                         :func:`allows` 过滤（第 12 环），并给每条引用附
                         ``permission_snapshot``；不给出时行为与改动前**逐字一致**。
        view_index:      ``{document_id: {chunk_key: ObjectACLView}}``（见
                         :func:`security_cascade.load_document_view_index`）。仅在
                         给出 ``pred`` 时生效。
        materialized:    对象权限行是否已物化；``False`` 时缺行回退允许（存量
                         未回填文档不因新逻辑集体不可见）。亦可传
                         ``{document_id: bool}``（``security_cascade.load_view_indexes``
                         的逐文档结论）—— 缺失的文档按 ``False`` 处理。

    Returns:
        BuiltContext —— context 为空字符串表示没有任何证据
    """
    settings = get_settings()
    # 图片 URL 口径与 multimodal 分支共用同一函数，避免两处各拼一遍（漂移）。
    # 惰性导入：仅在本函数被调用时解析，规避模块级循环导入。
    from app.services.nodes.multimodal_context_node import image_url_for

    # ── 【T4】第 12 环对象级最终校验（未给 pred 时整段跳过，零行为变化）────────
    acl_status = "clean"
    dropped_count = 0
    # 逐文档物化结论；提升到 if 之外，供下面的父块复核（发现问题 #14）复用。
    mat_map: dict[str, bool] = {}
    if pred is not None and view_index is not None:
        from app.services.nodes.final_check_node import filter_chunks_by_acl

        doc_ids = {str(getattr(c, "document_id", "") or "") for c in chunks}
        # 【T4-批量】``materialized`` 既接受 ``bool``（广播给所有文档，向后兼容），
        # 也接受 ``{document_id: bool}``（``load_view_indexes`` 的逐文档结论）。
        # Mapping 里**缺失**的文档按 ``False``（未物化 ⇒ 缺行回退允许）处理。
        if isinstance(materialized, Mapping):
            mat_map = {
                doc_id: bool(materialized.get(doc_id, False))
                for doc_id in doc_ids if doc_id
            }
        else:
            mat_map = {doc_id: bool(materialized) for doc_id in doc_ids if doc_id}
        outcome = filter_chunks_by_acl(
            chunks, pred, view_index, materialized=mat_map
        )
        # 【T5 上线前修复】第 12 环剔除必须留痕 —— 见 _schedule_acl_drop_audit。
        if outcome.dropped:
            _schedule_acl_drop_audit(
                outcome, pred, user_id=user_id, username=username
            )
        chunks = list(outcome.allowed)
        dropped_count = outcome.dropped_count
        acl_status = outcome.status

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
            dropped_count=dropped_count,
            acl_status=acl_status,
        )

    parts: list[str] = []
    sources: list[dict] = []
    masked = 0
    expanded = 0
    compressed = 0
    parent_blocked = 0

    for chunk in chunks:
        # ── small-to-big 回填：命中子块后，把父块完整文本交给 LLM ──────────
        # 检索打分用的是子块（短、准），但生成答案需要更完整的上下文，
        # 所以这里换成父块文本。父块不存在就退回子块，不影响功能。
        body = chunk.text
        if use_parent and chunk.parent_text:
            # 【#14】父块正文同样要过对象级判定：子块可见 ≠ 父块可见。不可见时
            # 退回子块正文（少一层上下文，而不是少一条证据，更不是把提级的整节
            # 正文送出去）。判定与 multimodal 组装入口**共用同一函数**。
            from app.services.nodes.final_check_node import parent_block_allows

            if parent_block_allows(chunk, pred, view_index, mat_map):
                body = chunk.parent_text
                expanded += 1
            else:
                parent_blocked += 1

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

    if masked:
        logger.warning(
            "Context builder masked %d document chunk(s) before generation", masked
        )
    if expanded:
        logger.info(
            "Context builder expanded %d/%d chunk(s) to parent context (small-to-big)",
            expanded, len(chunks),
        )
    if parent_blocked:
        # 这是一条**安全剔除**的留痕：父块被单独提级 / 剔除，正文不再进 LLM。
        # 打在 WARNING 上，运维看日志就知道"不是回填坏了，是权限拦下了"。
        logger.warning(
            "Context builder blocked %d parent block(s) by object-level ACL "
            "(small-to-big fell back to child text)",
            parent_blocked,
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
        dropped_count=dropped_count,
        acl_status=acl_status,
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
