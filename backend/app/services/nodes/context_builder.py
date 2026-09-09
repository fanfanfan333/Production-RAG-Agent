"""
Context Builder（架构图 Context Builder 节点）.

把检索到的 chunks 组装成喂给 LLM 的上下文块。抽出独立模块的理由：

- generate / document_summary / doc_relations 三条分支都要"组装上下文"，
  以前这段逻辑散落在各 node 里各写一遍，编号规则和安全过滤容易走偏。
- 统一在这里处理三件事，之后所有分支行为一致：
    1. 提示注入清洗（prompt_security.sanitize_document_context）
       —— 检索到的文档是不可信数据，永远不能当成指令执行
    2. 统一 [Source N] 编号（引用标记靠这个编号才能对上）
    3. small-to-big 父块回填（Hierarchical RAG）
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.services.prompt_security import sanitize_document_context
from app.services.retrieval_service import RetrievedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BuiltContext:
    """上下文组装结果."""

    context: str              # 可直接插入 system prompt 的上下文块
    sources: list[dict]       # 供前端引用展示的元数据（SSE sources 事件）
    masked_count: int         # 被安全过滤命中的片段数
    expanded_count: int       # 被 small-to-big 父块回填的片段数


def build_context(
    chunks: list[RetrievedChunk],
    *,
    snippet_chars: int = 300,
    expand_parent: bool | None = None,
) -> BuiltContext:
    """
    把 *chunks* 组装成编号上下文.

    Args:
        chunks:          检索结果（已精排，按分数降序）
        snippet_chars:   前端引用预览截断长度
        expand_parent:   是否启用 small-to-big 父块回填；None = 跟随
                         settings.HIERARCHICAL_RAG_ENABLED

    Returns:
        BuiltContext —— context 为空字符串表示没有任何证据
    """
    settings = get_settings()
    use_parent = (
        settings.HIERARCHICAL_RAG_ENABLED if expand_parent is None else expand_parent
    )

    if not chunks:
        return BuiltContext(
            context="No relevant documents were found in the knowledge base for this query.",
            sources=[],
            masked_count=0,
            expanded_count=0,
        )

    parts: list[str] = []
    sources: list[dict] = []
    masked = 0
    expanded = 0

    for i, chunk in enumerate(chunks, start=1):
        # ── small-to-big 回填：命中子块后，把父块完整文本交给 LLM ──────────
        # 检索打分用的是子块（短、准），但生成答案需要更完整的上下文，
        # 所以这里换成父块文本。父块不存在就退回子块，不影响功能。
        body = chunk.text
        if use_parent and chunk.parent_text:
            body = chunk.parent_text
            expanded += 1

        header = (
            f"[Source {i}] {chunk.filename}, page {chunk.page_number} "
            f"(relevance: {chunk.score:.2f})"
        )

        # 检索到的文本是数据，绝不能当成指令 —— 注入清洗
        safe_text, was_masked = sanitize_document_context(body)
        if was_masked:
            masked += 1

        parts.append(f"{header}\n{safe_text}")

        sources.append({
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            # 引用预览用子块原文（更贴切用户看到的高亮），不用父块
            "text_snippet": chunk.text[:snippet_chars],
            "score": round(chunk.score, 4),
            "expanded_to_parent": bool(use_parent and chunk.parent_text),
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

    return BuiltContext(
        context="\n\n---\n\n".join(parts),
        sources=sources,
        masked_count=masked,
        expanded_count=expanded,
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
