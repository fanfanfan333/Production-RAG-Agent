"""
Document Summary 节点（架构图 Document Summary 分支）.

与已有的 ``doc_relations`` 的区别
────────────────────────────────
两者都读全库文档摘要，但回答的焦点不同：

- doc_relations（已有）→ "这些文档之间有什么关联"
  输出：每个文档一节，重点是它和其他文档的关系
- document_summary（本模块）→ "这些文档 / 这份文档讲了什么"
  输出：每个文档一节，重点是内容本身的提炼（主题、要点、结论）

复用而非重写：摘要采样直接用 relation_service.collect_document_digests，
它已经处理好了"取哪些 chunk、截多少字符、权限过滤、上限控制"。
本模块只负责不同的 prompt 和不同的组织方式。
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.services.relation_service import (
    build_digest_context,
    collect_document_digests,
    digest_sources,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


_SUMMARY_SYSTEM_TEMPLATE = """\
你是一个企业私有知识库分析助手，正在对知识库中的文档做内容总结。

你将获得知识库中的文档；每个条目包含文件名，以及从其索引内容中采样得到的内容摘要。

任务 —— 对每个文档做内容提炼，严格按文档逐个组织回答：

1. 为每个文档单独设一节，以标题"### <文件名>"开头，内容包括：
   a. 核心主题 —— 一句话说明这个文档是关于什么的。
   b. 要点提炼 —— 用 3–5 个要点概括文档的关键信息、结论或数据。
   c. 适用对象 —— 简短说明什么样的人/场景需要读这份文档（可省略）。

2. 最后以"### 总体概览"一节收尾：用 3–5 句话概括整个知识库覆盖了哪些
   主题领域，以及这些文档合在一起构成了什么图景。

规则：
- 只基于给出的内容采样做总结，严禁编造采样中不存在的事实、数字或结论。
- 采样只是文档的部分内容，不要声称你的总结覆盖了文档的全部细节。
- 提到文档时必须使用其确切的文件名。
- 使用与用户提问相同的语言回答（默认中文）。
- 直接从第一个文档小节开始，不要任何开场白或客套。

── 知识库中的文档 ───────────────────────────────────────────────────────────────
{context}
──────────────────────────────────────────────────────────────────────────────
"""


async def collect_summary_digests(owner_id: str | None = None) -> list[dict]:
    """
    拉取全库文档摘要（供 Document Summary 分支使用）.

    复用 relation_service 的采样逻辑 —— 采样策略、权限过滤、上限控制
    全部沿用，不重复实现。
    """
    settings = get_settings()
    digests = await collect_document_digests(
        max_documents=settings.RELATION_MAX_DOCUMENTS,
        chunks_per_doc=settings.RELATION_CHUNKS_PER_DOC,
        digest_chars=min(
            settings.RELATION_DIGEST_CHARS,
            settings.DOC_SUMMARY_MAX_CHARS_PER_DOC,
        ),
        owner_id=owner_id,
    )
    logger.info("document_summary: collected %d document digests", len(digests))
    return digest_sources(digests)


def build_summary_messages(
    query: str,
    digests: list[dict],
    history_messages: list[BaseMessage],
) -> list[BaseMessage]:
    """组装 Document Summary 分支的 messages."""
    if digests:
        context = build_digest_context(digests)
    else:
        context = "The knowledge base currently contains no indexed documents."

    focused_query = (
        f"[指令：总结知识库中这些文档的内容，"
        f"并按要求的格式以文档为单位组织回答。]\n\n"
        f"{query}"
    )
    return [
        SystemMessage(content=_SUMMARY_SYSTEM_TEMPLATE.format(context=context)),
        *history_messages,
        HumanMessage(content=focused_query),
    ]


def build_summary_llm():
    """构造文档总结用的 LLM（temperature 略高，允许一定概括性措辞）."""
    settings = get_settings()
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.2,
        streaming=True,
        reasoning=True,
    )
