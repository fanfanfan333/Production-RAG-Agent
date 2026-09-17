"""LLM token 流转发判定（零第三方依赖，便于单测）.

背景
----
``master_graph`` 通过 ``graph.astream_events(..., version="v2")`` 拿到
LangGraph 的事件流，并把 ``on_chat_model_stream`` 事件转发成 SSE 的
``chunk`` / ``thinking_delta``。

问题在于：主图谱里有 **两类** 节点都会触发 ``on_chat_model_stream``——

* 生成类节点（generate / chat / summarize / analyze_relations）：
  输出就是给用户看的答案正文，**必须**转发；
* 决策类节点（route / rewrite / grade）：
  输出是内部中间数据（意图标签、改写后的查询、证据评分 JSON），
  **必须**丢弃。

更隐蔽的是，决策类节点用的是 ``llm.ainvoke(...)``。LangChain 的
``astream_events`` 会把 ``ainvoke`` 包装成 **一个** ``on_chat_model_stream``
chunk，内容就是完整的 JSON 字符串。所以一旦不加区分地转发，用户就会在
回答里看到 ``{"rewritten": ..., "variants": [...]}`` 这样的东西——
这不是模型"说胡话"，是内部数据漏出来了。

本模块把「这个 token 该不该转发给前端」抽成纯函数，不依赖 LangGraph /
LangChain，因此可以在没有后端依赖的环境里直接单测。
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "STREAMING_LLM_NODES",
    "INTERNAL_LLM_NODES",
    "resolve_llm_node",
    "should_stream_token",
]


# 会向用户输出答案正文的「生成类节点」。
STREAMING_LLM_NODES = frozenset({
    "generate",            # Knowledge QA 答案生成
    "chat",                # General Chat 闲聊生成
    "summarize",           # Document Summary 总结生成
    "analyze_relations",   # 跨文档关联分析生成
})

# 产出内部决策数据的「决策类节点」，其 LLM 输出一律不得外泄。
INTERNAL_LLM_NODES = frozenset({
    "route",     # 意图识别 → {"intent": "..."}
    "rewrite",   # 查询改写 → {"rewritten": ..., "variants": [...]}
    "grade",     # 证据评分 → 评分 JSON
})


def resolve_llm_node(
    event: Mapping[str, Any],
    streaming_node: str | None = None,
    internal_node: str | None = None,
) -> str:
    """判定一个 ``on_chat_model_stream`` 事件属于哪个节点.

    判定顺序（越靠前越可信）：

    1. ``event["metadata"]["langgraph_node"]`` —— LangGraph 为节点内事件
       注入的归属标记，最准；但部分版本不注入。
    2. ``streaming_node`` —— 由 ``on_chain_start`` / ``on_chain_end``
       追踪到的当前所在生成类节点。
    3. ``internal_node`` —— 同上，但针对决策类节点。

    三者都拿不到时返回 ``""``（归属不明）。
    """
    metadata = event.get("metadata") or {}
    node = metadata.get("langgraph_node")
    if node:
        return str(node)
    return streaming_node or internal_node or ""


def should_stream_token(
    event: Mapping[str, Any],
    streaming_node: str | None = None,
    internal_node: str | None = None,
) -> bool:
    """该 ``on_chat_model_stream`` 事件的 token 是否应转发给前端.

    策略：

    * 归属到决策类节点  → 丢弃（这是本次修复的核心）；
    * 归属到生成类节点  → 转发；
    * 归属不明          → 转发。宁可放行也不吞掉正常答案——元数据缺失
      只会导致旧的"泄漏"行为，不会造成更严重的"回答变空"。
    """
    node = resolve_llm_node(event, streaming_node, internal_node)
    if node in INTERNAL_LLM_NODES:
        return False
    # 生成类节点，以及归属不明 / 未知节点：放行。
    return True
