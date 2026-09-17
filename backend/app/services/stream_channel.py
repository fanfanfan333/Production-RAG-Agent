"""节点 → 外层流式通道：把**非 LLM token** 的确定性文本送进 SSE.

为什么需要它
────────────
LangGraph 的 ``get_stream_writer()`` 在 ``astream_events(version="v2")`` 下
**payload 会被直接丢弃**（langgraph 1.0.1 实测：只有 ``astream(stream_mode=
"custom")`` 能收到，而 ``astream_events`` 只把节点返回值包成
``on_chain_stream``）。master graph 的整个事件循环建立在 ``astream_events``
之上，为了一个节标题去重写事件循环不划算。

于是改用一条**显式旁路**：外层把一份列表注册成"当前请求的旁路缓冲"
（ContextVar，天然按 asyncio task 隔离，并发请求互不串扰），节点内调用
:func:`emit_text` 追加文本；外层在事件循环里每次拿到事件就把缓冲冲刷成
``chunk`` 下发。

使用方
──────
Document Summary 的 map-reduce：``### <文件名>`` / ``### 总体概览`` 这类
**由代码确定性生成**的节标题、兜底说明、截断提示，都不是模型 token，必须
走这条通道，否则前端只看得到正文、看不到小节划分（表现为"总结漏了文档"）。

顺序保证
────────
节点在调用 LLM **之前**先 ``emit_text(header)``，而外层在每个事件（包括
LLM 的首个 token）到达时先冲刷缓冲再处理事件 —— 因此节标题一定排在该节
正文之前。

无绑定时的行为
──────────────
``emit_text`` 返回 ``False`` 并丢弃。节点因此可以脱离流式上下文被单测直接
调用，不需要任何 mock。
"""

from __future__ import annotations

import contextvars

__all__ = ["bind_sink", "emit_text", "reset_sink"]

# 当前请求的旁路缓冲（list[str]）。None = 没有处在流式请求中。
_SINK: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "rag_stream_sink", default=None
)


def bind_sink(buffer: list[str]) -> contextvars.Token:
    """把 *buffer* 注册为当前上下文的旁路缓冲，返回可用于还原的 token."""
    return _SINK.set(buffer)


def reset_sink(token: contextvars.Token) -> None:
    """还原 :func:`bind_sink` 的设置（务必放在 ``finally`` 里，否则同 task
    后续的旁路文本会写进一份已经没人读的缓冲）。"""
    _SINK.reset(token)


def emit_text(text: str) -> bool:
    """把确定性文本交给外层流（无绑定则丢弃并返回 ``False``）."""
    if not text:
        return False
    buffer = _SINK.get()
    if buffer is None:
        return False
    buffer.append(text)
    return True
