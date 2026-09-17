"""Document Summary 流式节标题 + 旁路通道单元测试.

回归背景（实测 BUG）：
    map-reduce 总结里 `### <文件名>` 这类节标题不是 LLM token，原本靠
    `get_stream_writer()` 下发。而 langgraph 1.0.1 的
    `astream_events(version="v2")` 会**丢弃** StreamWriter 的 payload ——
    于是前端只看到各文档的正文、看不到任何小节标题，用户的第一观感就是
    "总结漏了文档 / 不是我要的那几份"。

本测试用假 LLM 直接驱动 `_summarize_node`，断言：
  * 每份文档都有一次 `### <文件名>` 节标题下发，且顺序与输入一致；
  * 多份文档时有 `### 总体概览`；
  * 返回值（落库的权威答案）里同样一份不缺 —— 展示与落库口径一致。

运行（容器内）：python /app/tests/test_stream_channel.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from app.services import master_graph as mg  # noqa: E402
from app.services.stream_channel import bind_sink, emit_text, reset_sink  # noqa: E402


def test_emit_text_requires_binding():
    """无绑定时静默丢弃并返回 False（节点因此可脱离流式上下文被单测调用）."""
    assert emit_text("x") is False


def test_bind_and_reset_sink():
    buf: list[str] = []
    token = bind_sink(buf)
    try:
        assert emit_text("A") is True
        assert emit_text("") is False      # 空串不入队
        assert emit_text("B") is True
        assert buf == ["A", "B"]
    finally:
        reset_sink(token)
    # 还原后必须重新变回"无绑定"
    assert emit_text("C") is False
    assert buf == ["A", "B"]


async def _run_summarize(digests: list[dict], scope: dict | None = None):
    """用假 LLM 驱动一次 _summarize_node，返回 (下发文本列表, 返回答案)."""
    calls: list[str] = []

    class _FakeLLM:  # 占位，_stream_llm_answer 已被替换
        pass

    async def _fake_stream_llm(llm, messages):
        calls.append("call")
        return "核心内容：要点一；要点二。"

    original_llm, original_stream = mg.build_summary_llm, mg._stream_llm_answer
    mg.build_summary_llm = lambda **_kw: _FakeLLM()
    mg._stream_llm_answer = _fake_stream_llm
    buf: list[str] = []
    token = bind_sink(buf)
    try:
        state = {
            "query": "总结所有文档",
            "digests": digests,
            "summary_scope": scope or {"targeted": False, "total_accessible": len(digests)},
            "history_messages": [],
        }
        out = await mg._summarize_node(state)
    finally:
        reset_sink(token)
        mg.build_summary_llm, mg._stream_llm_answer = original_llm, original_stream
    return buf, out.get("answer", ""), calls


def test_every_document_gets_a_streamed_section():
    digests = [
        {"filename": "A方案.docx", "text": "aaa"},
        {"filename": "B方案.docx", "text": "bbb"},
        {"filename": "C方案.docx", "text": "ccc"},
    ]
    buf, answer, calls = asyncio.run(_run_summarize(digests))

    streamed = "".join(buf)
    for d in digests:
        header = f"### {d['filename']}"
        assert header in streamed, f"流式输出缺节标题: {header} / got={streamed!r}"
        assert header in answer, f"落库答案缺节标题: {header}"
    assert "### 总体概览" in streamed, streamed
    assert "### 总体概览" in answer, answer
    # 每份文档一次 + 概览一次
    assert len(calls) == len(digests) + 1, calls

    # 顺序：节标题必须与输入顺序一致（避免"漏了中间那份"）
    positions = [streamed.index(f"### {d['filename']}") for d in digests]
    assert positions == sorted(positions), positions


def test_single_document_has_no_overview():
    """只有一份文档时不加"总体概览"（否则是废话一层）。"""
    buf, answer, calls = asyncio.run(_run_summarize([{"filename": "only.pdf", "text": "x"}]))
    assert "### only.pdf" in "".join(buf)
    assert "### 总体概览" not in answer
    assert len(calls) == 1, calls


def test_failed_document_still_emits_section():
    """单份文档 LLM 失败也要留下小节 —— 绝不静默跳过（这正是"漏文档"的成因）."""
    digests = [{"filename": "ok.docx", "text": "x"}, {"filename": "bad.docx", "text": "y"}]

    class _FakeLLM:
        pass

    async def _flaky(llm, messages):
        # 第一次成功，第二次抛错
        if getattr(_flaky, "called", False):
            raise RuntimeError("boom")
        _flaky.called = True
        return "正常内容"

    original_llm, original_stream = mg.build_summary_llm, mg._stream_llm_answer
    mg.build_summary_llm = lambda **_kw: _FakeLLM()
    mg._stream_llm_answer = _flaky
    buf: list[str] = []
    token = bind_sink(buf)
    try:
        out = asyncio.run(mg._summarize_node({
            "query": "总结所有文档",
            "digests": digests,
            "summary_scope": {"targeted": False, "total_accessible": 2},
            "history_messages": [],
        }))
    finally:
        reset_sink(token)
        mg.build_summary_llm, mg._stream_llm_answer = original_llm, original_stream
        _flaky.called = False

    answer = out.get("answer", "")
    assert "### bad.docx" in answer, answer
    assert "### bad.docx" in "".join(buf), "失败小节也必须在流式里出现"
    # 概览调用也会失败（_flaky 第二次抛错后一直抛），但不能阻塞交付
    assert "### ok.docx" in answer, answer


def test_truncation_is_disclosed():
    """被 DOC_SUMMARY_MAX_DOCUMENTS 截断时必须明示，不能让用户以为已覆盖全库."""
    digests = [{"filename": "a.docx", "text": "x"}]
    scope = {"targeted": False, "total_accessible": 42}
    _, answer, _ = asyncio.run(_run_summarize(digests, scope))
    assert "42" in answer and "1" in answer, answer
    assert "注：" in answer, answer


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
