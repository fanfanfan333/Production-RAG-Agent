"""
意图路由确定性规则单元测试（问题1：闲聊分支必须可达）.

覆盖 intent_rules.deterministic_route：
  * 纯打招呼 / 身份询问 → general_chat（且不依赖 LLM，路由超时也可用）；
  * 列出文档 → list_documents；
  * 跨文档关联 → doc_relations；
  * 整库总结 / 点名文档总结 → document_summary；
  * **关键回归**：带知识库语义的句子即使以"你好"开头，也不能被判成闲聊；
  * 其余情况返回 None，交给 LLM 路由。

零第三方依赖，CI 直接 python 运行即可。
通过 importlib 直接加载模块文件，避免触发
backend/app/services/routers/__init__.py 及 langchain 重依赖导入。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

_MOD_PATH = (
    Path(__file__).resolve().parent.parent
    / "app" / "services" / "routers" / "intent_rules.py"
)
_spec = importlib.util.spec_from_file_location("app.services.routers.intent_rules", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["app.services.routers.intent_rules"] = _mod
_spec.loader.exec_module(_mod)

deterministic_route = _mod.deterministic_route
looks_knowledge_seeking = _mod.looks_knowledge_seeking
VALID_INTENTS = _mod.VALID_INTENTS
DEFAULT_INTENT = _mod.DEFAULT_INTENT


# ── 1. 纯闲聊必须命中 general_chat（问题1 核心）────────────────────────────

def test_greeting_routes_to_general_chat():
    for q in ("你好", "您好", "hi", "Hello", "嗨", "早上好", "晚安", "在吗", "谢谢"):
        assert deterministic_route(q) == "general_chat", q


def test_identity_question_routes_to_general_chat():
    """截图1 里的"你是谁"必须走闲聊，而不是被塞进 RAG 检索。"""
    for q in ("你是谁", "你是谁？", "你叫什么名字", "你是什么模型", "介绍一下你自己"):
        assert deterministic_route(q) == "general_chat", q


def test_capability_question_routes_to_general_chat():
    for q in ("你能做什么", "你会什么", "你有什么功能"):
        assert deterministic_route(q) == "general_chat", q


def test_greeting_with_trailing_punctuation():
    assert deterministic_route("你好！") == "general_chat"
    assert deterministic_route("你好啊。") is None  # "你好啊"不在词表，交给 LLM


# ── 2. 关键回归：不能把知识库问题误判成闲聊 ─────────────────────────────────

def test_greeting_plus_real_question_is_not_chitchat():
    """以"你好"开头但带真实诉求 → 绝不能判成闲聊."""
    for q in (
        "你好，帮我看下合同第三条",
        "你好，知识库里有哪些文档",
        "你好，总结一下这份报告",
        "你好 请问营收增长率是多少",
    ):
        assert deterministic_route(q) != "general_chat", q


def test_long_query_never_chitchat():
    """超长提问直接跳过闲聊判定（长度闸门）."""
    long_q = "你好" + "啊" * 50
    assert deterministic_route(long_q) is None


def test_empty_query_is_none():
    assert deterministic_route("") is None
    assert deterministic_route("   ") is None


# ── 3. 列表 / 关联 / 整库总结 ────────────────────────────────────────────────

def test_document_list():
    assert deterministic_route("知识库里有哪些文档") == "list_documents"
    assert deterministic_route("列出所有文件") == "list_documents"


def test_doc_relations():
    assert deterministic_route("这些文档之间有什么关联") == "doc_relations"
    assert deterministic_route("分析知识库里各文档的共同点") == "doc_relations"


def test_whole_library_summary():
    """整库总结必须走 document_summary。

    回归背景：这类请求此前被 LLM 路由判成 knowledge_qa → 走检索链 →
    只把召回分最高的那一份文档的两三个片段拼成"总结"，用户看到的就是
    "只总结了内容最多的那份文档"。
    """
    for q in (
        "总结所有文档",
        "把所有文档总结一下",
        "总结一下全部文档",
        "帮我概括所有文件",
        "总结整个知识库",
        "总结一下知识库",
        "概览一下库里的文档",
    ):
        assert deterministic_route(q) == "document_summary", q


def test_named_document_summary():
    """点名带扩展名的文件 → 一定是"总结这份文档"。"""
    for q in (
        "总结《研发部-2024年度技术方案-nqkxx.docx》",
        "帮我总结研发部-2024年度技术方案-nqkxx.docx",
        "概括一下 2024年报.pdf",
    ):
        assert deterministic_route(q) == "document_summary", q


def test_summary_rule_does_not_overreach():
    """范围词缺席的句子不能被抢判 —— 这些交给 LLM 路由。"""
    for q in (
        "这份文档主要讲了什么",     # 已有断言：必须留给 LLM
        "总结一下第三章",
        "总结一下风险点",           # 范围词不指向文档集合
        "总结一下全部内容",         # 没有文档类名词
    ):
        assert deterministic_route(q) is None, q


# ── 4. 其余交给 LLM ─────────────────────────────────────────────────────────

def test_ambiguous_query_defers_to_llm():
    for q in ("这份文档主要讲了什么", "合同里约定的付款期限是多久", "总结一下第三章"):
        assert deterministic_route(q) is None, q


# ── 5. 知识提问守卫（LLM 判成闲聊时的纠偏依据）────────────────────────────

def test_knowledge_seeking_catches_domain_questions():
    """回归：这些问句曾被 LLM 路由判成 general_chat，导致整条检索链被绕过。

    判别口径是"回答该不该来自资料"，不是"听起来像不像通用知识"。
    """
    for q in (
        "反幻觉机制是怎么工作的？",
        "检索流程有哪些步骤？",
        "这个参数怎么配置？",
        "证据门控是什么？",
        "系统支持多租户吗？",
        "为什么引用会校验失败？",
        "RAG 和微调的区别是什么？",
        "介绍下文档切分策略",
        "how does the reranker work?",
    ):
        assert looks_knowledge_seeking(q), q


def test_knowledge_seeking_ignores_chitchat():
    """打招呼 / 身份询问 / 创作指令不能被纠偏成知识问答。"""
    for q in (
        "你好",
        "你是谁？",
        "你叫什么名字",
        "你能做什么",
        "谢谢",
        "帮我写一首关于春天的诗",
        "用 Python 写个快速排序",
        "",
        "   ",
    ):
        assert not looks_knowledge_seeking(q), q


def test_knowledge_seeking_ignores_offtopic_smalltalk():
    """带疑问语气的纯生活闲聊仍留给 general_chat（短句才豁免）。"""
    for q in ("今天天气怎么样？", "现在几点了", "今天星期几？", "讲个笑话给我听"):
        assert not looks_knowledge_seeking(q), q


def test_knowledge_seeking_long_query_still_checked():
    """长度闸门只用于"豁免闲聊话题"，不能把长问句一律放过。"""
    q = "请说明" + "很长的前缀" * 20 + "这个机制是怎么工作的？"
    assert len(q) > _mod._OFFTOPIC_MAX_LEN
    assert looks_knowledge_seeking(q), q


# ── 6. 常量一致性 ───────────────────────────────────────────────────────────

def test_default_intent_is_valid():
    assert DEFAULT_INTENT in VALID_INTENTS
    assert "general_chat" in VALID_INTENTS


def test_all_results_are_valid_intents():
    samples = [
        "你好", "你是谁", "知识库里有哪些文档", "这些文档之间有什么关联",
        "这份文档主要讲了什么", "",
    ]
    for q in samples:
        result = deterministic_route(q)
        assert result is None or result in VALID_INTENTS, (q, result)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
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
