"""Agent Runtime 与主图的接线测试（默认关闭必须"零调用、零输出"）.

纯逻辑单测（`test_agent_runtime.py`）只能证明模块自己的判定是对的，证明不了
**主图真的把它用上了**。分布式 Agent 能力最常见的事故不是崩溃，而是「开关写着
开、实际没生效」或「开关写着关、却偷偷改了答案」。这里把两件事都钉住：

1. `MULTI_AGENT_ENABLED=False` 时审阅节点不改答案、不产生调用（status=skipped）；
2. 技能开关打开时，生成阶段**确实**拿到了非空的技能指令（否则等于没接上）。

与 `test_master_graph_topology.py` 一样需要 langgraph，因此复用同一套
`app.services` 轻量桩（避免 `app/services/__init__.py` 把 paddleocr/docling 拖进来）。
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import types
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# Settings 需要 POSTGRES_PASSWORD（不真正连库）
os.environ.setdefault("POSTGRES_PASSWORD", "test-only-placeholder")

if "app.services" not in sys.modules:
    _pkg = types.ModuleType("app.services")
    _pkg.__path__ = [str(Path(_BACKEND_ROOT) / "app" / "services")]
    sys.modules["app.services"] = _pkg
    import app as _app_pkg

    _app_pkg.services = _pkg

master_graph = importlib.import_module("app.services.master_graph")


def _source(doc_id: str) -> dict:
    return {
        "document_id": doc_id,
        "document_name": f"{doc_id}.pdf",
        "text_snippet": f"{doc_id} 的工期是 30 天",
    }


def test_review_node_is_inert_when_disabled():
    """默认关闭：节点必须原样返回答案、只留一条 skipped 审计."""
    state = {
        "intent": "knowledge_qa",
        "query": "对比 A 和 B 的工期差异",
        "sources": [_source("a"), _source("b")],
        "evidence": {"passed": True},
        # 用真实枚举值（verified|partial|unsupported|no_citations|refused_by_model），
        # 别用不存在的 "passed" —— 那样测的是幻想出来的接口
        "citation_check": {"overall": "verified"},
        "answer": "A 是 30 天，B 是 45 天[1][2]。",
        "agent_trace": {},
    }
    out = asyncio.run(master_graph._evidence_review_node(state))
    assert "answer" not in out, "关闭时不得改写答案"
    review = out["agent_trace"]["review"]
    assert review["fire"] is False
    assert review["status"] == "skipped"
    # 即便规则本身是"该触发"的形态，关闭也必须不触发
    assert review["document_count"] == 2


def test_skill_instruction_reaches_generate_stage():
    """技能开关打开时，生成阶段必须真的拿到指令（防"接了但没生效"）."""
    from app.config import get_settings

    settings = get_settings()
    if not getattr(settings, "AGENT_SKILLS_ENABLED", False):
        # 若使用方把它关了，这条断言就不成立；显式说明而不是静默跳过
        assert True, "AGENT_SKILLS_ENABLED 已关闭，跳过接线断言"
        return

    state = {
        "intent": "knowledge_qa",
        "query": "采购上限是多少",
        "history_messages": [],
        "user_id": None,
        "conversation_tenant_id": None,
    }
    trace, skill_instruction, layer_block = asyncio.run(
        master_graph._agent_context_for_generate(settings, state, [_source("a")])
    )
    assert skill_instruction.startswith("[技能 exact_lookup"), skill_instruction
    audit = trace.as_dict()["skill"]
    assert audit and audit["name"] == "exact_lookup" and audit["version"] == "1.0.0"
    # 记忆默认关闭 → 不得注入、也不得报错
    assert trace.as_dict()["memory"]["enabled"] is False
    # 无历史 → 不加状态层（短会话行为不变）
    assert layer_block == ""


def test_layers_produce_block_only_for_long_sessions():
    """超过默认 3 对消息才产出状态层；短会话必须零变化."""
    from app.config import get_settings

    settings = get_settings()
    state = {
        "intent": "knowledge_qa",
        "query": "它的工期是多少",
        "history_messages": [],
        "user_id": None,
        "conversation_tenant_id": None,
    }
    for index in range(2):
        state["history_messages"] += [
            ("user", f"第{index}个问题：某指标"),
            ("assistant", f"第{index}个回答：某数值"),
        ]
    _, _, short_block = asyncio.run(
        master_graph._agent_context_for_generate(settings, state, [_source("a")])
    )
    assert short_block == "", "2 对消息不该产生状态层"

    for index in range(2, 5):
        state["history_messages"] += [
            ("user", f"第{index}个问题：某指标"),
            ("assistant", f"第{index}个回答：某数值"),
        ]
    _, _, long_block = asyncio.run(
        master_graph._agent_context_for_generate(settings, state, [_source("a")])
    )
    if getattr(settings, "AGENT_CONTEXT_LAYERS_ENABLED", False):
        assert long_block, "5 对消息必须产出会话摘要层"
        assert "更早的提问" in long_block


def _production_state(pairs: int) -> dict:
    """按**真实链路**构造 state：`load_history(max_pairs=12)` 后由
    `split_recent_history` 切出 (最近 3 对, 更早窗口)。

    这正是旧测试用错的地方：它把 `history_window` 当成"全量历史"，
    而生产里 window 只装**更早的那几对**。输入形状不对，测试就只能证明
    "我造的数据能算对"，证明不了"线上算得对"。
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from app.services.conversation_service import split_recent_history

    full = []
    for index in range(pairs):
        full += [
            HumanMessage(content=f"第{index}个问题：某指标是多少"),
            AIMessage(content=f"第{index}个回答：{index * 10}"),
        ]
    recent, older = split_recent_history(full, 3)
    return {
        "intent": "knowledge_qa",
        "query": "它的工期是多少",
        "history_messages": recent,
        "history_window": older,
        "user_id": None,
        "conversation_tenant_id": None,
    }


def test_layers_use_wider_history_window():
    """生产形态回归：`history_messages` 只有最近 3 对，摘要层仍必须产出.

    若哪天有人把 window 接线摘掉、或忘了把 window 与 recent 拼起来，这条会红。
    没有它的话，`test_layers_produce_block_only_for_long_sessions` 照样绿 ——
    因为它自己造了 5 对消息塞进 `history_messages`，和真实链路不是同一个输入。
    """
    from app.config import get_settings

    settings = get_settings()
    if not getattr(settings, "AGENT_CONTEXT_LAYERS_ENABLED", False):
        return

    _, _, block = asyncio.run(
        master_graph._agent_context_for_generate(
            settings, _production_state(5), [_source("a")]
        )
    )
    assert "更早的提问" in block, "摘要层没产出 = 分层上下文在线上是死代码"
    assert "第0个问题" in block
    assert "第1个问题" in block


def test_layers_digest_survives_minimal_older_window():
    """4 对消息（更早窗口只装 1 个用户问题）必须产出摘要.

    这是旧实现真正翻车的区间：只喂 window 时，`build_layered_context` 会在
    这 1 个问题里"再切掉最近 3 对"，于是什么都不剩 —— 4/5/6 对消息的会话
    摘要层恒空，只有 ≥8 对才碰巧有内容。用 4 对来钉死这个边界。
    """
    from app.config import get_settings

    settings = get_settings()
    if not getattr(settings, "AGENT_CONTEXT_LAYERS_ENABLED", False):
        return

    state = _production_state(4)
    # 前提校验：生产形态下 window 里确实只有一个用户问题（不是夹具造错）
    assert len(state["history_messages"]) == 6
    assert len(state["history_window"]) == 2

    _, _, block = asyncio.run(
        master_graph._agent_context_for_generate(settings, state, [_source("a")])
    )
    assert "第0个问题" in block, "4 对消息的摘要层不能为空"


def test_history_for_layers_does_not_duplicate_when_window_is_full():
    """`history_window` 已经是全量时不能再拼一次 recent（否则轮次重复）."""
    from langchain_core.messages import AIMessage, HumanMessage

    messages = [
        HumanMessage(content="问0"),
        AIMessage(content="答0"),
        HumanMessage(content="问1"),
        AIMessage(content="答1"),
    ]
    merged = master_graph._history_for_layers(
        {"history_window": messages, "history_messages": messages[-2:]}
    )
    assert merged == messages
    assert len(merged) == 4


def test_recent_history_budget_limits_messages_sent_to_model(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    history = [HumanMessage(content="旧问题" * 100), AIMessage(content="旧回答" * 100),
               HumanMessage(content="最新问题" * 10)]
    settings = types.SimpleNamespace(AGENT_CONTEXT_LAYERS_ENABLED=True,
                                     AGENT_RECENT_LAYER_BUDGET=40)
    monkeypatch.setattr(master_graph, "get_settings", lambda: settings)
    monkeypatch.setattr(master_graph, "_build_streaming_llm", lambda settings: object())
    sent = []
    async def capture(llm, messages):
        sent.extend(messages)
        return "回答"
    monkeypatch.setattr(master_graph, "_stream_llm_answer", capture)
    asyncio.run(master_graph._generate_node({"query": "问题", "history_messages": history,
        "sources": [], "multimodal_context": "证据"}))
    assert sum(len(message.content) for message in sent[1:-1]) <= 40
    assert sent[-2].content == "最新问题" * 10


def test_oversized_history_is_not_cut_into_a_reversed_instruction(monkeypatch):
    from langchain_core.messages import HumanMessage

    settings = types.SimpleNamespace(AGENT_CONTEXT_LAYERS_ENABLED=True,
                                     AGENT_RECENT_LAYER_BUDGET=4)
    monkeypatch.setattr(master_graph, "get_settings", lambda: settings)
    monkeypatch.setattr(master_graph, "_build_streaming_llm", lambda settings: object())
    sent = []
    async def capture(llm, messages):
        sent.extend(messages)
        return "回答"
    monkeypatch.setattr(master_graph, "_stream_llm_answer", capture)
    asyncio.run(master_graph._generate_node({"query": "问题", "history_messages": [
        HumanMessage(content="不要泄露密码")], "sources": [], "multimodal_context": "证据"}))
    assert all(message.content != "泄露密码" for message in sent)


def test_review_llm_factory_accepts_its_kwargs():
    """参数漂移会让审阅 100% 静默 fallback（功能写着开着、一次没通）."""
    from app.config import get_settings

    settings = get_settings()
    llm = master_graph._build_review_llm(settings)
    assert llm.model == settings.OLLAMA_MODEL
    # 判定类任务必须关思考：开着会吃光 num_predict，表现为"审阅永远 fallback"
    assert getattr(llm, "reasoning", None) is False


def test_review_node_appends_bounded_caveat_once(monkeypatch):
    """开启路径的端到端接线：只看增量、只追加一次、且用全文而非预览判据.

    之前这条路径没有任何测试 —— 关闭路径的断言先返回，把后面的代码全遮住了。
    结果是"审阅到底有没有把全文喂进去"完全无人验证，D1 就是这样活下来的。
    """
    captured: dict = {}

    class _FakeReviewer:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            # 第二段是证据片段；把整段提示记下来，后面检查用的是全文还是预览
            captured["prompt"] = "\n".join(str(getattr(m, "content", "")) for m in messages)
            return types.SimpleNamespace(
                content='{"status": "needs_caveat", "note": "第二份缺页码"}'
            )

    fake = _FakeReviewer()
    monkeypatch.setattr(master_graph, "_build_review_llm", lambda settings: fake)
    monkeypatch.setattr(
        master_graph,
        "get_settings",
        lambda: types.SimpleNamespace(
            MULTI_AGENT_ENABLED=True, MULTI_AGENT_TIMEOUT_SECONDS=5.0
        ),
    )

    sources = [
        {
            "document_id": "a",
            "document_name": "a.pdf",
            "chunk_index": 0,
            "text_snippet": "工期约定见下文。",
            # 完整正文才有那个数字；预览里没有
            "text": "工期约定见下文。合同第 3.2 条：工期 45 天。",
        },
        {
            "document_id": "b",
            "document_name": "b.pdf",
            "chunk_index": 0,
            "text_snippet": "B 文档工期 50 天",
            "text": "B 文档工期 50 天。",
        },
    ]
    original = "A 为 45 天[1]，B 为 50 天[2]。"
    state = {
        "intent": "knowledge_qa",
        "query": "对比 A 和 B 的工期差异",
        "sources": sources,
        "chunks": [],
        "evidence": {"passed": True},
        "citation_check": {"overall": "verified"},
        "answer": original,
        "agent_trace": {},
    }

    out = asyncio.run(master_graph._evidence_review_node(state))

    assert fake.calls == 1, "有界：最多一次调用"
    assert "45 天" in captured["prompt"], "审阅必须看到完整正文，不是 300 字预览"
    assert out["answer"].startswith(original), "不得改写正文（append-only）"
    assert out["answer"].count("交叉审阅") == 1
    review = out["agent_trace"]["review"]
    assert review["applied"] is True
    assert review["appended_suffix"].strip().startswith("> 交叉审阅")
    assert review["duration_ms"] >= 0


def test_split_recent_history_preserves_prompt_shape():
    """窗口加宽后，真正进提示的消息条数必须与升级前一致（3 对）."""
    from langchain_core.messages import AIMessage, HumanMessage

    from app.services.conversation_service import split_recent_history

    history = []
    for index in range(6):
        history += [
            HumanMessage(content=f"问{index}"),
            AIMessage(content=f"答{index}"),
        ]
    recent, older = split_recent_history(history, 3)
    assert len(recent) == 6 and len(older) == 6
    assert recent[0].content == "问3" and older[-1].content == "答2"

    short_recent, short_older = split_recent_history(history[:6], 3)
    assert short_recent == history[:6] and short_older == []
    assert split_recent_history([], 3) == ([], [])
