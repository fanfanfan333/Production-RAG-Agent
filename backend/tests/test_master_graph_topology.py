"""
Master Graph 拓扑回归测试（问题2 + 问题4：对照架构图锁定编排结构）.

把两张架构图的约定固化成可执行断言，防止后续改动悄悄破坏框架：

图2（主图谱 Master Graph —— 统一编排入口）：
    Query Router（LLM 分类 5 意图）
      ├─ summary           支路   → summary_digests → summarize
      ├─ knowledge_qa      主支路 → rewrite → retrieve → grade
      ├─ chat              闲聊   → chat
      ├─ doc_relations     关联分析 → collect_digests → analyze_relations
      └─ list_documents    文档列表 →（API 层 DB 直读，图内直达收尾）
    knowledge_qa 内部：
      rewrite（指代消解+多查询）→ retrieve（BM25+向量 ANN+RRF+精排）
      → Retrieval Grader → good → multimodal_context → Evidence Gate
                        → bad → rewrite（最多 N 次）→ 仍 bad → 礼貌拒答

图4（LangGraph 子图逻辑）：
    全部生成路径 → Generate → Citation Verifier → Output Guard（Citation Check）→ END
    拒答文本本身合规，不进 output_guard。

细粒度引用 + Evidence Gate + Citation Verifier（本次新增）：
    - Evidence Gate 是 Grader 之后的**确定性 fail-closed 兜底**，
      形态不达标直接拒答，不再交给模型硬答；
    - Citation Verifier 做五项校验（存在/位置/支持/数字/日期），
      插在 Generate 与 Output Guard 之间。

需要 langgraph / langchain 依赖（编译真实图谱，而不是看源码）。
CI 可 `python tests/test_master_graph_topology.py` 直接运行。
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# Settings 需要 POSTGRES_PASSWORD；测试环境给个占位值（不真正连库）
os.environ.setdefault("POSTGRES_PASSWORD", "test-placeholder")

# ── 轻量化导入 ────────────────────────────────────────────────────────────────
# 本测试只关心**图谱拓扑**，不跑上传管线。但 `app.services/__init__.py` 会
# 连带导入 document_service → parsers → paddleocr / docling 等重依赖，让这个
# 纯拓扑测试在最小环境里跑不起来。用一个带 __path__ 的桩包顶替它：Python
# 会直接从子模块路径加载 `app.services.master_graph`，跳过包初始化。
# 被测模块链上没有 `from app.services import X` 这种依赖包级导出的写法。
if "app.services" not in sys.modules:
    _pkg = types.ModuleType("app.services")
    _pkg.__path__ = [str(Path(_BACKEND_ROOT) / "app" / "services")]
    sys.modules["app.services"] = _pkg
    # ★ 必须同步父包属性：否则 pytest 的字符串式 monkeypatch 解析
    #   `app.services.x.y` 时 import 会成功，却在 getattr(app, "services")
    #   处抛 AttributeError —— 污染同一会话中后跑的测试（实测打断
    #   test_citation_open_recheck.py 的 10 个用例）。真实 app.services 被
    #   导入时会自动覆盖该属性，因此这里不引入额外持久污染。
    import app as _app_pkg

    _app_pkg.services = _pkg

get_master_graph = importlib.import_module(
    "app.services.master_graph"
).get_master_graph
VALID_INTENTS = importlib.import_module(
    "app.services.routers.intent_rules"
).VALID_INTENTS
_stream_filter = importlib.import_module("app.services.stream_filter")
INTERNAL_LLM_NODES = _stream_filter.INTERNAL_LLM_NODES
STREAMING_LLM_NODES = _stream_filter.STREAMING_LLM_NODES


def _edges(graph) -> set[tuple[str, str]]:
    """compiled graph 的 (source, target) 边集合（含条件边展开后的所有目标）."""
    drawn = graph.get_graph()
    return {(e.source, e.target) for e in drawn.edges}


def test_router_has_five_intents():
    """图2：Query Router 按 LLM 分类 6 意图（含 Document Agent 交付支路）."""
    assert set(VALID_INTENTS) == {
        "document_summary",
        "knowledge_qa",
        "general_chat",
        "doc_relations",
        "list_documents",
        "document_agent",
    }
    print("[OK] test_router_has_five_intents")


def test_all_framework_nodes_present():
    """图2/图4：框架里的每个节点都必须真实存在于编译后的图谱中."""
    graph = get_master_graph()
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "route",              # Query Router
        "rewrite",            # Query Rewrite（指代消解+多查询）
        "retrieve",           # Hybrid Retrieval（BM25+向量+RRF+精排）
        "grade",              # Retrieval Grader
        "generate",           # Generate
        "refuse",             # 礼貌拒答
        "summary_digests",    # Summary Retrieval
        "summarize",          # Document Summary 生成
        "collect_digests",    # doc_relations 摘要采样
        "analyze_relations",  # doc_relations 分析
        "chat",               # General Chat
        "multimodal_context",  # 图片上下文（部分6：Grade 与 Generate 之间）
        "evidence_gate",      # 确定性证据门控（形态不达标 → 拒答）
        "citation_verifier",  # 五项引用校验（Generate → Verifier → Output Guard）
        "build_document",     # Document Agent（Word 交付物生成）
        "output_guard",       # Citation Check / 输出防护
        "save_history",       # 收尾落库
    }
    missing = expected - nodes
    assert not missing, f"framework nodes missing from compiled graph: {missing}"
    print("[OK] test_all_framework_nodes_present")


def test_router_dispatches_five_branches():
    """图2：route 按 5 意图分流（list_documents 在图内直达收尾节点）."""
    edges = _edges(get_master_graph())
    for src, dst in [
        ("route", "rewrite"),            # knowledge_qa 主支路
        ("route", "summary_digests"),    # summary 支路
        ("route", "chat"),               # 闲聊
        ("route", "collect_digests"),    # doc_relations 关联分析
        ("route", "save_history"),       # list_documents（DB 直读快路径）
    ]:
        assert (src, dst) in edges, f"missing router branch {src} → {dst}"
    print("[OK] test_router_dispatches_five_branches")


def test_knowledge_qa_chain_with_retry_loop():
    """图2/图4：rewrite → retrieve → grade，grade 可回边 rewrite 构成 Retry 循环.

    部分6：Grade 与 Generate 之间新增 Image Context 节点（`multimodal_context`）。
    本次新增：`multimodal_context` 之后是 Evidence Gate（确定性门控），
    由它决定 generate / build_document / refuse。
    """
    edges = _edges(get_master_graph())
    assert ("rewrite", "retrieve") in edges
    assert ("retrieve", "grade") in edges
    # Image Context 节点插在 grade 与生成之间
    assert ("grade", "multimodal_context") in edges
    assert ("grade", "generate") not in edges, "grade 必须经 multimodal_context 再到 generate"
    # Evidence Gate 紧跟 Image Context：形态不达标直接拒答
    assert ("multimodal_context", "evidence_gate") in edges
    assert ("multimodal_context", "generate") not in edges, "必须先过 Evidence Gate"
    assert ("evidence_gate", "generate") in edges          # 证据充足 → 常规作答
    assert ("evidence_gate", "build_document") in edges    # document_agent 交付物
    assert ("evidence_gate", "refuse") in edges            # 证据不足 → 拒答
    assert ("grade", "rewrite") in edges    # bad → rewrite（回边，≤N 次）
    assert ("grade", "refuse") in edges     # 仍 bad → 礼貌拒答
    print("[OK] test_knowledge_qa_chain_with_retry_loop")


def test_evidence_gate_is_fail_closed_refusal():
    """Evidence Gate 必须是"拒答"支路，而不是回到重试 —— 它是最后一道闸."""
    edges = _edges(get_master_graph())
    assert ("evidence_gate", "refuse") in edges
    assert ("evidence_gate", "rewrite") not in edges, "门控失败不应重试（重试已由 Grader 负责）"
    print("[OK] test_evidence_gate_is_fail_closed_refusal")


def test_citation_verifier_sits_between_generate_and_guard():
    """图4：Generate → Citation Verifier → Output Guard（五项校验必经）."""
    edges = _edges(get_master_graph())
    assert ("generate", "citation_verifier") in edges
    assert ("citation_verifier", "output_guard") in edges
    assert ("generate", "output_guard") not in edges, "generate 必须经 Citation Verifier"
    # Document Agent 产物是代码确定性拼装的纯文本、不含引用，无需过 Verifier
    assert ("build_document", "citation_verifier") not in edges
    print("[OK] test_citation_verifier_sits_between_generate_and_guard")


def test_document_agent_shares_retrieval_chain():
    """部分6/Document Agent：document_agent 复用主检索链，在 Image Context 后分流."""
    edges = _edges(get_master_graph())
    assert ("route", "rewrite") in edges, "document_agent 应复用 rewrite 检索主链"
    assert ("build_document", "output_guard") in edges, "交付物同样过输出防护"
    print("[OK] test_document_agent_shares_retrieval_chain")


def test_all_generation_paths_pass_output_guard():
    """图4：所有生成路径 → output_guard（Citation Check）→ save_history → END.

    本次新增：generate 先经 citation_verifier 再进 output_guard。
    """
    edges = _edges(get_master_graph())
    for src in ("chat", "summarize", "analyze_relations"):
        assert (src, "output_guard") in edges, f"{src} must pass output_guard"
        assert (src, "save_history") not in edges, f"{src} must NOT bypass output_guard"
    # 问答主路径：generate → citation_verifier → output_guard
    assert ("generate", "citation_verifier") in edges
    assert ("citation_verifier", "output_guard") in edges
    assert ("generate", "save_history") not in edges
    assert ("citation_verifier", "save_history") not in edges
    assert ("output_guard", "save_history") in edges
    assert ("save_history", "__end__") in edges
    print("[OK] test_all_generation_paths_pass_output_guard")


def test_refusal_bypasses_output_guard():
    """图4：refuse 拒答文本本身合规，直接落库，不进 output_guard / Citation Verifier."""
    edges = _edges(get_master_graph())
    assert ("refuse", "save_history") in edges
    assert ("refuse", "output_guard") not in edges
    assert ("refuse", "citation_verifier") not in edges
    print("[OK] test_refusal_bypasses_output_guard")


def test_llm_node_classification_is_exhaustive():
    """图内 7 个会调 LLM 的节点必须全部归入 生成类/决策类，防止 JSON 再泄漏."""
    all_llm_nodes = STREAMING_LLM_NODES | INTERNAL_LLM_NODES
    assert all_llm_nodes == {
        "generate", "chat", "summarize", "analyze_relations",  # 生成类
        "route", "rewrite", "grade",                            # 决策类
    }
    assert not (STREAMING_LLM_NODES & INTERNAL_LLM_NODES)
    print("[OK] test_llm_node_classification_is_exhaustive")


def main():
    tests = [
        test_router_has_five_intents,
        test_all_framework_nodes_present,
        test_router_dispatches_five_branches,
        test_knowledge_qa_chain_with_retry_loop,
        test_evidence_gate_is_fail_closed_refusal,
        test_citation_verifier_sits_between_generate_and_guard,
        test_document_agent_shares_retrieval_chain,
        test_all_generation_paths_pass_output_guard,
        test_refusal_bypasses_output_guard,
        test_llm_node_classification_is_exhaustive,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
