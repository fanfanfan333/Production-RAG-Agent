"""
LLM token 流转发判定单元测试（问题1 续：内部 JSON 泄漏到答案）.

覆盖 stream_filter.should_stream_token / resolve_llm_node：
  * route / rewrite / grade 三个决策类节点的 LLM 输出必须被丢弃
    —— 这是截图1 中 ``{"rewritten": "你是谁", "variants": [...]}``
    出现在回答里的根因；
  * generate / chat / summarize / analyze_relations 的输出必须放行；
  * 归属判定：优先 metadata.langgraph_node，缺失时用 on_chain_start/end
    追踪到的节点名兜底；
  * 归属完全不明时保守放行（宁可维持旧行为，也不能吞掉正常答案）。

零第三方依赖，CI 直接 python 运行即可。
通过 importlib 直接加载模块文件，避免触发
backend/app/services/__init__.py 中的 SQLAlchemy/Postgres 重依赖导入。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

_MOD_PATH = Path(__file__).resolve().parent.parent / "app" / "services" / "stream_filter.py"
_spec = importlib.util.spec_from_file_location("app.services.stream_filter", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["app.services.stream_filter"] = _mod
_spec.loader.exec_module(_mod)

should_stream_token = _mod.should_stream_token
resolve_llm_node = _mod.resolve_llm_node
STREAMING_LLM_NODES = _mod.STREAMING_LLM_NODES
INTERNAL_LLM_NODES = _mod.INTERNAL_LLM_NODES


def _event(node: str | None = None) -> dict:
    """构造一个 on_chat_model_stream 事件骨架."""
    meta = {"langgraph_node": node} if node else {}
    return {"event": "on_chat_model_stream", "metadata": meta, "data": {"chunk": None}}


# ── 1. 决策类节点必须被拦截 ─────────────────────────────────────────────────

def test_route_token_blocked_by_metadata():
    """route 节点（意图识别 JSON）不得转发."""
    assert should_stream_token(_event("route")) is False


def test_rewrite_token_blocked_by_metadata():
    """rewrite 节点（{"rewritten":..., "variants":[...]}）不得转发."""
    assert should_stream_token(_event("rewrite")) is False


def test_grade_token_blocked_by_metadata():
    """grade 节点（证据评分）不得转发."""
    assert should_stream_token(_event("grade")) is False


def test_internal_nodes_blocked_via_tracking_without_metadata():
    """元数据缺失时，靠 on_chain_start/end 追踪到的 internal_node 也能拦住."""
    ev = _event(None)
    for node in ("route", "rewrite", "grade"):
        assert should_stream_token(ev, streaming_node=None, internal_node=node) is False


# ── 2. 生成类节点必须放行 ───────────────────────────────────────────────────

def test_streaming_nodes_allowed():
    for node in ("generate", "chat", "summarize", "analyze_relations"):
        assert should_stream_token(_event(node)) is True


def test_streaming_node_allowed_via_tracking():
    ev = _event(None)
    assert should_stream_token(ev, streaming_node="generate", internal_node=None) is True


# ── 3. 元数据优先于追踪值 ───────────────────────────────────────────────────

def test_metadata_wins_over_tracking():
    """元数据说 route（拦），追踪值说 generate（放）→ 以元数据为准，拦."""
    ev = _event("route")
    assert should_stream_token(ev, streaming_node="generate", internal_node=None) is False
    assert resolve_llm_node(ev, "generate", None) == "route"


def test_resolve_falls_back_to_streaming_then_internal():
    ev = _event(None)
    assert resolve_llm_node(ev, "summarize", "grade") == "summarize"
    assert resolve_llm_node(ev, None, "grade") == "grade"
    assert resolve_llm_node(ev, None, None) == ""


# ── 4. 归属不明时保守放行 ───────────────────────────────────────────────────

def test_unknown_origin_is_allowed():
    """拿不到任何归属信息时放行——避免元数据缺失导致回答变空."""
    assert should_stream_token(_event(None)) is True


def test_unknown_named_node_is_allowed():
    """未登记的新节点名放行（只有显式登记为决策类的才拦）."""
    assert should_stream_token(_event("some_future_node")) is True


# ── 5. 常量集合自身的一致性 ─────────────────────────────────────────────────

def test_node_sets_are_disjoint():
    assert STREAMING_LLM_NODES.isdisjoint(INTERNAL_LLM_NODES)


def test_all_llm_nodes_covered():
    """主图谱里 7 个会调 LLM 的节点必须都被归类，不能漏."""
    all_llm_nodes = {
        "route", "rewrite", "grade",
        "generate", "chat", "summarize", "analyze_relations",
    }
    assert STREAMING_LLM_NODES | INTERNAL_LLM_NODES == all_llm_nodes


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
