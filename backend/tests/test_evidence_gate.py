"""
Evidence Gate 单元测试（确定性证据门控 + 拒答）.

覆盖：
  - 证据充足 → 放行
  - 无证据 / 条数不足 → 拒答
  - 最高精排分低于阈值 → 拒答
  - 问题关键词覆盖率过低 → 拒答
  - 证据正文过短（空壳证据）→ 拒答
  - 关闭开关 → 一律放行
  - 拒答文案与 is_refusal() 的一致性（模型主动拒答 == 门控拒答）

纯函数测试，不启动后端、不连数据库。直接 python 运行即可。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 让被测模块能 `from app.services.hybrid_search import tokenize`，
# 又不触发 app/services/__init__.py 里的 SQLAlchemy 重依赖：
# 先注册一个带 __path__ 的桩包，Python 便会直接加载子模块。
if "app.services" not in sys.modules:
    _pkg = types.ModuleType("app.services")
    _pkg.__path__ = [str(_BACKEND_ROOT / "app" / "services")]
    sys.modules["app.services"] = _pkg

_MOD_PATH = _BACKEND_ROOT / "app" / "services" / "nodes" / "evidence_gate.py"
_spec = importlib.util.spec_from_file_location(
    "app.services.nodes.evidence_gate", _MOD_PATH
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["app.services.nodes.evidence_gate"] = _mod
_spec.loader.exec_module(_mod)

REFUSAL_ANSWER = _mod.REFUSAL_ANSWER
MODEL_REFUSAL_SENTINEL = _mod.MODEL_REFUSAL_SENTINEL
evaluate_evidence = _mod.evaluate_evidence
is_refusal = _mod.is_refusal
query_coverage = _mod.query_coverage


@dataclass
class _Chunk:
    """最小化复现 RetrievedChunk 的鸭子类型（只需要 text / score）."""

    text: str
    score: float


# ── 放行 ─────────────────────────────────────────────────────────────────────

def test_sufficient_evidence_passes():
    chunks = [
        _Chunk(
            "合同约定的付款期限为 30 天，自验收合格之日起算；"
            "若逾期付款，违约金按日万分之五计算，累计不超过合同总额的百分之五。"
            "本条约定不因合同其他条款的变更而失效。",
            0.82,
        ),
        _Chunk(
            "验收合格后由需方出具验收单，付款期限自验收单签署之日起算。"
            "供方应在收到款项后三个工作日内开具发票并寄送需方财务部。",
            0.61,
        ),
    ]
    d = evaluate_evidence("合同约定的付款期限是多少天？", chunks)
    assert d.passed is True, d
    assert d.reason == "evidence_sufficient"
    assert d.evidence_count == 2
    assert 0.0 < d.confidence <= 1.0
    assert not [s for s in d.signals if not s.passed]
    print("[OK] test_sufficient_evidence_passes")


def test_no_evidence_refuses():
    d = evaluate_evidence("公司的营收增长率是多少？", [])
    assert d.passed is False
    assert d.reason == "no_evidence"
    assert d.confidence == 0.0
    assert d.evidence_count == 0
    print("[OK] test_no_evidence_refuses")


# ── 逐项拒答 ─────────────────────────────────────────────────────────────────

def test_low_top_score_refuses():
    chunks = [_Chunk(
        "付款期限为 30 天，违约金按日万分之五计算，自验收合格之日起算；"
        "供方应在收款后三个工作日内开具发票并寄送需方财务部归档。",
        0.05,
    )]
    d = evaluate_evidence("付款期限是多少天？", chunks, min_top_score=0.25)
    assert d.passed is False
    assert "top_score" in d.reason
    assert any(s.name == "top_score" and not s.passed for s in d.signals)
    print("[OK] test_low_top_score_refuses")


def test_low_coverage_refuses():
    # 证据与问题关键词几乎无交集（问的是付款期限，证据讲的是员工考勤）
    chunks = [_Chunk("公司员工考勤制度规定，每日上下班需打卡，迟到早退按规定处理。" * 3, 0.9)]
    d = evaluate_evidence("合同约定的付款期限和违约金比例是多少？", chunks, min_coverage=0.5)
    assert d.passed is False
    assert "query_coverage" in d.reason
    print("[OK] test_low_coverage_refuses")


def test_short_evidence_refuses():
    chunks = [_Chunk("是的。", 0.9)]
    d = evaluate_evidence("合同约定的付款期限是多少天？", chunks, min_evidence_chars=80)
    assert d.passed is False
    assert "evidence_length" in d.reason
    print("[OK] test_short_evidence_refuses")


def test_short_structured_chunk_still_passes():
    """短但真实的结构化块（小表格）不应被"空壳检查"误杀.

    回归用例：长度信号曾经默认 80 字，会把一张 40 字的合法小表格判为
    "证据不足"从而拒答 —— 而它恰恰是能回答"营收是多少"的唯一证据。
    """
    chunks = [
        _Chunk("| 指标 | 数值 |\n| --- | --- |\n| 营收 | 1200 万 |", 0.88),
    ]
    d = evaluate_evidence("营收是多少？", chunks)
    assert d.passed is True, d
    assert len("| 指标 | 数值 |\n| --- | --- |\n| 营收 | 1200 万 |") < 80
    print("[OK] test_short_structured_chunk_still_passes")


def test_multiple_failed_signals_reported():
    chunks = [_Chunk("无关内容", 0.01)]
    d = evaluate_evidence(
        "合同约定的付款期限是多少天？",
        chunks,
        min_top_score=0.5,
        min_coverage=0.9,
        min_evidence_chars=500,
    )
    assert d.passed is False
    failed = {s.name for s in d.signals if not s.passed}
    assert {"top_score", "query_coverage", "evidence_length"} <= failed
    print("[OK] test_multiple_failed_signals_reported")


def test_min_chunks_threshold():
    chunks = [_Chunk(
        "付款期限为 30 天，违约金按日万分之五计算，自验收合格之日起算；"
        "供方应在收款后三个工作日内开具发票并寄送需方财务部归档留存。",
        0.9,
    )]
    d = evaluate_evidence("付款期限是多少天？", chunks, min_chunks=3)
    assert d.passed is False
    assert "has_evidence" in d.reason
    print("[OK] test_min_chunks_threshold")


# ── 开关与工具函数 ───────────────────────────────────────────────────────────

def test_disabled_gate_always_passes():
    d = evaluate_evidence("任何问题", [], enabled=False)
    assert d.passed is True
    assert d.reason == "evidence_gate_disabled"
    print("[OK] test_disabled_gate_always_passes")


def test_query_coverage_bounds():
    assert query_coverage("", "任何文本") == 1.0        # 问题无词 → 不判负
    assert query_coverage("付款期限", "") == 0.0        # 证据无词 → 0
    high = query_coverage("付款期限", "合同约定的付款期限为 30 天")
    assert high > 0.5, high
    print("[OK] test_query_coverage_bounds")


def test_refusal_text_is_consistent():
    """门控拒答文案必须能被 is_refusal 识别为拒答（与模型主动拒答同款）."""
    assert is_refusal(REFUSAL_ANSWER) is True
    assert MODEL_REFUSAL_SENTINEL in REFUSAL_ANSWER
    assert is_refusal("") is False
    assert is_refusal("合同约定的付款期限是 30 天 [Source 1]。") is False
    print("[OK] test_refusal_text_is_consistent")


def test_audit_payload_shape():
    """as_audit() 输出必须可 JSON 序列化（会经 SSE 下发到前端）."""
    import json

    chunks = [_Chunk(
        "付款期限为 30 天，违约金按日万分之五计算，自验收合格之日起算；"
        "供方应在收款后三个工作日内开具发票并寄送需方财务部归档留存。",
        0.8,
    )]
    d = evaluate_evidence("付款期限是多少天？", chunks)
    payload = json.dumps(d.as_audit(), ensure_ascii=False)
    assert "passed" in payload and "signals" in payload
    print("[OK] test_audit_payload_shape")


if __name__ == "__main__":
    tests = [
        test_sufficient_evidence_passes,
        test_no_evidence_refuses,
        test_low_top_score_refuses,
        test_low_coverage_refuses,
        test_short_evidence_refuses,
        test_short_structured_chunk_still_passes,
        test_multiple_failed_signals_reported,
        test_min_chunks_threshold,
        test_disabled_gate_always_passes,
        test_query_coverage_bounds,
        test_refusal_text_is_consistent,
        test_audit_payload_shape,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} evidence-gate tests passed.")
