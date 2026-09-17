"""
图片理解门控的两个真 bug 回归测试（2026-09-13）.

背景（实测踩坑）：一张深色 TensorBoard 数据流图（input → Fan → output），
PaddleOCR 只读出 3 个节点名，Vision（qwen2.5vl）读出了完整描述——
「主题：风扇系统 / 节点：output、Fan、input / 连线：input → Fan、Fan → output」。
但这份**完全正确**的描述被两道逻辑丢掉了：

    1. ``number-anchor`` 把 VLM 自己分点作答的编号（``1)`` ``2)`` ``3)``）
       当成了"报出的数字"，OCR 文本里当然没有 → 扣 0.25 并盖"疑似幻觉"。
    2. 接受阈值 0.75 恰好等于 VLM 的名义自评分；乘上任何质检折损都必然
       跌破阈值 → VLM 产出**永远**进不了 Accept；随后兜底路径又没有把
       主通道的产出放进候选 → "best is None" → 退回 3 个字的 OCR 标签。

本文件锁定这两条修复。

运行（容器内）：
    docker exec -u root -e HOME=/tmp rag_backend \
        sh -c "cd /app && python tests/test_image_vlm_gate.py"
"""

from __future__ import annotations

import sys

_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        _failures.append(name)


try:
    from app.services.image_understanding.confidence import gate
    from app.services.image_understanding.engines.base import EngineOutput
    from app.services.image_understanding.quality import (
        QualityReport,
        _strip_enumeration_markers,
        check_vlm_output,
    )
except ImportError as exc:      # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    # 不能用 sys.exit()：pytest 在收集阶段导入本模块，抛 SystemExit 会让整个
    # 会话 INTERNALERROR，同目录其它用例全部跑不了。
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# 实测那条被误判的 VLM 原文（带列表编号）
VLM_REAL_TEXT = (
    "1) 这张图表达的主题：风扇系统\n\n"
    "2) 图中出现的所有节点、模块、组件的名称：\n   - output\n   - Fan\n   - input\n\n"
    "3) 节点之间的连线关系与数据/控制流方向：\n   - input → Fan\n   - Fan → output\n\n"
    "4) 整体流程或结构的一句话概述：\n   输入数据通过风扇处理后输出。"
)
OCR_REAL_TEXT = "output\nFan\ninput"


def test_strip_enumeration_markers():
    check("剥掉 1) 2) 编号",
          _strip_enumeration_markers("1) 主题\n2) 节点") == "主题\n节点")
    check("剥掉 2. 编号", _strip_enumeration_markers("2. 节点") == "节点")
    check("剥掉 三、编号", _strip_enumeration_markers("三、连线") == "连线")
    check("剥掉 • 项目符号", _strip_enumeration_markers("• 输入") == "输入")
    check("剥掉 - 项目符号", _strip_enumeration_markers("- input") == "input")
    check("行内数字原样保留",
          _strip_enumeration_markers("准确率 92.5%，共 3 类") == "准确率 92.5%，共 3 类")
    check("行首非编号数字保留",
          _strip_enumeration_markers("2024 年数据") == "2024 年数据")


def test_number_anchor_not_tripped_by_list_markers():
    """核心回归：分点作答的编号不能再触发"疑似幻觉"."""
    report = check_vlm_output(VLM_REAL_TEXT, "diagram", ocr_text=OCR_REAL_TEXT)
    joined = "；".join(report.reasons)
    check("不再报『疑似幻觉』", "疑似幻觉" not in joined, joined)
    check("number-anchor 未被扣分触发",
          "number-anchor" not in (report.checks or []), str(report.checks))
    check("质检通过", report.ok, str(report.to_dict()))


def test_number_anchor_still_works_for_real_hallucination():
    """反面：正文里编造的数字仍然要被抓出来（不能修过头）."""
    text = (
        "该图表显示准确率 97.5%、召回率 88.2%、F1 91.4% 三条曲线。"
    )
    report = check_vlm_output(text, "diagram", ocr_text="准确率与召回率曲线")
    joined = "；".join(report.reasons)
    check("真幻觉数字仍被标记", "疑似幻觉" in joined, joined)


def test_vlm_gate_is_veto_style():
    """VLM 产出：质检通过即 Accept，不受 0.75 名义分 × 折损的影响."""
    ok_quality = QualityReport(ok=True, score=0.7, reasons=[], checks=["structure"])
    bad_quality = QualityReport(ok=False, score=0.3, reasons=["结构校验：描述过短且无结构词"])

    out = EngineOutput(text=VLM_REAL_TEXT, confidence=0.75, engine="vision", ok=True)
    v_ok = gate(out, "diagram", ok_quality)
    check("VLM + 质检通过 → accept", v_ok.accepted, v_ok.reason)

    v_bad = gate(out, "diagram", bad_quality)
    check("VLM + 质检不过 → fallback", not v_bad.accepted, v_bad.reason)

    v_none = gate(out, "diagram", None)
    check("VLM 无质检信息 → accept（否决式）", v_none.accepted, v_none.reason)


def test_ocr_gate_still_multiplies():
    """非 VLM 引擎保持乘法折损语义（不能被 VLM 的例外改坏）."""
    out = EngineOutput(text="一段足够长的 OCR 文本内容" * 3, confidence=0.9, engine="paddleocr", ok=True)
    q = QualityReport(ok=False, score=0.5, reasons=["x"])
    verdict = gate(out, "text", q)
    check("OCR 0.9 × 0.5 = 0.45 < 0.75 → fallback", not verdict.accepted, verdict.reason)
    check("折损分被算出来", abs(verdict.confidence - 0.45) < 0.001, str(verdict.confidence))


def main() -> int:
    test_strip_enumeration_markers()
    test_number_anchor_not_tripped_by_list_markers()
    test_number_anchor_still_works_for_real_hallucination()
    test_vlm_gate_is_veto_style()
    test_ocr_gate_still_multiplies()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("ALL IMAGE VLM-GATE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
