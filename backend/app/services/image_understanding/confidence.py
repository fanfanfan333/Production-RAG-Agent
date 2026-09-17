"""
置信度门控与校验（流程图里的 confidence / Validation 两个菱形）.

    Specialized Engine
          │
          ▼
      confidence ──高──► Accept ──► RAG
          │
          低
          ▼
      Fallback (Vision LLM / Second OCR)
          │
          ▼
      Validation
          │
          ▼
      confidence ──Pass──► RAG
          │
        Failed
          ▼
      Manual Review

本模块只做**判定**，不做调度（调度在 pipeline.py）。把判定单独抽出来有两个
好处：阈值可以集中调；判定逻辑可以单独测试 —— 这是整条链路里最容易出错、
也最该被测试覆盖的部分。

一个重要设计取舍：**不能只信引擎自报的 confidence**。OCR 引擎对"读错但很
自信"的输出照样给高分。所以 Validation 阶段做的是**结构校验**（表格有没有
对齐的管道符、公式有没有 LaTeX 记号、代码有没有缩进/围栏），用"内容长什么
样"交叉印证"引擎说它多准"。两者都过才算 Pass。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.image_understanding.engines.base import EngineOutput

# 判定结果
DECISION_ACCEPT = "accept"        # 置信度高，直接采用
DECISION_FALLBACK = "fallback"    # 置信度低，走兜底
DECISION_PASS = "pass"            # 兜底后校验通过
DECISION_MANUAL = "manual_review" # 兜底后仍不合格 → 人工复核


@dataclass
class Gate:
    """第一次 confidence 判定（专用引擎产出之后）."""

    confidence: float
    decision: str
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == DECISION_ACCEPT


@dataclass
class Validation:
    """Validation 阶段的校验结论."""

    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)


def _threshold(name: str, default: float) -> float:
    try:
        from app.config import get_settings

        return float(getattr(get_settings(), name, default))
    except Exception:      # noqa: BLE001
        return default


def accept_threshold() -> float:
    """高置信门限：>= 则 Accept."""
    return _threshold("IMAGE_CONFIDENCE_ACCEPT", 0.75)


def pass_threshold() -> float:
    """校验通过门限：>= 则 Pass，否则 Manual Review."""
    return _threshold("IMAGE_CONFIDENCE_PASS", 0.55)


def _effective_confidence(output: EngineOutput) -> float:
    """
    引擎自报置信度 → 可用于判定的置信度.

    失败/空产出直接归零；纯文本引擎在"文本极短"时要打折 —— "Hello" 这种
    5 个字符的输出，OCR 给 0.99 也不能算高置信，因为信息量太少。
    """
    if not output or not output.ok:
        return 0.0
    conf = float(output.confidence or 0.0)
    text = (output.text or "").strip()
    if not text:
        return 0.0
    if len(text) < 8:
        conf *= 0.5
    return round(min(1.0, max(0.0, conf)), 4)


def gate(output: EngineOutput, image_type: str, quality=None) -> Gate:
    """
    专用引擎产出后的第一次判定.

    *quality* 是 :mod:`app.services.image_understanding.quality` 的质检结论
    （可选）。传了就作为**乘数**折损置信度 —— 这一步很关键：OCR 对"读错但
    很自信"的输出照样给 0.9，只有把"代码语法没通过""OCR 行置信度整体偏低"
    这类**可验证的事实**乘进去，才能把它拉回 Fallback 分支。

    **VLM 例外**：多模态模型没有逐 token 置信度，它的自评分是名义值
    （有产出就给 0.75）。而接受阈值恰好也是 0.75 —— 一旦与质检分相乘，
    任何质检折损都会让它跌破阈值，于是"图表描述完全正确"也永远进不了
    Accept（实测踩过：深色流程图描述正确却落到 manual_review）。因此对
    VLM 产出改用**否决式**判定：可验证的质检通过 → Accept；不通过 → Fallback。
    """
    conf = _effective_confidence(output)
    if not output or not output.ok:
        return Gate(conf, DECISION_FALLBACK, output.error if output else "无产出")
    if _is_vlm_engine(output.engine):
        if quality is not None and not getattr(quality, "ok", True):
            reasons = "；".join(getattr(quality, "reasons", []) or [])
            return Gate(
                apply_quality(conf, quality), DECISION_FALLBACK,
                f"VLM 产出未通过质检：{reasons or '质检不合格'}",
            )
        return Gate(conf, DECISION_ACCEPT, "VLM 产出通过质检（否决式判定）")
    conf = apply_quality(conf, quality)
    if conf >= accept_threshold():
        return Gate(conf, DECISION_ACCEPT, "置信度达标")
    return Gate(conf, DECISION_FALLBACK, f"置信度 {conf:.2f} < {accept_threshold():.2f}")


def _is_vlm_engine(engine: str | None) -> bool:
    """产出是否来自多模态模型（vision / vision+… 这类命名）."""
    return "vision" in (engine or "").lower()


def apply_quality(confidence: float, quality) -> float:
    """
    把质检分折损进置信度（quality 为 None 时原样返回）.

    乘法而不是取小值：质检分 0.5 意味着"一半的可信度被可验证的事实否掉了"，
    直接相乘得到的衰减比 `min()` 更连续，也不会让一个 0.9 的引擎分在质检
    0.95 时毫无变化。
    """
    if quality is None:
        return round(float(confidence), 4)
    score = float(getattr(quality, "score", 1.0) or 0.0)
    return round(min(1.0, max(0.0, float(confidence) * score)), 4)


def validate(image_type: str, text: str) -> Validation:
    """
    结构校验（不依赖引擎自报分数）.

    每种类型查"内容形态是否成立"：
        table    → 有管道符、行为非空、列数一致
        formula  → 有 LaTeX 记号或 $ 包裹
        code     → 有围栏或缩进/符号密度足够
        chart    → 描述足够长（要能说清趋势/数据）
        diagram  → 描述里出现结构词（节点/箭头/流程/连接…）
        photo    → 有文本即可（门槛最低）
    """
    content = (text or "").strip()
    if not content:
        return Validation(False, 0.0, ["内容为空"])

    from app.services.image_understanding.structured_content import (
        IMAGE_TYPE_CHART,
        IMAGE_TYPE_CODE,
        IMAGE_TYPE_DIAGRAM,
        IMAGE_TYPE_FORMULA,
        IMAGE_TYPE_TABLE,
    )

    if image_type == IMAGE_TYPE_TABLE:
        return _validate_table(content)
    if image_type == IMAGE_TYPE_FORMULA:
        return _validate_formula(content)
    if image_type == IMAGE_TYPE_CODE:
        return _validate_code(content)
    if image_type == IMAGE_TYPE_CHART:
        return _validate_length(content, minimum=40, label="图表描述")
    if image_type == IMAGE_TYPE_DIAGRAM:
        return _validate_diagram(content)
    return Validation(len(content) >= 2, 0.7 if len(content) >= 2 else 0.0, ["通用文本"])


def _validate_table(content: str) -> Validation:
    rows = [r for r in content.splitlines() if r.strip()]
    piped = [r for r in rows if r.count("|") >= 2]
    if len(piped) < 2:
        return Validation(False, 0.2, ["未形成表格结构（管道符行不足）"])
    widths = [r.count("|") for r in piped]
    consistent = sum(1 for w in widths if w == max(set(widths), key=widths.count)) / len(widths)
    cells = sum(1 for r in piped for c in r.split("|") if c.strip())
    score = round(0.5 * consistent + 0.5 * min(1.0, cells / (len(piped) * 3)), 4)
    return Validation(score >= 0.5, score, [f"行数={len(piped)}", f"列一致度={consistent:.2f}"])


def _validate_formula(content: str) -> Validation:
    latex_markers = re.findall(r"\\[a-zA-Z]+|\^|_|\{|\}|\$", content)
    if len(latex_markers) < 2:
        return Validation(False, 0.2, ["缺少 LaTeX 记号"])
    score = round(min(1.0, len(latex_markers) / 8), 4)
    return Validation(True, max(score, 0.5), [f"LaTeX 记号 {len(latex_markers)} 个"])


def _validate_code(content: str) -> Validation:
    """
    代码校验：**能用解析器就用解析器**.

    ``ast.parse`` / ``json.loads`` 是确定性判定，比"缩进/符号密度"这类启发式
    可靠得多 —— 一段看起来很像代码的乱码在启发式下能拿高分，但过不了
    Python 解析器。启发式只作为"语言未知"时的降级路径保留。
    """
    if content.startswith("```"):
        body = "\n".join(content.splitlines()[1:-1]) or content
    else:
        body = content

    from app.services.image_understanding.quality import check_code_syntax

    report = check_code_syntax(content)
    if report.passed:
        # 解析器通过 → 高置信；启发式通过 → 中等置信
        parser_checked = any(c in report.checks for c in ("ast.parse", "json.loads", "yaml.safe_load"))
        score = 0.9 if parser_checked else max(0.6, report.score)
        return Validation(True, score, [f"代码校验通过({report.language})"])

    # 解析器/配平都不过 → 再看"像不像代码"，避免把普通文本误杀
    from app.services.image_understanding.engines.code_parser import code_likeness

    likeness = code_likeness(body.splitlines())
    score = round(0.4 * likeness, 4)
    reasons = [f"代码校验未通过({report.language})", *report.errors[:2]]
    return Validation(likeness >= 0.6, score, reasons)


def _validate_diagram(content: str) -> Validation:
    structure_words = ("节点", "箭头", "流程", "连接", "输入", "输出", "模块",
                       "步骤", "分支", "架构", "层", "方向", "→", "->")
    hits = sum(1 for w in structure_words if w in content)
    if len(content) < 20 and hits == 0:
        return Validation(False, 0.3, ["描述过短且无结构词"])
    score = round(min(1.0, 0.4 + 0.1 * hits), 4)
    return Validation(True, score, [f"结构词命中 {hits}"])


def _validate_length(content: str, *, minimum: int, label: str) -> Validation:
    if len(content) < minimum:
        return Validation(False, 0.3, [f"{label}过短（{len(content)} 字）"])
    return Validation(True, 0.75, [f"{label} {len(content)} 字"])


def final_confidence(engine_conf: float, validation: Validation) -> float:
    """
    兜底后的最终置信度 = 引擎自评 × 结构校验.

    相乘而不是平均：任意一方很差，结果就该很差。这也让"引擎很自信但结构不
    成立"的输入（典型的静默错误）拿不到高分。
    """
    return round(min(1.0, max(0.0, float(engine_conf) * float(validation.score))), 4)


def decide_after_fallback(engine_conf: float, validation: Validation) -> str:
    """Validation 之后：Pass 还是 Manual Review."""
    score = final_confidence(engine_conf, validation)
    return DECISION_PASS if (validation.passed and score >= pass_threshold()) else DECISION_MANUAL


__all__ = [
    "Gate",
    "Validation",
    "gate",
    "validate",
    "final_confidence",
    "decide_after_fallback",
    "apply_quality",
    "accept_threshold",
    "pass_threshold",
    "DECISION_ACCEPT",
    "DECISION_FALLBACK",
    "DECISION_PASS",
    "DECISION_MANUAL",
]
