"""
证据可信度（反幻觉：让"本身就不可信的解析产物"自己沉下去）.

要解决的问题
────────────
规模场景下最隐蔽的幻觉来源**不是模型编造**，而是**检索回来的证据本身就是错的**：

  · 扫描件 OCR 把 ``1`` 读成 ``l``、``0`` 读成 ``O`` —— 文字通顺、数字全错；
  · VLM 把代码截图转写成"看起来很像"的代码 —— 括号不配平、把 ``!=`` 读成 ``=``；
  · 结构解析失败只留下一段残片 —— 语义上仍然通顺，精排分照样很高。

这类 chunk 在**语义层面完全合理**，所以向量检索会把它们排上来，cross-encoder
精排也会给高分 —— 因为精排只看"这段文字与问题是否相关"，它无从知道
"这段文字是不是被读错的"。结果就是：一份引用完全自洽、却与原文不符的答案。

而项目在**解析期其实已经算出了**这些可信度信号（``analyze_confidence`` /
``analyze_quality`` / ``manual_review`` / 产出引擎），只是一直没有接到排序上。
本模块就是那座桥：把解析期"已知的不可信"变成检索期"排不上去"。

设计原则：**只用已知的、可解释的信号**
──────────────────────────────────────
不引入任何新的、需要猜测的分数。每一个降权因子都对应一条解析期已经记录在案的
事实（"这次 OCR 平均行置信度 0.44"、"质检判定代码语法未通过"、"已标记待人工
复核"），因此：

  · 可解释 —— 前端可以逐条告诉用户"这条证据为什么被降权"；
  · 可回归 —— 每个因子都能被单测直接构造；
  · 不会漂移 —— 不依赖模型版本或阈值调参。

刻意**不做**的事：不因为"文档老/页数多/来源部门"降权。那些是与事实相符性无关
的元数据，拿它们当可信度只会引入偏见。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# 各降权因子的权重（乘法链，最终夹到 [0.05, 1.0]）。
# 数值全部保守：宁可让一条真的低质证据留在候选里（用户能自己核对），
# 也不要误杀一条正确的证据（用户会得到"知识库没有"的错误印象）。
_FACTOR_MANUAL_REVIEW = 0.45      # 解析期已判"兜底后仍未通过校验，需人工复核"
_FACTOR_QUALITY_FAILED = 0.70     # 质检未通过（代码语法错、结构校验失败…）
_FACTOR_NO_ENGINE = 0.85          # 连产出引擎都没记录（旧索引 / 降级路径）
_FACTOR_TINY_TEXT = 0.60          # 正文过短：作为"证据"的信息量不足
_FACTOR_LOW_OCR = 0.75            # 平均行置信度低于可接受线

# 低于该长度的 chunk 视为"碎片"（不是"短"，是"不足以支撑结论"）。
# 20 字符量级对应"第 3 页"这类残片；一张 40 字的小表格是合法短证据，
# 因此阈值刻意压得很低，避免误伤结构化块。
_TINY_TEXT_CHARS = 20


@dataclass
class TrustReport:
    """一条证据的可信度评估（分数 + 逐条理由，理由会进前端展示）."""

    score: float = 1.0
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.score > 0.0

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "reasons": list(self.reasons)}


def compute_trust(
    *,
    text: str,
    content_type: str = "text",
    image_id: str | None = None,
    analyze_engine: str | None = None,
    analyze_confidence: float = 0.0,
    manual_review: bool = False,
    analyze_quality: dict | None = None,
) -> TrustReport:
    """
    给一条检索到的证据打可信度分（0~1，1 = 解析期未发现任何问题）.

    纯函数、无 I/O、无模型调用 —— 因此可以被单测穷举，也可以在检索路径上
    零成本地对每一条候选执行。
    """
    score = 1.0
    reasons: list[str] = []
    quality = analyze_quality if isinstance(analyze_quality, dict) else {}

    # ── 1. 人工复核标记（最强信号）───────────────────────────────────────────
    # 这个标记是解析期**主动**得出的结论："所有引擎都试过、校验仍未通过"。
    # 它比任何置信度数字都更直接 —— 置信度是引擎自己报的，复核标记是流程判的。
    if manual_review:
        score *= _FACTOR_MANUAL_REVIEW
        reasons.append("解析期已标记「需人工复核」，内容准确性未确认")

    # ── 2. 质检结论 ─────────────────────────────────────────────────────────
    if quality.get("ok") is False:
        score *= _FACTOR_QUALITY_FAILED
        for reason in (quality.get("reasons") or [])[:3]:
            if str(reason).strip():
                reasons.append(f"解析质检未通过：{reason}")

    # ── 3. 图片派生证据：引擎与置信度 ────────────────────────────────────────
    is_image_derived = bool(image_id) or content_type == "image"
    if is_image_derived:
        if not analyze_engine:
            score *= _FACTOR_NO_ENGINE
            reasons.append("未记录产出引擎（疑似降级路径或旧索引）")
        try:
            confidence = float(analyze_confidence or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence > 0:
            # 置信度直接线性参与：0.5 → 半可信，0.9 → 基本可信。
            # 用 (0.5 + 0.5*conf) 而不是 conf 本身：引擎自报的低置信度往往
            # 仍然是对的（只是模型不自信），直接乘会让好证据掉到阈值之下。
            score *= (0.5 + 0.5 * min(1.0, confidence))
            if confidence < 0.6:
                reasons.append(f"解析置信度偏低（{confidence:.2f}）")
        ocr = quality.get("ocr") or {}
        if isinstance(ocr, dict) and ocr.get("passed") is False:
            score *= _FACTOR_LOW_OCR
            reasons.append("OCR 行置信度不达标，可能读错数字或字母")

    # ── 4. 碎片正文 ─────────────────────────────────────────────────────────
    body = (text or "").strip()
    if len(body) < _TINY_TEXT_CHARS:
        score *= _FACTOR_TINY_TEXT
        reasons.append(f"正文过短（{len(body)} 字），不足以支撑结论")

    return TrustReport(score=max(0.05, min(1.0, score)), reasons=reasons)


def apply_trust_weighting(
    score: float,
    trust: float,
    weight: float,
) -> float:
    """
    把可信度以 *weight* 的强度混进最终排序分.

    ``final = score × (1 - weight + weight × trust)``

    为什么是**乘法**而不是加法：精排分（0~1 的概率语义）里，低分意味着"不相关"，
    高分意味着"相关"。可信度低意味着"相关，但可能是错的" —— 两者相乘表达的是
    "既要相关又要可信"，而相加会让一条高分低可信的证据压过一条中分高可信的证据，
    恰好把最危险的证据抬上来。

    *weight* 默认很小（0.15）：这里要做的是"同分/接近时让可信的胜出"，
    而不是"用可信度取代相关性" —— 相关性仍应由精排分主导。
    """
    w = max(0.0, min(1.0, float(weight)))
    if w <= 0:
        return score
    return score * (1.0 - w + w * max(0.0, min(1.0, trust)))
