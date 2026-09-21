"""
OCR / VLM 双通道融合（把"串行兜底"升级为"并行互补"）.

## 为什么要改

原设计是**串行**的：

    Specialized Engine → 置信度低 → Fallback(Vision) → Validation

问题在于"低置信"这个触发条件太钝。实践中最常见的失败模式是**专用引擎
自信地错**：OCR 把代码里的 ``!=`` 读成 ``=``、把 ``0`` 读成 ``O``，然后给出
0.9 的置信度 —— 门控直接 Accept，VLM 根本没机会看一眼。

改成**双通道**后，两条通道各自独立产出，再由融合器交叉验证：

    ┌─ 通道 A：OCR(+结构化引擎)  → 文本 / 缩进 / 表格网格
    │
    └─ 通道 B：VLM               → 语义 / 代码校正 / 图表数据

    两个通道 → 融合（按图片类型选策略）→ 最终产出

## 融合策略（按图片类型）

    ┌───────────────┬──────────────────────────────────────────────────┐
    │ complementary │ table / code / formula —— 结构通道给骨架，VLM 做   │
    │               │ 交叉校验与校正。代码用**语法校验**当裁判：谁的代码  │
    │               │ 能通过 ast.parse，就用谁的。                       │
    │ vision-first  │ chart / diagram / screenshot —— VLM 是主力，OCR 只  │
    │               │ 作为"图里的文字/数字锚点"补充。                    │
    │ ocr-first     │ photo —— 文字为主，OCR 优先；OCR 质量差才让 VLM 上 │
    └───────────────┴──────────────────────────────────────────────────┘

## 成本控制（很重要）

VLM 是整条管线里最贵的一步（一次推理 1-5 秒 + 显存）。因此双通道**不是
无条件开**：受 ``IMAGE_DUAL_CHANNEL_ENABLED`` 与 ``IMAGE_DUAL_CHANNEL_TYPES``
白名单控制，默认只对"结构复杂、OCR 容易错"的类型开（code / table / formula /
chart / diagram / screenshot），photo 默认不开。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import get_settings
from app.services.image_understanding.engines.base import EngineOutput
from app.services.image_understanding.quality import QualityReport
from app.utils.logging import get_logger

logger = get_logger(__name__)

CHANNEL_OCR = "ocr"
CHANNEL_VLM = "vlm"

#: 融合策略名
STRATEGY_COMPLEMENTARY = "complementary"
STRATEGY_VISION_FIRST = "vision-first"
STRATEGY_OCR_FIRST = "ocr-first"


@dataclass
class ChannelResult:
    """一个通道的产出 + 它的质检结论."""

    channel: str
    output: EngineOutput
    quality: QualityReport

    @property
    def text(self) -> str:
        return (self.output.text or "").strip()

    @property
    def rank(self) -> float:
        """
        通道排序分 = 引擎自评置信度 × 质检折损.

        质检分做**折损**而不是**平均**：引擎说自己 0.9 准，但代码语法校验
        没通过，那 0.9 就应当被打下来 —— 平均会让"引擎很自信但产出是错的"
        拿到虚高的排序分，正是我们要避免的。
        """
        conf = float(self.output.confidence or 0.0)
        return round(conf * (0.5 + 0.5 * float(self.quality.score)), 4)

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "engine": self.output.engine,
            "ok": bool(self.output.ok),
            "chars": len(self.text),
            "confidence": round(float(self.output.confidence or 0.0), 4),
            "rank": self.rank,
            "quality": self.quality.to_dict(),
        }


@dataclass
class FusionResult:
    """
    双通道融合结论.

    ``confidence`` 刻意保持**引擎原始自评**（不做质检折损）—— 因为管线会把它
    连同 ``quality`` 一起交给门控 ``gate()``，由门控做**唯一一次**折损。
    如果这里先折一次、门控再折一次，等于平方衰减，会把好产出也压到阈值以下。
    通道之间的**选择**用 :attr:`ChannelResult.rank`（已含折损），两者分工不同。
    """

    text: str
    engine: str
    confidence: float
    strategy: str
    chosen: str = ""
    rank: float = 0.0
    notes: list[str] = field(default_factory=list)
    channels: list[ChannelResult] = field(default_factory=list)

    @property
    def agreed(self) -> bool:
        """两个通道是否**都**通过了质检（= 交叉验证成功）."""
        return bool(self.channels) and all(c.quality.ok for c in self.channels)

    def chosen_channel(self) -> ChannelResult | None:
        """被选中的那个通道（拿它的 quality 交给门控）."""
        for c in self.channels:
            if c.channel == self.chosen:
                return c
        return self.channels[0] if self.channels else None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "chosen": self.chosen,
            "engine": self.engine,
            "confidence": round(float(self.confidence), 4),
            "rank": round(float(self.rank), 4),
            "agreed": self.agreed,
            "notes": list(self.notes),
            "channels": [c.to_dict() for c in self.channels],
        }


def dual_channel_types() -> set[str]:
    """允许跑双通道的图片类型白名单（控制 VLM 成本）."""
    raw = (get_settings().IMAGE_DUAL_CHANNEL_TYPES or "").strip()
    if not raw:
        return set()
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def dual_channel_enabled(image_type: str) -> bool:
    """
    这张图要不要跑双通道.

    两个条件都要满足：总开关打开、类型在白名单里。VLM 是否可用由调用方
    （pipeline 的 ``_run_vlm_channel``）在真正调用前探测，不可用时会写
    ``result.meta["vlm_channel_skipped"]``，这里不做可用性探测。
    """
    settings = get_settings()
    if not getattr(settings, "IMAGE_DUAL_CHANNEL_ENABLED", True):
        return False
    return image_type in dual_channel_types()


def strategy_for(image_type: str) -> str:
    """图片类型 → 融合策略."""
    from app.services.image_understanding.structured_content import (
        IMAGE_TYPE_CHART,
        IMAGE_TYPE_CODE,
        IMAGE_TYPE_DIAGRAM,
        IMAGE_TYPE_FORMULA,
        IMAGE_TYPE_SCREENSHOT,
        IMAGE_TYPE_TABLE,
    )

    if image_type in (IMAGE_TYPE_TABLE, IMAGE_TYPE_CODE, IMAGE_TYPE_FORMULA):
        return STRATEGY_COMPLEMENTARY
    if image_type in (IMAGE_TYPE_CHART, IMAGE_TYPE_DIAGRAM, IMAGE_TYPE_SCREENSHOT):
        return STRATEGY_VISION_FIRST
    return STRATEGY_OCR_FIRST


def _reliable(side: ChannelResult) -> bool:
    """
    OCR / 结构化通道是否**可信到可以不看 VLM**.

    比 ``quality.ok`` 严一档：还要求那次 OCR 的逐行置信度评估没有失败。
    理由是"结构通道的骨架来自 OCR"—— 若 OCR 行置信度整体偏低，那么基于它
    还原出来的表格网格 / 代码缩进同样不可信，即使形态看起来是对的。

    注意这个判据只用于 OCR / 结构化通道。VLM 通道不依赖 OCR，用它对 VLM
    是不公平的（VLM 看得懂图，OCR 读不准并不影响它）。
    """
    if not side.quality.ok:
        return False
    ocr = side.quality.ocr
    if ocr is not None and ocr.reported and not ocr.passed:
        return False
    return True


def fuse_channels(
    image_type: str,
    channels: list[ChannelResult],
) -> FusionResult:
    """
    按策略融合两个通道.

    *channels* 里可以只有一个通道（另一个不可用 / 被白名单挡掉），此时融合器
    退化为"透传 + 说明"，不做任何猜测。
    """
    usable = [c for c in channels if c.text]
    if not usable:
        return FusionResult(
            text="", engine="none", confidence=0.0,
            strategy=strategy_for(image_type), chosen="none",
            notes=["两个通道都没有产出"], channels=list(channels),
        )

    if len(usable) == 1:
        only = usable[0]
        return FusionResult(
            text=only.text, engine=only.output.engine or only.channel,
            confidence=float(only.output.confidence or 0.0),
            rank=only.rank,
            strategy=strategy_for(image_type),
            chosen=only.channel,
            notes=[f"仅 {only.channel} 通道可用"],
            channels=list(channels),
        )

    strategy = strategy_for(image_type)
    by_channel = {c.channel: c for c in usable}
    ocr_side = by_channel.get(CHANNEL_OCR)
    vlm_side = by_channel.get(CHANNEL_VLM)

    if strategy == STRATEGY_COMPLEMENTARY:
        return _fuse_complementary(image_type, ocr_side, vlm_side, channels)
    if strategy == STRATEGY_VISION_FIRST:
        return _fuse_vision_first(ocr_side, vlm_side, channels)
    return _fuse_ocr_first(ocr_side, vlm_side, channels)


def _fuse_complementary(
    image_type: str,
    ocr_side: ChannelResult | None,
    vlm_side: ChannelResult | None,
    channels: list[ChannelResult],
) -> FusionResult:
    """
    互补型（table / code / formula）：结构通道给骨架，VLM 做交叉校验.

    选择规则（顺序即优先级）：

        1. 结构通道质检通过 → 用它（**结构**比描述更接近原图，表格尤其如此）
        2. 结构通道不过、VLM 通过 → 用 VLM（典型的"OCR 读崩了但模型看懂了"）
        3. 都不过 → 取排序分高的，并明确标注"两通道均未通过校验"

    代码类型额外加一条：**谁的代码能通过语法校验就用谁的**。这一条比
    "谁分高"更硬 —— 语法是可判定的，分数不是。
    """
    from app.services.image_understanding.structured_content import IMAGE_TYPE_CODE

    notes: list[str] = []
    if ocr_side is None or vlm_side is None:
        only = ocr_side or vlm_side
        assert only is not None
        return FusionResult(
            text=only.text, engine=only.output.engine or only.channel,
            confidence=only.rank, strategy=STRATEGY_COMPLEMENTARY,
            chosen=only.channel, notes=["另一通道未产出"], channels=channels,
        )

    # ── 代码：语法校验当裁判 ────────────────────────────────────────────────
    if image_type == IMAGE_TYPE_CODE:
        ocr_syntax = (ocr_side.quality.code.passed
                      if ocr_side.quality.code else None)
        vlm_syntax = (vlm_side.quality.code.passed
                      if vlm_side.quality.code else None)
        if vlm_syntax and not ocr_syntax:
            notes.append("OCR 代码未通过语法校验，采用 VLM 校正后的版本")
            return _chosen(vlm_side, STRATEGY_COMPLEMENTARY, notes, channels)
        if ocr_syntax and vlm_syntax:
            notes.append("两通道代码均通过语法校验")
        elif ocr_syntax and not vlm_syntax:
            notes.append("VLM 代码未通过语法校验，采用 OCR+CodeParser 版本")

    if _reliable(ocr_side) and not vlm_side.quality.ok:
        notes.append("VLM 通道未通过质检（已忽略其产出）")
        return _chosen(ocr_side, STRATEGY_COMPLEMENTARY, notes, channels)
    if vlm_side.quality.ok and not _reliable(ocr_side):
        notes.append("结构通道未通过质检，改用 VLM 产出")
        return _chosen(vlm_side, STRATEGY_COMPLEMENTARY, notes, channels)
    if _reliable(ocr_side) and vlm_side.quality.ok:
        notes.append("双通道均通过质检（交叉验证成功）")
        chosen = ocr_side if ocr_side.rank >= vlm_side.rank else vlm_side
        return _chosen(chosen, STRATEGY_COMPLEMENTARY, notes, channels)

    notes.append("两通道均未通过质检 → 取排序分高者并建议人工复核")
    chosen = ocr_side if ocr_side.rank >= vlm_side.rank else vlm_side
    return _chosen(chosen, STRATEGY_COMPLEMENTARY, notes, channels)


def _fuse_vision_first(
    ocr_side: ChannelResult | None,
    vlm_side: ChannelResult | None,
    channels: list[ChannelResult],
) -> FusionResult:
    """
    视觉优先型（chart / diagram / screenshot）：VLM 是主力.

    图表/流程图的"信息"在结构与语义里，OCR 只能读到零散标注文字 —— 所以
    即使 OCR 分更高也不该赢。只有 VLM 不可用或质检不过时才退回 OCR。
    """
    notes: list[str] = []
    if vlm_side is None:
        assert ocr_side is not None
        notes.append("VLM 通道不可用，退回 OCR 文本")
        return _chosen(ocr_side, STRATEGY_VISION_FIRST, notes, channels)
    if ocr_side is None:
        return _chosen(vlm_side, STRATEGY_VISION_FIRST, ["仅 VLM 通道产出"], channels)

    if vlm_side.quality.ok:
        if _reliable(ocr_side):
            notes.append("双通道均通过质检；图表语义以 VLM 为准")
        else:
            notes.append("OCR 通道未通过质检，以 VLM 为准")
        return _chosen(vlm_side, STRATEGY_VISION_FIRST, notes, channels)

    if _reliable(ocr_side):
        notes.append("VLM 产出未通过质检，退回 OCR 文本")
        return _chosen(ocr_side, STRATEGY_VISION_FIRST, notes, channels)

    notes.append("两通道均未通过质检 → 取排序分高者并建议人工复核")
    chosen = ocr_side if ocr_side.rank >= vlm_side.rank else vlm_side
    return _chosen(chosen, STRATEGY_VISION_FIRST, notes, channels)


def _fuse_ocr_first(
    ocr_side: ChannelResult | None,
    vlm_side: ChannelResult | None,
    channels: list[ChannelResult],
) -> FusionResult:
    """OCR 优先型（photo）：文字为主，OCR 够用就不花 VLM 的钱."""
    notes: list[str] = []
    if ocr_side is None:
        assert vlm_side is not None
        notes.append("OCR 通道无产出，改用 VLM")
        return _chosen(vlm_side, STRATEGY_OCR_FIRST, notes, channels)
    if _reliable(ocr_side):
        if vlm_side is not None and not vlm_side.quality.ok:
            notes.append("VLM 通道未通过质检（已忽略）")
        return _chosen(ocr_side, STRATEGY_OCR_FIRST, notes, channels)
    if vlm_side is not None and vlm_side.quality.ok:
        notes.append("OCR 置信度不足，改用 VLM 产出")
        return _chosen(vlm_side, STRATEGY_OCR_FIRST, notes, channels)

    notes.append("OCR 未通过质检且 VLM 无有效产出 → 保留 OCR 文本待复核")
    return _chosen(ocr_side, STRATEGY_OCR_FIRST, notes, channels)


def _chosen(
    side: ChannelResult,
    strategy: str,
    notes: list[str],
    channels: list[ChannelResult],
) -> FusionResult:
    return FusionResult(
        text=side.text,
        engine=side.output.engine or side.channel,
        confidence=float(side.output.confidence or 0.0),
        rank=side.rank,
        strategy=strategy,
        chosen=side.channel,
        notes=notes,
        channels=list(channels),
    )


def merge_captions(primary: str, secondary: str | None) -> str:
    """
    把两个通道的**自然语言**产出合并成一段（去重后拼接）.

    只用于 chart / diagram / screenshot 这类"描述型"产出 —— 结构化产出
    （表格 Markdown、代码围栏）绝不能拼接，否则检索时会拿到两份互相打架的
    网格。结构化场景请直接用 :func:`fuse_channels` 选中的那一个。
    """
    first = (primary or "").strip()
    second = (secondary or "").strip()
    if not second:
        return first
    if not first:
        return second
    # 已经包含（或高度重叠）就不重复叠加
    if second in first or first in second:
        return first if len(first) >= len(second) else second
    return f"{first}\n\n{second}"


__all__ = [
    "ChannelResult",
    "FusionResult",
    "CHANNEL_OCR",
    "CHANNEL_VLM",
    "STRATEGY_COMPLEMENTARY",
    "STRATEGY_VISION_FIRST",
    "STRATEGY_OCR_FIRST",
    "dual_channel_enabled",
    "dual_channel_types",
    "strategy_for",
    "fuse_channels",
    "merge_captions",
]
