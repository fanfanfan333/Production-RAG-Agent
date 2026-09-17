"""
图片理解管线 —— 置信度门控的多引擎调度（设计稿的核心）.

    Image
      │
      ▼
    Preprocessing                      ← preprocess.py（放大/去噪/对比度/纠偏）
      │  ├─ image      → 分类 / Vision / 表格结构识别（保持彩色与层次）
      │  └─ ocr_image  → OCR（可选二值化）
      ▼
    OCR（唯一一次）                     ← 文本 + 行坐标 + 逐行置信度
      │
      ▼
    Picture Classification              ← classifier.py（table/code/formula/chart/…）
      │
      ▼
    ┌─ 通道 A：Specialized Engine        ┐
    │  （表格→Table Parser，代码→Code   │   ← dual_channel.py
    │   Parser，图表→Vision，…）        │      两个通道各自独立产出 + 各自质检
    └─ 通道 B：VLM（白名单类型才跑）      ┘
      │
      ▼
    Fusion（按图片类型选策略）           ← 互补 / 视觉优先 / OCR 优先
      │
      ▼
    Quality（代码语法 / OCR confidence / VLM 幻觉）
      │
      ▼
    confidence  ──── 高 ────► Accept ──► RAG
      │
      低
      ▼
    Fallback ──┬── Vision LLM
               └── Second OCR
      │
      ▼
    Validation                          ← confidence.py（结构校验 + 语法校验）
      │
      ▼
    confidence  ──── Pass ───► RAG
      │
    Failed
      ▼
    Manual Review                       ← manual_review=True 落到 chunk metadata

按类型选哪个"专用引擎"（对应设计稿的引擎清单）：

    ┌──────────┬───────────────────────┬────────────────────────────────┐
    │ 图片类型  │ 专用引擎                │ 低于阈值时兜底                  │
    ├──────────┼───────────────────────┼────────────────────────────────┤
    │ table    │ Table Transformer      │ 规则表格 → Vision 转写          │
    │ formula  │ PaddleOCR Formula      │ Vision（LaTeX）                │
    │ code     │ OCR + Code Parser      │ Vision（逐行转写 + 语法校正）    │
    │ chart    │ Vision                 │ Second OCR                     │
    │ diagram  │ Vision                 │ Second OCR                     │
    │ screenshot│ Vision                │ Second OCR                     │
    │ photo    │ PaddleOCR              │ Vision                         │
    └──────────┴───────────────────────┴────────────────────────────────┘

**只跑一次 OCR**：分类需要的行坐标、代码缩进重建、表格单元格、兜底校验全都
复用同一次 OCR 结果。几十张图的一篇文档，这一步的差异就是分钟级的。

**双通道的额外成本也要算**：VLM 是这条链路里最贵的一步，所以通道 B 只在
``IMAGE_DUAL_CHANNEL_TYPES`` 白名单里的类型上跑，且 chart/diagram/screenshot
的"专用引擎"本来就是 Vision —— 那种情况下不重复推理，直接把 primary 当作
通道 B。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from app.config import get_settings
from app.services.image_understanding.classifier import (
    ImageClassification,
    classify_image_safe,
    compute_signals,
    vision_analyze_types,
)
from app.services.image_understanding.confidence import (
    DECISION_ACCEPT,
    DECISION_FALLBACK,
    DECISION_MANUAL,
    DECISION_PASS,
    decide_after_fallback,
    final_confidence,
    gate,
    validate,
)
from app.services.image_understanding.dual_channel import (
    CHANNEL_OCR,
    CHANNEL_VLM,
    ChannelResult,
    dual_channel_enabled,
    fuse_channels,
)
from app.services.image_understanding.engines.base import EngineOutput
from app.services.image_understanding.engines.registry import get_registry
from app.services.image_understanding.engines.vision_engine import VisionEngine
from app.services.image_understanding.preprocess import (
    pick_mode,
    preprocess,
)
from app.services.image_understanding.quality import (
    QualityReport,
    assess_ocr_confidence,
    verify_output,
)
from app.services.image_understanding.structured_content import (
    IMAGE_TYPE_CODE,
    IMAGE_TYPE_FORMULA,
    IMAGE_TYPE_PHOTO,
    IMAGE_TYPE_TABLE,
)
from app.services.image_understanding.table_recognizer import (
    TableStructure,
    recognize_table_safe,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 处理路径。注意 ROUTE_TABLE 的历史取值 "table_parser" 被下游（入库元数据、
# 测试）依赖，**不要**改成 "table"。
ROUTE_TABLE = "table_parser"
ROUTE_FORMULA = "formula_parser"
ROUTE_CODE = "code_parser"
ROUTE_VISION = "vision"
ROUTE_OCR = "ocr"


@dataclass
class ImageUnderstanding:
    """一张图片走完"预处理 → 双通道 → 融合 → 质检 → 门控 → 兜底"的全部产出."""

    image_type: str
    route: str
    classification: ImageClassification
    ocr_text: str = ""
    ocr_engine: str | None = None
    # 表格 → Markdown 表格 / 代码 → 代码块 / 公式 → LaTeX；其余为 None
    structured_content: str | None = None
    # 图表 / 流程图 / 截图的"结构化描述"（Vision 产出）
    vision_caption: str | None = None
    # 真正产出内容的引擎名（写进 chunk metadata，前端展示）
    analyze_engine: str = "ocr"
    table: TableStructure | None = None
    ocr_line_count: int = 0
    meta: dict = field(default_factory=dict)

    # ── 置信度门控（新增）──────────────────────────────────────────────────
    # 最终置信度：Accept 时 = 专用引擎自评（已按质检折损）；兜底时 = 引擎自评 × 结构校验
    confidence: float = 0.0
    # accept | fallback | pass | manual_review
    decision: str = ""
    # True 表示兜底后仍未通过校验 → 需要人工复核
    manual_review: bool = False
    # 预处理做了哪些操作（可审计）
    preprocess: list[str] = field(default_factory=list)
    # 每个引擎的尝试轨迹：[{engine, ok, confidence, decision, error}]
    attempts: list[dict] = field(default_factory=list)

    # ── 噪声处理 / 双通道 / 质量校验（新增）────────────────────────────────
    # 预处理档位与噪声信号（为什么选了这个档）
    preprocess_mode: str = ""
    preprocess_meta: dict = field(default_factory=dict)
    # OCR 逐行置信度评估（均值/最低/低置信占比/引擎是否上报）
    ocr_confidence: dict = field(default_factory=dict)
    # 最终产出的质检结论（代码语法、VLM 幻觉、数字锚点…）
    quality: dict = field(default_factory=dict)
    # 双通道融合结论（策略、选中通道、两个通道各自的质检）
    fusion: dict = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        """是否发生了降级（没跑到理想引擎）。"""
        return self.analyze_engine in ("ocr-degraded", "none") or not self.analyze_engine

    def to_dict(self) -> dict:
        return {
            "image_type": self.image_type,
            "route": self.route,
            "analyze_engine": self.analyze_engine,
            "ocr_engine": self.ocr_engine,
            "ocr_lines": self.ocr_line_count,
            "confidence": round(float(self.confidence), 4),
            "decision": self.decision,
            "manual_review": self.manual_review,
            "preprocess": list(self.preprocess),
            "preprocess_mode": self.preprocess_mode,
            "preprocess_meta": dict(self.preprocess_meta),
            "ocr_confidence": dict(self.ocr_confidence),
            "quality": dict(self.quality),
            "fusion": dict(self.fusion),
            "classification": self.classification.to_dict(),
            "table": self.table.to_dict() if self.table else None,
            "attempts": list(self.attempts),
            "degraded": self.degraded,
        }


def resolve_route(image_type: str) -> str:
    """图片类型 → 处理路径."""
    if image_type == IMAGE_TYPE_TABLE:
        return ROUTE_TABLE
    if image_type == IMAGE_TYPE_FORMULA:
        return ROUTE_FORMULA
    if image_type == IMAGE_TYPE_CODE:
        return ROUTE_CODE
    if image_type in vision_analyze_types():
        return ROUTE_VISION
    return ROUTE_OCR


def understand_image(
    img: Image.Image,
    *,
    page_number: int = 1,
    filename: str = "",
    png_bytes: bytes | None = None,
) -> ImageUnderstanding:
    """
    对一张图片执行完整管线.

    *png_bytes* 可以复用调用方已经编码好的 PNG（落盘时通常已经编过），
    避免为了喂 Vision 再编码一次大图。
    """
    settings = get_settings()
    png_cache: dict[str, bytes] = {}
    if png_bytes:
        png_cache["png"] = png_bytes

    def png() -> bytes:
        if "png" not in png_cache:
            from app.services.image_understanding.imaging import encode_png_safe

            png_cache["png"] = encode_png_safe(img)
        return png_cache["png"]

    # ── 0. Preprocessing（按噪声信号自动选档）──────────────────────────────
    # 先量噪声再动手：干净的导出图一步都不做，翻拍件才去噪/去边框/纠偏。
    # pick_mode 只看内容，与图片类型无关 —— 类型此时还没判出来。
    mode, noise = pick_mode(img)
    pre = preprocess(img, mode=mode)
    # 纠偏只对"文档感"很强的图做：线稿占比高说明是扫描件/截图而不是照片。
    # 这样判定不需要 OCR，成本只有一次缩略图像素遍历。
    if (
        getattr(settings, "IMAGE_PREPROCESS_GEOMETRIC", True)
        and mode in ("light", "noisy")          # scan/geometric 档位已经纠过偏
    ):
        try:
            preview = compute_signals(pre.image)
            if preview["line_art_ratio"] >= 0.8:
                geometric = preprocess(img, mode="geometric")
                if "deskew" in geometric.applied:
                    pre = geometric
                    mode = "geometric"
        except Exception as exc:      # noqa: BLE001
            logger.debug("Geometric preprocessing skipped: %s", exc)

    work = pre.image            # 给分类 / Vision / 表格结构识别（保持彩色）
    ocr_input = pre.ocr_input   # 给 OCR（scan/text 档位会带二值化）

    # ── 1. 唯一一次 OCR（文本 + 行坐标 + 逐行置信度）────────────────────────
    ocr_out = _run_ocr(ocr_input)
    ocr_text = (ocr_out.text or "").strip()[: settings.MAX_IMAGE_OCR_CHARS]
    # 逐行置信度评估：均值会被大量"轻松识别的行"拉高，所以同时看最低行与
    # 低置信行占比 —— 表格里一个数字读错就足以让结论错。
    ocr_confidence = assess_ocr_confidence(ocr_out.lines)

    # ── 2. Picture Classification ─────────────────────────────────────────
    classification = classify_image_safe(work, ocr_lines=ocr_out.lines, filename=filename)

    # ── 3. 分派专用引擎（= 通道 A）─────────────────────────────────────────
    route = resolve_route(classification.image_type)
    result = ImageUnderstanding(
        image_type=classification.image_type,
        route=route,
        classification=classification,
        ocr_text=ocr_text,
        ocr_engine=ocr_out.engine if ocr_out.ok else None,
        ocr_line_count=len(ocr_out.lines or []),
        preprocess=list(pre.applied),
        preprocess_mode=mode,
        preprocess_meta=noise,
        ocr_confidence=ocr_confidence.to_dict(),
    )

    primary = _dispatch(
        classification.image_type, work, ocr_out, png,
        page_number=page_number, filename=filename, result=result,
    )
    _record(result, primary)

    # ── 4. 双通道 + 融合 ───────────────────────────────────────────────────
    # 通道 A（结构/OCR）与通道 B（VLM）各自独立质检，再按图片类型融合。
    # 注意 chart/diagram/screenshot 的"专用引擎"本身就是 Vision —— 那种情况
    # 下 primary 就是通道 B，不能再调一次模型（白花钱且结果一样）。
    ocr_side, vlm_side = _build_channels(
        classification.image_type, route, primary, ocr_out, ocr_text, work, png,
        filename=filename,
    )
    channels = [c for c in (ocr_side, vlm_side) if c is not None]

    if len(channels) > 1:
        fusion = fuse_channels(classification.image_type, channels)
        result.fusion = fusion.to_dict()
        for note in fusion.notes:
            logger.info(
                "Image pipeline%s: 双通道融合 type=%s strategy=%s → %s (%s)",
                f" [{filename}]" if filename else "",
                classification.image_type, fusion.strategy, fusion.chosen, note,
            )
        chosen = fusion.chosen_channel()
        final_out = EngineOutput(
            text=fusion.text,
            confidence=fusion.confidence,
            engine=fusion.engine,
            ok=bool(fusion.text),
            lines=primary.lines or ocr_out.lines,
        )
        quality = chosen.quality if chosen is not None else _empty_quality()
    else:
        # 只有一个通道 → 退化为历史行为（单通道 + 质检）
        final_out = primary
        quality = ocr_side.quality if ocr_side is not None else _empty_quality()
        if vlm_side is not None:
            result.fusion = {
                "strategy": "single", "chosen": vlm_side.channel,
                "notes": ["仅 VLM 通道产出"],
            }

    result.quality = quality.to_dict()

    # ── 5. confidence → Accept / Fallback ─────────────────────────────────
    verdict = gate(final_out, classification.image_type, quality)

    if verdict.accepted:
        result.decision = DECISION_ACCEPT
        result.confidence = verdict.confidence
        result.analyze_engine = final_out.engine or route
        _apply_output(result, final_out)
    else:
        result.decision = DECISION_FALLBACK
        logger.info(
            "Image pipeline%s: type=%s engine=%s conf=%.2f → fallback (%s)",
            f" [{filename}]" if filename else "", classification.image_type,
            final_out.engine, verdict.confidence, verdict.reason,
        )
        _run_fallback(
            work, ocr_out, png, result,
            page_number=page_number, filename=filename, primary=final_out,
            # 通道 B 已经调过 VLM 了：它没产出可用结果，再用通用提示词试一次
            # 大概率还是不行，不如省下这次推理（几十张图的文档差别很大）。
            skip_vision=vlm_side is not None,
        )

    logger.info(
        "Image pipeline%s: type=%s route=%s mode=%s engine=%s conf=%.2f decision=%s "
        "quality=%.2f noise=%s",
        f" [{filename}]" if filename else "",
        result.image_type, result.route, mode, result.analyze_engine,
        result.confidence, result.decision, float(quality.score),
        noise.get("noise_sigma"),
    )
    return result


def _empty_quality() -> QualityReport:
    """没有任何通道产出时的空质检（分数 0，理由明确）."""
    return QualityReport(ok=False, score=0.0, reasons=["没有可评估的产出"])


def _build_channels(
    image_type: str,
    route: str,
    primary: EngineOutput,
    ocr_out: EngineOutput,
    ocr_text: str,
    work: Image.Image,
    png,
    *,
    filename: str,
) -> tuple[ChannelResult | None, ChannelResult | None]:
    """
    组装双通道.

    返回 ``(ocr_side, vlm_side)``，任一为 None 表示该通道没有产出。

    三种情形：

        route == vision          primary 就是 VLM → 通道 B = primary，
                                 通道 A = 那次 OCR 的文本（作为文字锚点）
        白名单类型 + VLM 可用     通道 A = primary，通道 B = 额外跑一次 VLM
        其余                     只有通道 A（单通道，不额外花 VLM 的钱）
    """
    # ── 情形 1：专用引擎本身就是 Vision ────────────────────────────────────
    if route == ROUTE_VISION:
        primary_quality = verify_output(
            primary.text, image_type, engine=primary.engine,
            ocr_lines=ocr_out.lines, ocr_text=ocr_text,
        )
        vlm_side = ChannelResult(CHANNEL_VLM, primary, primary_quality) if primary.text else None

        if not ocr_text:
            return None, vlm_side
        ocr_output = EngineOutput(
            text=ocr_text, confidence=float(ocr_out.confidence or 0.0),
            engine=ocr_out.engine or "ocr", ok=True, lines=ocr_out.lines,
        )
        ocr_quality = verify_output(
            ocr_text, image_type, engine=ocr_output.engine, ocr_lines=ocr_out.lines,
        )
        return ChannelResult(CHANNEL_OCR, ocr_output, ocr_quality), vlm_side

    # ── 情形 2/3：结构通道 = primary，VLM 通道按白名单决定要不要跑 ──────────
    primary_quality = verify_output(
        primary.text, image_type, engine=primary.engine,
        ocr_lines=ocr_out.lines, ocr_text=ocr_text,
    )
    ocr_side = ChannelResult(CHANNEL_OCR, primary, primary_quality) if primary.text else None

    if not dual_channel_enabled(image_type):
        return ocr_side, None
    return ocr_side, _run_vlm_channel(
        work, png, image_type, ocr_out, ocr_text, filename=filename,
    )


def _run_vlm_channel(
    image: Image.Image,
    png,
    image_type: str,
    ocr_out: EngineOutput,
    ocr_text: str,
    *,
    filename: str,
) -> ChannelResult | None:
    """
    通道 B：主动跑一次 VLM（**不是**"失败才兜底"）.

    这是本次改造的要点。原设计里 VLM 只在专用引擎置信度低时才登场，于是
    "OCR 自信地读错"这类失败永远轮不到模型纠正 —— 代码截图最典型：OCR 把
    ``!=`` 读成 ``=``、把缩进丢了，仍给出 0.9 的置信度，门控直接 Accept。

    现在按图片类型**主动**跑一遍，产出与通道 A 交叉校验（见 dual_channel）。
    """
    vision = VisionEngine()
    if not vision.is_available():
        return None
    try:
        out = vision.process(image, png_bytes=png(), image_type=image_type, role="primary")
    except Exception as exc:      # noqa: BLE001
        logger.warning("VLM channel failed for '%s': %s", filename, exc)
        return None
    if not out.ok or not (out.text or "").strip():
        return None
    quality = verify_output(
        out.text, image_type, engine=out.engine,
        ocr_lines=ocr_out.lines, ocr_text=ocr_text,
    )
    return ChannelResult(CHANNEL_VLM, out, quality)


# ─────────────────────────────────────────────────────────────────────────────
# OCR 与分派
# ─────────────────────────────────────────────────────────────────────────────

def _run_ocr(image: Image.Image) -> EngineOutput:
    """
    主 OCR：PaddleOCR 优先，Tesseract 兜底.

    直接走引擎层（而不是 ocr service）是为了拿到 **confidence** —— 门控需要它，
    而旧的 ``extract_text_with_lines`` 只返回文本。
    """
    registry = get_registry()
    for name in ("paddleocr", "tesseract"):
        engine = registry.get(name)
        if engine is None or not engine.is_available():
            continue
        out = engine.process(image)
        if out.ok and (out.text or "").strip():
            return out
    return EngineOutput(engine="none", ok=False, error="没有可用的 OCR 引擎")


def _dispatch(
    image_type: str,
    image: Image.Image,
    ocr_out: EngineOutput,
    png,
    *,
    page_number: int,
    filename: str,
    result: ImageUnderstanding,
) -> EngineOutput:
    """按图片类型选专用引擎."""
    settings = get_settings()
    registry = get_registry()

    if image_type == IMAGE_TYPE_TABLE:
        return _table_primary(image, ocr_out, page_number, result)

    if image_type == IMAGE_TYPE_CODE:
        engine = registry.get("ocr+code-parser")
        if engine is not None:
            out = engine.process(image, ocr_lines=ocr_out.lines)
            if out.ok:
                return out
        return EngineOutput(engine="ocr+code-parser", ok=False, error="代码解析失败")

    if image_type == IMAGE_TYPE_FORMULA:
        engine = registry.get("paddleocr-formula")
        if engine is not None and engine.is_available():
            out = engine.process(image)
            if out.ok:
                return out
        if getattr(settings, "ENABLE_IMAGE_OCR", True):
            # 没有公式引擎时把 OCR 文本作为弱产出交给门控 → 大概率触发兜底
            return EngineOutput(
                text=ocr_out.text, confidence=0.3,
                engine="ocr", ok=bool((ocr_out.text or "").strip()),
                lines=ocr_out.lines,
            )
        return EngineOutput(engine="paddleocr-formula", ok=False, error="公式引擎不可用")

    if image_type == IMAGE_TYPE_PHOTO:
        # 普通图片：那次 OCR 的结果本身就是最终产出
        return EngineOutput(
            text=ocr_out.text,
            confidence=ocr_out.confidence,
            engine=ocr_out.engine,
            ok=ocr_out.ok and bool((ocr_out.text or "").strip()),
            lines=ocr_out.lines,
            error=ocr_out.error,
        )

    # chart / diagram / screenshot → Vision（按类型提示词）
    vision = VisionEngine()
    if vision.is_available():
        out = vision.process(image, png_bytes=png(), image_type=image_type, role="primary")
        if out.ok:
            return out
        result.meta["vision_error"] = out.error
    # 记下"不是这张图有问题，是这台机器没有多模态模型"—— 图表/流程图的语义
    # （节点、连线方向、趋势）只能靠 Vision 拿到，OCR 只能捡回图里的文字。
    # 不写这个标记的话，下游只能看到一个笼统的 manual_review，无从判断
    # 该去补模型还是去改图。
    result.meta["vision_unavailable"] = not vision.is_available()
    return EngineOutput(engine="vision", ok=False, error="Vision 不可用")


def _table_primary(
    image: Image.Image, ocr_out: EngineOutput, page_number: int,
    result: ImageUnderstanding,
) -> EngineOutput:
    """
    表格专用引擎：Table Transformer 优先，规则表格兜底.

    Table Transformer 是"研究型方案"：对无边框表、跨行跨列更稳，但需要模型。
    规则方案（框线投影）零依赖且对有线表格又快又准。两者是**互补**关系，
    所以这里不是"二选一"而是"能上就上，上不了就退"。
    """
    settings = get_settings()
    registry = get_registry()
    ocr_fn = _cell_ocr_fn()

    if getattr(settings, "TABLE_TRANSFORMER_ENABLED", True):
        engine = registry.get("table-transformer")
        if engine is not None and engine.is_available():
            out = engine.process(image, ocr_fn=ocr_fn)
            if out.ok:
                result.meta["table_engine"] = "table-transformer"
                return out
            logger.info("Table Transformer produced nothing — falling back to rules")

    structure = recognize_table_safe(
        image, lines=ocr_out.lines, page_number=page_number, ocr_fn=ocr_fn,
    )
    result.table = structure
    if structure.ok:
        return EngineOutput(
            text=structure.markdown,
            confidence=0.85 if structure.method == "rules+cell-ocr" else 0.65,
            engine=f"table_parser:{structure.method}",
            ok=True,
            lines=ocr_out.lines,
            meta={"table_method": structure.method},
        )
    return EngineOutput(
        engine="table-parser", ok=False,
        error=str(structure.meta.get("reason") or "结构还原失败"),
        meta={"table_method": "failed"},
    )


def _cell_ocr_fn():
    """给表格用的"单格 OCR"回调：小格先放大再识别（与 TT 路径一致）."""
    registry = get_registry()
    engine = registry.get("paddleocr")
    if engine is None or not engine.is_available():
        engine = registry.get("tesseract")
    if engine is None or not engine.is_available():
        return None

    def _run(cell_image):
        try:
            if cell_image.height < 32:
                scale = max(2, min(4, 32 // max(cell_image.height, 1)))
                cell_image = cell_image.resize(
                    (cell_image.width * scale, cell_image.height * scale), Image.BICUBIC
                )
            out = engine.process(cell_image)
            return (out.text or "").replace("\n", " ").strip()
        except Exception:      # noqa: BLE001
            return ""

    return _run


# ─────────────────────────────────────────────────────────────────────────────
# Fallback → Validation → Pass / Manual Review
# ─────────────────────────────────────────────────────────────────────────────

def _run_fallback(
    image: Image.Image,
    ocr_out: EngineOutput,
    png,
    result: ImageUnderstanding,
    *,
    page_number: int,
    filename: str,
    primary: EngineOutput,
    skip_vision: bool = False,
) -> None:
    """
    兜底：先 Vision LLM，再 Second OCR；然后 Validation 决定 Pass / Manual Review.

    顺序是有讲究的：Vision 能"读懂图意"（表格被还原、公式被转写），上限更高；
    但它需要模型。Second OCR 换一个引擎重读，解决了"引擎 A 系统性读错"这类
    问题，但解决不了"这张图根本不是文字"。

    :param skip_vision: 双通道阶段已经调过 VLM 且没拿到可用产出时置 True ——
        再用通用提示词试一次大概率还是不行，省下这次推理。
    """
    settings = get_settings()
    image_type = result.image_type
    candidates: list[EngineOutput] = []

    # ── 路径 0：主通道已有产出（只是没过 Accept 门）─────────────────────────
    # 之前这里是空的：primary 被门控拒了就直接进兜底，兜底若也没产出就
    # "best is None" → 保留三五个字的 OCR 标签，把主通道真正读懂的描述
    # （如 VLM 写的"input→Fan→output 的数据流"）整段丢掉。门控严格不等于
    # 内容没有价值 —— 这段产出必须留在候选里，让下面的校验给它一个公平的
    # 分数（大概率仍判 manual_review，但内容保住了）。
    if (primary.text or "").strip():
        candidates.append(primary)
        result.meta["primary_kept_as_candidate"] = True

    # ── 路径 A：Vision LLM（通用转写提示词）────────────────────────────────
    vision = VisionEngine()
    if not skip_vision and vision.is_available():
        out = vision.process(image, png_bytes=png(), image_type=image_type, role="fallback")
        if out.ok:
            candidates.append(out)
            _record(result, out)

    # ── 路径 B：Second OCR（换引擎重读）──────────────────────────────────
    second = _second_ocr(image, ocr_out)
    if second is not None:
        candidates.append(second)
        _record(result, second)

    # ── 表格额外保留"规则兜底"：TV 失败时规则往往仍能救回来 ─────────────────
    if image_type == IMAGE_TYPE_TABLE and result.table is None:
        structure = recognize_table_safe(
            image, lines=ocr_out.lines, page_number=page_number,
            ocr_fn=_cell_ocr_fn(),
        )
        result.table = structure
        if structure.ok and structure.markdown:
            rules_out = EngineOutput(
                text=structure.markdown, confidence=0.7,
                engine=f"table_parser:{structure.method}", ok=True,
            )
            candidates.append(rules_out)
            _record(result, rules_out)

    # ── Validation：结构校验 × 引擎自评 × 质检 → Pass / Manual Review ────────
    # 三个因子相乘：结构形态（表格对齐/代码围栏）、引擎自评、以及**可验证的
    # 事实**（代码能不能解析、OCR 行置信度够不够）。只靠前两个的话，"形态对
    # 但内容错"的产出（语法不成立的代码）会一路 Pass 到 RAG 里。
    best: tuple[float, EngineOutput, object, QualityReport] | None = None
    for out in candidates:
        check = validate(image_type, out.text)
        quality = verify_output(
            out.text, image_type, engine=out.engine,
            ocr_lines=ocr_out.lines, ocr_text=result.ocr_text,
        )
        score = round(final_confidence(out.confidence, check) * quality.score, 4)
        if best is None or score > best[0]:
            best = (score, out, check, quality)

    if best is None:
        # 连兜底都没产出 → 保留 OCR 文本，标记人工复核
        result.decision = DECISION_MANUAL
        result.manual_review = True
        result.confidence = 0.0
        result.analyze_engine = "ocr-degraded"
        result.meta["fallback_failed"] = True
        _keep_ocr_text(result, ocr_out)
        return

    score, chosen, check, quality = best
    decision = decide_after_fallback(chosen.confidence, check)
    # 质检不过时把 Pass 降级为 Manual Review：语法/置信度这类硬事实比
    # "结构形态看起来对"更值得采信。
    if decision == DECISION_PASS and not quality.ok:
        decision = DECISION_MANUAL
    result.decision = decision
    result.confidence = score
    result.analyze_engine = chosen.engine
    result.quality = quality.to_dict()
    _apply_output(result, chosen)
    result.meta["validation"] = {"passed": check.passed, "score": check.score, "reasons": check.reasons}

    if decision == DECISION_MANUAL:
        result.manual_review = True
        # 校验不合格时仍保留 OCR 文本兜底，避免"图彻底变空"
        _keep_ocr_text(result, ocr_out)
        if result.meta.get("vision_unavailable"):
            # 把根因写进质检理由，前端/排查时能直接看到"是缺模型，不是图坏了"
            result.quality.setdefault("reasons", []).append(
                "Vision 不可用（未安装多模态模型）：图表/流程图的图意无法解读，"
                "已退化为纯 OCR，仅保留图内文字"
            )
        logger.info(
            "Image pipeline%s: type=%s 兜底后校验未通过 → 人工复核 (score=%.2f, %s)",
            f" [{filename}]" if filename else "", image_type, score, check.reasons,
        )


def _second_ocr(image: Image.Image, ocr_out: EngineOutput) -> EngineOutput | None:
    """
    换一个 OCR 引擎重读（跳过已经用过的那个）.

    对**深色底 / 混合极性**的图再补一次"反色重读"：OCR 引擎的识别模型都在
    "深字浅底"上训练，深底浅字的图（深色主题流程图截图、终端窗口）直接读
    通常大面积漏检 —— Tesseract 甚至直接返回空串。反色后纸与字换到模型熟悉
    的那一侧，同一张图往往就能读出字来。这里取"字数更多"的那次结果，避免
    反色把本来读得好的图弄坏。

    只在兜底路径上多跑这几次 OCR，而兜底本身就是"已经出问题了"的分支，
    相对后面可能触发的 VLM 推理，这点成本可以忽略。
    """
    registry = get_registry()
    used = (ocr_out.engine or "").lower()

    variants: list[tuple[str, Image.Image]] = [("", image)]
    try:
        polarity = _page_polarity_of(image)
        if polarity != "light":
            variants.append(("inverted", _invert_for_ocr(image)))
    except Exception as exc:      # noqa: BLE001
        logger.debug("Polarity probe for second OCR skipped: %s", exc)

    best: EngineOutput | None = None
    for suffix, variant in variants:
        for name in ("tesseract", "paddleocr"):
            if name in used:
                continue
            engine = registry.get(name)
            if engine is None or not engine.is_available():
                continue
            out = engine.process(variant)
            if not out.ok or not (out.text or "").strip():
                continue
            if suffix and out.engine:
                out.engine = f"{out.engine}+{suffix}"
            if best is None or len(out.text or "") > len(best.text or ""):
                best = out
    if best is not None:
        logger.info(
            "Second OCR produced %d chars via %s", len(best.text or ""), best.engine
        )
    return best


def _page_polarity_of(image: Image.Image) -> str:
    """读页面极性（容错包装：探测失败一律当"浅底"，即不做反色重读）."""
    from app.services.image_understanding.preprocess import page_polarity

    return page_polarity(image)


def _invert_for_ocr(image: Image.Image) -> Image.Image:
    from app.services.image_understanding.preprocess import invert_for_ocr

    return invert_for_ocr(image)


def _apply_output(result: ImageUnderstanding, out: EngineOutput) -> None:
    """把引擎产出落到统一的结构化字段上."""
    text = (out.text or "").strip()
    if not text:
        return
    if result.image_type == IMAGE_TYPE_TABLE:
        result.structured_content = text
    elif result.image_type in (IMAGE_TYPE_CODE, IMAGE_TYPE_FORMULA):
        result.structured_content = text
    else:
        result.vision_caption = text


def _keep_ocr_text(result: ImageUnderstanding, ocr_out: EngineOutput) -> None:
    """人工复核/彻底失败时，至少把 OCR 文本留下来（不让图变空）."""
    text = (ocr_out.text or "").strip()
    if not text or result.structured_content or result.vision_caption:
        return
    if result.image_type == IMAGE_TYPE_TABLE:
        result.structured_content = None
    else:
        result.vision_caption = text


def _record(result: ImageUnderstanding, out: EngineOutput) -> None:
    """记录一次引擎尝试（排查"这张图到底走了哪条路"）."""
    result.attempts.append({
        "engine": out.engine,
        "ok": bool(out.ok),
        "confidence": round(float(out.confidence or 0.0), 4),
        "chars": len(out.text or ""),
        "error": out.error,
    })


def fallback_understanding(img: Image.Image, *, reason: str = "") -> ImageUnderstanding:
    """整条理解链路异常时的保守兜底：当普通图片处理（只走 OCR）."""
    from app.services.image_understanding.classifier import ImageClassification

    return ImageUnderstanding(
        image_type=IMAGE_TYPE_PHOTO,
        route=ROUTE_OCR,
        classification=ImageClassification(
            image_type=IMAGE_TYPE_PHOTO, confidence=0.0,
            signals={"fallback": reason}, engine="fallback",
        ),
        analyze_engine="ocr",
        meta={"fallback": reason},
    )


__all__ = [
    "ImageUnderstanding",
    "understand_image",
    "resolve_route",
    "fallback_understanding",
    "ROUTE_TABLE",
    "ROUTE_FORMULA",
    "ROUTE_CODE",
    "ROUTE_VISION",
    "ROUTE_OCR",
    # 便于测试 / 复用
    "DECISION_ACCEPT",
    "DECISION_FALLBACK",
    "DECISION_PASS",
    "DECISION_MANUAL",
    "CHANNEL_OCR",
    "CHANNEL_VLM",
]
