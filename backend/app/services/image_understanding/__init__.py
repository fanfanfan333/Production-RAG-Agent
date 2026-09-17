"""
Image Understanding（置信度门控的多引擎图片处理）.

文档里的图片不再"一律 OCR"，而是先按噪声情况预处理、判类型，再走 **OCR /
VLM 双通道**，融合后做质量校验，最后用置信度决定是直接采用还是走兜底：

    Image → Preprocessing（去噪 / 对比度 / 去边框 / 纠偏 / 二值化）
              │
              ▼
            OCR（唯一一次，带逐行置信度）
              │
              ▼
        Picture Classification
              │
              ▼
        ┌─ 通道 A：Specialized Engine ─┐
        └─ 通道 B：VLM（白名单类型）    ┘
              │
              ▼
        Fusion（互补 / 视觉优先 / OCR 优先）
              │
              ▼
        Quality（代码语法 / OCR confidence / VLM 幻觉）
              │
              ▼
        confidence ──高──► Accept ──► RAG
              │
              低
              ▼
        Fallback (Vision / Second OCR)
              │
              ▼
        Validation → confidence
              │
    ┌── Pass ──► RAG
    │
  Failed
    ▼
  Manual Review

模块划分：

    engines/          引擎适配层（Docling / PP-Structure / PaddleOCR /
                      Table Transformer / Code Parser / Formula / Vision）
    classifier.py     图片类型判断（规则，不依赖模型）
    preprocess.py     预处理 + **噪声处理**（噪声估计 / 自适应去噪 / 对比度 /
                      去边框 / 摩尔纹抑制 / 纠偏 / 二值化）
    dual_channel.py   **OCR / VLM 双通道**与按类型的融合策略
    quality.py        **产出质量校验**（代码语法 / OCR confidence / VLM 幻觉）
    confidence.py     置信度门控与结构校验（Accept/Fallback/Pass/Manual）
    pipeline.py       调度：把上面几块串成流程图
    table_recognizer.py 规则表格识别（Table Transformer 的零依赖兜底）
    analyzer.py       Vision 提示词与调用
    structured_content.py 结构化内容对象与 Qdrant payload 映射
"""

from app.services.image_understanding.analyzer import (
    FALLBACK_PROMPT,
    analyze_image_sync,
    prompt_for,
)
from app.services.image_understanding.classifier import (
    ImageClassification,
    classify_image,
    classify_image_safe,
    compute_signals,
    vision_analyze_types,
)
from app.services.image_understanding.confidence import (
    DECISION_ACCEPT,
    DECISION_FALLBACK,
    DECISION_MANUAL,
    DECISION_PASS,
    Gate,
    Validation,
    apply_quality,
    decide_after_fallback,
    final_confidence,
    gate,
    validate,
)
from app.services.image_understanding.dual_channel import (
    CHANNEL_OCR,
    CHANNEL_VLM,
    STRATEGY_COMPLEMENTARY,
    STRATEGY_OCR_FIRST,
    STRATEGY_VISION_FIRST,
    ChannelResult,
    FusionResult,
    dual_channel_enabled,
    dual_channel_types,
    fuse_channels,
    merge_captions,
    strategy_for,
)
from app.services.image_understanding.engines import (
    EngineKind,
    EngineOutput,
    engine_report,
    get_registry,
)
from app.services.image_understanding.pipeline import (
    ROUTE_CODE,
    ROUTE_FORMULA,
    ROUTE_OCR,
    ROUTE_TABLE,
    ROUTE_VISION,
    ImageUnderstanding,
    fallback_understanding,
    resolve_route,
    understand_image,
)
from app.services.image_understanding.preprocess import (
    NOISE_SIGMA_STRONG,
    NOISE_SIGMA_THRESHOLD,
    PreprocessResult,
    border_trim,
    contrast_span,
    estimate_noise,
    invert_for_ocr,
    noise_signals,
    page_polarity,
    pick_mode,
    preprocessing_mode_for,
    preprocess,
)
from app.services.image_understanding.quality import (
    CodeSyntaxReport,
    OcrConfidenceReport,
    QualityReport,
    assess_ocr_confidence,
    check_code_syntax,
    check_vlm_output,
    verify_output,
)
from app.services.image_understanding.structured_content import (
    ALL_IMAGE_TYPES,
    DEFAULT_VISION_TYPES,
    IMAGE_TYPE_CHART,
    IMAGE_TYPE_CODE,
    IMAGE_TYPE_DIAGRAM,
    IMAGE_TYPE_FORMULA,
    IMAGE_TYPE_LABELS,
    IMAGE_TYPE_PHOTO,
    IMAGE_TYPE_SCREENSHOT,
    IMAGE_TYPE_TABLE,
    StructuredContent,
    content_type_for,
)
from app.services.image_understanding.table_recognizer import (
    TableStructure,
    detect_rules,
    recognize_table,
    recognize_table_safe,
    to_markdown,
)

__all__ = [
    # classifier
    "ImageClassification",
    "classify_image",
    "classify_image_safe",
    "compute_signals",
    "vision_analyze_types",
    # table
    "TableStructure",
    "recognize_table",
    "recognize_table_safe",
    "detect_rules",
    "to_markdown",
    # analyzer
    "analyze_image_sync",
    "prompt_for",
    "FALLBACK_PROMPT",
    # pipeline
    "ImageUnderstanding",
    "understand_image",
    "resolve_route",
    "fallback_understanding",
    "ROUTE_TABLE",
    "ROUTE_FORMULA",
    "ROUTE_CODE",
    "ROUTE_VISION",
    "ROUTE_OCR",
    # preprocess / 噪声处理
    "PreprocessResult",
    "preprocess",
    "preprocessing_mode_for",
    "pick_mode",
    "noise_signals",
    "estimate_noise",
    "contrast_span",
    "border_trim",
    "page_polarity",
    "invert_for_ocr",
    "NOISE_SIGMA_THRESHOLD",
    "NOISE_SIGMA_STRONG",
    # dual channel
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
    # quality
    "CodeSyntaxReport",
    "OcrConfidenceReport",
    "QualityReport",
    "check_code_syntax",
    "check_vlm_output",
    "assess_ocr_confidence",
    "verify_output",
    # confidence
    "Gate",
    "Validation",
    "gate",
    "validate",
    "final_confidence",
    "decide_after_fallback",
    "apply_quality",
    "DECISION_ACCEPT",
    "DECISION_FALLBACK",
    "DECISION_PASS",
    "DECISION_MANUAL",
    # engines
    "EngineKind",
    "EngineOutput",
    "get_registry",
    "engine_report",
    # content
    "StructuredContent",
    "content_type_for",
    "IMAGE_TYPE_TABLE",
    "IMAGE_TYPE_FORMULA",
    "IMAGE_TYPE_CODE",
    "IMAGE_TYPE_CHART",
    "IMAGE_TYPE_DIAGRAM",
    "IMAGE_TYPE_SCREENSHOT",
    "IMAGE_TYPE_PHOTO",
    "IMAGE_TYPE_LABELS",
    "ALL_IMAGE_TYPES",
    "DEFAULT_VISION_TYPES",
]
