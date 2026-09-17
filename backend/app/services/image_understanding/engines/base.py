"""
引擎适配层的统一契约.

设计目标：把"用什么引擎"从"怎么调度"里彻底剥离开。管线（pipeline）只认
``EngineOutput`` 这一个返回形状，具体是 PaddleOCR、Table Transformer 还是
Vision，对上层完全透明 —— 换引擎 = 改配置，不改调度逻辑。

    ┌─ EngineKind ────────────────────────────────────────────────────────┐
    │  BASE_PARSE  整篇 PDF/DOCX 基础解析        → Docling                 │
    │  LAYOUT      版面检测（文字/表格/图分区）   → PP-StructureV3          │
    │  OCR         通用文字识别                  → PaddleOCR / Tesseract   │
    │  TABLE       表格结构识别（研究型方案）      → Table Transformer       │
    │  CODE        代码截图 → 代码               → OCR + Code Parser        │
    │  FORMULA     公式 → LaTeX                  → PaddleOCR Formula        │
    │  VISION      流程图/架构图/图片描述          → 多模态模型              │
    └─────────────────────────────────────────────────────────────────────┘

两种基类：

* :class:`ImageEngine` —— 输入单张图，产出文本/结构（LAYOUT/OCR/TABLE/CODE/
  FORMULA/VISION）。
* :class:`DocumentEngine` —— 输入整篇文档字节，产出正文（BASE_PARSE/Docling）。

**可用性探测必须无副作用**：``is_available()`` 只能做 import 探测与缓存，
绝不能触发模型下载或真正推理 —— 否则启动期就会被拖住，甚至被 segfault
（PaddleOCR 在本环境的历史教训）带走整个进程。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum

from PIL import Image


class EngineKind(str, Enum):
    """引擎族 —— 决定它接哪一类输入、产出什么。"""

    BASE_PARSE = "base_parse"
    LAYOUT = "layout"
    OCR = "ocr"
    TABLE = "table"
    CODE = "code"
    FORMULA = "formula"
    VISION = "vision"


@dataclass
class EngineOutput:
    """
    所有引擎的统一产出.

    ``confidence`` 是整条"置信度门控"链路的燃料：管线用它在
    "Accept / Fallback"、"Pass / Manual Review" 之间做决策。取值 [0, 1]，
    0 表示"没跑出来"，1 表示"高置信"。

    约定：
        * OCR 引擎 → 各文本行置信度的加权均值（行越长权重越高）。
        * 结构化引擎（表格/公式/代码）→ 由结构线索折算（网格完整度、语法
          可解析比例、符号密度等），见 confidence.py。
        * Vision → 有产出即给一个中性偏高值（模型没有逐 token 置信度）。
    """

    text: str = ""
    confidence: float = 0.0
    engine: str = ""
    ok: bool = True
    error: str | None = None
    # 带坐标的文字行（OCRLine），表格结构识别与版面分析会复用
    lines: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.text or "").strip()

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "ok": self.ok,
            "confidence": round(float(self.confidence), 4),
            "chars": len(self.text or ""),
            "lines": len(self.lines or []),
            "error": self.error,
        }


class Engine(abc.ABC):
    """引擎公共部分（名称 + 可用性探测）."""

    #: 人类可读的引擎名（会写进 chunk metadata，前端展示）
    name: str = "engine"
    #: 引擎族
    kind: EngineKind = EngineKind.OCR

    @abc.abstractmethod
    def is_available(self) -> bool:
        """
        当前环境是否可用.

        实现**必须**轻量、可缓存、绝无副作用：只做 import 探测 / 环境变量
        检查。不允许在这里下载模型或跑推理。
        """

    def unavailable_reason(self) -> str:
        """不可用原因（日志用），默认空串。"""
        return ""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "available": self.is_available(),
            "reason": self.unavailable_reason(),
        }


class ImageEngine(Engine):
    """处理单张图片的引擎."""

    @abc.abstractmethod
    def process(self, image: Image.Image, **kwargs) -> EngineOutput:
        """对一张图跑推理，返回统一产出."""


class DocumentEngine(Engine):
    """处理整篇文档的引擎（Docling）."""

    @abc.abstractmethod
    def parse(self, content: bytes, filename: str, **kwargs):
        """
        解析整篇文档.

        返回 :class:`~app.services.parsers.base.ExtractionResult`；无法处理时
        抛异常，由调用方决定回退到哪个解析器。
        """


__all__ = [
    "EngineKind",
    "EngineOutput",
    "Engine",
    "ImageEngine",
    "DocumentEngine",
]
