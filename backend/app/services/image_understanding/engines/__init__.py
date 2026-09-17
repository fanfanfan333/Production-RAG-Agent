"""
引擎适配层（Image Understanding Engines）.

把"用哪个模型/工具"封装成统一契约的引擎，管线只调度、不关心实现。

    ┌────────────────────┬──────────────────────┬──────────────────────────┐
    │ 引擎                │ 族                    │ 负责                     │
    ├────────────────────┼──────────────────────┼──────────────────────────┤
    │ Docling            │ BASE_PARSE           │ PDF/DOCX 正文与结构解析   │
    │ PP-Structure(V3/V2)│ LAYOUT               │ 版面检测（区块划分）       │
    │ PaddleOCR          │ OCR                  │ 通用 OCR（中英）          │
    │ Table Transformer  │ TABLE                │ 表格结构（研究型方案）     │
    │ OCR + Code Parser  │ CODE                 │ 代码截图 → 可读代码块      │
    │ PaddleOCR Formula  │ FORMULA              │ 公式 → LaTeX             │
    │ Vision             │ VISION               │ 流程图/架构图/图片描述     │
    │ Tesseract          │ OCR（第二引擎）        │ Fallback 的第二 OCR      │
    └────────────────────┴──────────────────────┴──────────────────────────┘

    >>> from app.services.image_understanding.engines import engine_report
    >>> for item in engine_report():
    ...     print(item["name"], item["available"])
"""

from app.services.image_understanding.engines.base import (
    DocumentEngine,
    Engine,
    EngineKind,
    EngineOutput,
    ImageEngine,
)
from app.services.image_understanding.engines.code_parser import (
    CodeParserEngine,
    code_likeness,
    detect_language,
    rebuild_indentation,
)
from app.services.image_understanding.engines.docling_engine import DoclingEngine
from app.services.image_understanding.engines.paddle_engines import (
    PaddleFormulaEngine,
    PaddleOCREngine,
    PPStructureEngine,
)
from app.services.image_understanding.engines.registry import (
    EngineRegistry,
    build_default_registry,
    engine_report,
    get_registry,
)
from app.services.image_understanding.engines.table_transformer import (
    TableTransformerEngine,
)
from app.services.image_understanding.engines.tesseract_engine import TesseractEngine
from app.services.image_understanding.engines.vision_engine import VisionEngine

__all__ = [
    # base
    "Engine",
    "EngineKind",
    "EngineOutput",
    "ImageEngine",
    "DocumentEngine",
    # registry
    "EngineRegistry",
    "build_default_registry",
    "get_registry",
    "engine_report",
    # engines
    "DoclingEngine",
    "PPStructureEngine",
    "PaddleOCREngine",
    "TableTransformerEngine",
    "CodeParserEngine",
    "PaddleFormulaEngine",
    "VisionEngine",
    "TesseractEngine",
    # code parser helpers
    "detect_language",
    "code_likeness",
    "rebuild_indentation",
]
