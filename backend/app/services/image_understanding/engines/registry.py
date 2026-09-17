"""
引擎注册表 —— "有哪些引擎、现在能不能用"的唯一事实来源.

注册表**只**管登记与探测，不管决策。"哪种图片用哪个引擎、置信度低怎么兜底"
是 :mod:`app.services.image_understanding.confidence` 与 ``pipeline`` 的事。

可用性探测全程无副作用（只查 import / 环境变量），因此可以安全地在健康检查、
启动日志、管理接口里反复调用。
"""

from __future__ import annotations

from app.services.image_understanding.engines.base import Engine, EngineKind
from app.utils.logging import get_logger

logger = get_logger(__name__)


class EngineRegistry:
    """名称 → 引擎实例。同族可以有多个（OCR 就有 PaddleOCR 和 Tesseract）。"""

    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {}

    def register(self, engine: Engine) -> Engine:
        self._engines[engine.name] = engine
        return engine

    def get(self, name: str | None) -> Engine | None:
        if not name:
            return None
        return self._engines.get(name)

    def by_kind(self, kind: EngineKind) -> list[Engine]:
        return [e for e in self._engines.values() if e.kind == kind]

    def names(self) -> list[str]:
        return sorted(self._engines)

    def report(self) -> list[dict]:
        """每个引擎的可用性与原因（日志 / 健康检查用）."""
        return [e.describe() for e in sorted(self._engines.values(), key=lambda x: x.name)]

    def log_summary(self) -> None:
        """启动/首次使用时打一条汇总，方便一眼看出"现在到底有哪些能力"。"""
        lines = []
        for item in self.report():
            mark = "OK  " if item["available"] else "MISS"
            reason = f"  ← {item['reason']}" if item["reason"] else ""
            lines.append(f"    [{mark}] {item['name']:<18} ({item['kind']}){reason}")
        logger.info("Image engines:\n%s", "\n".join(lines))


def build_default_registry() -> EngineRegistry:
    """
    组装默认引擎表.

    设计稿里的七个引擎一一对应：

        Docling            → BASE_PARSE
        PP-StructureV3     → LAYOUT
        PaddleOCR          → OCR
        Table Transformer  → TABLE
        OCR + Code Parser  → CODE
        PaddleOCR Formula  → FORMULA
        Vision             → VISION

    外加 Tesseract 作为 "Second OCR" 兜底（流程图的 Fallback 分支）。
    """
    from app.services.image_understanding.engines.code_parser import CodeParserEngine
    from app.services.image_understanding.engines.docling_engine import DoclingEngine
    from app.services.image_understanding.engines.paddle_engines import (
        PaddleFormulaEngine,
        PaddleOCREngine,
        PPStructureEngine,
    )
    from app.services.image_understanding.engines.table_transformer import (
        TableTransformerEngine,
    )
    from app.services.image_understanding.engines.tesseract_engine import TesseractEngine
    from app.services.image_understanding.engines.vision_engine import VisionEngine

    registry = EngineRegistry()
    for engine in (
        DoclingEngine(),
        PPStructureEngine(),
        PaddleOCREngine(),
        TableTransformerEngine(),
        CodeParserEngine(),
        PaddleFormulaEngine(),
        VisionEngine(),
        TesseractEngine(),
    ):
        registry.register(engine)
    return registry


_registry: EngineRegistry | None = None


def get_registry() -> EngineRegistry:
    """进程级单例注册表（首次调用时组装）."""
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def engine_report() -> list[dict]:
    """便捷函数：当前环境的引擎可用性报告."""
    return get_registry().report()


__all__ = [
    "EngineRegistry",
    "build_default_registry",
    "get_registry",
    "engine_report",
]
