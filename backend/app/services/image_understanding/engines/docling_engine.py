"""
Docling 基础解析引擎（PDF / DOCX 的正文与结构解析）.

对应设计稿"PDF/DOCX 基础解析 → Docling"。Docling 负责**正文层**：版面感知的
阅读顺序、标题层级、列表、原生表格 → 统一导出 Markdown。

注意与图片管线的分工：Docling **不**接管图片理解。文档里的图片仍然走
"解析 → 图片分类 → 分流"那三层（见 ``image_understanding/``）——因为那才是
表格图片、公式、代码截图能被正确还原的地方。这里只把正文解析换掉。

        PDF / DOCX bytes
               │
        ┌──────┴───────┐
        │ Docling      │  正文 + 结构 → Markdown   ← 本模块
        └──────┬───────┘
               │
        ┌──────┴───────┐
        │ 图片三层管线   │  table / formula / code / chart / photo
        └──────────────┘

Docling 不可用时调用方（pdf_parser / docx_parser）自动回退到 PyMuPDF /
python-docx，功能不缺失，只是正文质量略降。
"""

from __future__ import annotations

import io
import threading

from app.services.image_understanding.engines.base import DocumentEngine, EngineKind
from app.utils.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_converter = None
_shim_applied = False


def _apply_pipeline_hash_shim() -> None:
    """
    给 Docling 的 pipeline 缓存哈希打一个**保守的**兼容补丁.

    上游 bug：``docling/utils/pipeline_cache.py`` 用
    ``pipeline_options.model_dump_json(serialize_as_any=True)`` 生成缓存键，在某些
    pydantic 版本上会因为选项对象里的 serializer lambda 触发

        PydanticSerializationError: Circular reference detected (id repeated)

    缓存键本身只是"同一套选项复用同一条 pipeline"的优化 —— 它挂掉不该让整篇
    文档解析失败。因此这里把该函数包一层：**先走原实现**，只有原实现抛异常时
    才退回一个基于 ``repr`` 的等价哈希。行为完全一致，只是哈希稳定性略降。

    只改这一个函数、只在本引擎首次使用时生效，且不改变任何解析结果。
    """
    global _shim_applied
    if _shim_applied:
        return

    # 仅当 numpy/docling 可用时才导入（is_available 已保证）
    import hashlib

    from docling.datamodel.pipeline_options import PipelineOptions
    from docling.utils import pipeline_cache

    original = pipeline_cache.create_pipeline_options_hash

    def _safe_hash(pipeline_options: PipelineOptions) -> str:
        try:
            return original(pipeline_options)
        except Exception:      # noqa: BLE001
            # repr 里含全部字段值，作为缓存键足够稳定
            payload = f"{type(pipeline_options).__qualname__}|{pipeline_options!r}"
            return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()

    pipeline_cache.create_pipeline_options_hash = _safe_hash

    # document_converter 在模块顶层 import 了该函数，需同步替换其命名空间里的引用
    try:
        from docling import document_converter

        if getattr(document_converter, "create_pipeline_options_hash", None) is original:
            document_converter.create_pipeline_options_hash = _safe_hash
    except Exception:      # noqa: BLE001
        pass

    _shim_applied = True
    logger.info("Docling pipeline-hash compatibility shim applied")


class DoclingEngine(DocumentEngine):
    """Docling 文档转换器包装（懒加载 + 进程级单例）."""

    name = "docling"
    kind = EngineKind.BASE_PARSE

    def is_available(self) -> bool:
        try:
            import docling  # noqa: F401  # noqa: PLC0415

            return True
        except Exception:      # noqa: BLE001
            return False

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "未安装 docling（pip install docling）"

    def _get_converter(self):
        global _converter
        if _converter is not None:
            return _converter
        with _lock:
            if _converter is None:
                from docling.document_converter import DocumentConverter  # noqa: PLC0415

                _apply_pipeline_hash_shim()
                _converter = DocumentConverter()
                logger.info("Docling DocumentConverter initialized")
        return _converter

    def parse(self, content: bytes, filename: str, **kwargs):
        """
        bytes → :class:`ExtractionResult`.

        解析失败时抛异常，由调用方回退到原生解析器（**不要**在这里吞掉异常
        返回半成品 —— 半成品正文比解析失败更难排查）。
        """
        from docling.datamodel.base_models import DocumentStream
        from app.services.parsers.base import ExtractedPage, ExtractionResult

        converter = self._get_converter()
        stream = DocumentStream(name=filename, stream=io.BytesIO(content))
        result = converter.convert(stream)
        document = result.document

        markdown = document.export_to_markdown() or ""
        markdown = markdown.strip()
        if not markdown:
            raise ValueError(f"Docling produced empty output for '{filename}'")

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
        page = ExtractedPage(
            page_number=1, text=markdown, char_start=0, char_end=len(markdown)
        )
        return ExtractionResult(
            pages=[page],
            full_text=markdown,
            page_count=1,
            char_count=len(markdown),
            file_type=ext,
            parser_used="docling",
            ocr_used=False,
            extraction_method="docling",
        )


__all__ = ["DoclingEngine"]
