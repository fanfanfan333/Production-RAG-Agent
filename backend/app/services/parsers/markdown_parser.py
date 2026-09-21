from app.services.parsers.base import (
    DocumentParser,
    ExtractionResult,
    ExtractedPage,
    decode_text_with_fallback,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

class MarkdownParser(DocumentParser):
    def parse(self, content: bytes, filename: str, *, document_id: str | None = None, tenant_id: str | None = None) -> ExtractionResult:
        # 编码回退链（utf-8-sig → gb18030 → big5 → utf-8+replace），
        # 非 UTF-8 中文不再静默变 U+FFFD 入库（见 base 模块）。
        raw_text = decode_text_with_fallback(content, filename=filename)
        full_text = raw_text.strip()

        if not full_text:
            raise ValueError(f"Markdown '{filename}' contains no extractable text.")
            
        page = ExtractedPage(
            page_number=1,
            text=full_text,
            char_start=0,
            char_end=len(full_text)
        )
        
        return ExtractionResult(
            pages=[page],
            full_text=full_text,
            page_count=1,
            char_count=len(full_text),
            file_type="markdown",
            parser_used="markdown",
            ocr_used=False,
            ocr_engine=None,
            extraction_method="native"
        )
