from app.services.parsers.base import (
    DocumentParser,
    ExtractionResult,
    ExtractedPage,
    decode_text_with_fallback,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

class TxtParser(DocumentParser):
    def parse(self, content: bytes, filename: str, *, document_id: str | None = None, tenant_id: str | None = None) -> ExtractionResult:
        # 编码回退链（utf-8-sig → gb18030 → big5 → utf-8+replace），
        # 命中非 UTF-8 时打 warning，乱码占比过高直接拒收（见 base 模块）。
        full_text = decode_text_with_fallback(content, filename=filename).strip()
        if not full_text:
            raise ValueError(f"TXT '{filename}' contains no extractable text.")
            
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
            file_type=self._file_type_for(filename),
            parser_used="utf8",
            ocr_used=False,
            ocr_engine=None,
            extraction_method="native"
        )

    @staticmethod
    def _file_type_for(filename: str) -> str:
        """按真实扩展名记录 file_type（.json / .log 不再被记成 'txt'，见 B12）."""
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        return ext or "txt"
