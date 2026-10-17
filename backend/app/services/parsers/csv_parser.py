import io
import pandas as pd
from app.services.parsers.base import DocumentParser, ExtractionResult, ExtractedPage
from app.utils.logging import get_logger

logger = get_logger(__name__)

class CsvParser(DocumentParser):
    def parse(self, content: bytes, filename: str, *, document_id: str | None = None, tenant_id: str | None = None) -> ExtractionResult:
        try:
            # 保留标识符前导零、空单元格和列边界，不能把 CSV 压成对齐空白文本。
            df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
            if df.empty:
                raise ValueError("CSV has no data rows")
            def row_text(values):
                return "| " + " | ".join(str(value).replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>") for value in values) + " |"
            full_text = "\n".join([
                row_text(df.columns), row_text(["---"] * len(df.columns)),
                *(row_text(row) for row in df.itertuples(index=False, name=None)),
            ])
        except Exception as exc:
            raise ValueError(f"Cannot open CSV '{filename}': {exc}") from exc

        if not full_text or full_text.strip() == "Empty DataFrame\nColumns: []\nIndex: []":
            raise ValueError(f"CSV '{filename}' contains no extractable text.")
            
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
            file_type="csv",
            parser_used="pandas",
            ocr_used=False,
            ocr_engine=None,
            extraction_method="native",
            is_tabular=True,
        )
