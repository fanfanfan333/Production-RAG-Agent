from dataclasses import dataclass, field
import abc

@dataclass
class ExtractedPage:
    """Text content and position metadata for a single page or section."""
    page_number: int        # 1-indexed
    text: str
    char_start: int         # offset of this page's text in the full document string
    char_end: int           # exclusive end offset

@dataclass
class ExtractionResult:
    """Result returned by extractors."""
    pages: list[ExtractedPage] = field(default_factory=list)
    full_text: str = ""
    page_count: int = 0
    char_count: int = 0

    # Metadata about how it was extracted
    file_type: str = "unknown"
    parser_used: str = "unknown"
    ocr_used: bool = False
    ocr_engine: str | None = None
    extraction_method: str = "native"

    # Embedded-image recognition (问题3): text recognised from images that
    # live INSIDE the document (PDF/DOCX/PPTX). Already merged into
    # full_text/pages by the parsers — kept here for stats and logging.
    image_count: int = 0                 # embedded images recognised
    image_texts: list[str] = field(default_factory=list)

    def page_for_offset(self, char_offset: int) -> int:
        """Return the 1-indexed page number that contains the given character offset."""
        for page in self.pages:
            if page.char_start <= char_offset < page.char_end:
                return page.page_number
        # Fallback: return last page
        return self.pages[-1].page_number if self.pages else 1

class DocumentParser(abc.ABC):
    """
    Abstract base class for all document parsers.
    """
    @abc.abstractmethod
    def parse(self, content: bytes, filename: str) -> ExtractionResult:
        """Parse raw bytes and return an ExtractionResult."""
        pass
