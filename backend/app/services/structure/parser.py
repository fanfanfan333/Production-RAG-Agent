"""
结构解析调度器：按配置链逐个尝试 Provider，链尾恒为 native.

调用点（document_service）只写一行：

    structured = await asyncio.to_thread(parse_structure, content, filename, ...)

它保证**永远返回一个可用的 StructuredDocument** —— 外部引擎全不可用时返回
native 结果（= 现有 PyMuPDF/python-docx 的正文 + 页码，行为与改造前完全一致）。
因此这次改造对没有装 MinerU/Marker 的部署是**零回归**的：收益是拿到结构树
（章节路径 / 元素类型）用于父子分块与元数据，页码仍由原生解析保证。
"""

from __future__ import annotations

from app.services.structure.model import (
    PageMark,
    StructuredDocument,
)
from app.services.structure.outline import build_nodes_from_markdown
from app.services.structure.providers import (
    available_providers,
    build_provider,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_chain(settings) -> list[str]:
    raw = getattr(settings, "STRUCTURE_PARSER_CHAIN", "mineru,marker,docling,native")
    chain = [p.strip().lower() for p in str(raw).split(",") if p.strip()]
    if "native" not in chain:
        chain.append("native")          # native 是兜底，必须存在
    return chain


def _allowed_extension(filename: str, settings) -> bool:
    raw = getattr(settings, "STRUCTURE_PARSER_EXTENSIONS", "pdf,docx,pptx")
    allowed = {e.strip().lower().lstrip(".") for e in str(raw).split(",") if e.strip()}
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in allowed


def native_structure(extraction) -> StructuredDocument:
    """
    把已有的 ``ExtractionResult`` 包装成 ``StructuredDocument``.

    不重新解析 —— 只是把已经算好的 ``pages``（含真实 char_start/char_end）
    翻译成 page_marks，并把 full_text 当作 Markdown 抽取结构节点。
    所以这一步不可能引入页码回归。
    """
    markdown = extraction.full_text or ""
    marks: list[PageMark] = []
    for page in getattr(extraction, "pages", []) or []:
        try:
            marks.append(PageMark(
                page_number=int(page.page_number),
                char_start=max(0, int(page.char_start)),
            ))
        except (TypeError, ValueError):
            continue
    marks.sort(key=lambda m: m.char_start)
    # 极端情况：解析器没给出任何页（空文档 / 纯图片）——给一个第 1 页的标记，
    # 下游 page_for_offset 才有东西可查。
    if not marks:
        marks = [PageMark(page_number=1, char_start=0)]

    nodes = build_nodes_from_markdown(markdown, marks)
    return StructuredDocument(
        markdown=markdown,
        page_marks=marks,
        nodes=nodes,
        provider="native",
        page_count=max((m.page_number for m in marks), default=1),
        page_marks_trustworthy=True,
    )


def parse_structure(
    content: bytes,
    filename: str,
    *,
    extraction,
    settings,
) -> StructuredDocument:
    """
    按 ``STRUCTURE_PARSER_CHAIN`` 顺序尝试结构解析，返回可用的结果.

    Args:
        content:    原始文件字节（外部 provider 需要原始文件才能真正解析版面）
        filename:   原始文件名（决定扩展名与临时文件名）
        extraction: 原生解析的 ``ExtractionResult``（native 兜底 + 元数据来源）
        settings:   Settings 实例

    Returns:
        StructuredDocument —— 保证非空且页码可信。
    """
    native = native_structure(extraction)

    if not getattr(settings, "STRUCTURE_PARSER_ENABLED", True):
        return native
    if not _allowed_extension(filename, settings):
        return native

    for name in _parse_chain(settings):
        if name == "native":
            break
        provider = build_provider(name, settings)
        if provider is None:
            logger.warning("structure chain lists unknown provider '%s' — skipped", name)
            continue
        if not provider.available():
            logger.debug("structure provider '%s' not available — skipped", name)
            continue

        doc = provider.parse(content, filename)
        if doc is None:
            continue

        # 节点统一从 Markdown 还原（provider 之间不共享结构语义，见 outline 模块）
        doc.nodes = build_nodes_from_markdown(doc.markdown, doc.page_marks)
        logger.info(
            "structure parser '%s' succeeded: provider=%s pages=%d nodes=%d chars=%d",
            name, doc.provider, doc.page_count, len(doc.nodes), len(doc.markdown),
        )
        return doc

    return native


def structure_capability_report(settings) -> dict:
    """
    当前环境的结构解析能力（诊断/健康检查用）.

    输出会直接出现在 /health 或诊断脚本里，让"到底是没装工具还是链配置错了"
    一目了然 —— 否则结构解析静默退回 native 时没人知道原因。
    """
    providers = available_providers(settings)
    return {
        "enabled": bool(getattr(settings, "STRUCTURE_PARSER_ENABLED", True)),
        "chain": _parse_chain(settings),
        "extensions": str(getattr(settings, "STRUCTURE_PARSER_EXTENSIONS", "")),
        "providers": providers,
        "usable": [n for n, ok in providers.items() if ok],
    }
