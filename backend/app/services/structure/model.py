"""
结构解析的统一数据模型.

为什么要有这一层
────────────────
入库链路上游有四个可能的解析器（MinerU / Marker / Docling / 原生 PyMuPDF），
下游只有一个消费者（分块器 + 元数据 + 引用溯源）。如果让每个 provider 各自
往 ``ExtractionResult`` 里塞东西，页码、行号、章节路径就会各说各话。

所以约定：**所有 provider 只产出一种东西**

    StructuredDocument
      ├ markdown     统一 Markdown 正文（分块器只认这个）
      ├ page_marks   每一页在 markdown 中的起始字符偏移（**逐页归属的唯一来源**）
      └ nodes        章节/元素树（元数据大纲 + 结构感知父子分块的基础）

为什么 page_marks 是整个设计的命门
──────────────────────────────────
本项目的细粒度引用（"《年报.pdf》第 3 页，第 12–28 行"）完全建立在
``ExtractedPage.char_start/char_end`` 之上 —— ``page_for_offset`` 靠它把字符
偏移映射回页码，行号也是同一套坐标。

项目里 ``DOCLING_PDF_ENABLED`` 默认关闭，注释写得很直白：Docling 是**整篇**
转换，套进来会让正文全部归到第 1 页，引用就废了。

因此本模块立一条硬规矩：**拿不到真实逐页偏移的 provider 结果一律作废**
（``page_marks_trustworthy=False``），由调度器跳到下一个 provider。
宁可退回原生解析（页码正确、结构稍弱），也不要一份页码全错的漂亮 Markdown
—— 前者损失一点检索质量，后者会让每一条引用都在撒谎。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── 元素类型（与 content_type / 分块器保持一致的词汇表）───────────────────────
class NodeKind:
    """结构节点类型。值同时用于 payload 过滤与前端徽标，保持稳定不随版本变。"""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CODE = "code"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"


@dataclass
class PageMark:
    """一页在 ``StructuredDocument.markdown`` 中的起始字符偏移。"""

    page_number: int          # 1-based
    char_start: int           # 页首在 markdown 中的偏移


@dataclass
class StructureNode:
    """
    一个结构元素（标题 / 段落 / 表格 / 图 …）.

    只保留**扁平**列表 + ``level``，不建嵌套 children —— 树的形状可以随时由
    level 复原，而嵌套结构在跨 provider 合并、跨页切断时极易损坏。
    """

    kind: str
    level: int                       # 标题层级 1-6；非标题元素恒为 0
    title: str                       # 标题文字；非标题元素为空串
    char_start: int                  # 在 markdown 中的起始偏移
    char_end: int                    # 开区间上界
    page_number: int = 1             # 1-based，来自 provider 的真实页码
    section_path: tuple[str, ...] = ()  # 祖先标题链，如 ("第 3 章", "3.2 核算方法")
    metadata: dict = field(default_factory=dict)

    @property
    def is_heading(self) -> bool:
        return self.kind == NodeKind.HEADING

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "level": self.level,
            "title": self.title,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "page_number": self.page_number,
            "section_path": list(self.section_path),
        }


@dataclass
class StructuredDocument:
    """
    结构解析的统一产物.

    ``page_marks_trustworthy`` 的语义很关键：它表示 **page_marks 如实反映了
    原文的分页**。当 provider 只能吐出整篇 Markdown（无法定位页边界）时，
    它必须把该标志置为 False —— 调度器会据此丢弃这份结果。伪造"平均分页"
    比没有页码更糟：用户会照着错误的页码去核对原文。
    """

    markdown: str
    page_marks: list[PageMark] = field(default_factory=list)
    nodes: list[StructureNode] = field(default_factory=list)
    provider: str = "native"
    page_count: int = 0
    page_marks_trustworthy: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """是否有实质内容可供下游使用（空正文一律视为不可用）。"""
        return bool(self.markdown.strip()) and self.page_marks_trustworthy

    def page_for_offset(self, char_offset: int) -> int:
        """
        把 markdown 中的字符偏移映射回页码（二分）.

        与 ``ExtractionResult.page_for_offset`` 语义一致，但基于本结构的
        page_marks —— 所以上游换成 MinerU/Marker 后，下游的页码行为不变。
        """
        if not self.page_marks:
            return 1
        marks = self.page_marks
        if char_offset < marks[0].char_start:
            return marks[0].page_number
        lo, hi = 0, len(marks)
        while lo < hi:
            mid = (lo + hi) // 2
            if marks[mid].char_start <= char_offset:
                lo = mid + 1
            else:
                hi = mid
        idx = max(0, lo - 1)
        return marks[idx].page_number

    def page_spans(self) -> list[tuple[int, int, int]]:
        """
        返回 ``[(page_number, char_start, char_end), ...]``（char_end 开区间）.

        下游据此重建 ``ExtractedPage`` 列表；不重建的话 page_for_offset 与
        分块器的 page_resolver 会继续读旧的（已失效的）偏移。
        """
        if not self.page_marks:
            return []
        marks = self.page_marks
        total = len(self.markdown)
        spans: list[tuple[int, int, int]] = []
        for i, mark in enumerate(marks):
            end = marks[i + 1].char_start if i + 1 < len(marks) else total
            spans.append((mark.page_number, mark.char_start, max(mark.char_start, end)))
        return spans

    def outline(self) -> list[dict]:
        """章节大纲（只含标题节点），供元数据与前端目录使用."""
        return [
            {
                "level": n.level,
                "title": n.title,
                "page_number": n.page_number,
                "section_path": list(n.section_path),
            }
            for n in self.nodes
            if n.is_heading and n.title
        ]
