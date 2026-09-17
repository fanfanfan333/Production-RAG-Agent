"""
结构化内容对象（Structured Content）.

对应设计稿"图片 → 结构化内容 → RAG"这一步的产物。无论一张图片走的是
Table Parser 还是 Vision，产出都统一收敛成同一个形状，便于下游建块：

    {
      "type": "table",                                   # table|chart|diagram|screenshot|photo
      "page": 1,
      "content": "| 指标 | 数值 |\\n|---|---|\\n| 准确率 |95%|..."
    }

入库时再映射成 Qdrant payload：

    {
      "content_type": "table",
      "text":         "| 指标 | 数值 |...",
      "page_number":  1,
      "source":       "照片.docx"
    }

这样"图片"与"正文表格"在检索层是同一种东西 —— 都能被 BM25 与向量召回到，
这也是"表格检索"能对**图片里的表格**生效的原因。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── 图片类型（Picture Classification 的取值域）───────────────────────────────
IMAGE_TYPE_TABLE = "table"           # 表格图片 → Table Transformer / 规则表格
IMAGE_TYPE_FORMULA = "formula"       # 公式图片 → PaddleOCR Formula（LaTeX）
IMAGE_TYPE_CODE = "code"             # 代码截图 → OCR + Code Parser
IMAGE_TYPE_CHART = "chart"           # 柱状/折线/饼图 → Vision
IMAGE_TYPE_DIAGRAM = "diagram"       # 流程图 / 结构图 / 架构图 → Vision
IMAGE_TYPE_SCREENSHOT = "screenshot" # 界面截图 → OCR / Vision
IMAGE_TYPE_PHOTO = "photo"           # 普通照片 → OCR

ALL_IMAGE_TYPES = (
    IMAGE_TYPE_TABLE,
    IMAGE_TYPE_FORMULA,
    IMAGE_TYPE_CODE,
    IMAGE_TYPE_CHART,
    IMAGE_TYPE_DIAGRAM,
    IMAGE_TYPE_SCREENSHOT,
    IMAGE_TYPE_PHOTO,
)

# 这些类型走 Vision（"看图理解"），其余走 OCR / 专用引擎。
# 说明：设计稿"普通图片 → OCR"；只有图表 / 流程图 / 结构图这类需要"读懂图意"
# 的才交给多模态模型。代码截图默认走 Code Parser，公式默认走 Formula 引擎，
# 它们拿不到专用引擎时会**降级**到 Vision（见 pipeline 的 fallback 分支）。
# 实际走哪些类型由 settings.VISION_ANALYZE_TYPES 决定。
DEFAULT_VISION_TYPES = (IMAGE_TYPE_CHART, IMAGE_TYPE_DIAGRAM, IMAGE_TYPE_SCREENSHOT)

# 各类型的中文名（日志 / 前端展示）
IMAGE_TYPE_LABELS = {
    IMAGE_TYPE_TABLE: "表格图片",
    IMAGE_TYPE_FORMULA: "公式",
    IMAGE_TYPE_CODE: "代码截图",
    IMAGE_TYPE_CHART: "图表",
    IMAGE_TYPE_DIAGRAM: "流程图/结构图",
    IMAGE_TYPE_SCREENSHOT: "截图",
    IMAGE_TYPE_PHOTO: "普通图片",
}


@dataclass
class StructuredContent:
    """一张图片被"结构化"之后的内容（设计稿的 {"type","page","content"}）."""

    type: str                    # 图片类型，见 ALL_IMAGE_TYPES
    page: int = 1
    content: str = ""            # Markdown 表格 / 结构化描述全文
    # 该结构化内容是由谁产出的：table_parser | vision | ocr
    engine: str | None = None
    # 额外元信息（行数/列数/置信度等），仅用于日志与排查
    meta: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.content.strip()

    def to_dict(self) -> dict:
        """设计稿里那个 JSON 形状."""
        return {"type": self.type, "page": int(self.page), "content": self.content}

    def to_qdrant_payload(self, *, source: str = "") -> dict:
        """入库前的 payload 映射（设计稿"然后存入 Qdrant"）."""
        return {
            "content_type": content_type_for(self.type),
            "text": self.content,
            "page_number": int(self.page),
            "source": source,
        }


def content_type_for(image_type: str) -> str:
    """
    图片类型 → Qdrant ``content_type``.

    表格图片产出的是**真表格**（Markdown），因此它的 ``content_type`` 也是
    "table" 而不是 "image" —— 这样"表格检索"对图片表格与正文表格一视同仁。

    公式 / 代码虽然也是结构化文本，但它们在检索层仍然按"图片"对待
    （``content_type="image"``）：它们需要连同原图一起回显（看图问答、
    引用卡片），细化区别由独立的 ``image_type`` 字段承载，避免下游按
    content_type 分流时漏掉这两类。
    """
    return "table" if image_type == IMAGE_TYPE_TABLE else "image"


__all__ = [
    "StructuredContent",
    "content_type_for",
    "IMAGE_TYPE_TABLE",
    "IMAGE_TYPE_FORMULA",
    "IMAGE_TYPE_CODE",
    "IMAGE_TYPE_CHART",
    "IMAGE_TYPE_DIAGRAM",
    "IMAGE_TYPE_SCREENSHOT",
    "IMAGE_TYPE_PHOTO",
    "ALL_IMAGE_TYPES",
    "DEFAULT_VISION_TYPES",
    "IMAGE_TYPE_LABELS",
]
