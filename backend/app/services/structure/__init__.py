"""
文档结构解析（Document Structure Parsing）.

统一封装 MinerU / Marker / Docling / 原生 四种解析后端，对外只暴露一个
``parse_structure()``，并保证：

  * **逐页归属**：结构解析不得让引用退化成"第 1 页"（见 model 模块的论证）；
  * **优雅降级**：外部引擎缺失/超时/输出不可信 → 自动退回原生，入库不受影响；
  * **统一结构树**：章节路径、元素类型在所有后端之间语义一致（由 outline 保证）。

    from app.services.structure import parse_structure, structure_capability_report
"""

from app.services.structure.model import (
    NodeKind,
    PageMark,
    StructuredDocument,
    StructureNode,
)
from app.services.structure.outline import (
    build_nodes_from_markdown,
    section_path_for_offset,
)
from app.services.structure.parser import (
    native_structure,
    parse_structure,
    structure_capability_report,
)
from app.services.structure.providers import (
    available_providers,
    build_provider,
    reset_probe_cache,
)

__all__ = [
    "NodeKind",
    "PageMark",
    "StructureNode",
    "StructuredDocument",
    "build_nodes_from_markdown",
    "section_path_for_offset",
    "native_structure",
    "parse_structure",
    "structure_capability_report",
    "available_providers",
    "build_provider",
    "reset_probe_cache",
]
