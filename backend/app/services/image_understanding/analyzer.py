"""
Vision Analyzer（三层路由里的 Vision 支路）.

设计稿里 Chart 与 Diagram 都流向 Vision，产出"结构化描述"再进 embedding：

    Image → Vision Model → 结构化描述 → Embedding

这里的关键点是**提示词随图片类型变化**：让模型"描述这张图"和让它"把图里的
数据点、坐标轴含义、数值都列出来"得到的结果质量差一个量级。流程图同理 ——
需要的是"节点和连线"，不是"这是一张流程图"。

视觉模型不可用时的降级（设计稿"没有就先放弃 vision"）：
    · 表格图片 —— 走 Table Parser（完全不依赖模型），只有结构识别失败才需要救场；
    · 图表 / 流程图 / 截图 —— 退回图内 OCR 文本，并明确标注"视觉分析不可用"，
      让 LLM 知道这是"没能看图"而不是"图中没有信息"。
"""

from __future__ import annotations

from app.config import get_settings
from app.services.image_understanding.structured_content import (
    IMAGE_TYPE_CHART,
    IMAGE_TYPE_CODE,
    IMAGE_TYPE_DIAGRAM,
    IMAGE_TYPE_FORMULA,
    IMAGE_TYPE_SCREENSHOT,
    IMAGE_TYPE_TABLE,
)
from app.services.vision import get_vision_service
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 各类型的提示词。共同要求：只输出内容本身、用中文、结构化罗列。
#
# CHART_PROMPT 先做一次"性质判别"再分流：分类器是**基于版面/色彩统计**的规则，
# 对深色底的节点图（TensorBoard graph / Keras plot_model）偶尔会判成 chart。与其
# 让模型硬套"横轴/纵轴/数据点"模板吐出一堆"无具体数值"，不如让提示词自带这条
# 兜底 —— 代价只是多一句话，却能把误分类的损失降到最低。
CHART_PROMPT = (
    "先判断这张图片属于下面哪种情况，再按对应要求作答：\n"
    "【情况一】它是数据图表（柱状图、折线图、饼图、散点图、热力图等）。请提取：\n"
    "1) 图表的类型与标题；\n"
    "2) 横轴、纵轴、图例各自代表什么；\n"
    "3) 图中的主要数据点及其数值（尽量给出具体数字与单位）；\n"
    "4) 由数据能得出的结论。\n"
    "【情况二】它其实不是数据图表，而是流程图 / 结构图 / 节点图 —— 图里没有坐标轴、"
    "没有数值，只有方框、模块、箭头或连线。此时**不要**套用数据点模板，改为输出：\n"
    "1) 这张图表达的主题；\n"
    "2) 图中出现的所有节点 / 模块 / 组件的名称（逐个列出，按流程顺序）；\n"
    "3) 节点之间的连线关系与方向（A → B 的形式）；\n"
    "4) 整体流程或结构的一句话概述。\n"
    "判断依据只看图里**实际有什么**：没有坐标轴与数值就不要按图表描述。\n"
    "用中文输出，分点罗列，只输出内容本身，不要任何前缀或解释。"
)

DIAGRAM_PROMPT = (
    "这是一张流程图 / 结构图 / 架构图 / 模型示意图。请提取：\n"
    "1) 这张图表达的主题；\n"
    "2) 图中出现的所有节点、模块、组件的名称（逐个列出）；\n"
    "3) 节点之间的连线关系与数据/控制流方向（A → B 的形式）；\n"
    "4) 整体流程或结构的一句话概述。\n"
    "用中文输出，分点罗列，只输出内容本身，不要任何前缀或解释。"
)

SCREENSHOT_PROMPT = (
    "这是一张软件界面截图或代码截图。请提取：\n"
    "1) 截图展示的是什么界面 / 什么代码；\n"
    "2) 其中可见的关键文字、字段、命令或函数名；\n"
    "3) 这张截图能说明的信息要点。\n"
    "用中文输出，分点罗列，只输出内容本身，不要任何前缀或解释。"
)

# 表格救场提示词：只有当框线/对齐还原失败时才用
TABLE_RESCUE_PROMPT = (
    "这是一张表格图片。请把表格内容完整转写成 Markdown 表格，"
    "保留所有行列与表头。只输出 Markdown 表格本身，不要任何说明文字。"
)

# 公式提示词：专用公式引擎（PaddleOCR Formula）不可用时由 Vision 兜底
FORMULA_PROMPT = (
    "这是一张数学公式图片。请把它转写成 LaTeX：\n"
    "1) 只输出公式本身的 LaTeX 代码（行内用 $...$ 包裹）；\n"
    "2) 保留上下标、分式、求和/积分号、希腊字母；\n"
    "3) 不要输出任何解释、编号或 Markdown 围栏。\n"
    "若图中不止一个公式，每个公式单独一行输出。"
)

# 代码兜底提示词：Code Parser 拿不到结构时由 Vision 转写
CODE_RESCUE_PROMPT = (
    "这是一张代码截图。请把代码**逐行**转写出来：\n"
    "1) 保留原始缩进（缩进是代码语法的一部分）；\n"
    "2) 保留符号、括号、字符串与注释；\n"
    "3) 用 Markdown 代码围栏包裹，并标注语言；\n"
    "4) 只输出代码块本身，不要任何解释。"
)

# 通用兜底提示词：流程图里 "Fallback → Vision LLM" 分支用。
# 与上面按类型定制的提示词不同 —— 兜底时我们**已经知道**专用引擎失败了，
# 所以让模型"尽力把内容完整转写出来"，而不是做某一种特定结构的抽取。
FALLBACK_PROMPT = (
    "请尽最大努力把这张图片里的全部信息完整转写为文字：\n"
    "1) 如果有文字，逐行忠实转写（保留原有顺序与换行）；\n"
    "2) 如果是表格，用 Markdown 表格输出；\n"
    "3) 如果是图表，列出数据点与数值；\n"
    "4) 如果是流程图/结构图，列出节点与连线关系；\n"
    "5) 如果是公式，用 LaTeX 输出；\n"
    "6) 如果是照片，描述画面内容与其中可读的文字。\n"
    "用中文输出，只输出转写内容本身，不要任何前缀或解释。"
)

_PROMPTS = {
    IMAGE_TYPE_CHART: CHART_PROMPT,
    IMAGE_TYPE_DIAGRAM: DIAGRAM_PROMPT,
    IMAGE_TYPE_SCREENSHOT: SCREENSHOT_PROMPT,
    IMAGE_TYPE_TABLE: TABLE_RESCUE_PROMPT,
    IMAGE_TYPE_FORMULA: FORMULA_PROMPT,
    IMAGE_TYPE_CODE: CODE_RESCUE_PROMPT,
}


def prompt_for(image_type: str) -> str:
    return _PROMPTS.get(image_type, CHART_PROMPT)


def analyze_image_sync(
    image_bytes: bytes,
    image_type: str,
    *,
    prompt: str | None = None,
    timeout: float | None = None,
) -> str | None:
    """
    按图片类型调用 Vision，返回"结构化描述"（模型/开关不可用时返回 None）.

    :param prompt: 显式提示词。传了就完全覆盖按类型的默认提示词 ——
        流程图的 fallback 分支（``FALLBACK_PROMPT``）靠它实现。

    与 ``VisionService.describe_image_sync`` 的区别只在提示词 —— 复用同一个
    单例与同一套可用性探测，避免出现两套 vision 配置。
    """
    if not image_bytes:
        return None
    settings = get_settings()
    limit = (
        settings.VISION_DIAGRAM_MAX_CHARS
        if image_type in (IMAGE_TYPE_DIAGRAM, IMAGE_TYPE_SCREENSHOT)
        else settings.VISION_CHART_MAX_CHARS
    )
    if image_type == IMAGE_TYPE_TABLE:
        limit = max(limit, settings.VISION_CAPTION_MAX_CHARS)
    try:
        text = get_vision_service().describe_image_sync(
            image_bytes,
            prompt=prompt or prompt_for(image_type),
            timeout=timeout or settings.VISION_TIMEOUT_SECONDS,
        )
    except Exception as exc:      # noqa: BLE001
        logger.warning("Vision analysis failed (type=%s): %s", image_type, exc)
        return None
    if not text:
        return None
    return text[:limit].strip() or None


__all__ = [
    "analyze_image_sync",
    "prompt_for",
    "CHART_PROMPT",
    "DIAGRAM_PROMPT",
    "SCREENSHOT_PROMPT",
    "TABLE_RESCUE_PROMPT",
    "FORMULA_PROMPT",
    "CODE_RESCUE_PROMPT",
    "FALLBACK_PROMPT",
]
