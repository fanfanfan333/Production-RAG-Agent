"""
Vision 引擎适配器（流程图/架构图/图片描述 → 多模态模型）.

对应设计稿"流程图/架构图/图片描述 → Vision"。它同时承担两个角色：

* **专用引擎**：图片类型判为 chart / diagram / formula / photo 时的首选；
* **兜底引擎**：其它引擎置信度低时，流程图里的 "Fallback → Vision LLM"。

两个角色走的提示词不同（前者按图类型选，后者是"尽力转写"通用提示词），
因此 ``process`` 收一个 ``role`` 参数。
"""

from __future__ import annotations

from PIL import Image

from app.services.image_understanding.analyzer import (
    FALLBACK_PROMPT,
    analyze_image_sync,
)
from app.services.image_understanding.engines.base import (
    EngineKind,
    EngineOutput,
    ImageEngine,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class VisionEngine(ImageEngine):
    """多模态理解。模型缺失时**如实报告不可用**，由管线继续往下降级。"""

    name = "vision"
    kind = EngineKind.VISION

    def is_available(self) -> bool:
        try:
            from app.services.vision.vision_service import get_vision_service

            # 必须用同步版：async is_available() 同步调用会返回 coroutine（恒真），
            # 会让"模型没装"也被判成可用。
            return bool(get_vision_service().is_available_sync())
        except Exception:      # noqa: BLE001
            return False

    def unavailable_reason(self) -> str:
        if self.is_available():
            return ""
        return "Ollama 中没有可用的多模态模型（如 ollama pull qwen2.5vl:7b）"

    def process(self, image: Image.Image, *, png_bytes: bytes | None = None,
                image_type: str = "photo", role: str = "primary", **kwargs) -> EngineOutput:
        """
        :param role: ``primary`` = 按类型的专用提示词；``fallback`` = 兜底转写。
        """
        # primary 用"按类型定制"的提示词（问什么答什么）；fallback 用通用
        # 转写提示词（此时已确认专用引擎失败，目标是"尽量把内容捞回来"）。
        prompt = None if role == "primary" else FALLBACK_PROMPT

        payload = png_bytes
        if payload is None:
            from app.services.image_understanding.imaging import encode_png_safe

            payload = encode_png_safe(image)

        try:
            text = analyze_image_sync(payload, image_type, prompt=prompt)
        except Exception as exc:      # noqa: BLE001
            logger.warning("Vision inference failed: %s", exc)
            return EngineOutput(engine=self.name, ok=False, error=str(exc))

        text = (text or "").strip()
        if not text:
            return EngineOutput(engine=self.name, ok=False, error=self.unavailable_reason() or "空结果")

        # 多模态模型不给逐 token 置信度；有产出即给中性偏高值，让 Validation
        # 阶段的"结构校验"（而不是这个数字）去决定最终 Pass/Fail。
        return EngineOutput(
            text=text,
            confidence=0.75,
            engine=self.name,
            ok=True,
            meta={"role": role, "prompt_type": image_type if role == "primary" else "fallback"},
        )


__all__ = ["VisionEngine"]
