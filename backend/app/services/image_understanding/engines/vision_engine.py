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
    TEXTLESS_SUMMARY_PROMPT,
    analyze_image_sync,
    looks_like_prompt_echo,
    prompt_for,
)
from app.services.image_understanding.engines.base import (
    EngineKind,
    EngineOutput,
    ImageEngine,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# role → 提示词。``None`` 表示"按图片类型的专用提示词"（由 analyzer 决定）。
#
# ``summarize`` 是"图内一个文字都没读出来"时用的**图意总结**提示词（实施手册
# 3.3.1 的三段式）：此时再拿"把图里的字转写出来"去问模型只会又得到空串，
# 换成"这张图是什么"才有产出，图片也才能拿到可检索文本、不被分块阶段跳过。
_ROLE_PROMPTS: dict[str, str | None] = {
    "primary": None,
    "fallback": FALLBACK_PROMPT,
    "summarize": TEXTLESS_SUMMARY_PROMPT,
}


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
        :param role: ``primary`` = 按类型的专用提示词；``fallback`` = 兜底转写；
            ``summarize`` = 图内没有任何文字时的图意总结（三段式）。
        """
        # primary 用"按类型定制"的提示词（问什么答什么）；fallback 用通用
        # 转写提示词（此时已确认专用引擎失败，目标是"尽量把内容捞回来"）；
        # summarize 用于"整张图一个字都没读出来"（转写提示词必然又是空串）。
        # 未知 role 保持历史行为（通用转写提示词），不改变既有调用方语义。
        prompt = _ROLE_PROMPTS.get(role, FALLBACK_PROMPT)
        prompt_type = image_type if role == "primary" else (
            role if role in _ROLE_PROMPTS else "fallback"
        )

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

        # 复读自检（真跑实测的缺陷）：模型偶尔会把**提示词原文**吐回来。实测
        # qwen2.5vl:3b 对"没有文字的图"会逐条复读兜底转写提示词，且丢掉了带
        # "请尽最大努力"的开头句 —— quality.py 的关键词检测一个都不命中，于是
        # 这段指令被当成"图意描述"写进 vision_caption 并随图片分块**入库**：
        # 检索命中的是提示词，不是图里的内容。
        # 在引擎层判掉最省事也最彻底：primary / fallback / summarize 三个 role
        # 都走这里，判成失败后，管线上"ok=False 就不进候选"的既有逻辑会自然接管
        # （对无文字的图，接着就会走"图意总结"那条分支）。
        if looks_like_prompt_echo(text, prompt or prompt_for(image_type)):
            logger.warning(
                "Vision 复读了提示词（role=%s type=%s, %d 字）→ 判为失败",
                role, image_type, len(text),
            )
            return EngineOutput(
                engine=self.name, ok=False,
                error="提示词复读：模型把指令原文吐了回来，等于没读取图片",
            )

        # 多模态模型不给逐 token 置信度；有产出即给中性偏高值，让 Validation
        # 阶段的"结构校验"（而不是这个数字）去决定最终 Pass/Fail。
        return EngineOutput(
            text=text,
            confidence=0.75,
            engine=self.name,
            ok=True,
            meta={"role": role, "prompt_type": prompt_type},
        )


__all__ = ["VisionEngine"]
