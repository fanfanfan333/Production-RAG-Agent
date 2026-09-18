"""
Vision service（部分4：把 Vision 打开，没有模型就先放弃 vision）.

一个进程内单例，统一承担两类多模态调用：

    1. 入库期 —— 描述文档内嵌图片（caption），让"图是什么"进入向量库；
    2. 检索期 —— 拿到命中的原始图片 + 用户问题，做"看图回答"（VQA），
       把视觉结论作为证据喂给最终 LLM（multimodal_context 节点）。

设计要点
────────
- **默认打开**：``VISION_ENABLED=true`` 且 ``OLLAMA_VISION_MODEL`` 非空即视为开启。
- **可用性探测**：启动/首次调用时查询 Ollama ``/api/tags``，确认该多模态模型
  确实已拉取。模型不存在时**不报错**，而是标记为不可用并降级 —— 这是
  需求里"如果没有，就先放弃 vision"的落地方式：
      · 图片仍被抽成独立检索对象（OCR 文本 + 原图回显照常工作）；
      · 只有"视觉理解 / 看图问答"这两项依赖模型的能力被跳过。
- **失败隔离**：任何网络 / 解析异常都返回 None，绝不让多模态环节拖垮主链路。
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time

import httpx

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 入库期默认提示词：只输出描述本身。
CAPTION_PROMPT = (
    "请用中文简要描述这张图片的内容（如果是图表，请说明图表类型和主要数据；"
    "如果是照片，请描述其中的对象和场景）。只输出描述本身，不要任何前缀。"
)

# 检索期默认提示词：带着用户问题去看图，产出"可当证据用"的结论。
_VQA_PROMPT_TEMPLATE = (
    "你是一个文档图像分析助手。下面这张图片来自知识库，是回答用户问题时检索到的证据。\n"
    "请只依据图片本身的内容作答，用中文简洁地说明：\n"
    "1) 图片展示的是什么；\n"
    "2) 其中哪些信息与用户问题相关（若无关，直接说明无关）。\n"
    "用户问题：{query}\n"
    "只输出结论，不要复述这段指令。"
)


class VisionService:
    """Ollama 多模态模型调用封装（线程安全单例）。"""

    _instance: "VisionService | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "VisionService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._probe_done = False
        self._available: bool | None = None
        self._probe_error: str | None = None
        self._initialized = True

    # ── 配置 / 可用性 ──────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return (get_settings().OLLAMA_VISION_MODEL or "").strip()

    @property
    def configured(self) -> bool:
        """是否在配置层面开启了 vision（开关打开且填了模型名）."""
        settings = get_settings()
        if not getattr(settings, "VISION_ENABLED", False):
            return False
        if not self.model:
            return False
        # 允许显式关闭探测（离线环境跳过 /api/tags 查询）
        return True

    def _probe_sync(self) -> bool:
        """查询 Ollama /api/tags，确认多模态模型已经存在."""
        settings = get_settings()
        if not self.configured:
            self._probe_error = "vision disabled or model not configured"
            return False
        try:
            resp = httpx.get(
                f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags",
                timeout=min(5.0, float(settings.VISION_TIMEOUT_SECONDS)),
            )
            resp.raise_for_status()
            models = resp.json().get("models", []) or []
        except Exception as exc:      # noqa: BLE001
            self._probe_error = f"ollama unreachable: {exc}"
            logger.warning(
                "Vision probe failed (ollama at %s): %s — vision will be disabled",
                settings.OLLAMA_BASE_URL, exc,
            )
            return False

        wanted = self.model
        wanted_base = wanted.split(":")[0]
        for entry in models:
            name = str(entry.get("name") or entry.get("model") or "")
            if not name:
                continue
            caps = entry.get("capabilities") or []
            if name != wanted and name.split(":")[0] != wanted_base:
                continue
            # 若 Ollama 明确报告了能力列表，要求其包含 vision
            if caps and "vision" not in caps:
                self._probe_error = f"model '{name}' has no vision capability"
                logger.warning(
                    "Configured vision model '%s' exists but reports capabilities=%s "
                    "(no 'vision') — vision disabled",
                    name, caps,
                )
                return False
            return True

        self._probe_error = f"model '{wanted}' not pulled"
        logger.warning(
            "Vision model '%s' is not installed in Ollama (available: %s). "
            "Vision understanding / image Q&A will be skipped — "
            "run `ollama pull %s` to enable it.",
            wanted,
            ", ".join(str(m.get("name")) for m in models) or "none",
            wanted,
        )
        return False

    async def is_available(self, *, refresh: bool = False) -> bool:
        if not self.configured:
            return False
        if self._available is not None and not refresh:
            return self._available
        if self._probe_done and not refresh:
            return bool(self._available)
        self._available = await asyncio.to_thread(self._probe_sync)
        self._probe_done = True
        if self._available:
            logger.info("Vision service ready (model=%s)", self.model)
        return self._available

    def status(self) -> dict:
        """供 /health 展示的 vision 状态快照."""
        return {
            "configured": self.configured,
            "model": self.model or None,
            "available": bool(self._available) if self._available is not None else None,
            "error": self._probe_error,
        }

    # ── 调用 ───────────────────────────────────────────────────────────────

    def _generate_sync(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        timeout: float,
    ) -> str | None:
        settings = get_settings()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        try:
            resp = httpx.post(
                f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [b64],
                    "stream": False,
                    # 多模态解码开销大，给足上下文但不做无谓放大
                    "options": {"temperature": 0.2, "num_ctx": settings.chat_num_ctx},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            text = str(resp.json().get("response", "")).strip()
            return text or None
        except Exception as exc:      # noqa: BLE001
            logger.warning("Vision generate failed (model=%s): %s", self.model, exc)
            return None

    async def describe_image(
        self,
        image_bytes: bytes,
        *,
        prompt: str | None = None,
        timeout: float | None = None,
    ) -> str | None:
        """入库期：描述一张图片（caption）."""
        if not image_bytes:
            return None
        if not await self.is_available():
            return None
        settings = get_settings()
        return await asyncio.to_thread(
            self._generate_sync,
            image_bytes,
            prompt or CAPTION_PROMPT,
            timeout=float(timeout or settings.VISION_TIMEOUT_SECONDS),
        )

    async def analyze_image(
        self,
        image_bytes: bytes,
        query: str,
        *,
        timeout: float | None = None,
    ) -> str | None:
        """
        检索期：带着用户问题看图（VQA），产出可当证据用的结论.

        对应设计稿 ``multimodal_context`` 节点里的 ``analyze_image(image, query)``。
        """
        if not image_bytes:
            return None
        if not await self.is_available():
            return None
        settings = get_settings()
        prompt = _VQA_PROMPT_TEMPLATE.format(query=query or "（未提供问题）")
        return await asyncio.to_thread(
            self._generate_sync,
            image_bytes,
            prompt,
            timeout=float(timeout or settings.VISION_TIMEOUT_SECONDS),
        )

    # 同步版本，供 OCR/caption 这类本就在线程里跑的入库路径使用
    def ensure_probed_sync(self) -> bool:
        """确保可用性探测已执行（同步），返回是否可用."""
        if not self.configured:
            return False
        if not self._probe_done:
            self._available = self._probe_sync()
            self._probe_done = True
            if self._available:
                logger.info("Vision service ready (model=%s)", self.model)
        return bool(self._available)

    def is_available_sync(self) -> bool:
        """
        同步可用性判定（供引擎层的 ``is_available()`` 使用）.

        ⚠️ 不要用 ``is_available()`` —— 它是 **async** 的，同步调用会拿到一个
        coroutine 对象，而 coroutine 恒为真值，于是"模型没装"也会被判成可用，
        兜底分支白跑一趟。引擎探测一律走本方法。
        """
        return self.ensure_probed_sync()

    def describe_image_sync(self, image_bytes: bytes, **kwargs) -> str | None:
        """同步入库期 caption（probe 失败 / 未配置时返回 None）."""
        if not image_bytes:
            return None
        if not self.ensure_probed_sync():
            return None
        settings = get_settings()
        return self._generate_sync(
            image_bytes,
            kwargs.get("prompt") or CAPTION_PROMPT,
            timeout=float(kwargs.get("timeout") or settings.VISION_TIMEOUT_SECONDS),
        )


_service: VisionService | None = None
_service_lock = threading.Lock()


def get_vision_service() -> VisionService:
    """
    进程级单例.

    历史上这里每次 ``return VisionService()`` —— 于是可用性缓存（``_probe_done``
    / ``_available``）形同虚设，**每次调用都会重新探测一遍 Ollama**。图片管线一
    张图会多次问"vision 可用吗"，这个开销被放大得很厉害。
    """
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = VisionService()
    return _service


__all__ = ["VisionService", "get_vision_service", "CAPTION_PROMPT"]
