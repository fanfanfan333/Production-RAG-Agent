"""
「图里一个字都没有时，不要直接跳过这张图」—— 回归测试（实施手册 3.3.1）.

被验证的行为（改动前的旧行为是"无文字图片在分块阶段被直接跳过"）：

  1. ``TEXTLESS_SUMMARY_PROMPT`` 必须是手册要求的三段式：图注 + 关键要素 + 数值信息，
     并要求"看不清就说不确定、不得编造"。
  2. ``VisionEngine`` 的 role 分流：``summarize`` 走总结提示词；``primary`` 仍走
     按类型的专用提示词（prompt=None）；``fallback`` 与**未知 role** 仍走通用转写
     提示词 —— 保证既有调用方语义零变化。
  3. 管线：三条文本字段（structured_content / ocr_text / vision_caption）全空时，
     调一次总结推理并把它写进 ``vision_caption``；有文字时**绝不**多花这次推理。
  4. 总结文本**不得**写进 ``structured_content``：那是"可核对的结构"（Markdown
     表格 / 代码 / LaTeX），塞散文进去会把 content_type 误标成 table。
  5. ``route`` 必须反映真实产出：总结生效时记成 vision，原 route 留在 meta。
  6. 端到端：总结生效后这张图在 ``build_image_chunks`` 里**真的建出了 chunk**
     （旧行为是 0 个 chunk = 图片消失）。
  7. 两个"保持旧行为"的开关：``IMAGE_SUMMARY_WHEN_TEXTLESS=false`` 与
     vision 不可用 —— 都不得产出总结，也不得崩。

宿主机缺依赖时优雅跳过（与既有单测同一约定）；推荐在 backend 容器内运行：
    docker exec -w /app -e PYTHONPATH=/app rag_backend \
        python -m pytest tests/test_image_textless_summary.py -q
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from PIL import Image

    from app.config import get_settings
    from app.services.chunker import build_image_chunks
    from app.services.image_understanding import IMAGE_TYPE_TABLE
    from app.services.image_understanding.classifier import ImageClassification
    from app.services.image_understanding.engines.base import EngineOutput
    from app.services.image_understanding.pipeline import (
        ImageUnderstanding,
        _has_any_text,
        _summarize_if_textless,
    )
    from app.services.parsers.base import ExtractedImage

    pl = importlib.import_module("app.services.image_understanding.pipeline")
    an = importlib.import_module("app.services.image_understanding.analyzer")
    ve = importlib.import_module("app.services.image_understanding.engines.vision_engine")
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")


SUMMARY_TEXT = (
    "图注：这是一张系统架构示意图，展示检索链路。\n"
    "关键要素：用户 → 检索服务 → 向量库。\n"
    "数值信息：图中未标注数值。"
)


class _RecordingVision:
    """可配置的 VisionEngine 替身：记录 role、可指定"什么 role 才给产出"."""

    available = True
    # None = 所有 role 都给产出；否则只有该 role 给产出（其余返回 ok=False）
    only_role: str | None = "summarize"
    # True = 一律失败（模拟模型/传输层故障）
    force_fail: bool = False
    calls: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    def is_available(self) -> bool:
        return type(self).available

    def process(self, image, *, png_bytes=None, image_type="photo", role="primary", **kw):
        type(self).calls.append(role)
        if type(self).force_fail:
            return EngineOutput(engine="vision", ok=False, error="Vision 超时（120s）")
        if type(self).only_role is not None and role != type(self).only_role:
            return EngineOutput(engine="vision", ok=False, error="空结果")
        return EngineOutput(text=SUMMARY_TEXT, confidence=0.75, engine="vision", ok=True)


def _img() -> "Image.Image":
    return Image.new("RGB", (320, 200), (255, 255, 255))


def _understanding(image_type: str = "photo", route: str = "ocr") -> ImageUnderstanding:
    return ImageUnderstanding(
        image_type=image_type,
        route=route,
        classification=ImageClassification(
            image_type=image_type, confidence=0.8, signals={}, engine="rules",
        ),
        quality={"ok": False, "score": 0.0, "reasons": ["没有可评估的产出"]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. 提示词本身：手册 3.3.1 要求的「图注 + 关键要素 + 数值信息」三段式
# ─────────────────────────────────────────────────────────────────────────────


def test_summary_prompt_is_manual_three_part() -> None:
    p = an.TEXTLESS_SUMMARY_PROMPT
    for token in ("图注", "关键要素", "数值信息"):
        assert token in p, f"实施手册 3.3.1 要求三段式，提示词缺 {token}: {p!r}"
    # 手册表 3-2："描述中缺少数值"是不推荐做法 → 提示词必须逼模型交代数值
    assert "图中未标注数值" in p, "图里没有数字时也要有明确交代，不能假装有"
    # VLM 幻觉是这条链路最贵的错误：看不清必须说看不清
    assert "不要编造" in p and "不确定" in p, p
    print("  ok test_summary_prompt_is_manual_three_part")


# ─────────────────────────────────────────────────────────────────────────────
# 2. VisionEngine 的 role 分流：新增 summarize，不动 primary / fallback / 未知
# ─────────────────────────────────────────────────────────────────────────────


def test_vision_engine_role_prompts() -> None:
    seen: list[str | None] = []
    orig = ve.analyze_image_sync

    def _fake(payload, image_type, *, prompt=None, timeout=None):
        seen.append(prompt)
        return "文本"

    ve.analyze_image_sync = _fake  # type: ignore[assignment]
    try:
        engine = ve.VisionEngine()
        out_primary = engine.process(_img(), image_type="chart", role="primary")
        out_fallback = engine.process(_img(), image_type="chart", role="fallback")
        out_summary = engine.process(_img(), image_type="photo", role="summarize")
        out_unknown = engine.process(_img(), image_type="chart", role="legacy-role")
    finally:
        ve.analyze_image_sync = orig  # type: ignore[assignment]

    assert seen[0] is None, "primary 必须保持 None（由 analyzer 按类型选提示词）"
    assert seen[1] == an.FALLBACK_PROMPT, "fallback 提示词不得被改动"
    assert seen[2] == an.TEXTLESS_SUMMARY_PROMPT, "summarize 必须用三段式总结提示词"
    assert seen[3] == an.FALLBACK_PROMPT, "未知 role 必须保持历史行为（通用转写）"

    assert out_primary.meta["prompt_type"] == "chart"
    assert out_fallback.meta["prompt_type"] == "fallback"
    assert out_summary.meta["prompt_type"] == "summarize"
    assert out_unknown.meta["prompt_type"] == "fallback"
    assert out_summary.ok is True and out_summary.text == "文本"
    print("  ok test_vision_engine_role_prompts")


# ─────────────────────────────────────────────────────────────────────────────
# 3~6. 管线分支：无文字 → 总结；有文字 → 一次推理都不多花
# ─────────────────────────────────────────────────────────────────────────────

#: 真跑实测（qwen2.5vl:3b + 一张无文字示意图）时，模型把兜底转写提示词**逐条
#: 复读**回来的原文 —— 开头那句"请尽最大努力…"被丢掉了，所以 quality.py 里按
#: 关键词匹配的泄漏检测一个都没命中。原样留在这里当回归夹具。
OBSERVED_PROMPT_ECHO = (
    "1) 如果有文字，逐行忠实转写（保留原有顺序与换行）；\n"
    "2) 如果是表格，用 Markdown 表格输出；\n"
    "3) 如果是图表，列出数据点与数值；\n"
    "4) 如果是流程图/结构图，列出节点与连线关系；\n"
    "5) 如果是公式，用 LaTeX 输出；\n"
    "6) 如果是照片，描述画面内容与其中可读的文字。"
)

#: 真跑实测的真实描述（同一次测试里用总结提示词拿到的输出）
OBSERVED_REAL_DESCRIPTION = (
    "1) 图注：这是一张流程图，展示了三个步骤之间的关系，每个步骤用不同颜色的圆形表示。\n"
    "2) 关键要素：第一个圆形：蓝色，表示步骤1；第二个圆形：红色，表示步骤2；"
    "第三个圆形：绿色，表示步骤3；第四个区域：灰色，表示步骤4。\n"
    "3) 数值信息：图中未标注数值。"
)


def test_prompt_echo_detection_on_real_samples() -> None:
    """复读检测必须认出实测样本，且不能冤枉真实描述."""
    assert an.looks_like_prompt_echo(OBSERVED_PROMPT_ECHO, an.FALLBACK_PROMPT) is True
    assert an.looks_like_prompt_echo(
        OBSERVED_REAL_DESCRIPTION, an.TEXTLESS_SUMMARY_PROMPT
    ) is False, "真实描述不能被当成复读拒掉"
    assert an.looks_like_prompt_echo(OBSERVED_PROMPT_ECHO, "") is False, "没有提示词就无从比对"
    assert an.looks_like_prompt_echo("", an.FALLBACK_PROMPT) is False
    # 单行输出不判（短文本"碰巧"重合的概率太高）
    assert an.looks_like_prompt_echo(
        "如果是表格，用 Markdown 表格输出；", an.FALLBACK_PROMPT
    ) is False
    print("  ok test_prompt_echo_detection_on_real_samples")


def test_vision_engine_rejects_prompt_echo(monkeypatch) -> None:
    """引擎层把复读判为失败 —— 这一段文字绝不能落成 vision_caption."""
    monkeypatch.setattr(ve, "analyze_image_sync", lambda *a, **kw: OBSERVED_PROMPT_ECHO)
    out = ve.VisionEngine().process(_img(), image_type="photo", role="fallback")

    assert out.ok is False, "复读必须判失败（否则指令会被当成图意描述入库）"
    assert "复读" in (out.error or ""), out.error
    assert not (out.text or ""), "复读的文本不得随产出带出去"

    monkeypatch.setattr(ve, "analyze_image_sync", lambda *a, **kw: OBSERVED_REAL_DESCRIPTION)
    out2 = ve.VisionEngine().process(
        _img(), image_type="photo", role="summarize",
    )
    assert out2.ok is True, "真实描述必须照常通过"
    print("  ok test_vision_engine_rejects_prompt_echo")


def test_echoed_fallback_no_longer_blocks_the_summary(monkeypatch) -> None:
    """
    复现真跑缺陷 + 证明修复：兜底转写被复读时，整条管线应转而产出图意总结.

    这里**不替换 VisionEngine**（用真的引擎层），只把模型换成"和真机一样的行为"：
    非总结提示词 → 复读指令；总结提示词 → 真实三段式描述。改动前这段复读会被
    当成"有文字"，于是总结分支永远不触发、入库的是提示词原文。
    """
    def _fake_model(payload, image_type, *, prompt=None, timeout=None):
        return OBSERVED_REAL_DESCRIPTION if prompt == an.TEXTLESS_SUMMARY_PROMPT else OBSERVED_PROMPT_ECHO

    monkeypatch.setattr(ve, "analyze_image_sync", _fake_model)
    monkeypatch.setattr(pl, "_run_ocr", lambda image: EngineOutput(
        text="", confidence=0.0, engine="tesseract", ok=True, lines=[],
    ))
    monkeypatch.setattr(pl, "classify_image_safe", lambda image, ocr_lines=None, filename="": ImageClassification(
        image_type="photo", confidence=0.7, signals={}, engine="rules",
    ))
    monkeypatch.setattr(pl, "_dispatch", lambda *a, **kw: EngineOutput(
        text="", confidence=0.0, engine="paddleocr", ok=False,
    ))
    monkeypatch.setattr(pl, "_second_ocr", lambda image, ocr_out: None)
    monkeypatch.setattr(ve.VisionEngine, "is_available", lambda self: True)

    result = pl.understand_image(_img(), page_number=1, filename="textless.png")

    assert result.vision_caption == OBSERVED_REAL_DESCRIPTION, (
        f"应产出图意总结而不是提示词原文（实际 {result.vision_caption!r}）"
    )
    assert OBSERVED_PROMPT_ECHO not in (result.vision_caption or "")
    assert result.meta.get("textless_summary") is True, result.meta
    assert result.analyze_engine == "vision:summary"
    chunk_owner = ExtractedImage(
        image_id="d-p1-i1", page_number=1, ocr_text="",
        vision_caption=result.vision_caption, image_path="images/a.png",
        image_type=result.image_type, analyze_engine=result.analyze_engine,
    )
    chunks = build_image_chunks([chunk_owner])
    assert len(chunks) == 1 and "图注" in chunks[0].text, chunks
    print("  ok test_echoed_fallback_no_longer_blocks_the_summary")


def _patch_vision(monkeypatch, *, available: bool = True) -> type[_RecordingVision]:
    monkeypatch.setattr(pl, "VisionEngine", _RecordingVision)
    _RecordingVision.available = available
    _RecordingVision.only_role = "summarize"
    _RecordingVision.force_fail = False
    _RecordingVision.calls = []
    return _RecordingVision


def test_textless_image_is_summarized_not_skipped(monkeypatch) -> None:
    stub = _patch_vision(monkeypatch)
    result = _understanding(image_type="photo", route="ocr")

    assert _has_any_text(result) is False, "前置：这张图确实一个字都没有"
    ok = _summarize_if_textless(
        _img(), lambda: b"png", result, page_number=1, filename="arch.png",
    )

    assert ok is True, "无文字图片必须拿到总结，而不是被丢掉"
    assert stub.calls == ["summarize"], f"只应调一次总结推理（实际 {stub.calls}）"
    assert result.vision_caption == SUMMARY_TEXT
    assert result.meta.get("textless_summary") is True, result.meta
    assert result.meta.get("textless_summary_chars") == len(SUMMARY_TEXT)
    assert result.analyze_engine == "vision:summary"
    # route 必须反映真实产出：这段文字确实来自多模态模型
    assert result.route == "vision", result.route
    assert result.meta.get("original_route") == "ocr", result.meta
    # 有产出了就不该再标人工复核
    assert result.manual_review is False
    assert result.meta.get("fallback_failed") is None
    # 质检里留下"这是总结不是识别"的可机读痕迹，且**过期结论必须被替换掉**
    reasons = [str(r) for r in result.quality.get("reasons", [])]
    assert len(reasons) == 1 and "图意总结" in reasons[0] and "未经结构校验" in reasons[0], reasons
    assert result.quality.get("summary_generated") is True
    assert "manual_review_reason" not in result.quality, (
        "走到这一步时旧结论（如 no_engine_output）已过期，留着会让前端显示与事实相反的话"
    )
    assert result.attempts and result.attempts[-1]["chars"] == len(SUMMARY_TEXT)
    print("  ok test_textless_image_is_summarized_not_skipped")


def test_image_with_text_is_not_summarized(monkeypatch) -> None:
    """有文字（哪怕只有几个字）→ 不得多花这次最贵的推理."""
    stub = _patch_vision(monkeypatch)
    result = _understanding()
    result.ocr_text = "检索服务"

    assert _has_any_text(result) is True
    ok = _summarize_if_textless(
        _img(), lambda: b"png", result, page_number=1, filename="x.png",
    )

    assert ok is False
    assert stub.calls == [], f"有文字的图不应触发总结推理（实际 {stub.calls}）"
    assert result.vision_caption is None
    assert result.route == "ocr", "未总结时 route 不得被动过"
    print("  ok test_image_with_text_is_not_summarized")


def test_vision_caption_alone_also_counts_as_text(monkeypatch) -> None:
    """已有 vision_caption（上游 VLM 已产出）→ 同样不算"无文字"."""
    stub = _patch_vision(monkeypatch)
    result = _understanding(image_type="chart", route="vision")
    result.vision_caption = "柱状图：Q1 120 万，Q2 150 万"

    assert _summarize_if_textless(_img(), lambda: b"p", result, page_number=1, filename="c.png") is False
    assert stub.calls == []
    print("  ok test_vision_caption_alone_also_counts_as_text")


def test_summary_never_pollutes_structured_content(monkeypatch) -> None:
    """
    表格/代码/公式类型：总结只能落在 vision_caption.

    否则 ``ExtractedImage.content_type`` 会因 structured_content 非空而被判成
    "table"，一段散文就会混进表格检索通道。
    """
    _patch_vision(monkeypatch)
    result = _understanding(image_type=IMAGE_TYPE_TABLE, route="table_parser")

    ok = _summarize_if_textless(
        _img(), lambda: b"png", result, page_number=2, filename="tbl.png",
    )

    assert ok is True
    assert result.structured_content is None, "总结绝不能写进 structured_content"
    assert result.vision_caption == SUMMARY_TEXT

    extracted = ExtractedImage(
        image_id="d-p1-i1", page_number=2,
        vision_caption=result.vision_caption, structured_content=None,
        image_path="images/a.png", image_type=IMAGE_TYPE_TABLE,
    )
    assert extracted.content_type == "image", (
        "无结构化产出的图片表格不能冒充真表格"
    )
    print("  ok test_summary_never_pollutes_structured_content")


def test_summary_switch_off_keeps_old_behaviour(monkeypatch) -> None:
    """``IMAGE_SUMMARY_WHEN_TEXTLESS=false`` → 回到旧行为，且不多花推理."""
    stub = _patch_vision(monkeypatch)
    settings = get_settings()
    monkeypatch.setattr(settings, "IMAGE_SUMMARY_WHEN_TEXTLESS", False, raising=False)
    result = _understanding()

    assert _summarize_if_textless(_img(), lambda: b"p", result, page_number=1, filename="x.png") is False
    assert stub.calls == [], "开关关掉后不得再调模型"
    assert result.vision_caption is None
    assert result.meta.get("textless_summary") is None
    print("  ok test_summary_switch_off_keeps_old_behaviour")


def test_vision_unavailable_keeps_old_behaviour(monkeypatch) -> None:
    """多模态不可用 → 无法总结，如实留痕、不崩、不伪造描述."""
    stub = _patch_vision(monkeypatch, available=False)
    result = _understanding()

    assert _summarize_if_textless(_img(), lambda: b"p", result, page_number=1, filename="x.png") is False
    assert stub.calls == [], "引擎不可用时连 process 都不该进"
    assert result.vision_caption is None
    assert result.meta.get("textless_summary_skipped") == "vision-unavailable", result.meta
    print("  ok test_vision_unavailable_keeps_old_behaviour")


def test_flagged_vision_unavailable_short_circuits(monkeypatch) -> None:
    """meta 里已标 vision_unavailable（如超时降级）→ 不再重复烧推理."""
    stub = _patch_vision(monkeypatch, available=True)
    result = _understanding(image_type="chart", route="ocr")
    result.meta["vision_unavailable"] = True

    assert _summarize_if_textless(_img(), lambda: b"p", result, page_number=1, filename="c.png") is False
    assert stub.calls == []
    print("  ok test_flagged_vision_unavailable_short_circuits")


def test_summary_failure_reports_reason(monkeypatch) -> None:
    """总结推理失败 → 不产出、留痕、保持旧行为（该图仍不建块）."""
    stub = _patch_vision(monkeypatch)
    stub.force_fail = True         # 所有 role 都失败（模拟超时/传输层故障）
    result = _understanding()

    assert _summarize_if_textless(_img(), lambda: b"p", result, page_number=1, filename="x.png") is False
    assert result.meta.get("textless_summary_failed"), result.meta
    assert result.vision_caption is None
    assert result.route == "ocr", "失败时不得改 route"
    print("  ok test_summary_failure_reports_reason")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 端到端：这张图**真的**建出了独立 chunk（旧行为是 0 个 chunk）
# ─────────────────────────────────────────────────────────────────────────────


def test_textless_summary_yields_image_chunk_not_skipped(monkeypatch) -> None:
    _patch_vision(monkeypatch)
    result = _understanding()
    _summarize_if_textless(_img(), lambda: b"png", result, page_number=3, filename="a.png")

    with_summary = ExtractedImage(
        image_id="doc-p3-i1", page_number=3,
        ocr_text=result.ocr_text, vision_caption=result.vision_caption,
        image_path="images/a.png", image_type=result.image_type,
        analyze_engine=result.analyze_engine, position=2,
    )
    chunks = build_image_chunks([with_summary])
    assert len(chunks) == 1, f"总结后的图片必须建出 1 个 chunk（实际 {len(chunks)}）"
    assert SUMMARY_TEXT in chunks[0].text, chunks[0].text
    assert chunks[0].text.startswith("图片描述:"), chunks[0].text
    assert chunks[0].position == 2, "图片位置（第几张图）必须随分块带出"

    # 对照组：旧行为（三处全空）确实会被跳过 —— 证明上面那条不是"本来就通过"
    empty = ExtractedImage(image_id="doc-p3-i2", page_number=3, image_path="images/b.png")
    assert build_image_chunks([empty]) == [], "无文字且无总结的图片仍会被跳过（旧行为）"
    print("  ok test_textless_summary_yields_image_chunk_not_skipped")


# ─────────────────────────────────────────────────────────────────────────────
# 8. 整条 explain_image 管线：分支真的被接上了（不只是函数能被调用）
# ─────────────────────────────────────────────────────────────────────────────


def test_understand_image_wires_the_textless_summary(monkeypatch) -> None:
    """OCR 空 + 分类 photo + 专用引擎空 + VLM 转写空 → 仍应产出图意总结."""
    saved = {
        "run_ocr": pl._run_ocr,
        "classify": pl.classify_image_safe,
        "dispatch": pl._dispatch,
        "second": pl._second_ocr,
        "vision": pl.VisionEngine,
    }
    stub = _RecordingVision()
    try:
        pl._run_ocr = lambda image: EngineOutput(  # type: ignore[assignment]
            text="", confidence=0.0, engine="tesseract", ok=True, lines=[],
        )
        pl.classify_image_safe = lambda image, ocr_lines=None, filename="": ImageClassification(  # type: ignore[assignment]
            image_type="photo", confidence=0.7, signals={}, engine="rules",
        )
        pl._dispatch = lambda *a, **kw: EngineOutput(  # type: ignore[assignment]
            text="", confidence=0.0, engine="paddleocr", ok=False,
        )
        pl._second_ocr = lambda image, ocr_out: None  # type: ignore[assignment]
        pl.VisionEngine = _RecordingVision  # type: ignore[assignment]
        _RecordingVision.available = True
        _RecordingVision.only_role = "summarize"
        _RecordingVision.force_fail = False
        _RecordingVision.calls = []

        result = pl.understand_image(_img(), page_number=3, filename="arch.png")
    finally:
        pl._run_ocr = saved["run_ocr"]            # type: ignore[assignment]
        pl.classify_image_safe = saved["classify"]  # type: ignore[assignment]
        pl._dispatch = saved["dispatch"]          # type: ignore[assignment]
        pl._second_ocr = saved["second"]          # type: ignore[assignment]
        pl.VisionEngine = saved["vision"]         # type: ignore[assignment]

    assert "summarize" in _RecordingVision.calls, (
        f"整条管线里必须真的走到总结分支（实际调用 role={_RecordingVision.calls}）"
    )
    assert result.vision_caption == SUMMARY_TEXT, result.vision_caption
    assert result.meta.get("textless_summary") is True, result.meta
    assert result.route == "vision" and result.meta.get("original_route") == "ocr", (
        result.route, result.meta,
    )
    assert result.analyze_engine == "vision:summary"
    assert result.manual_review is False
    assert result.decision == "pass", result.decision
    print("  ok test_understand_image_wires_the_textless_summary")


if __name__ == "__main__":
    import traceback

    class _MP:
        """最小 monkeypatch 替身（直接跑 python 本文件时用）."""

        def setattr(self, obj, name, value, raising=True):
            setattr(obj, name, value)

    ok = fail = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn(_MP())
            ok += 1
        except Exception:
            fail += 1
            print(f"  FAIL {_name}")
            traceback.print_exc()
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)
