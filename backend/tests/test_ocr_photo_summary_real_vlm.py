"""
真实模型端到端：一张**真实的文档照片** → 识别 → 文档总结.

与 ``test_ocr_photo_summary_e2e.py`` 的分工
────────────────────────────────────────────
* 那个文件用替身把**链路**钉死：快、每次必跑、带突变对照，回答"结构上通不通"。
* 本文件用**真实模型**跑一次，回答"端到端到底出不出内容"：慢，默认跳过。

门控
────
``RUN_REAL_VLM=1`` 才执行（图像理解 40~60s + 总结 10~30s）：

    set RUN_REAL_VLM=1 && python -m pytest tests/test_ocr_photo_summary_real_vlm.py -q -s

依赖本机 Ollama 提供视觉模型（``OLLAMA_VISION_MODEL``，默认 qwen2.5vl:3b）与文本模型
（``OLLAMA_MODEL``）。任一不可用 → skip，**不算失败**（缺环境 ≠ 功能坏了）。

⚠️ 宿主机没有真实 OCR 引擎（pytesseract / paddleocr 由 conftest 垫片顶替），
所以本文件真实覆盖的是**多模态识别那一段**（VLM 走 HTTP，是真的）；
容器内运行时同一段会带上真实 OCR。链路本身（OCR 文本 → chunk → payload → digest
→ prompt）由 ``test_ocr_photo_summary_e2e.py`` 用真值替身覆盖。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_VLM") != "1",
    reason="真实模型端到端默认跳过 —— 设 RUN_REAL_VLM=1 开启（需本机 Ollama）",
)

try:
    from PIL import Image

    from app.config import get_settings
    from app.services.chunker import build_image_chunks
    from app.services.image_understanding import understand_image
    from app.services.nodes.document_summary_node import (
        build_single_doc_messages,
        build_summary_llm,
    )
    from app.services.parsers.base import ExtractedImage
except ImportError as exc:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")


_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "_doc_photo_fixtures"
_PHOTO = _FIXTURE_DIR / "t0_clean_fullpage.png"
_TRUTH = _FIXTURE_DIR / "groundtruth.json"


def _groundtruth() -> dict:
    data = json.loads(_TRUTH.read_text(encoding="utf-8"))
    return next(f for f in data["fixtures"] if f["name"] == "t0_clean_fullpage")


def _require_fixtures() -> None:
    if not _PHOTO.exists():
        pytest.skip(f"缺少文档照片夹具 {_PHOTO}")
    if not _TRUTH.exists():
        pytest.skip(f"缺少真值文件 {_TRUTH}")


def _ollama_available() -> bool:
    import urllib.request

    base = get_settings().OLLAMA_BASE_URL.rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=5).read()
        return True
    except Exception:  # noqa: BLE001
        return False


def test_real_doc_photo_flows_into_document_summary() -> None:
    """真实照片 → 真实识别 → 真实总结：总结必须包含照片上的关键数字."""
    _require_fixtures()
    if not _ollama_available():
        pytest.skip("本机 Ollama 不可用")

    truth = _groundtruth()
    settings = get_settings()
    image = Image.open(_PHOTO).convert("RGB")

    # ── 1. 真实识别（VLM 走 HTTP）────────────────────────────────────────────
    understanding = understand_image(image, page_number=1, filename=_PHOTO.name)

    extracted = ExtractedImage(
        image_id="real-p1-i1",
        page_number=1,
        ocr_text=understanding.ocr_text or "",
        vision_caption=understanding.vision_caption,
        image_path=None,
        image_type=understanding.image_type,
        structured_content=understanding.structured_content,
        ocr_engine=understanding.ocr_engine,
        analyze_engine=understanding.analyze_engine,
        analyze_confidence=understanding.confidence,
        manual_review=understanding.manual_review,
        position=1,
    )

    searchable = extracted.searchable_text.strip()
    print(f"\n[识别] route={understanding.route} engine={understanding.analyze_engine}")
    print(f"[识别] 正文长度={len(searchable)}")
    print(f"[识别] 正文前 400 字:\n{searchable[:400]}")

    assert searchable, (
        "真实照片没有产出任何可检索文本 —— 这张图在文档总结里会彻底消失"
        f"（meta={dict(understanding.meta or {})}）"
    )

    # ── 2. 建块 + 载荷（与线上同一路径）──────────────────────────────────────
    chunks = build_image_chunks([extracted])
    assert len(chunks) == 1, f"识别出内容却没建出 chunk（实际 {len(chunks)}）"
    payload = chunks[0].to_dict()
    assert payload["text"].strip(), "image chunk 的 payload 没有正文"

    # ── 3. 真实总结（与 document_summary 分支同一 prompt / 同一模型）──────────
    digest = {
        "document_id": "real-doc",
        "filename": _PHOTO.name,
        "digest": payload["text"][:1200],
        "page_count": 1,
        "chunk_count": 1,
        "sampled_chunks": 1,
        "warnings": [],
        "index": 1,
    }
    llm = build_summary_llm(reasoning=False)
    messages = build_single_doc_messages("总结一下这份文档讲了什么", digest)
    reply = llm.invoke(messages)
    summary = reply.content if isinstance(reply.content, str) else str(reply.content)

    print(f"\n[总结]\n{summary}\n")

    assert summary.strip(), "总结为空"
    assert len(summary.strip()) >= 30, f"总结过短，像是没吃到内容: {summary!r}"

    # ── 4. 内容真的进去了吗：总结里应能对上照片上的数字 ───────────────────────
    numbers = {row[2] for row in truth["rows"]} | {row[3] for row in truth["rows"]}
    hit = [n for n in numbers if n.replace("%", "").replace("+", "") in summary]
    print(f"[核对] 真值数字命中: {hit} / 全部 {sorted(numbers)}")
    assert hit, (
        "总结里一个真值数字都没有 —— 说明照片内容没进 LLM 输入"
        f"\n总结={summary!r}\n正文={searchable[:300]!r}"
    )
