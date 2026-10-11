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
（``OLLAMA_MODEL``）。显式开启后缺模型或夹具必须失败，防止发布门禁静默跳过。

⚠️ 宿主机没有真实 OCR 引擎（pytesseract / paddleocr 由 conftest 垫片顶替），
所以本文件真实覆盖的是**多模态识别那一段**（VLM 走 HTTP，是真的）；
容器内运行时同一段会带上真实 OCR。链路本身（OCR 文本 → chunk → payload → digest
→ prompt）由 ``test_ocr_photo_summary_e2e.py`` 用真值替身覆盖。
"""

from __future__ import annotations

import json
import asyncio
import importlib
import os
import re
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
    if os.getenv("RUN_REAL_VLM") == "1":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")


_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "_doc_photo_fixtures"
_PHOTO = _FIXTURE_DIR / "t0_clean_fullpage.png"
_TRUTH = _FIXTURE_DIR / "groundtruth.json"


def _groundtruth(fixture_name: str) -> dict:
    data = json.loads(_TRUTH.read_text(encoding="utf-8"))
    return next(f for f in data["fixtures"] if f["name"] == fixture_name)


def _require_fixtures() -> None:
    if not _PHOTO.exists():
        pytest.fail(f"缺少文档照片夹具 {_PHOTO}")
    if not _TRUTH.exists():
        pytest.fail(f"缺少真值文件 {_TRUTH}")


def _ollama_available() -> bool:
    import urllib.request

    base = get_settings().OLLAMA_BASE_URL.rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=5).read()
        return True
    except Exception:  # noqa: BLE001
        return False


def _unload_test_model(model: str) -> None:
    import urllib.request
    request = urllib.request.Request(
        get_settings().OLLAMA_BASE_URL.rstrip("/") + "/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        assert json.load(response).get("done") is True
    import time
    for _ in range(60):
        with urllib.request.urlopen(get_settings().OLLAMA_BASE_URL.rstrip("/") + "/api/ps", timeout=5) as response:
            models = json.load(response).get("models")
        assert isinstance(models, list), "cannot verify model memory release"
        if not any(item.get("name") == model or item.get("model") == model for item in models):
            time.sleep(1)
            return
        time.sleep(.25)
    raise AssertionError(f"test model did not unload: {model}")


@pytest.fixture(autouse=True)
def release_real_test_models():
    yield
    settings = get_settings()
    for model in {settings.OLLAMA_VISION_MODEL, settings.OLLAMA_MODEL}:
        _unload_test_model(model)


def test_real_image_dependencies_are_not_host_shims() -> None:
    from app.services.image_understanding.dual_channel import dual_channel_enabled
    assert dual_channel_enabled("table"), "发布配置必须开启表格交叉校验"
    for name in ("cv2", "paddleocr", "torch", "docling", "asyncpg"):
        module = importlib.import_module(name)
        assert not vars(module).get("__host_shim__", False), f"真实图片门禁缺少 {name}"


@pytest.mark.parametrize("fixture_name", ["t1_clean_titled", "t2_photo_jpeg", "t5_skew5", "t6_skew15", "t7_hlines_only", "t8_header_filled"])
def test_real_degraded_table_cells_or_explicit_review(fixture_name, record_property):
    _require_fixtures()
    assert _ollama_available(), "真实视觉模型不可用"
    truth = _groundtruth(fixture_name)
    image = Image.open(_FIXTURE_DIR / f"{fixture_name}.png").convert("RGB")
    result = understand_image(image, page_number=1, filename=f"{fixture_name}.png")
    grid = []
    for line in (result.structured_content or "").splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip()[1:-1])]
            if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                grid.append(cells)
    expected = [truth["header"], *truth["rows"]]
    mismatches = [f"{row}:{column}={cell}" for row, cells in enumerate(expected) for column, cell in enumerate(cells)
                  if row >= len(grid) or column >= len(grid[row]) or grid[row][column] != cell]
    record_property("mismatched_cells", json.dumps(mismatches, ensure_ascii=False))
    record_property("manual_review", result.manual_review)
    assert result.structured_content or result.ocr_text or result.vision_caption, "整张图片丢失"
    assert not mismatches or result.manual_review, f"存在未提示的单元格错误：{mismatches}"


@pytest.mark.parametrize("fixture_name", ["t0_clean_fullpage", "t3_photo_shadow", "t4_photo_perspective"])
def test_real_doc_photo_flows_into_document_summary(fixture_name: str, monkeypatch) -> None:
    """真实照片 → 真实识别 → 真实总结：总结必须包含照片上的关键数字."""
    _require_fixtures()
    if not _ollama_available():
        pytest.fail("本机 Ollama 不可用，真实模型门禁不能跳过")

    truth = _groundtruth(fixture_name)
    settings = get_settings()
    photo = _FIXTURE_DIR / f"{fixture_name}.png"
    assert photo.exists(), f"缺少真实夹具 {photo}"
    image = Image.open(photo).convert("RGB")

    # ── 1. 真实识别（VLM 走 HTTP）────────────────────────────────────────────
    understanding = understand_image(image, page_number=1, filename=photo.name)

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
        analyze_fusion=understanding.fusion,
        analyze_quality=understanding.quality,
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
    assert understanding.table is not None and understanding.table.cols == len(truth["header"]), (
        f"表格列数错位：expected={len(truth['header'])}, actual={understanding.table}"
    )
    assert all(value in searchable for value in truth["header"]), "真实表头未完整恢复"
    assert all(row[2] in searchable for row in truth["rows"]), "2024 数值识别存在遗漏"
    missing_cells = [cell for row in truth["rows"] for cell in row if cell not in searchable]
    assert not missing_cells or understanding.manual_review, (
        f"单元格存在误识别但未提示复核：{missing_cells}"
    )
    if understanding.fusion.get("numeric_conflict"):
        assert understanding.manual_review and not understanding.fusion["agreed"]
        assert "不能据此确定数值" in searchable

    # ── 2. 建块 + 载荷（与线上同一路径）──────────────────────────────────────
    chunks = build_image_chunks([extracted])
    assert len(chunks) == 1, f"识别出内容却没建出 chunk（实际 {len(chunks)}）"
    payload = chunks[0].to_dict()
    assert payload["text"].strip(), "image chunk 的 payload 没有正文"

    # ── 3. 真实总结（与 document_summary 分支同一 prompt / 同一模型）──────────
    digest = {
        "document_id": "real-doc",
        "filename": photo.name,
        "digest": payload["text"][:1200],
        "page_count": 1,
        "chunk_count": 1,
        "sampled_chunks": 1,
        "warnings": ["图片识别结果需复核，请核对原文"] if understanding.manual_review else [],
        "index": 1,
    }
    from app.services import master_graph
    monkeypatch.setattr(master_graph, "build_summary_llm", lambda **kwargs: build_summary_llm(reasoning=False))
    # Keep vision and text weights from occupying host RAM together.
    _unload_test_model(get_settings().OLLAMA_VISION_MODEL)
    summary = asyncio.run(master_graph._summarize_node({"query": "总结一下这份文档讲了什么", "digests": [digest]}))["answer"]

    print(f"\n[总结]\n{summary}\n")

    assert summary.strip(), "总结为空"
    assert "总结生成失败" not in summary, "真实总结超时或报错，不能计作通过"
    assert len(summary.strip()) >= 30, f"总结过短，像是没吃到内容: {summary!r}"

    # ── 4. 内容真的进去了吗：总结里应能对上照片上的数字 ───────────────────────
    numbers = {row[2] for row in truth["rows"]} | {row[3] for row in truth["rows"]}
    hit = [n for n in numbers if n.replace("%", "").replace("+", "") in summary]
    print(f"[核对] 真值数字命中: {hit} / 全部 {sorted(numbers)}")
    value_hits = sum(row[2] in summary for row in truth["rows"])
    if understanding.manual_review:
        assert any(word in summary for word in ("复核", "核对", "不一致", "分歧")), "总结未传达识别不确定性"
    assert value_hits >= 3 or understanding.manual_review, (
        "总结里一个真值数字都没有 —— 说明照片内容没进 LLM 输入"
        f"\n总结={summary!r}\n正文={searchable[:300]!r}"
    )
