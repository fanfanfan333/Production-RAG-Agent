"""#17「无依据句显式标注」回归 —— 每个不误标/要标的情形留一条能跑的检查.

用户裁定：答案里「找不到来源支持」的句子要**显式标注**（不静默、不改答案文字）。
前端只对 `supported=false 且 evidence_available!==false` 的引用做标注；本文件
把这条语义固化在后端：

  1. 中文句（正常文本来源）       → supported=True（不标）
  2. 图片块（结论来自 vision）    → supported=True（不标）——来源含 vision 即可比对
  3. 表格块来源（Markdown 表格）  → supported=True（不标）
  4. 不透明来源（无任何 token）   → evidence_available=False（**不得**标"无依据"）
  5. 真正无依据（原文不支持）     → supported=False 且 evidence_available=True（要标）
  6. payload 契约：as_dict 带 evidence_available
  7. 前端渲染通道：source-citations.tsx 有 UnsupportedSentences 且接进 SourceCitations

纯函数/静态检查，容器内运行：
    docker exec -w /app rag_backend python -m pytest -q tests/test_unsupported_annotation_17.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _report(answer: str, sources: list[dict]):
    from app.services.nodes.citation_verifier import verify_citations

    return verify_citations(answer, sources, annotate=False)


def _annotate(verdicts) -> list[str]:
    """前端 UnsupportedSentences 的判定口径（supported=False 且可比对）."""
    return [v.sentence for v in verdicts if (not v.supported) and v.evidence_available]


# ── 1. 中文句 ────────────────────────────────────────────────────────────────

def test_chinese_sentence_supported_not_annotated():
    rep = _report(
        "公司2024年度营业收入为100万元[Source 1]。",
        [{"text": "公司2024年度营业收入为100万元，同比增长12%。"}],
    )
    assert rep.verdicts[0].supported is True
    assert rep.verdicts[0].evidence_available is True
    assert _annotate(rep.verdicts) == []


# ── 2. 图片块：结论来自 vision ───────────────────────────────────────────────

def test_vision_source_supported_not_annotated():
    rep = _report(
        "图中显示2023年销量为120万辆[Source 1]。",
        [
            {
                "content_type": "image",
                "image_type": "chart",
                "text_snippet": "OCR 文本：销量趋势图。",
                "vision": "图中显示 2023 年销量为 120 万辆。",
                "image_caption": "一张销量柱状图",
            }
        ],
    )
    v = rep.verdicts[0]
    assert v.supported is True, v.as_dict()
    assert v.evidence_available is True
    assert _annotate(rep.verdicts) == []


# ── 3. 表格块来源 ────────────────────────────────────────────────────────────

def test_table_source_supported_not_annotated():
    rep = _report(
        "第三季度华东区销售额为350万元[Source 1]。",
        [
            {
                "content_type": "table",
                "text_snippet": "| 区域 | 季度 | 销售额 |\n|---|---|---|\n| 华东 | Q3 | 350万元 |",
            }
        ],
    )
    v = rep.verdicts[0]
    assert v.supported is True, v.as_dict()
    assert v.evidence_available is True
    assert _annotate(rep.verdicts) == []


# ── 4. 不透明来源：拿不到 token/片段 => 不得标"无依据" ──────────────────────

def test_opaque_source_not_annotated():
    rep = _report(
        "该方案分为三期实施[Source 1]。",
        [{"content_type": "image", "image_type": "photo"}],
    )
    v = rep.verdicts[0]
    # supported 仍为 False（支持度阈值语义不变），但 evidence_available=False
    # 表明这是"无从判断"——前端据此**不**标无依据（宁可少标，不可错标）。
    assert v.supported is False
    assert v.evidence_available is False, v.as_dict()
    assert _annotate(rep.verdicts) == []


def test_punctuation_only_source_not_annotated():
    """来源文本全是标点/符号 → tokenize 为空 → 同样属无从判断."""
    rep = _report(
        "这段话没有任何依据[Source 1]。",
        [{"text": "……！！！"}],
    )
    assert rep.verdicts[0].evidence_available is False
    assert _annotate(rep.verdicts) == []


# ── 5. 真正无依据 => 要标 ────────────────────────────────────────────────────

def test_genuinely_unsupported_is_annotated():
    rep = _report(
        "公司计划在火星建立生产基地[Source 1]。",
        [{"text": "公司2024年度营业收入为100万元。"}],
    )
    v = rep.verdicts[0]
    assert v.supported is False
    assert v.evidence_available is True
    assert _annotate(rep.verdicts) == [v.sentence]


# ── 6. payload 契约 ──────────────────────────────────────────────────────────

def test_as_dict_exposes_evidence_available():
    rep = _report("A[Source 1]。", [{"text": "A。"}])
    d = rep.verdicts[0].as_dict()
    assert "evidence_available" in d
    # as_audit 也要带上（SSE citation_check 走的就是它）
    audit = rep.as_audit()
    assert "evidence_available" in audit["verdicts"][0]


# ── 7. 前端渲染通道 ──────────────────────────────────────────────────────────
#
# 后端镜像里没有前端源码（/app 只有 backend），所以这里按候选根目录查找；
# 都找不到就 skip（前端回归的权威手段是 `tsc --noEmit`）。验证前端改动时用
#   RAG_REPO_ROOT=/repo 并 `docker cp` 出前端快照，即可让本检查真正执行。

import os

import pytest


def _find_frontend(rel: str) -> Path | None:
    roots = [os.getenv("RAG_REPO_ROOT"), str(_REPO_ROOT), "/repo"]
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / rel
        if candidate.exists():
            return candidate
    return None


def test_frontend_renders_unsupported_annotation():
    markers = _find_frontend("components/chat/source-citations.tsx")
    types_path = _find_frontend("lib/types.ts")
    normalize_path = _find_frontend("lib/api/normalize.ts")
    if markers is None or types_path is None or normalize_path is None:
        pytest.skip("前端源码未挂载到本容器（用 RAG_REPO_ROOT 指向仓库根以执行）")

    src = markers.read_text(encoding="utf-8")
    assert "function UnsupportedSentences" in src
    assert "<UnsupportedSentences" in src
    assert "evidence_available !== false" in src
    # 类型定义必须带上该字段，否则前端读不到
    assert "evidence_available" in types_path.read_text(encoding="utf-8")
    # SSE 归一化必须**透传**该字段，否则后端发了也会被 normalizeVerdicts 丢掉
    assert "evidence_available" in normalize_path.read_text(encoding="utf-8")
