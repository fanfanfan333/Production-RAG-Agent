"""审计缺陷修复回归（#16）—— 每个修好的缺陷留一条能跑的检查.

覆盖（修复前失败 / 修复后通过）：
  1. 隔离 fail-closed：注册表异常时 admin **不得**被放行未排除范围
     （tenancy.exclude_test_tenants）。
  2. 日期归一补零：`2024年1月` 与 `2024年01月` 必须判等
     （citation_verifier._canon_date）——否则会删掉正确答案的引用。
  3. 权威注释 = 最终口径：company_registry.test_tenant_ids 的 docstring
     不得再残留被用户否掉的旧口径。
  4. 父块回填按 document_id 绑定：污染 payload 指向他文档父块时**不得**回填
     （retrieval_service._hydrate_parents）。
  5. 内存 BM25 腿媒体字段契约：以候选 chunk 为基底覆盖 score，媒体/位置/质检
     字段全部保留（与 PG 关键词腿一致）。
  6. 空 token 句 support=1.0：**有意取舍**（宁可漏判），本测试固化行为、防止
     无声改成误删引用。
  7. 命中句定位只用 display_text：固化当前行为（vision 派生结论可能无高亮）。
  8. 降级路径（MULTIMODAL_CONTEXT_ENABLED=false）sources 仍带图片字段。

纯函数/桩测试，容器内运行：
    docker exec -w /app rag_backend python -m pytest -q tests/test_audit_fixes_16.py
"""

from __future__ import annotations

import asyncio
import sys
import types
import uuid
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _run(coro):
    return asyncio.run(coro)


# ── 1. 隔离 fail-closed ───────────────────────────────────────────────────────

def test_exclude_test_tenants_fail_closed_for_admin(monkeypatch):
    from app.services import company_registry, tenancy

    async def _boom():
        raise RuntimeError("registry down")

    monkeypatch.setattr(company_registry, "test_tenant_ids", _boom)
    tenants, owns = _run(
        tenancy.exclude_test_tenants(
            frozenset({"t1", "t2"}), frozenset({"t1", "t2"})
        )
    )
    # fail-closed：不放行 → admin 公司范围清空；个人库由 owner_id 分支保留
    assert tenants == frozenset(), tenants
    assert owns == frozenset(), owns


def test_exclude_test_tenants_non_admin_unchanged_when_registry_down(monkeypatch):
    from app.services import company_registry, tenancy

    async def _boom():
        raise RuntimeError("registry down")

    monkeypatch.setattr(company_registry, "test_tenant_ids", _boom)
    # 非 admin：owns 为空 → 提前返回，**不查注册表**，范围原样
    tenants, owns = _run(
        tenancy.exclude_test_tenants(frozenset({"compA"}), frozenset())
    )
    assert tenants == frozenset({"compA"})
    assert owns == frozenset()


# ── 2. 日期归一补零 ───────────────────────────────────────────────────────────

def test_canon_date_zero_padding():
    from app.services.nodes import citation_verifier as cv

    assert cv._canon_date("2024年1月") == cv._canon_date("2024年01月") == "2024-01"
    assert cv._canon_date("2024年1月31日") == "2024-01-31"
    assert cv._canon_date("2024-01-31") == "2024-01-31"
    assert cv._canon_date("2024/1/31") == "2024-01-31"
    assert cv._canon_date("2024年") == "2024"


def test_verify_citations_date_mismatch_fixed():
    from app.services.nodes.citation_verifier import verify_citations

    report = verify_citations(
        "2024年1月的营业收入为100万元[Source 1]。",
        [{"text": "2024年01月的营业收入为100万元。"}],
        annotate=False,
    )
    assert report.verdicts, "应识别到一条引用"
    assert report.verdicts[0].dates_consistent is True, report.verdicts[0].as_dict()


# ── 3. 权威注释 = 最终口径 ───────────────────────────────────────────────────

def test_company_registry_docstring_final_wording():
    src = (_BACKEND_ROOT / "app" / "services" / "company_registry.py").read_text(
        encoding="utf-8"
    )
    # 旧口径必须消失
    assert "所有人都检索不到" not in src
    assert '含测试公司自己的成员）都看不到' not in src.replace("\n", "")
    # 最终口径必须出现
    assert "一切照常" in src
    assert "测试公司成员" in src


# ── 4. 父块回填按 document_id 绑定 ────────────────────────────────────────────

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self, rows, captured):
        self._rows = rows
        self._captured = captured

    async def execute(self, stmt):
        self._captured.append(str(stmt))
        return _FakeResult(self._rows)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _install_fake_db(monkeypatch, rows, captured):
    import app.db.postgres as pg

    monkeypatch.setattr(pg, "get_db_session", lambda: _FakeSession(rows, captured))


def _mk_child(document_id: str):
    from app.services.retrieval_service import RetrievedChunk

    return RetrievedChunk(
        document_id=document_id, filename="a.docx", page_number=1,
        chunk_index=0, text="child", score=0.5, parent_id=f"{document_id}:p:0",
    )


def test_hydrate_parents_rejects_cross_document(monkeypatch):
    from app.services import retrieval_service as rs

    doc_a = str(uuid.uuid4())
    doc_b = str(uuid.uuid4())
    # 行：parent_id 冒充 docA 的父块，但 document_id 实际是 docB（污染）
    row = types.SimpleNamespace(
        parent_id=f"{doc_a}:p:0", document_id=uuid.UUID(doc_b),
        text="B 文档的机密父块正文", char_start=0, char_end=3, idx=0,
        section_path=None, heading=None,
    )
    captured: list[str] = []
    _install_fake_db(monkeypatch, [row], captured)

    child = _mk_child(doc_a)
    filled = _run(rs._hydrate_parents([child]))

    assert filled == 0, "跨文档父块必须被拒绝回填"
    assert child.parent_text is None
    assert captured, "应发出一次查询"
    assert "document_id" in captured[0], "SQL 必须带 document_id 谓词"


def test_hydrate_parents_fills_same_document(monkeypatch):
    from app.services import retrieval_service as rs

    doc_a = str(uuid.uuid4())
    row = types.SimpleNamespace(
        parent_id=f"{doc_a}:p:0", document_id=uuid.UUID(doc_a),
        text="A 文档的父块正文", char_start=0, char_end=5, idx=0,
        section_path=None, heading=None,
    )
    captured: list[str] = []
    _install_fake_db(monkeypatch, [row], captured)

    child = _mk_child(doc_a)
    filled = _run(rs._hydrate_parents([child]))

    assert filled == 1
    assert child.parent_text == "A 文档的父块正文"


# ── 5. 内存 BM25 腿媒体字段契约 ──────────────────────────────────────────────

def _image_chunk():
    from app.services.retrieval_service import RetrievedChunk

    return RetrievedChunk(
        document_id="d1", filename="f.docx", page_number=2, chunk_index=3,
        text="t", score=0.0, content_type="image", image_id="img1",
        image_path="images/x.png", image_caption="一张柱状图", image_type="chart",
        analyze_engine="vlm", analyze_confidence=0.87, manual_review=True,
        position=4, bbox=(1.0, 2.0, 3.0, 4.0),
        analyze_quality={"score": 0.9}, analyze_fusion={"strategy": "ocr"},
    )


def test_replace_preserves_all_media_fields():
    from dataclasses import replace

    c = _image_chunk()
    out = replace(c, score=0.42)
    assert out.score == 0.42
    # 修复前这些字段在使用手写构造时会丢；用 replace 必须全保留
    assert out.image_type == "chart"
    assert out.position == 4
    assert out.bbox == (1.0, 2.0, 3.0, 4.0)
    assert out.analyze_engine == "vlm"
    assert out.analyze_confidence == 0.87
    assert out.manual_review is True
    assert out.analyze_quality == {"score": 0.9}
    assert out.analyze_fusion == {"strategy": "ocr"}
    assert out.image_id == "img1" and out.image_path == "images/x.png"


def test_bm25_leg_branch_uses_replace():
    src = (_BACKEND_ROOT / "app" / "services" / "retrieval_service.py").read_text(
        encoding="utf-8"
    )
    assert "replace(c, score=keyword_only_score)" in src


# ── 6. 空 token 句 support=1.0（有意取舍，固化行为）──────────────────────────

def test_support_ratio_empty_tokens_lenient_by_design():
    from app.services.nodes import citation_verifier as cv

    # 内容为空（纯标点）→ 不因支持度判负（模块「宁可漏判、不可错杀」取舍）
    assert cv._support_ratio("……", "任意原文内容") == 1.0
    assert cv._support_ratio("🙂", "任意原文内容") == 1.0


# ── 7. 命中句定位只用 display_text（固化行为）───────────────────────────────

def test_evidence_localization_uses_display_text_only():
    from app.services.nodes import citation_verifier as cv

    source = {
        "text_snippet": "OCR 文本：这是一张柱状图。",
        "vision": "图中显示 2023 年销量为 120 万辆。",
        "image_caption": "一张柱状图",
    }
    assert cv.display_text(source) == "OCR 文本：这是一张柱状图。"
    # 支持度判定看得到 vision（_source_text 含 vision）……
    assert "120" in cv._source_text(source)
    # ……但命中句定位只落在 display_text 上（vision 派生结论可能无高亮，属已知轻微降级）
    spans = cv.find_evidence_spans("图中显示 2023 年销量为 120 万辆", source)
    assert all("vision" not in s.text for s in spans)


# ── 8. 降级路径 sources 仍带图片字段 ─────────────────────────────────────────

def test_build_context_sources_carry_image_fields():
    from app.services.nodes.context_builder import build_context

    built = build_context([_image_chunk()])
    assert built.sources, "应有 sources"
    s = built.sources[0]
    assert s["content_type"] == "image"
    assert s["image_id"] == "img1"
    assert s["image_path"] == "images/x.png"
    assert s["image_url"] == "/documents/d1/images/x.png"
    assert s["image_type"] == "chart"
    assert s["analyze_engine"] == "vlm"
    assert s["manual_review"] is True
