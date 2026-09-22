"""
「OCR 识别出来的文档照片，能不能被文档总结功能总结出来」—— 端到端链路回归.

用户场景
────────
上传一张**文档照片**（纸质报表 / 合同 / 发票用手机拍的照片）→ OCR 识别出图内文字
→ 追问「总结一下这份文档」→ 走 ``document_summary`` 分支。

这条链路跨四个模块，任何一处断掉，用户看到的现象都一样：**总结里没有这张照片的内容**。
所以必须逐跳钉死，而不是只测其中一跳：

  ① 入库分块   ``chunker.build_image_chunks``
       照片的 OCR 文本必须成为 image chunk 的**正文**（不能只落盘、正文留空）
  ② 向量载荷   ``TextChunk.to_dict()["text"]``
       内容必须真的写进 Qdrant payload —— 摘要是从 payload 读正文的
  ③ 摘要采样   ``relation_service.collect_document_digests``
       scroll 必须把该正文取回来、拼进 digest
  ④ 提示词组装 ``document_summary_node.build_single_doc_messages``
       必须出现在送给 LLM 的 HumanMessage 里

为什么这条链路值得单独测
────────────────────────
独立上传的照片文档，``IMAGE_AS_INDEPENDENT_OBJECT`` 默认为 True ⇒ ``ImageParser``
把 ``full_text`` 置为**空串**，正文侧一个文本块都不产生。也就是说：
**照片文档的全部可总结内容都压在 image chunk 这一条路上**。这条路一断，
文档仍会正常 ``COMPLETED``、列表里看得见、点得开、原文预览也能出图，
只是总结里空空如也 —— 又一个"看起来一切正常"的静默失效。

对照：``test_image_textless_summary.py`` 测的是**入库阶段**的图片级图意总结
（三处文本全空时生成 ``vision_caption``）；本文件测的是**问答阶段**的文档级总结
（``document_summary`` 分支）。两者不是一回事，互不覆盖。

替身边界
────────
四个被测模块全部用**真实实现**；只有 IO 用替身：
PG session（返回文档行）/ Qdrant client（返回 scroll 点）/ LLM（只组装 messages 不调用）。
"""

from __future__ import annotations

import asyncio
import io
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from PIL import Image

    from app.config import get_settings
    from app.services.chunker import build_image_chunks
    from app.services.nodes.document_summary_node import (
        build_single_doc_messages,
    )
    from app.services.parsers.base import ExtractedImage
    import app.services.relation_service as rs
except ImportError as exc:  # pragma: no cover - 宿主机缺依赖 → 整份跳过
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ── 夹具：一张"文档照片" ─────────────────────────────────────────────────────

DOC_ID = "3f2a91c4-5b6d-4e77-9a10-8c2d4f6e1357"
PHOTO_FILENAME = "2024年度经营分析报告-照片.png"

#: 模拟 OCR 从照片里认出来的文字（保持 <150 字符：单块截断上限 = 1200/8）
OCR_TEXT = (
    "2024年度经营分析报告\n"
    "营业收入 1286 万元 同比增长 12.4%\n"
    "净利润 213 万元 同比增长 8.1%\n"
    "研发投入 96 万元 占营收 7.5%"
)
PHOTO_CAPTION = "图片描述: 一张打印好的纸质报表照片，页眉有公司名称与报告年度。"


def _photo(
    *,
    image_id: str = "doc-p1-i1",
    ocr: str = OCR_TEXT,
    caption: str | None = None,
    image_type: str = "photo",
    structured: str | None = None,
    position: int | None = 1,
) -> ExtractedImage:
    """构造一张照片对象（等价于 ``ImageParser`` 对一张有文字的照片的产出）."""
    return ExtractedImage(
        image_id=image_id,
        page_number=1,
        ocr_text=ocr,
        vision_caption=caption,
        image_path="images/p1.png",
        width=1200,
        height=1600,
        ocr_engine="tesseract",
        position=position,
        image_type=image_type,
        structured_content=structured,
        analyze_engine="ocr",
        classify_engine="rules",
        analyze_confidence=0.86,
        analyze_decision="pass",
        manual_review=False,
    )


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def _understanding() -> SimpleNamespace:
    """``understand_image`` 的替身：一张 OCR 出中文的文档照片."""
    return SimpleNamespace(
        ocr_text=OCR_TEXT,
        vision_caption=None,
        ocr_engine="tesseract",
        image_type="photo",
        structured_content=None,
        analyze_engine="ocr",
        route="ocr",
        classification=SimpleNamespace(engine="rules", signals={"text_density": 0.4}),
        confidence=0.86,
        decision="pass",
        manual_review=False,
    )


# ── 替身：PG session / Qdrant client ─────────────────────────────────────────


class _FakeResult:
    def __init__(self, rows: list):
        self._rows = rows

    def all(self):
        return self._rows

    def scalars(self):
        return self

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, rows: list):
        self._rows = rows

    async def execute(self, _stmt):
        return _FakeResult(self._rows)


class _FakeDB:
    def __init__(self, session: _FakeSession):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


class _FakePoint:
    def __init__(self, payload: dict):
        self.payload = payload


class _FakeQdrant:
    """记录每次 scroll 的 filter，并**按 must 的真实语义**过滤后返回该文档的点.

    为什么不只按 ``document_id`` 取、把其余条件忽略掉：那样替身就没有过滤语义，
    一旦有人给采样加了 ``content_type="text"`` 之类的条件，替身照样把照片块返回，
    测试全绿 —— 而线上照片文档的总结会**整批变空**。替身必须能"感觉到"过滤条件，
    突变测试才有意义（见 test_digest_scroll_is_not_restricted_by_content_type）。
    """

    def __init__(self, payloads_by_doc: dict[str, list[dict]]):
        self._payloads = payloads_by_doc
        self.filters: list = []

    def _apply_must(self, scroll_filter) -> list[dict]:
        doc_id = None
        #: 除 document_id 之外的 must 条件 —— 按 MatchValue 等值语义逐个筛
        extra: list[tuple[str, object]] = []
        for cond in scroll_filter.must or []:
            key = getattr(cond, "key", None)
            match = getattr(cond, "match", None)
            if key is None or match is None:
                continue
            value = getattr(match, "value", None)
            if key == "document_id":
                doc_id = value
            else:
                extra.append((key, value))

        payloads = list(self._payloads.get(doc_id, []))
        for key, value in extra:
            payloads = [p for p in payloads if p.get(key) == value]
        return payloads

    async def scroll(
        self,
        *,
        collection_name,
        scroll_filter,
        limit=256,
        with_payload=True,
        with_vectors=False,
    ):
        self.filters.append(scroll_filter)
        payloads = self._apply_must(scroll_filter)
        return [_FakePoint(p) for p in payloads][:limit], None


def _document_rows(*, page_count: int = 1, chunk_count: int = 1) -> list[tuple]:
    return [
        (
            uuid.UUID(DOC_ID),
            PHOTO_FILENAME,
            page_count,
            chunk_count,
            datetime(2026, 9, 21, 8, 0, tzinfo=timezone.utc),
        )
    ]


def _collect(payloads_by_doc: dict[str, list[dict]], **kwargs):
    """跑一次真实的 ``collect_document_digests``（PG / Qdrant 用替身）."""
    session = _FakeSession(_document_rows())
    qdrant = _FakeQdrant(payloads_by_doc)
    orig_db = rs.get_db_session
    orig_qdrant = rs.get_qdrant_client
    # ⚠️ 补丁必须打在 relation_service 上：模块里写的是
    # ``from app.db.qdrant import get_qdrant_client`` 的**直接绑定**，
    # 改 app.db.qdrant / app.db.postgres 对它无效。
    rs.get_db_session = lambda: _FakeDB(session)
    rs.get_qdrant_client = lambda: qdrant
    try:
        digests = asyncio.run(
            rs.collect_document_digests(max_documents=5, chunks_per_doc=8, **kwargs)
        )
    finally:
        rs.get_db_session = orig_db
        rs.get_qdrant_client = orig_qdrant
    return digests, qdrant


# ═════════════════════════════════════════════════════════════════════════════
# ① 入库分块：照片的 OCR 文本必须成为 chunk 正文
# ═════════════════════════════════════════════════════════════════════════════


def test_photo_ocr_text_becomes_image_chunk_body() -> None:
    chunks = build_image_chunks([_photo()])

    assert len(chunks) == 1, f"有 OCR 文本的照片必须建出 1 个 chunk（实际 {len(chunks)}）"
    assert OCR_TEXT in chunks[0].text, chunks[0].text
    assert chunks[0].content_type == "image"
    assert chunks[0].image_id == "doc-p1-i1"
    assert chunks[0].image_path == "images/p1.png"
    assert chunks[0].position == 1
    print("  ok test_photo_ocr_text_becomes_image_chunk_body")


def test_photo_caption_is_appended_after_ocr_text() -> None:
    """有图意描述时，描述追加在 OCR 文本之后（两者都在正文里，都可被总结）."""
    chunks = build_image_chunks([_photo(caption="纸质报表")])

    body = chunks[0].text
    assert OCR_TEXT in body
    assert "图片描述: 纸质报表" in body
    assert body.index(OCR_TEXT) < body.index("图片描述:"), body
    print("  ok test_photo_caption_is_appended_after_ocr_text")


def test_uploaded_photo_document_relies_on_the_image_chunk(monkeypatch) -> None:
    """
    独立上传的照片文档：正文侧为空 ⇒ 可总结内容 100% 落在 image chunk 上.

    这是本文件存在的根本理由：``IMAGE_AS_INDEPENDENT_OBJECT=True``（默认）
    让 ``ImageParser`` 把 ``full_text`` 置空，正文一个 block 都不产生。
    如果 image chunk 这条链断了，文档照样 COMPLETED，但总结里什么都没有。
    """
    from app.services.parsers import image_parser

    settings = get_settings()
    monkeypatch.setattr(settings, "ENABLE_IMAGE_SAVE", False, raising=False)
    # 补丁打在 image_parser 上（模块内是直接绑定导入）
    monkeypatch.setattr(image_parser, "understand_image", lambda *a, **kw: _understanding())

    result = image_parser.ImageParser().parse(
        _png_bytes(), PHOTO_FILENAME, document_id=DOC_ID, tenant_id="t1"
    )

    assert result.full_text == "", "照片文档的正文侧应为空（IMAGE_AS_INDEPENDENT_OBJECT）"
    assert result.page_count == 1
    assert result.extraction_method == "ocr", result.extraction_method
    assert len(result.images) == 1
    assert OCR_TEXT in result.images[0].searchable_text

    chunks = build_image_chunks(result.images)
    assert len(chunks) == 1, "正文为空时，内容必须由 image chunk 承载"
    assert OCR_TEXT in chunks[0].text
    print("  ok test_uploaded_photo_document_relies_on_the_image_chunk")


# ═════════════════════════════════════════════════════════════════════════════
# ② 向量载荷：内容必须真的写进 payload["text"]
# ═════════════════════════════════════════════════════════════════════════════


def test_photo_chunk_payload_carries_text_for_digest() -> None:
    chunk = build_image_chunks([_photo()])[0]
    payload = chunk.to_dict()

    assert payload["text"] == chunk.text
    assert OCR_TEXT in payload["text"], "payload 里没有 OCR 文本，摘要就无从采样"
    assert payload["content_type"] == "image"
    assert payload["chunk_index"] == 0
    assert payload["image_id"] == "doc-p1-i1"
    print("  ok test_photo_chunk_payload_carries_text_for_digest")


# ═════════════════════════════════════════════════════════════════════════════
# ③ 摘要采样：scroll 必须把照片正文取回来
# ═════════════════════════════════════════════════════════════════════════════


def test_digest_sampling_returns_photo_ocr_text() -> None:
    payload = build_image_chunks([_photo()])[0].to_dict()
    digests, _qdrant = _collect({DOC_ID: [payload]})

    assert len(digests) == 1, "照片文档应产出 1 份 digest"
    digest = digests[0]
    assert OCR_TEXT in digest.digest, digest.digest
    assert digest.sampled_chunks == 1, digest.sampled_chunks
    assert digest.warnings == [], digest.warnings
    assert digest.filename == PHOTO_FILENAME
    assert digest.page_count == 1 and digest.chunk_count == 1
    print("  ok test_digest_sampling_returns_photo_ocr_text")


def test_digest_scroll_is_not_restricted_by_content_type() -> None:
    """
    照片块是 ``content_type="image"`` 入库的 —— 采样条件**不得**按类型过滤.

    这条断言防的是"以后有人为了给摘要排除图片而加 ``content_type="text"`` 过滤"
    这类改动：它会静默把**所有照片文档**从总结里抹掉，而单元测试全绿。
    """
    payload = build_image_chunks([_photo()])[0].to_dict()
    assert payload["content_type"] == "image"

    digests, qdrant = _collect({DOC_ID: [payload]})
    assert OCR_TEXT in digests[0].digest, "content_type=image 的块被采样漏掉了"

    scroll_filter = qdrant.filters[0]
    keys = [getattr(cond, "key", None) for cond in (scroll_filter.must or [])]
    assert keys == ["document_id"], f"采样条件只应限定文档，实际 {keys}"
    print("  ok test_digest_scroll_is_not_restricted_by_content_type")


def test_multiple_photos_all_land_in_digest() -> None:
    """一份文档里多张照片 → 每张的 OCR 文本都要进 digest."""
    photos = [
        _photo(image_id="doc-p1-i1", ocr="照片一：2024 年营业收入 1286 万元"),
        _photo(image_id="doc-p1-i2", ocr="照片二：2023 年营业收入 1144 万元", position=2),
        _photo(image_id="doc-p1-i3", ocr="照片三：2022 年营业收入 1021 万元", position=3),
    ]
    payloads = [c.to_dict() for c in build_image_chunks(photos)]
    digests, _ = _collect({DOC_ID: payloads})

    body = digests[0].digest
    for token in ("照片一", "照片二", "照片三", "1286", "1144", "1021"):
        assert token in body, f"digest 漏了 {token}: {body}"
    assert digests[0].sampled_chunks == 3
    print("  ok test_multiple_photos_all_land_in_digest")


def test_table_photo_is_also_summarised() -> None:
    """表格照片走 ``content_type="table"``，采样同样不看类型 → 照样能总结."""
    table = _photo(
        ocr="",
        image_type="table",
        structured="| 指标 | 2024 年 |\n|---|---|\n| 营业收入 | 1286 万元 |",
    )
    chunks = build_image_chunks([table])
    assert len(chunks) == 1 and chunks[0].content_type == "table"

    digests, _ = _collect({DOC_ID: [chunks[0].to_dict()]})
    assert "营业收入" in digests[0].digest
    assert "1286" in digests[0].digest
    print("  ok test_table_photo_is_also_summarised")


# ═════════════════════════════════════════════════════════════════════════════
# ④ 提示词组装：必须出现在送给 LLM 的输入里
# ═════════════════════════════════════════════════════════════════════════════


def test_summary_prompt_carries_photo_ocr_text() -> None:
    payload = build_image_chunks([_photo()])[0].to_dict()
    digests, _ = _collect({DOC_ID: [payload]})
    serialised = rs.digest_sources(digests)[0]

    messages = build_single_doc_messages("总结一下这份文档", serialised)
    human = messages[1].content

    assert OCR_TEXT in human, human
    assert PHOTO_FILENAME in human, human
    assert "1 页" in human and "1 个文本块" in human, human
    # 总结指令本身不能被内容挤掉
    assert "核心主题" in messages[0].content
    print("  ok test_summary_prompt_carries_photo_ocr_text")


def test_ocr_photo_reaches_summary_prompt_end_to_end() -> None:
    """①→④ 串成一条：一张照片对象 → 最终总结提示词里能看到它的 OCR 文本."""
    # ① 分块
    chunks = build_image_chunks([_photo(caption="纸质报表照片")])
    assert len(chunks) == 1

    # ② 载荷
    payload = chunks[0].to_dict()

    # ③ 采样
    digests, _ = _collect({DOC_ID: [payload]})
    assert len(digests) == 1 and OCR_TEXT in digests[0].digest

    # ④ 提示词
    serialised = rs.digest_sources(digests)[0]
    human = build_single_doc_messages("总结这份文档", serialised)[1].content

    assert OCR_TEXT in human
    assert "1286" in human and "213" in human, "关键数字必须能被总结到"
    assert "纸质报表照片" in human
    print("  ok test_ocr_photo_reaches_summary_prompt_end_to_end")


# ═════════════════════════════════════════════════════════════════════════════
# 阴性对照：证明上面几条不是"本来就通过"
# ═════════════════════════════════════════════════════════════════════════════


def test_textless_photo_builds_no_chunk() -> None:
    """三处文本全空的照片 → 不建块（旧行为），与上面的阳性组形成对照."""
    bare = _photo(ocr="", caption=None)
    assert bare.searchable_text.strip() == ""
    assert build_image_chunks([bare]) == []
    print("  ok test_textless_photo_builds_no_chunk")


def test_digest_warns_when_photo_has_no_text() -> None:
    """照片无文字（Qdrant 也没点）→ digest 明确留痕，不假装"总结了但没内容"."""
    digests, _ = _collect({DOC_ID: []})

    digest = digests[0]
    assert digest.sampled_chunks == 0
    assert "未采样到文本内容" in digest.warnings, digest.warnings
    assert digest.digest == "（无可用文本内容）"
    print("  ok test_digest_warns_when_photo_has_no_text")


def test_low_quality_ocr_is_dropped_in_favour_of_caption() -> None:
    """低质 OCR 被质检判否且有描述兜底 → 正文只留描述（不给总结灌噪声）."""
    noisy = _photo(ocr="H |=+2 x padding|0|—dilation...", caption="一张流程图")
    noisy.analyze_quality = {"ocr": {"passed": False}}
    chunks = build_image_chunks([noisy])

    assert len(chunks) == 1
    body = chunks[0].text
    assert "dilation" not in body, "质检判否的 OCR 噪声不该进正文"
    assert "图片描述: 一张流程图" in body
    print("  ok test_low_quality_ocr_is_dropped_in_favour_of_caption")


if __name__ == "__main__":
    import traceback

    class _MP:
        """最小 monkeypatch 替身（直接跑 python 本文件时用）."""

        def setattr(self, obj, name, value, raising=True):
            setattr(obj, name, value)

    _ok = _fail = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn(_MP())
            _ok += 1
        except Exception:
            _fail += 1
            print(f"  FAIL {_name}")
            traceback.print_exc()
    print(f"\n{_ok} passed, {_fail} failed")
    raise SystemExit(1 if _fail else 0)
