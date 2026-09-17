"""旧版 ``.doc``（OLE2 二进制 Word）解析与上传校验.

覆盖三件事：
  1. ``.doc`` 真的被路由到 :class:`DocParser`（而不是掉到 TXT 兜底读成乱码）；
  2. 上传校验接受 OLE2 魔数、拒绝"改名/损坏"的文件，且错误提示**可操作**；
  3. 正文抽取（FIB → Clx → PlcPcd）能还原中文正文；拿不到真实样本时用
     合成的最小 OLE 结构校验"识别失败要报可读错误"这条通路。

真实样本按需跳过：夹具是"用户桌面上碰巧存在的 .doc"，不该成为整套件的前置条件。
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from app.services.parsers.doc_parser import DocParser
from app.services.parsers.factory import get_parser_for_file
from app.utils.file_utils import _ALLOWED_EXTENSIONS, validate_document_upload

OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# 常见位置找一份真实 .doc；找不到就跳过依赖样本的用例。
# 可用 LEGACY_DOC_FIXTURE 指向任意一份真实 .doc（CI / 临时验证用）。
_CANDIDATES = [
    Path(os.getenv("LEGACY_DOC_FIXTURE", "")),
    Path("C:/Users/86187/Desktop/进展/嵌入式软件成神手册.doc"),
    Path("samples/legacy.doc"),
]


def _real_doc() -> Path | None:
    for p in _CANDIDATES:
        if p.is_file():
            return p
    return None


# ── 路由 ────────────────────────────────────────────────────────────────────

def test_doc_extension_is_in_allowlist() -> None:
    assert "doc" in _ALLOWED_EXTENSIONS


def test_doc_routes_to_doc_parser() -> None:
    """`.doc` 必须走 DocParser；掉到 TxtParser 会把二进制读成乱码."""
    assert isinstance(get_parser_for_file("manual.doc"), DocParser)
    assert isinstance(get_parser_for_file("manual.DOC"), DocParser)
    # 回归：不要因为新增 msword 分支而抢走 docx
    assert type(get_parser_for_file("manual.docx")).__name__ == "DocxParser"


def test_doc_mime_also_routes_to_doc_parser() -> None:
    assert isinstance(
        get_parser_for_file("noext", "application/msword"), DocParser
    )


# ── 上传校验 ─────────────────────────────────────────────────────────────────

class _FakeUpload:
    """最小 UploadFile 替身：只需 ``read`` 与 ``filename``."""

    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._buf = io.BytesIO(content)

    async def read(self, _size: int = -1) -> bytes:
        return self._buf.read()


def _validate(filename: str, content: bytes) -> bytes:
    """同步壳：本仓库的测试约定是 asyncio.run 包一层，不用异步插件."""
    import asyncio

    return asyncio.run(validate_document_upload(_FakeUpload(filename, content)))


def test_upload_accepts_ole2_doc() -> None:
    content = OLE2_MAGIC + b"\x00" * 512
    assert _validate("a.doc", content) == content


def test_upload_rejects_doc_with_wrong_magic_with_actionable_hint() -> None:
    """扩展名是 .doc 但内容不是 OLE2 → 中文可操作提示（不是一句英文 unsupported）."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _validate("fake.doc", b"%PDF-1.7 nope")
    assert exc.value.status_code == 415
    assert "损坏" in exc.value.detail or "扩展名" in exc.value.detail


def test_unsupported_extension_hint_suggests_conversion() -> None:
    """不支持的格式要告诉用户"下一步做什么"，而不是只报一声不支持."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _validate("old.xls", b"whatever")
    assert exc.value.status_code == 415
    assert ".xlsx" in exc.value.detail

# ── 正文抽取 ─────────────────────────────────────────────────────────────────

def test_clean_normalizes_controls_and_drops_anchors() -> None:
    """`\\r`→`\\n`、单元格分隔符→制表符、私有区锚点字符被丢弃（正文保留）."""
    # 注意用 \uf0ff（私有区）而不是 \xf0ff —— 后者是 \xf0 + "ff"，即 'ðff'。
    raw = "第一行\r第二行\x07单元格\uf0ff锚点\x0c第三行"
    out = DocParser._clean(raw)
    assert "\r" not in out and "\x07" not in out and "\x0c" not in out
    assert "\uf0ff" not in out
    assert out.splitlines() == ["第一行", "第二行\t单元格锚点", "第三行"]


def test_ole_extraction_recovers_chinese_text() -> None:
    sample = _real_doc()
    if sample is None:
        pytest.skip("no legacy .doc fixture available")

    parser = DocParser()
    text = parser._extract_text_via_ole(sample.read_bytes(), sample.name)

    assert len(text) > 1000, "正文太短，piece table 解析可能没走对"
    assert text.count("\ufffd") == 0, "出现解码失败字符，说明编码分支判断错了"
    # 中文正文必须真的还原成中文（而不是被当作 cp1252 单字节读出来）
    assert any("\u4e00" <= ch <= "\u9fff" for ch in text)


def test_missing_worddocument_stream_raises_actionable_error(monkeypatch) -> None:
    """
    OLE2 但不是 Word（例如 .xls 被改名成 .doc）→ 提示改用 .docx.

    用桩替掉 olefile 的打开结果，而不是去合成一个 OLE 文件：olefile 的
    write_mode 只能改已有文件、不能新建，"造一个合法 OLE"在测试里成本远高于
    收益，而这里真正要覆盖的是**我们自己的分支**（没有 WordDocument 流怎么报错）。
    """
    olefile = pytest.importorskip("olefile")

    class _NotWordOle:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exists(self, _name: str) -> bool:
            return False

        def openstream(self, _name: str):      # pragma: no cover
            raise AssertionError("不应走到读取流的分支")

        def close(self) -> None:
            pass

    monkeypatch.setattr(olefile, "OleFileIO", _NotWordOle)

    with pytest.raises(ValueError) as exc:
        DocParser()._extract_text_via_ole(OLE2_MAGIC + b"\x00" * 32, "renamed.doc")
    assert ".docx" in str(exc.value)


def test_garbage_input_never_leaks_raw_exception() -> None:
    """损坏的 .doc 必须转成中文 ValueError，而不是把底层异常直接抛给用户."""
    with pytest.raises(ValueError) as exc:
        DocParser()._extract_text_via_ole(OLE2_MAGIC + b"\x00" * 1024, "broken.doc")
    assert ".docx" in str(exc.value)
