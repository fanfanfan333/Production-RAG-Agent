"""旧版二进制 Word（``.doc`` / OLE2 复合文档）解析器.

**为什么需要它**：``.doc`` 是历史资料、教材、内部规范里极常见的格式，但
``factory`` 里没有对应解析器 —— 上传时会被 ``_ALLOWED_EXTENSIONS`` 直接
415 拒收，用户手里一整批存量资料根本进不了库。

**两条通路（按保真度排序）**：

1. **LibreOffice 转换（首选）** —— 若容器内有 ``soffice``，先无头转成
   ``.docx`` 再交给 :class:`DocxParser`。这是唯一能保住**表格结构 + 内嵌
   图片**的通路（图片对 RAG 很关键：电路图/截图/表格往往承载正文之外的信息）。
2. **纯 Python 兜底** —— 用 OLE 复合文档的 piece table 直接抽 ``WordDocument``
   流的正文。零额外依赖（``olefile``），但**只有文字**：图片、表格网格、
   分栏版式全部丢失。日志会明确提示这一点。

**为什么不用"启发式扫字节"**：直接在流里找可打印字符会产出乱序垃圾，
污染向量库比拒收更糟。这里严格按 FIB → Clx → PlcPcd → 文本片 解析，
每条片段的编码（cp1252 压缩 / UTF-16LE 未压缩）由 fCompressed 位决定，
所以中英混排、表格内文字都能正确还原。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from app.services.parsers.base import DocumentParser, ExtractionResult, ExtractedPage
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 抽出的文本里 U+FFFD（解码失败）占比超过这个阈值就认为抽取失败 ——
# 与其把一坨乱码写进向量库，不如报错让用户换 .docx。
_MAX_REPLACEMENT_RATIO = 0.05
# 正文短于这个长度视为"没抽到东西"（.doc 的正文通常远不止这些）。
_MIN_CHARS = 8

_SOFFICE_TIMEOUT_S = 180


class DocParser(DocumentParser):
    """旧版 ``.doc`` 解析：LibreOffice 优先，OLE 兜底。"""

    def parse(
        self,
        content: bytes,
        filename: str,
        *,
        document_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ExtractionResult:
        converted = self._convert_with_libreoffice(content, filename)
        if converted is not None:
            # 交给 DocxParser：图片落盘 / 表格识别 / OCR 分流全部复用既有通路。
            from app.services.parsers.docx_parser import DocxParser

            result = DocxParser().parse(
                converted,
                filename=self._docx_name(filename),
                document_id=document_id,
                tenant_id=tenant_id,
            )
            # 记录真实来源格式，便于排查"为什么这份 docx 的图像路径看着奇怪"。
            result.file_type = "doc"
            result.parser_used = f"libreoffice→{result.parser_used}"
            return result

        text = self._extract_text_via_ole(content, filename)
        page = ExtractedPage(
            page_number=1,
            text=text,
            char_start=0,
            char_end=len(text),
        )
        logger.warning(
            "'.doc' parsed without LibreOffice — text-only, embedded images and "
            "table grids are NOT indexed. Install 'soffice' in the image (or save "
            "the file as .docx) for full fidelity. filename=%s chars=%d",
            filename, len(text),
        )
        return ExtractionResult(
            pages=[page],
            full_text=text,
            page_count=1,
            char_count=len(text),
            file_type="doc",
            parser_used="ole-text",
            ocr_used=False,
            ocr_engine=None,
            extraction_method="ole_piece_table",
        )

    # ── 通路 1：LibreOffice ───────────────────────────────────────────────────

    @staticmethod
    def _docx_name(filename: str) -> str:
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        return f"{stem}.docx"

    def _convert_with_libreoffice(self, content: bytes, filename: str) -> bytes | None:
        """有 soffice 就转 .docx；没有/失败返回 None（不抛异常）。"""
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            return None

        tmpdir = tempfile.mkdtemp(prefix="docconv_")
        src = os.path.join(tmpdir, "input.doc")
        try:
            with open(src, "wb") as fh:
                fh.write(content)
            proc = subprocess.run(  # noqa: S603 — 参数为受控常量
                [
                    soffice, "--headless", "--norestore", "--convert-to", "docx",
                    "--outdir", tmpdir, src,
                ],
                capture_output=True,
                timeout=_SOFFICE_TIMEOUT_S,
            )
            out = os.path.join(tmpdir, "input.docx")
            if proc.returncode == 0 and os.path.exists(out):
                with open(out, "rb") as fh:
                    return fh.read()
            logger.warning(
                "LibreOffice conversion failed for '%s' (rc=%s): %s",
                filename, proc.returncode, proc.stderr[:200],
            )
            return None
        except Exception as exc:      # noqa: BLE001 — 转换是"更优路径"，失败要能安静降级
            logger.warning("LibreOffice conversion error for '%s': %s", filename, exc)
            return None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 通路 2：OLE piece table ──────────────────────────────────────────────

    def _extract_text_via_ole(self, content: bytes, filename: str) -> str:
        try:
            import olefile
        except ImportError as exc:
            raise ValueError(
                f"无法解析旧版 .doc 文件「{filename}」：缺少 olefile 依赖。"
                "请用 Word 另存为 .docx 后重新上传。"
            ) from exc

        try:
            text = self._read_text_fib(content, olefile)
        except Exception as exc:      # noqa: BLE001
            raise ValueError(
                f"无法解析旧版 .doc 文件「{filename}」（{exc}）。"
                "该文件可能已加密、损坏或使用了不支持的旧格式；"
                "请用 Word 另存为 .docx 后重新上传。"
            ) from exc

        text = self._clean(text)
        replacement_ratio = text.count("\ufffd") / max(1, len(text))
        if len(text) < _MIN_CHARS or replacement_ratio > _MAX_REPLACEMENT_RATIO:
            raise ValueError(
                f"旧版 .doc 文件「{filename}」的正文无法可靠还原"
                f"（提取 {len(text)} 字，异常字符占比 {replacement_ratio:.0%}）。"
                "请用 Word 另存为 .docx 后重新上传，以获得完整正文与内嵌图片。"
            )
        return text

    @staticmethod
    def _read_text_fib(content: bytes, olefile_module) -> str:
        """
        按 Word 97+ 的 FIB → Clx → PlcPcd 读正文.

        FIB 偏移：``0x000A`` 的 bit9 选 0Table/1Table；``0x01A2``/``0x01A6``
        是 Clx 的 fc/lcb。Clx 由若干 Prc 后跟一个 Pcdt(0x02) 组成，Pcdt 里是
        PlcPcd：``n+1`` 个 CP 后跟 ``n`` 个 8 字节 PCD。每个 PCD 的 fc 位 30
        = fCompressed（1 字节/字符，cp1252）否则 2 字节/字符（UTF-16LE）。
        """
        import io

        ole = olefile_module.OleFileIO(io.BytesIO(content))
        try:
            if not ole.exists("WordDocument"):
                raise ValueError("not an OLE Word document (no WordDocument stream)")
            wd = ole.openstream("WordDocument").read()

            flags = int.from_bytes(wd[0x000A:0x000C], "little")
            table_name = "1Table" if (flags & 0x0200) else "0Table"
            if not ole.exists(table_name):
                table_name = "0Table" if ole.exists("0Table") else "1Table"
            tbl = ole.openstream(table_name).read()

            fc_clx = int.from_bytes(wd[0x01A2:0x01A6], "little")
            lcb_clx = int.from_bytes(wd[0x01A6:0x01AA], "little")
            clx = tbl[fc_clx: fc_clx + lcb_clx]

            cursor = 0
            while cursor < len(clx) and clx[cursor] == 0x01:      # Prc*
                cb = int.from_bytes(clx[cursor + 1: cursor + 3], "little")
                cursor += 3 + cb
            if cursor >= len(clx) or clx[cursor] != 0x02:
                raise ValueError("malformed Clx (no Pcdt)")
            lcb_plcpcd = int.from_bytes(clx[cursor + 1: cursor + 5], "little")
            plc = clx[cursor + 5: cursor + 5 + lcb_plcpcd]
        finally:
            ole.close()

        piece_count = (lcb_plcpcd - 4) // 12
        if piece_count <= 0:
            raise ValueError("empty piece table")

        cps = [
            int.from_bytes(plc[k * 4: k * 4 + 4], "little")
            for k in range(piece_count + 1)
        ]
        pieces: list[str] = []
        for k in range(piece_count):
            base = 4 * (piece_count + 1) + k * 8
            fc = int.from_bytes(plc[base + 2: base + 6], "little")
            compressed = bool(fc & 0x40000000)
            fc &= 0x3FFFFFFF
            length = cps[k + 1] - cps[k]
            if compressed:
                raw = wd[fc // 2: fc // 2 + length]
                pieces.append(raw.decode("cp1252", errors="replace"))
            else:
                raw = wd[fc: fc + length * 2]
                pieces.append(raw.decode("utf-16-le", errors="replace"))
        return "".join(pieces)

    @staticmethod
    def _clean(text: str) -> str:
        """
        归一化控制字符：``\\r`` → ``\\n``，丢弃 .doc 里表示"图/表/脚注锚点"的
        私有区字符（U+F000-U+F0FF）与其它不可见控制符。

        这些锚点字符如果留在正文里，会污染 chunk 边界与 BM25 分词。
        """
        out: list[str] = []
        for ch in text:
            code = ord(ch)
            if ch == "\r":
                out.append("\n")
            elif ch == "\x07":              # cell/row end marker
                out.append("\t")
            elif ch == "\x0c":              # page break
                out.append("\n")
            elif ch in ("\t", "\n") or code >= 0x20:
                if 0xF000 <= code <= 0xF0FF:    # 私有区：图/表/脚注锚点
                    continue
                out.append(ch)
        return "\n".join(
            line.rstrip() for line in "".join(out).split("\n")
        ).strip()
