"""
结构解析的调度器（Provider 链 + 能力探测 + 优雅降级）.

设计原则（每一条都对应一次真实事故）
────────────────────────────────────
1. **探测先于调用**。``shutil.which`` / import 探测 + 结果缓存，缺工具时
   一次探测就够，不会每份文档都去拉起一个必然失败的子进程。

2. **失败只是降级，不是错误**。任一 provider 抛异常、超时、输出为空、
   或拿不到可信逐页偏移 → 记录 warning 后跳到下一个。链尾是 native，
   它永远可用。入库链路不会因为外部解析器挂了而失败。

3. **页码不可信即作废**（见 model.StructuredDocument 的注释）。这是与
   项目里 ``DOCLING_PDF_ENABLED`` 默认关闭同一个理由：整篇转换会把正文
   全归到第 1 页，细粒度引用立刻失效。

4. **子进程必须限时**。MinerU 在 CPU 上解析一份 100 页 PDF 可以跑十几分钟，
   GPU 排队时更久。没有超时的话入库会永久卡住（文档停在 PARSING 状态、
   占着 embedding 信号量、把整个上传队列拖死）。
"""

from __future__ import annotations

import abc
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from app.utils.logging import get_logger
from app.services.structure.model import (
    NodeKind,
    PageMark,
    StructuredDocument,
)

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 探测结果缓存
# ═══════════════════════════════════════════════════════════════════════════════

_probe_cache: dict[str, tuple[float, bool]] = {}
_probe_lock = threading.Lock()


def _cached_probe(key: str, ttl: float, fn) -> bool:
    """带 TTL 的能力探测缓存（进程内）。探测本身可能要 import torch（秒级）。"""
    now = time.monotonic()
    with _probe_lock:
        hit = _probe_cache.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
    try:
        ok = bool(fn())
    except Exception:      # noqa: BLE001 — 探测失败即"不可用"，不是错误
        ok = False
    with _probe_lock:
        _probe_cache[key] = (now, ok)
    return ok


def reset_probe_cache() -> None:
    """供测试与"装完工具后不重启"的场景使用."""
    with _probe_lock:
        _probe_cache.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# 小型 Markdown 辅助（provider 之间共用）
# ═══════════════════════════════════════════════════════════════════════════════

_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_HTML_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.DOTALL | re.IGNORECASE)
_HTML_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ROWSPAN_RE = re.compile(
    r'<t[dh]\b[^>]*\browspan\s*=\s*"?(\d+)"?[^>]*>', re.IGNORECASE
)


def html_table_to_markdown(html: str) -> str:
    """
    把 HTML 表格转成 Markdown 网格.

    MinerU 的 ``table_body`` 是 HTML（``<table><tr><td>…``）。本项目整条下游链路
    （分块器的 ``_TABLE_LINE_RE``、``detect_content_type``、Document Agent 的
    表格还原、前端的表格预览）**只认 Markdown 表格**，直接落 HTML 会让表格
    退化成一段"看起来像文本"的普通段落 —— 表格检索和表格渲染同时失效。

    处理能力刻意保守：去标签、按 ``<tr>/<td>`` 切分、补齐列宽。``rowspan``/``colspan``
    的**合并语义不去还原**（还原需要完整网格推演，出错时会造出比原表更错的表，
    违反"宁缺毋假"）；只需保证单元格文字与行列关系不丢。
    """
    def _one(match: re.Match) -> str:
        rows: list[list[str]] = []
        for row_html in _HTML_ROW_RE.findall(match.group(0)):
            cells = [
                _HTML_TAG_RE.sub("", c).strip().replace("|", "\\|").replace("\n", " ")
                for c in _HTML_CELL_RE.findall(row_html)
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        out = ["| " + " | ".join(rows[0]) + " |"]
        out.append("| " + " | ".join("---" for _ in range(width)) + " |")
        for r in rows[1:]:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)

    return _HTML_TABLE_RE.sub(_one, html)


def normalize_mineru_block_type(raw: str) -> str:
    """MinerU 的 type 字段 → 本项目 NodeKind 词汇表."""
    t = (raw or "").strip().lower()
    return {
        "text": NodeKind.PARAGRAPH,
        "title": NodeKind.HEADING,
        "table": NodeKind.TABLE,
        "image": NodeKind.FIGURE,
        "figure": NodeKind.FIGURE,
        "equation": NodeKind.FORMULA,
        "formula": NodeKind.FORMULA,
        "interline_equation": NodeKind.FORMULA,
        "list": NodeKind.LIST,
        "code": NodeKind.CODE,
    }.get(t, NodeKind.PARAGRAPH)


# ═══════════════════════════════════════════════════════════════════════════════
# Provider 基类
# ═══════════════════════════════════════════════════════════════════════════════


class StructureProvider(abc.ABC):
    """
    一个结构解析后端.

    子类只需实现 ``probe`` 与 ``_parse_bytes``。页码可信性由基类的
    ``wrap`` 统一把关 —— 各 provider 不能"自行决定"页码可不可信，
    否则这条硬规矩迟早被绕过。
    """

    name: str = "?"

    def __init__(self, settings) -> None:
        self.settings = settings

    # ── 能力探测 ──────────────────────────────────────────────────────────────
    @abc.abstractmethod
    def probe(self) -> bool:
        """该后端在当前环境下是否可用（不得抛异常）."""

    def available(self) -> bool:
        return _cached_probe(
            f"structure::{self.name}",
            self.settings.STRUCTURE_PROBE_CACHE_SECONDS,
            self.probe,
        )

    # ── 解析 ──────────────────────────────────────────────────────────────────
    @abc.abstractmethod
    def _parse_bytes(self, content: bytes, filename: str) -> StructuredDocument | None:
        """真正干活；返回 None 表示"这个后端这次没产出"。"""

    def parse(self, content: bytes, filename: str) -> StructuredDocument | None:
        """带统一兜底与体检的解析入口（调度器只调这个方法）."""
        if not content:
            return None
        try:
            doc = self._parse_bytes(content, filename)
        except Exception as exc:      # noqa: BLE001 — 外部工具，什么都能抛
            logger.warning("structure provider '%s' failed: %s", self.name, exc)
            return None
        if doc is None:
            return None
        doc.provider = self.name
        if not doc.markdown.strip():
            logger.warning("structure provider '%s' returned empty markdown", self.name)
            return None
        # 页码体检：拿不到真实分页就作废（宁可退回原生，也不给错页码）
        if not doc.page_marks_trustworthy or not doc.page_marks:
            logger.warning(
                "structure provider '%s': page marks not trustworthy — discarded "
                "(page numbers must never be guessed)", self.name,
            )
            return None
        return doc

    # ── 公共工具 ──────────────────────────────────────────────────────────────
    def _run_cli(
        self,
        argv: list[str],
        cwd: str,
        timeout: float,
    ) -> tuple[int, str]:
        """跑一个外部 CLI，返回 (returncode, 合并输出)。超时返回 (-9, ...)."""
        env = dict(os.environ)
        # 有些工具（modelscope / huggingface）在容器里默认走代理会卡死；
        # 尊重调用方已有的环境，这里只确保输出不被缓冲。
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.run(          # noqa: S603 — argv 由本模块构造，无 shell
                argv,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "structure provider '%s' timed out after %.0fs", self.name, timeout,
            )
            return -9, "timeout"
        except FileNotFoundError:
            logger.warning("structure provider '%s': command not found (%s)", self.name, argv[0])
            return -2, "not found"
        except Exception as exc:            # noqa: BLE001
            return -3, str(exc)
        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        return proc.returncode, out[-4000:]


# ═══════════════════════════════════════════════════════════════════════════════
# MinerU
# ═══════════════════════════════════════════════════════════════════════════════


class MinerUProvider(StructureProvider):
    """
    MinerU（opendatalab）—— 版面 + 公式 + 表格 + 阅读顺序.

    为什么 MinerU 放在链首：它的 ``*_content_list.json`` **逐块带 ``page_idx``**，
    因此我们可以**从块重建 Markdown**，从而拿到精确到字符的页边界。这是本项目
    对结构解析器的硬要求（见 model.py）。Marker 的 CLI 产物做不到这点。

    正文从 ``content_list`` 重建而**不是**直接读 ``.md``：``.md`` 是整篇拼接，
    页边界信息已经丢失，读它就只能猜页码。
    """

    name = "mineru"

    def probe(self) -> bool:
        return shutil.which(self.settings.MINERU_COMMAND) is not None

    def _parse_bytes(self, content: bytes, filename: str) -> StructuredDocument | None:
        with tempfile.TemporaryDirectory(prefix="mineru_") as tmp:
            src = Path(tmp) / filename
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_bytes(content)
            out_dir = Path(tmp) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            argv = [
                self.settings.MINERU_COMMAND,
                "-p", str(src),
                "-o", str(out_dir),
                "-b", self.settings.MINERU_BACKEND,
            ]
            code, tail = self._run_cli(
                argv, cwd=tmp, timeout=self.settings.STRUCTURE_PARSER_TIMEOUT_SECONDS,
            )
            if code != 0:
                logger.warning("mineru exited %s: %s", code, tail[-500:])
                return None

            # content_list 文件名在不同版本间有出入，宽松匹配
            lists = glob.glob(str(out_dir / "**" / "*content_list*.json"), recursive=True)
            lists = [p for p in lists if not p.endswith("_middle.json")]
            if not lists:
                logger.warning(
                    "mineru produced no content_list.json — cannot recover per-page "
                    "offsets, discarded (page numbers would be wrong)"
                )
                return None
            return self._from_content_list(lists[0])

    @staticmethod
    def _from_content_list(path: str) -> StructuredDocument | None:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:                     # noqa: BLE001
            return None
        if not isinstance(raw, list):
            return None

        chunks: list[str] = []
        marks: list[PageMark] = []
        warnings: list[str] = []
        cursor = 0
        current_page: int | None = None

        def emit(text: str, page_idx: int) -> None:
            nonlocal cursor, current_page
            page_number = int(page_idx) + 1        # MinerU page_idx 是 0-based
            if current_page is None or page_number != current_page:
                marks.append(PageMark(page_number=page_number, char_start=cursor))
                current_page = page_number
            body = text.strip()
            if not body:
                return
            chunks.append(body)
            cursor += len(body) + 2               # "\n\n" 连接
            chunks.append("\n\n")
            cursor += 0

        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                page_idx = int(item.get("page_idx") or 0)
            except (TypeError, ValueError):
                page_idx = 0
            kind = normalize_mineru_block_type(item.get("type", "text"))
            text = item.get("text") or ""

            if kind == NodeKind.TABLE:
                body = item.get("table_body") or text or ""
                if body.lstrip().lower().startswith("<table"):
                    body = html_table_to_markdown(body)
                caption = " ".join(
                    str(c) for c in (item.get("table_caption") or []) if str(c).strip()
                )
                emit((caption + "\n" if caption else "") + body, page_idx)
            elif kind == NodeKind.FORMULA:
                emit(f"$$\n{text}\n$$" if text.strip() else "", page_idx)
            elif kind == NodeKind.FIGURE:
                img = item.get("img_path") or ""
                caption = " ".join(
                    str(c) for c in (item.get("img_caption") or []) if str(c).strip()
                )
                if img.strip():
                    emit(f"![{caption}]({img})", page_idx)
                elif caption.strip():
                    emit(caption, page_idx)
            else:
                if not text.strip():
                    continue
                # text_level 存在时是标题；MinerU 的 text 里有时已带 "#"
                level = item.get("text_level")
                if level and not text.lstrip().startswith("#"):
                    try:
                        n = max(1, min(6, int(level)))
                    except (TypeError, ValueError):
                        n = 1
                    text = "#" * n + " " + text.strip()
                emit(text, page_idx)

        markdown = "".join(chunks).strip()
        if not markdown:
            return None
        if not marks:
            marks = [PageMark(page_number=1, char_start=0)]
        marks.sort(key=lambda m: m.char_start)
        # 去重同页重复出现（MinerU 偶尔会跨块乱序回跳）
        dedup: list[PageMark] = []
        for m in marks:
            if not dedup or dedup[-1].page_number != m.page_number:
                dedup.append(m)
            elif m.char_start < dedup[-1].char_start:
                dedup[-1] = m
        page_count = max((m.page_number for m in dedup), default=1)
        return StructuredDocument(
            markdown=markdown,
            page_marks=dedup,
            provider="mineru",
            page_count=page_count,
            page_marks_trustworthy=True,
            warnings=warnings,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Marker
# ═══════════════════════════════════════════════════════════════════════════════


class MarkerProvider(StructureProvider):
    """
    Marker（VikParuchuri/marker）—— 深度学习的 PDF → Markdown 转换.

    走 **Python API**（``marker.converters.pdf.PdfConverter``）而不是 CLI：
    CLI 只产出整篇 ``.md`` + ``_meta.json``，页边界已丢失；API 的
    ``build_document()`` 会给出按页组织的 block 列表，我们逐页渲染并累计偏移，
    才能得到可信的 ``page_marks``。

    API 在不同 marker 版本间签名有变动（``render()`` / ``export_to_markdown()`` /
    ``raw_text``），这里逐个尝试；全都拿不到就返回 None 交给下一个 provider。
    """

    name = "marker"

    def probe(self) -> bool:
        try:
            import marker  # noqa: F401
        except Exception:            # noqa: BLE001
            return False
        return True

    def _parse_bytes(self, content: bytes, filename: str) -> StructuredDocument | None:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
        except Exception:            # noqa: BLE001
            return None

        suffix = Path(filename).suffix or ".pdf"
        with tempfile.TemporaryDirectory(prefix="marker_") as tmp:
            src = Path(tmp) / f"input{suffix}"
            src.write_bytes(content)
            try:
                converter = PdfConverter(artifact_dict=create_model_dict())
                document = converter.build_document(str(src))
            except Exception as exc:      # noqa: BLE001
                logger.warning("marker build_document failed: %s", exc)
                return None

        pages = getattr(document, "pages", None) or getattr(document, "children", None)
        if not pages:
            logger.warning(
                "marker: document has no per-page structure — page numbers would be "
                "guessed, discarded"
            )
            return None

        parts: list[str] = []
        marks: list[PageMark] = []
        cursor = 0
        for idx, page in enumerate(pages):
            page_parts: list[str] = []
            for block in (getattr(page, "children", None) or []):
                page_parts.append(_render_marker_block(block))
            body = "\n\n".join(p for p in page_parts if p and p.strip()).strip()
            if not body:
                continue
            marks.append(PageMark(page_number=idx + 1, char_start=cursor))
            parts.append(body)
            cursor += len(body) + 2
            parts.append("\n\n")

        markdown = "".join(parts).strip()
        if not markdown or not marks:
            return None
        return StructuredDocument(
            markdown=markdown,
            page_marks=marks,
            provider="marker",
            page_count=max(m.page_number for m in marks),
            page_marks_trustworthy=True,
        )


def _render_marker_block(block) -> str:
    """尽力把 marker 的一个 block 渲染成 Markdown（版本差异都吞在这里）."""
    for attr in ("render", "export_to_markdown"):
        fn = getattr(block, attr, None)
        if callable(fn):
            try:
                out = fn(None) if attr == "render" else fn()
                if isinstance(out, str) and out.strip():
                    return out
            except Exception:        # noqa: BLE001
                pass
    for attr in ("raw_text", "text"):
        val = getattr(block, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Docling
# ═══════════════════════════════════════════════════════════════════════════════


class DoclingProvider(StructureProvider):
    """
    Docling —— 版面感知的 PDF/DOCX 解析.

    与项目里既有的 ``docling_support.docling_text()``（只取整篇 Markdown、
    因而丢掉逐页归属）不同：这里走 ``iterate_items()`` 读每个元素的
    ``prov[0].page_no``，**按页分桶**再拼接，从而产出可信 page_marks。

    这也是在修正一处既有的设计妥协 —— ``docling_support`` 的模块注释明确写了
    "Docling 是整篇转换，套进来会丢掉逐页归属"，所以 ``DOCLING_PDF_ENABLED``
    只能默认关。按页渲染后这个取舍不再必要：结构收益与页码正确性可以兼得。
    """

    name = "docling"

    def probe(self) -> bool:
        try:
            import docling  # noqa: F401
        except Exception:            # noqa: BLE001
            return False
        return True

    def _parse_bytes(self, content: bytes, filename: str) -> StructuredDocument | None:
        try:
            from docling.document_converter import DocumentConverter
        except Exception:            # noqa: BLE001
            return None

        suffix = Path(filename).suffix or ".pdf"
        with tempfile.TemporaryDirectory(prefix="docling_") as tmp:
            src = Path(tmp) / f"input{suffix}"
            src.write_bytes(content)
            try:
                result = DocumentConverter().convert(str(src))
                doc = result.document
            except Exception as exc:     # noqa: BLE001
                logger.warning("docling convert failed: %s", exc)
                return None
            try:
                items = list(doc.iterate_items())
            except Exception:            # noqa: BLE001
                return None

        # 按页码分桶（保持元素原始顺序）
        buckets: dict[int, list[str]] = {}
        order: list[int] = []
        for item in items:
            node, _level = item if isinstance(item, tuple) and len(item) == 2 else (item, None)
            page_no = _docling_page_no(node)
            text = _docling_markdown(node)
            if not text.strip():
                continue
            if page_no not in buckets:
                buckets[page_no] = []
                order.append(page_no)
            buckets[page_no].append(text.strip())

        if not order:
            return None

        parts: list[str] = []
        marks: list[PageMark] = []
        cursor = 0
        for page_no in order:
            body = "\n\n".join(buckets[page_no]).strip()
            if not body:
                continue
            marks.append(PageMark(page_number=page_no, char_start=cursor))
            parts.append(body)
            cursor += len(body) + 2
            parts.append("\n\n")

        markdown = "".join(parts).strip()
        if not markdown or not marks:
            return None
        return StructuredDocument(
            markdown=markdown,
            page_marks=marks,
            provider="docling",
            page_count=max(m.page_number for m in marks),
            page_marks_trustworthy=True,
        )


def _docling_page_no(node) -> int:
    prov = getattr(node, "prov", None)
    if prov:
        first = prov[0] if isinstance(prov, (list, tuple)) and prov else prov
        page = getattr(first, "page_no", None)
        try:
            return max(1, int(page))
        except (TypeError, ValueError):
            return 1
    return 1


def _docling_markdown(node) -> str:
    for attr in ("export_to_markdown",):
        fn = getattr(node, attr, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, str) and out.strip():
                    return out
            except Exception:        # noqa: BLE001
                pass
    text = getattr(node, "text", None)
    if isinstance(text, str) and text.strip():
        label = str(getattr(node, "label", "") or "").lower()
        if label.startswith("section_header") or label == "title":
            return "## " + text.strip()
        return text.strip()
    # Docling 的表格对象没有 .text，只有 data
    data = getattr(node, "data", None)
    if data is not None:
        try:
            grid = getattr(data, "grid", None)
            if grid:
                rows = []
                for row in grid:
                    cells = [str(getattr(c, "text", "") or "").strip() for c in row]
                    rows.append(cells)
                width = max(len(r) for r in rows)
                rows = [r + [""] * (width - len(r)) for r in rows]
                out = ["| " + " | ".join(rows[0]) + " |",
                       "| " + " | ".join("---" for _ in range(width)) + " |"]
                out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                return "\n".join(out)
        except Exception:            # noqa: BLE001
            pass
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 注册表
# ═══════════════════════════════════════════════════════════════════════════════

PROVIDER_CLASSES: dict[str, type[StructureProvider]] = {
    "mineru": MinerUProvider,
    "marker": MarkerProvider,
    "docling": DoclingProvider,
}


def build_provider(name: str, settings) -> StructureProvider | None:
    cls = PROVIDER_CLASSES.get((name or "").strip().lower())
    if cls is None:
        return None
    return cls(settings)


def available_providers(settings) -> dict[str, bool]:
    """诊断用：当前环境里每个 provider 是否可用（不做任何解析）."""
    out: dict[str, bool] = {}
    for name in PROVIDER_CLASSES:
        provider = build_provider(name, settings)
        out[name] = bool(provider and provider.available())
    return out
