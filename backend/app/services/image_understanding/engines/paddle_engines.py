"""
Paddle 系引擎适配器：通用 OCR / 版面检测 / 公式识别.

三个引擎共用一个前置模块（:mod:`paddle_common`），因为它们的坑是同一套
（导入顺序 / HOME 可写 / PP-OCR 版本）。所有 PaddleOCR 实例都是**进程级单例**：
加载一次要好几秒，绝不能每张图新建一个。

────────────────────────────────────────────────────────────────────────────
坑 4（重要，勿回退）：Paddle 原生推理**必须串行化**
────────────────────────────────────────────────────────────────────────────
Paddle 的 predictor **不是线程安全的** —— 它内部有共享的 workspace 与 oneDNN
primitive cache，两个线程同时跑推理会把原生内存改坏。症状是：

    could not create a primitive descriptor for a reorder primitive
    InvalidArgumentError: Broadcast dimension mismatch ... db_fpn.py:249
    FatalError: `Segmentation fault` is detected by the operating system.

前两条只是"错"，第三条是**致命的**：SIGSEGV 是原生崩溃，Python 侧任何
try/except 都抓不住，**整个 API 进程直接消失**。容器被重启后，
`recover_stuck_documents` 把所有在跑的文档标成 FAILED（stuck_recovered），
用户看到的就是"传上去一会儿就失败"，日志里甚至没有 Python 异常栈。

为什么以前没炸、现在炸：入库改成后台 asyncio 任务 + `asyncio.to_thread` 之后，
多份文档的 OCR 会**真的并行**跑在不同线程上，而它们共用同一个 `_ocr_instance`。
单文档跑永远不会触发，所以极易漏测。

因此所有原生推理调用都要过模块级的 `_predict_lock`（三套模型共用一把锁：
它们跑在同一个 Paddle runtime 上，共享底层内存池）。

复现脚本：``backend/scripts/repro_paddle_threads.py``（4 线程共享实例 ⇒ 秒崩
退出码 139；加锁 ⇒ 288 次调用零崩溃）。
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

from app.services.image_understanding.engines import paddle_common
from app.services.image_understanding.engines.base import (
    EngineKind,
    EngineOutput,
    ImageEngine,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 构造（加载模型）用：只保证"同一个实例不会被建两遍"
_build_lock = threading.Lock()

# 推理锁是**进程级共享**的，定义在 app.utils.paddle_env（坑 4 的说明也在那里）
_predict_lock = paddle_common.predict_lock()

_ocr_instance = None
_layout_instance = None
_formula_instance = None


def _to_bgr(image: Image.Image) -> "np.ndarray":
    """PIL(RGB) → PaddleOCR 期望的 BGR ndarray."""
    arr = np.array(image.convert("RGB"))
    return arr[:, :, ::-1].copy()


def _weighted_confidence(lines) -> float:
    """
    行置信度 → 整图置信度.

    按**字符数加权**而不是简单平均：一行"的"和一个 20 字的句子权重不该相同，
    长行识别对了更能说明问题。
    """
    total = sum(max(len(getattr(l, "text", "") or ""), 1) for l in lines)
    if not total:
        return 0.0
    acc = sum(
        max(len(getattr(l, "text", "") or ""), 1) * float(getattr(l, "confidence", 0.0) or 0.0)
        for l in lines
    )
    return round(acc / total, 4)


class PaddleOCREngine(ImageEngine):
    """通用 OCR（设计稿里的"普通 OCR → PaddleOCR"）."""

    name = "paddleocr"
    kind = EngineKind.OCR

    def is_available(self) -> bool:
        return paddle_common.paddle_available()

    def unavailable_reason(self) -> str:
        if not paddle_common.paddle_available():
            return "paddleocr 未安装或导入失败（需 albumentations；且须先导入 numpy/cv2）"
        return ""

    def _instance(self):
        global _ocr_instance
        if _ocr_instance is None:
            with _build_lock:
                if _ocr_instance is None:
                    _ocr_instance = paddle_common.build_paddle_ocr()
        return _ocr_instance

    def process(self, image: Image.Image, **kwargs) -> EngineOutput:
        """一次推理产出 文本 + 带坐标行 + 置信度（分类与表格还原共用）."""
        # 有意只在真正要用时才构造实例 —— 探测阶段绝不触发模型加载
        from app.services.ocr.base import OCRLine

        ocr = self._instance()
        if ocr is None:
            return EngineOutput(
                engine=self.name, ok=False,
                error="PaddleOCR 不可用",
            )

        try:
            # 串行化：共享 predictor 并发调用会 SIGSEGV，见模块 docstring 坑 4
            with _predict_lock:
                raw = ocr.ocr(_to_bgr(image), cls=False)
        except Exception as exc:      # noqa: BLE001
            logger.warning("PaddleOCR inference failed: %s", exc)
            return EngineOutput(engine=self.name, ok=False, error=str(exc))

        lines: list[OCRLine] = []
        for item in (raw[0] if raw else None) or []:
            try:
                coords, payload = item[0], item[1]
                text = str(payload[0]) if payload else ""
                if not text.strip():
                    continue
                conf = float(payload[1]) if len(payload) > 1 else 1.0
            except (TypeError, ValueError, IndexError):
                continue
            lines.append(OCRLine(text=text, box=_bbox(coords), confidence=conf))

        text = "\n".join(l.text for l in lines)
        return EngineOutput(
            text=text,
            confidence=_weighted_confidence(lines),
            engine=self.name,
            ok=True,
            lines=lines,
            meta={"lang": paddle_common.paddle_lang(), "version": paddle_common.paddle_ocr_version()},
        )


class PPStructureEngine(ImageEngine):
    """
    版面检测（设计稿：Layout Detection → PP-StructureV3）.

    能力探测：装了 paddleocr ≥3.0 用 **PP-StructureV3**（版式+表格+公式+
    阅读顺序一体），只有 2.9.x 时退回 **PPStructure（v2）**，并把实际用的是
    哪一版如实写进 ``meta``/``engine`` —— 不假装是 V3。
    """

    name = "pp-structure"
    kind = EngineKind.LAYOUT

    def _version(self) -> str | None:
        if paddle_common.pp_structure_v3_class() is not None:
            return "v3"
        if paddle_common.pp_structure_v2_class() is not None:
            return "v2"
        return None

    def is_available(self) -> bool:
        return self._version() is not None

    def unavailable_reason(self) -> str:
        return "" if self.is_available() else "paddleocr 未提供 PPStructure / PPStructureV3"

    @property
    def engine_label(self) -> str:
        v = self._version()
        return f"pp-structure{v}" if v else self.name

    def _instance(self):
        global _layout_instance
        if _layout_instance is None:
            with _build_lock:
                if _layout_instance is None:
                    paddle_common.ensure_model_home()
                    v3 = paddle_common.pp_structure_v3_class()
                    if v3 is not None:
                        try:
                            _layout_instance = v3()
                        except Exception as exc:      # noqa: BLE001
                            logger.warning("PP-StructureV3 init failed: %s", exc)
                            _layout_instance = None
                    if _layout_instance is None:
                        v2 = paddle_common.pp_structure_v2_class()
                        if v2 is not None:
                            try:
                                _layout_instance = v2(
                                    lang=paddle_common.paddle_lang(),
                                    layout=True, table=False, ocr=True, show_log=False,
                                )
                            except Exception as exc:      # noqa: BLE001
                                logger.warning("PPStructure(v2) init failed: %s", exc)
                                _layout_instance = None
        return _layout_instance

    def process(self, image: Image.Image, **kwargs) -> EngineOutput:
        """返回版面区块（type / bbox / 文本），供分区路由使用."""
        struct = self._instance()
        if struct is None:
            return EngineOutput(engine=self.name, ok=False, error="PP-Structure 不可用")

        try:
            # 串行化：见模块 docstring 坑 4
            with _predict_lock:
                regions_raw = struct(_to_bgr(image))
        except Exception as exc:      # noqa: BLE001
            logger.warning("PP-Structure inference failed: %s", exc)
            return EngineOutput(engine=self.name, ok=False, error=str(exc))

        regions: list[dict] = []
        for region in regions_raw or []:
            box = region.get("bbox") or region.get("coords")
            if box is None:
                continue
            rtype = str(region.get("type") or region.get("label") or "text")
            res = region.get("res")
            text = ""
            if isinstance(res, dict):
                text = str(res.get("text") or res.get("html") or "")
            elif isinstance(res, list):
                text = "\n".join(str(r.get("text", "")) for r in res if isinstance(r, dict))
            elif res is not None:
                text = str(res)
            regions.append({
                "type": rtype,
                "bbox": [int(v) for v in box],
                "text": text.strip(),
            })

        # 版面检测的"置信度" = 是否真的分出了区块（无区块说明这次没看懂）
        conf = 0.85 if regions else 0.0
        return EngineOutput(
            text="",
            confidence=conf,
            engine=self.engine_label,
            ok=bool(regions),
            meta={"regions": regions, "count": len(regions)},
        )


class PaddleFormulaEngine(ImageEngine):
    """
    公式识别（设计稿：公式 → PaddleOCR Formula）.

    诚实起见：paddleocr **2.9.x 只带公式的训练/数据工具，没有推理入口**，
    真正能跑公式的是 ≥3.0 的 ``FormulaRecognition``（PP-FormulaNet）或
    PP-StructureV3 内置的公式分支。拿不到时本引擎如实报告不可用，管线会转
    Vision 兜底 —— 而不是返回一个假的 LaTeX。
    """

    name = "paddleocr-formula"
    kind = EngineKind.FORMULA

    def is_available(self) -> bool:
        return paddle_common.formula_class() is not None

    def unavailable_reason(self) -> str:
        if self.is_available():
            return ""
        return "paddleocr 未暴露公式推理 API（需 ≥3.0 的 FormulaRecognition / PP-StructureV3）"

    def _instance(self):
        global _formula_instance
        if _formula_instance is None:
            with _build_lock:
                if _formula_instance is None:
                    cls = paddle_common.formula_class()
                    if cls is not None:
                        try:
                            _formula_instance = cls()
                        except Exception as exc:      # noqa: BLE001
                            logger.warning("Formula engine init failed: %s", exc)
                            _formula_instance = None
        return _formula_instance

    def process(self, image: Image.Image, **kwargs) -> EngineOutput:
        engine = self._instance()
        if engine is None:
            return EngineOutput(engine=self.name, ok=False, error=self.unavailable_reason())

        try:
            # 串行化：见模块 docstring 坑 4
            with _predict_lock:
                raw = engine(_to_bgr(image))
        except Exception as exc:      # noqa: BLE001
            logger.warning("Formula inference failed: %s", exc)
            return EngineOutput(engine=self.name, ok=False, error=str(exc))

        latex = ""
        conf = 0.0
        if isinstance(raw, list) and raw:
            item = raw[0]
            if isinstance(item, dict):
                latex = str(item.get("rec_text") or item.get("latex") or "")
                conf = float(item.get("rec_score") or item.get("score") or 0.0)
            elif isinstance(item, (list, tuple)) and item:
                latex = str(item[0])
                conf = float(item[1]) if len(item) > 1 else 0.0
        elif isinstance(raw, str):
            latex = raw
            conf = 0.8

        latex = latex.strip()
        if not latex:
            return EngineOutput(engine=self.name, ok=False, error="未识别出公式")

        return EngineOutput(
            text=_wrap_latex(latex),
            confidence=conf or 0.6,
            engine=self.name,
            ok=True,
            meta={"latex": latex},
        )


def _bbox(coords) -> tuple[float, float, float, float] | None:
    """PaddleOCR 的 4 点多边形 → 外接矩形 (x0, y0, x1, y1)."""
    try:
        xs = [float(p[0]) for p in coords]
        ys = [float(p[1]) for p in coords]
    except (TypeError, ValueError, IndexError):
        return None
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _wrap_latex(latex: str) -> str:
    """把 LaTeX 包成行内公式（``$...$``），已是公式环境则不重复包裹。"""
    stripped = latex.strip()
    if stripped.startswith("$") or stripped.startswith("\\["):
        return stripped
    return f"${stripped}$"


__all__ = ["PaddleOCREngine", "PPStructureEngine", "PaddleFormulaEngine"]
