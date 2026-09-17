import pytesseract
from PIL import Image

from app.config import get_settings
from app.services.ocr.base import OCRLine, OCRProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)

# 已解析过的语言集合（避免每次 OCR 都查一遍）
_resolved_lang: str | None = None


def _resolve_lang() -> str:
    """
    选出实际可用的识别语言.

    配置默认是 ``chi_sim+eng``（中文知识库），但镜像里未必装了 chi_sim。
    如果直接把不存在的语言传给 Tesseract，它会报错或被降级成只会认英文 ——
    表现为"中文全部识别为空"，而且**不报错**，很难排查。这里主动探测一次
    并打印告警，让问题在日志里可见。
    """
    global _resolved_lang
    if _resolved_lang is not None:
        return _resolved_lang

    wanted = (get_settings().OCR_TESSERACT_LANG or "eng").strip() or "eng"
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception as exc:      # noqa: BLE001
        logger.warning("Cannot query tesseract languages (%s) — using '%s'", exc, wanted)
        _resolved_lang = wanted
        return _resolved_lang

    wanted_parts = [p.strip() for p in wanted.split("+") if p.strip()]
    usable = [p for p in wanted_parts if p in available]
    missing = [p for p in wanted_parts if p not in available]
    if missing:
        logger.warning(
            "Tesseract language(s) %s are NOT installed (available: %s). "
            "Chinese text in images will be missing — install the language pack "
            "(e.g. apt-get install tesseract-ocr-chi-sim) or use PaddleOCR.",
            missing, sorted(available),
        )
    if usable:
        _resolved_lang = "+".join(usable)
    else:
        _resolved_lang = "eng" if "eng" in available else (sorted(available)[0] if available else "eng")
    logger.info("Tesseract OCR language = '%s' (configured '%s')", _resolved_lang, wanted)
    return _resolved_lang


def _join_words(words: list[str]) -> str:
    """
    拼接同一行的词.

    英文单词之间需要空格；中文单字之间不能有空格（Tesseract 在中文模式下
    常把一个词切成多个单字）。因此只在"两侧都是 ASCII 字母/数字"时补空格。
    """
    out = ""
    for word in words:
        if not word:
            continue
        if out and out[-1].isascii() and out[-1].isalnum() \
                and word[0].isascii() and word[0].isalnum():
            out += " "
        out += word
    return out


def _split_cells(parts: list[dict]) -> list[list[dict]]:
    """
    把 Tesseract 的"一行"按 x 方向的大间隙切成若干单元格.

    为什么必须切：Tesseract 的行分组是**按基线**的，一行里三个列的内容会被
    合并成同一个 line（例如 "Model Params Accuracy"）。如果不切开，无边框表格
    的列信息就彻底丢了 —— 三个不同列的值会被当成同一句话。

    阈值取 **字高的 0.8 倍**。依据：正常词间距约为字高的 0.25–0.4 倍，而跨列
    间隙通常 ≥ 1.5 倍字高，用 0.8 倍可以干净地分开两者。

    （不要用"行内间隙中位数"当基准 —— 如果这一行本身就是一整行表格数据，
    那么行内**全部**间隙都是列间隙，中位数会被抬高到列间隙量级，
    结果一个都切不开。）
    """
    if len(parts) <= 1:
        return [parts]

    heights = sorted(p["height"] for p in parts)
    median_h = heights[len(heights) // 2] or 10.0
    threshold = max(median_h * 0.8, 6.0)

    groups: list[list[dict]] = [[parts[0]]]
    for previous, current in zip(parts, parts[1:]):
        gap = current["left"] - previous["right"]
        if gap > threshold:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def _bucket_to_lines(bucket: dict) -> list[OCRLine]:
    """一个 (block, par, line) 桶 → 一条或多条带坐标的 OCRLine."""
    parts = bucket["parts"]
    groups = _split_cells(parts)

    lines: list[OCRLine] = []
    for group in groups:
        text = _join_words([p["word"] for p in group]).strip()
        if not text:
            continue
        left = min(p["left"] for p in group)
        top = min(p["top"] for p in group)
        right = max(p["right"] for p in group)
        bottom = max(p["bottom"] for p in group)
        confs = [p["conf"] for p in group] or [0.0]
        lines.append(
            OCRLine(
                text=text,
                box=(left, top, right, bottom),
                confidence=sum(confs) / len(confs) / 100.0,
            )
        )
    return lines


class TesseractProvider(OCRProvider):
    def name(self) -> str:
        return "Tesseract"

    def analyze(self, image: Image.Image) -> tuple[str, list[OCRLine]]:
        """
        用 image_to_data 拿到词级坐标，再按 (block, par, line) 聚成行.

        这样 Tesseract 回退路径也能支撑表格结构识别 —— 否则 PaddleOCR 不可用
        时"表格图片 → 结构化表格"会静默退化成一段没有对齐信息的文本。
        """
        rgb = image.convert("RGB")
        data = pytesseract.image_to_data(
            rgb, output_type=pytesseract.Output.DICT, lang=_resolve_lang()
        )

        buckets: dict[tuple[int, int, int], dict] = {}
        order: list[tuple[int, int, int]] = []
        n = len(data.get("text", []))
        for i in range(n):
            word = (data["text"][i] or "").strip()
            if not word:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf is not None and conf < 0:
                continue
            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            left, top = float(data["left"][i]), float(data["top"][i])
            right = left + float(data["width"][i])
            bottom = top + float(data["height"][i])
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {"parts": []}
                buckets[key] = bucket
                order.append(key)
            bucket["parts"].append(
                {
                    "word": word, "conf": max(conf, 0.0),
                    "left": left, "top": top, "right": right, "bottom": bottom,
                    "height": bottom - top,
                }
            )

        lines: list[OCRLine] = []
        for key in order:
            lines.extend(_bucket_to_lines(buckets[key]))
        return "\n".join(item.text for item in lines), lines

    def extract_text(self, image: Image.Image) -> str:
        # Convert to RGB just in case it's RGBA or something else
        return self.analyze(image)[0]

    def extract_lines(self, image: Image.Image) -> list[OCRLine]:
        return self.analyze(image)[1]
