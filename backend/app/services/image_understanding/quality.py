"""
产出质量校验（VLM / OCR / 结构化引擎的"事后质检"）.

流程图里的 Validation 菱形只做了一件粗活：**结构形态**对不对（表格有没有
对齐的管道符、公式有没有 LaTeX 记号）。但真实世界里最常见的失败不是"形态
不对"，而是"形态对但内容是错的"：

    · 代码截图 → VLM 转写出**看起来**很合理的代码，但括号不配平、`def` 少了
      冒号 —— 语法根本不成立。形态校验（有 ``` 围栏）100% 通过。
    · 扫描件 → OCR 把 "1" 读成 "l"、"0" 读成 "O"，文本读起来通顺，但每个数字
      都是错的。而 OCR 引擎自己对这些行照样给 0.9 的置信度。
    · VLM → 幻觉出图里根本没有的数字，或把提示词原文当答案吐回来。

本模块就是补这一层：**能形式化验证的，就真的去验证**。

    ┌──────────────────┬────────────────────────────────────────────────┐
    │ check_code_syntax│ Python 用 ast.parse（真解析器！）、JSON 用      │
    │                  │ json.loads、YAML 用 safe_load、其余语言用       │
    │                  │ 字符串感知的括号/引号配平 + 结构启发式           │
    │ assess_ocr_       │ 行置信度均值 / 最低行 / 低置信行占比。          │
    │ confidence       │ 全部为 0 时判"未上报"而不是"全错"（重要）        │
    │ check_vlm_output │ 提示词泄漏、复读、长度、数字锚点、围栏缺失       │
    └──────────────────┴────────────────────────────────────────────────┘

设计原则与 confidence.py 一致：**只信可验证的事实**。任何一项校验都返回
``passed/score/reasons``，由上层决定是"打折"还是"打回"。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CodeSyntaxReport:
    """代码语法校验结论."""

    language: str
    passed: bool
    score: float
    errors: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "passed": self.passed,
            "score": round(float(self.score), 4),
            "errors": list(self.errors),
            "checks": list(self.checks),
        }


@dataclass
class OcrConfidenceReport:
    """OCR 行置信度评估结论."""

    lines: int
    mean: float
    minimum: float
    low_ratio: float
    reported: bool          # 引擎是否真的给了置信度
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lines": self.lines,
            "mean": round(float(self.mean), 4),
            "min": round(float(self.minimum), 4),
            "low_ratio": round(float(self.low_ratio), 4),
            "reported": self.reported,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


@dataclass
class QualityReport:
    """一张图片产出的完整质检结论（写进 ImageUnderstanding.quality）."""

    ok: bool
    score: float
    engine: str = ""
    reasons: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    code: CodeSyntaxReport | None = None
    ocr: OcrConfidenceReport | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "score": round(float(self.score), 4),
            "engine": self.engine,
            "reasons": list(self.reasons),
            "checks": list(self.checks),
            "code": self.code.to_dict() if self.code else None,
            "ocr": self.ocr.to_dict() if self.ocr else None,
            "meta": dict(self.meta),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 代码语法校验
# ─────────────────────────────────────────────────────────────────────────────

#: 代码围栏：```lang\n...\n```
_FENCE_RE = re.compile(r"^\s*```([A-Za-z0-9_+#.-]*)\s*\n(.*?)(?:\n\s*```\s*)?$", re.DOTALL)

#: 语言别名 → 规范化名（VLM 常写 "py" / "c++" / "C#"）
_LANGUAGE_ALIASES = {
    "py": "python", "python3": "python", "python": "python",
    "js": "javascript", "javascript": "javascript", "jsx": "javascript",
    "ts": "typescript", "typescript": "typescript", "tsx": "typescript",
    "json": "json", "yaml": "yaml", "yml": "yaml",
    "java": "java", "go": "go", "golang": "go", "rust": "rust", "rs": "rust",
    "c": "c_cpp", "cpp": "c_cpp", "c++": "c_cpp", "c_cpp": "c_cpp",
    "h": "c_cpp", "hpp": "c_cpp",
    "sql": "sql", "sh": "shell", "bash": "shell", "shell": "shell",
    "zsh": "shell", "html": "html", "xml": "html",
}

#: 每种语言的"必须配平"的括号对
_BRACKETS = {"(": ")", "[": "]", "{": "}"}

#: 注释起始符（用于把注释内容排除在括号配平之外）
_LINE_COMMENT = {"python": "#", "shell": "#", "yaml": "#", "sql": "--",
                 "javascript": "//", "typescript": "//", "java": "//",
                 "c_cpp": "//", "go": "//", "rust": "//"}

#: 语言是**猜**出来的时候，内容至少要有这么"像代码"才认（见 check_code_syntax）。
#: 配平检查对自然语言毫无鉴别力 —— 中文说明的括号天然配平，不设这道门就会
#: 把"这是一段说明文字。"判成"通过的 JavaScript"。
_MIN_GUESSED_LIKENESS = 0.4

#: 硬失败（提示词泄漏 / 复读 / 代码语法不成立）时的分数上限.
#: 刻意低于默认通过阈值 0.5 —— 硬失败必须判"未通过"，不能因为"扣的分刚好
#: 等于阈值"而蒙混过关。
HARD_FAIL_CAP = 0.45


def _normalize_language(language: str) -> str:
    return _LANGUAGE_ALIASES.get((language or "").strip().lower(), "unknown")


def split_fence(text: str) -> tuple[str, str]:
    """
    拆开 Markdown 代码围栏 → ``(语言, 代码体)``.

    没有围栏时返回 ``("", 原文)``。语言标签可能是空的（```\\n...）。
    """
    body = text or ""
    match = _FENCE_RE.match(body)
    if not match:
        return "", body.strip()
    return (match.group(1) or "").strip(), (match.group(2) or "").strip()


def _scan_balance(code: str, language: str) -> list[str]:
    """
    字符串感知的括号 / 引号配平检查.

    "字符串感知"是必须的：``print("(")`` 里的括号不算括号，``x = "it's"``
    里的单引号不是字符串起始。天真地数 ``code.count("(")`` 会在这两处
    同时误判 —— 而代码截图里带括号的字符串恰恰非常常见。

    返回错误描述列表（空列表 = 配平通过）。
    """
    errors: list[str] = []
    stack: list[tuple[str, int]] = []
    line = 1
    quote: str | None = None
    comment = _LINE_COMMENT.get(language)
    block_comment = language in ("javascript", "typescript", "java", "c_cpp", "go", "rust")
    i = 0
    length = len(code)
    in_block = False

    while i < length:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < length else ""

        if ch == "\n":
            line += 1
            i += 1
            continue

        if in_block:
            if ch == "*" and nxt == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue

        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue

        # 行注释
        if comment and code.startswith(comment, i):
            while i < length and code[i] != "\n":
                i += 1
            continue
        # 块注释
        if block_comment and ch == "/" and nxt == "*":
            in_block = True
            i += 2
            continue

        if ch in ("'", '"', "`"):
            quote = ch
            i += 1
            continue

        if ch in _BRACKETS:
            stack.append((ch, line))
        elif ch in _BRACKETS.values():
            if not stack:
                errors.append(f"第 {line} 行：多余的右括号 '{ch}'")
            else:
                opener, opened_at = stack.pop()
                if _BRACKETS[opener] != ch:
                    errors.append(
                        f"第 {line} 行：'{ch}' 与第 {opened_at} 行的 '{opener}' 不匹配"
                    )
        i += 1

    if quote is not None:
        errors.append(f"字符串引号 {quote} 未闭合")
    for opener, opened_at in stack:
        errors.append(f"第 {opened_at} 行的 '{opener}' 未闭合")
    return errors


def _check_python(code: str) -> CodeSyntaxReport:
    """
    Python：**真的用解析器**（``ast.parse``），不是猜.

    这是本模块最有价值的一项 —— 它是"确定性判定"，不存在阈值调参。
    解析失败时把出错行号带出来，便于人工复核时直接定位。
    """
    checks = ["ast.parse"]
    try:
        ast.parse(code)
    except SyntaxError as exc:
        detail = f"第 {exc.lineno} 行：{exc.msg}" if exc.lineno else str(exc.msg)
        return CodeSyntaxReport(
            language="python", passed=False, score=0.2,
            errors=[detail], checks=checks,
            meta={"lineno": exc.lineno, "offset": exc.offset},
        )
    except (ValueError, MemoryError) as exc:      # 空字节 / 超深嵌套
        return CodeSyntaxReport(
            language="python", passed=False, score=0.3,
            errors=[f"解析异常：{exc}"], checks=checks,
        )
    return CodeSyntaxReport(language="python", passed=True, score=1.0, checks=checks)


def _check_json(code: str) -> CodeSyntaxReport:
    checks = ["json.loads"]
    try:
        json.loads(code)
    except json.JSONDecodeError as exc:
        return CodeSyntaxReport(
            language="json", passed=False, score=0.2,
            errors=[f"第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}"],
            checks=checks,
        )
    return CodeSyntaxReport(language="json", passed=True, score=1.0, checks=checks)


def _check_yaml(code: str) -> CodeSyntaxReport:
    """YAML：装了 PyYAML 就真解析，没装则退化为缩进一致性启发式."""
    checks: list[str] = []
    try:
        import yaml      # type: ignore

        checks.append("yaml.safe_load")
        try:
            yaml.safe_load(code)
        except Exception as exc:      # noqa: BLE001 — PyYAML 的异常类型很多
            return CodeSyntaxReport(
                language="yaml", passed=False, score=0.25,
                errors=[str(exc).splitlines()[0]], checks=checks,
            )
        return CodeSyntaxReport(language="yaml", passed=True, score=1.0, checks=checks)
    except ImportError:
        checks.append("indent-heuristic")
        # 退化为"缩进是否稳定为偶数 / 有无 tab 混用"这类结构性检查
        errors: list[str] = []
        for i, raw in enumerate(code.splitlines(), start=1):
            if "\t" in raw[: len(raw) - len(raw.lstrip())]:
                errors.append(f"第 {i} 行：缩进混用 Tab（YAML 不允许）")
        balance = _scan_balance(code, "yaml")
        errors.extend(balance)
        if errors:
            return CodeSyntaxReport(
                language="yaml", passed=False, score=0.4, errors=errors, checks=checks,
            )
        return CodeSyntaxReport(language="yaml", passed=True, score=0.7, checks=checks)


def _check_braced(code: str, language: str) -> CodeSyntaxReport:
    """
    C 系 / JS / Go / Rust / SQL 等：括号配平 + 结构启发式.

    这些语言没有"标准库自带解析器"，所以只能做**形式检查**。刻意不引入
    tree-sitter 之类的重型依赖：入库期每张图都要跑，依赖越重越难部署。
    形式检查抓不出"语义错误"，但能抓住 OCR/VLM 最常见的失败 —— 括号丢失、
    引号未闭合、代码被截断。
    """
    checks = ["bracket-balance", "quote-balance"]
    errors = _scan_balance(code, language)
    score = 1.0
    if errors:
        score = max(0.15, 1.0 - 0.25 * len(errors))
        return CodeSyntaxReport(
            language=language, passed=False, score=score, errors=errors, checks=checks,
        )

    # 额外的弱信号：代码被截断（最后一行以运算符/逗号结尾且无后续）
    lines = [l for l in code.splitlines() if l.strip()]
    if lines:
        tail = lines[-1].rstrip()
        if tail.endswith((",", "+", "-", "&&", "||", "=", "->", ".")):
            checks.append("truncation")
            return CodeSyntaxReport(
                language=language, passed=False, score=0.5,
                errors=[f"最后一行疑似被截断：…{tail[-30:]}"], checks=checks,
            )
    return CodeSyntaxReport(language=language, passed=True, score=0.9, checks=checks)


def check_code_syntax(code: str, language: str = "") -> CodeSyntaxReport:
    """
    代码语法校验（可形式化验证的语言就真的验证）.

    :param code: 代码文本，可以带 Markdown 围栏（会自动拆开并采用围栏里的
        语言标签 —— VLM 通常会标语言，比调用方猜更准）。
    :param language: 语言提示；为空时用围栏标签，再为空则按内容猜。

    空文本返回 ``passed=False`` —— "没产出代码"当然不算"代码语法正确"。
    """
    fenced_lang, body = split_fence(code)
    if not body.strip():
        return CodeSyntaxReport(
            language=_normalize_language(language or fenced_lang),
            passed=False, score=0.0, errors=["代码内容为空"], checks=["non-empty"],
        )

    raw_lang = language or fenced_lang
    guessed = not raw_lang
    if guessed:
        from app.services.image_understanding.engines.code_parser import detect_language

        raw_lang, _ = detect_language(body.splitlines())
    lang = _normalize_language(raw_lang)

    if lang == "python":
        report = _check_python(body)
    elif lang == "json":
        report = _check_json(body)
    elif lang == "yaml":
        report = _check_yaml(body)
    else:
        # 语言是**猜**出来的（调用方没给、围栏也没标）时，先确认"这确实是代码".
        # 否则一段括号天然配平的中文说明会被判成"通过的 JavaScript" ——
        # 配平检查对自然语言毫无鉴别力（自然语言本来就没有不配平的括号）。
        if guessed:
            from app.services.image_understanding.engines.code_parser import code_likeness

            likeness = code_likeness(body.splitlines())
            if likeness < _MIN_GUESSED_LIKENESS:
                return CodeSyntaxReport(
                    language=lang, passed=False, score=round(likeness, 4),
                    errors=[f"未识别出编程语言，且内容不像代码（相似度 {likeness:.2f}）"],
                    checks=["language-detect"],
                )
        report = _check_braced(body, lang if lang != "unknown" else "javascript")

    if fenced_lang:
        report.checks.append("fence-language")
    report.meta.setdefault("chars", len(body))
    report.meta.setdefault("lines", len(body.splitlines()))
    return report


# ─────────────────────────────────────────────────────────────────────────────
# OCR 置信度评估
# ─────────────────────────────────────────────────────────────────────────────

#: 低于该值的 OCR 行算"低置信行"
LOW_LINE_CONFIDENCE = 0.60
#: 低置信行占比上限：超过就认为这次 OCR 整体不可信
LOW_LINE_RATIO_MAX = 0.35


def assess_ocr_confidence(lines: list | None) -> OcrConfidenceReport:
    """
    从 OCR 行置信度分布判断"这次 OCR 到底能不能信".

    为什么不只看均值：均值会被大量"轻松识别的空白行"拉高。真正决定成败的是
    **最差的那几行** —— 表格里一个数字读错就足以让结论错。因此同时看
    均值、最低行、低置信行占比。

    关键细节：**全部为 0 时判"未上报"而不是"全错"**。部分 OCR 后端（某些
    Tesseract 配置、Paddle 的检测阶段）不返回逐行置信度，此时一刀切判失败
    会把整条链路误伤成"人工复核"。这类情况如实标记 ``reported=False``，
    由上层决定是否采信。
    """
    values: list[float] = []
    for line in lines or []:
        text = (getattr(line, "text", "") or "").strip()
        if not text:
            continue
        try:
            values.append(float(getattr(line, "confidence", 0.0) or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)

    if not values:
        return OcrConfidenceReport(
            lines=0, mean=0.0, minimum=0.0, low_ratio=1.0,
            reported=False, passed=False, reasons=["没有可用的 OCR 行"],
        )

    if all(v <= 0.0 for v in values):
        return OcrConfidenceReport(
            lines=len(values), mean=0.0, minimum=0.0, low_ratio=0.0,
            reported=False, passed=True,
            reasons=["OCR 引擎未上报逐行置信度（无法评估，不作否决）"],
        )

    mean = sum(values) / len(values)
    minimum = min(values)
    low = sum(1 for v in values if v < LOW_LINE_CONFIDENCE)
    low_ratio = low / len(values)

    reasons: list[str] = []
    passed = True
    if mean < _threshold("IMAGE_OCR_CONFIDENCE_MIN", 0.65):
        passed = False
        reasons.append(f"平均行置信度 {mean:.2f} 偏低")
    if low_ratio > _threshold("IMAGE_OCR_LOWCONF_RATIO_MAX", LOW_LINE_RATIO_MAX):
        passed = False
        reasons.append(f"{low_ratio:.0%} 的行置信度低于 {LOW_LINE_CONFIDENCE:.2f}")
    if minimum < 0.25 and len(values) >= 3:
        # 极低行不直接否决（可能只是一行噪声），但记下来供人工复核参考
        reasons.append(f"存在极低置信行（最低 {minimum:.2f}）")

    return OcrConfidenceReport(
        lines=len(values), mean=round(mean, 4), minimum=round(minimum, 4),
        low_ratio=round(low_ratio, 4), reported=True, passed=passed, reasons=reasons,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VLM 输出校验
# ─────────────────────────────────────────────────────────────────────────────

#: 提示词泄漏特征 —— 模型把提示词原文当答案吐回来
_PROMPT_LEAK_MARKERS = (
    "请提取", "请把", "用中文输出", "只输出", "不要任何前缀",
    "分点罗列", "这是一张", "请尽最大努力",
)
#: 数字（含小数、百分比、千分位）
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")

# 列表编号 / 分点标记：``1)`` ``2.`` ``三、`` ``(4)`` ``•`` —— 这些是**排版**，
# 不是图里的数字。数字锚点校验前必须先剥掉它们。
_LIST_MARKER_RE = re.compile(
    r"^[ \t]*(?:\(\s*\d+\s*\)|\d+\s*[).、:：]|[一二三四五六七八九十]+\s*[)）、.:：]|[-*•·])\s*",
    re.MULTILINE,
)


def _strip_enumeration_markers(text: str) -> str:
    """
    剥掉行首的列表编号/项目符号.

    只处理**行首**标记：正文里的数字（``准确率 92.5%``）必须原样保留，
    否则数字锚点就失去意义了。
    """
    if not text:
        return ""
    return _LIST_MARKER_RE.sub("", text)


def _repetition_ratio(text: str) -> float:
    """
    复读率：重复行占比.

    多模态模型在"看不清图"时会退化成复读（同一句输出十几遍），这是最典型的
    幻觉形态之一。只统计**非平凡行**（长度 ≥ 6）以避免把代码里重复的 ``}``
    误判成复读。
    """
    lines = [l.strip() for l in (text or "").splitlines() if len(l.strip()) >= 6]
    if len(lines) < 4:
        return 0.0
    unique = len(set(lines))
    return round(1.0 - unique / len(lines), 4)


def check_vlm_output(
    text: str,
    image_type: str,
    *,
    ocr_text: str | None = None,
) -> QualityReport:
    """
    校验 VLM 的产出是否"可信".

    检查项按"能否形式化验证"排序，从硬到软：

        1. 非空 / 长度下限           —— 硬
        2. 提示词泄漏                 —— 硬（模型复述了指令，等于没干活）
        3. 复读率                     —— 硬（幻觉的典型形态）
        4. 代码围栏 + 语法校验         —— 硬（代码类型专用，见 check_code_syntax）
        5. 数字锚点                   —— **软**（VLM 报的数字若一个都不在 OCR
                                        文本里，可能是幻觉；但 OCR 也可能没
                                        读到那个数字，所以只降分不否决）
    """
    from app.services.image_understanding.structured_content import (
        IMAGE_TYPE_CHART,
        IMAGE_TYPE_CODE,
        IMAGE_TYPE_DIAGRAM,
        IMAGE_TYPE_FORMULA,
        IMAGE_TYPE_TABLE,
    )

    content = (text or "").strip()
    checks: list[str] = ["non-empty"]
    reasons: list[str] = []
    meta: dict = {}
    score = 1.0
    # 硬失败的分数上限。**硬失败必须落在通过阈值（IMAGE_VLM_QUALITY_MIN，
    # 默认 0.5）以下** —— 否则"扣了 0.5 分"的 0.5 会正好卡在阈值上被判通过，
    # 变成"发现了问题却不处理"。用"封顶"而不是"扣分"就是为了不受阈值调整影响。
    hard_fail_cap: float | None = None

    if not content:
        return QualityReport(
            ok=False, score=0.0, reasons=["VLM 产出为空"], checks=checks,
        )

    # ── 2. 提示词泄漏 ──────────────────────────────────────────────────────
    leaked = [m for m in _PROMPT_LEAK_MARKERS if m in content[:200]]
    if len(leaked) >= 2:
        checks.append("prompt-leak")
        reasons.append(f"疑似复述提示词（命中 {len(leaked)} 个指令词）")
        hard_fail_cap = HARD_FAIL_CAP
    elif leaked:
        checks.append("prompt-leak")
        score -= 0.15

    # ── 3. 复读 ────────────────────────────────────────────────────────────
    rep = _repetition_ratio(content)
    meta["repetition_ratio"] = rep
    if rep >= 0.5:
        checks.append("repetition")
        reasons.append(f"复读率 {rep:.0%}（模型疑似未看清图片）")
        hard_fail_cap = HARD_FAIL_CAP if hard_fail_cap is None else min(hard_fail_cap, HARD_FAIL_CAP)
        score -= 0.4

    # ── 4. 代码类型：必须有围栏，且语法要成立 ───────────────────────────────
    code_report: CodeSyntaxReport | None = None
    if image_type == IMAGE_TYPE_CODE:
        checks.append("code-fence")
        if not content.lstrip().startswith("```"):
            reasons.append("代码产出缺少 Markdown 围栏")
            score -= 0.2
        if _threshold_bool("IMAGE_CODE_SYNTAX_CHECK", True):
            code_report = check_code_syntax(content)
            meta["code_syntax"] = code_report.to_dict()
            if not code_report.passed:
                checks.append("code-syntax")
                reasons.append(
                    "代码语法校验未通过：" + "；".join(code_report.errors[:3])
                )
                score -= 0.5
                hard_fail_cap = HARD_FAIL_CAP if hard_fail_cap is None else min(hard_fail_cap, HARD_FAIL_CAP)
            else:
                checks.append("code-syntax")

    # ── 5. 数字锚点（软信号）───────────────────────────────────────────────
    if ocr_text and image_type in (IMAGE_TYPE_CHART, IMAGE_TYPE_TABLE,
                                   IMAGE_TYPE_FORMULA, IMAGE_TYPE_DIAGRAM):
        # 先剥掉**列表编号**：VLM 的提示词模板要求"1) 2) 3)"分点作答，这些
        # 序号是模型自己的排版，不是图里的数字。不剥的话它们会被当成"报出的
        # 数字"，而 OCR 文本里当然找不到 → 一顶"疑似幻觉"的帽子扣下来，
        # 一份完全正确的图表描述被扣 0.25 分甚至挡在门外（实测踩过）。
        vlm_numbers = set(_NUMBER_RE.findall(_strip_enumeration_markers(content)))
        if len(vlm_numbers) >= 3:
            ocr_numbers = set(_NUMBER_RE.findall(_strip_enumeration_markers(ocr_text)))
            # 归一化后再比：去掉千分位逗号
            norm = lambda s: {n.replace(",", "") for n in s}      # noqa: E731
            shared = norm(vlm_numbers) & norm(ocr_numbers)
            anchor = len(shared) / len(norm(vlm_numbers))
            meta["number_anchor"] = round(anchor, 4)
            checks.append("number-anchor")
            if anchor == 0.0:
                reasons.append("VLM 报出的数字在 OCR 文本中一个都找不到（疑似幻觉）")
                score -= 0.25

    if hard_fail_cap is not None:
        score = min(score, hard_fail_cap)
    score = round(max(0.0, min(1.0, score)), 4)
    ok = score >= _threshold("IMAGE_VLM_QUALITY_MIN", 0.5)
    return QualityReport(
        ok=ok, score=score, reasons=reasons, checks=checks,
        code=code_report, meta=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────


def verify_output(
    text: str,
    image_type: str,
    *,
    engine: str = "",
    ocr_lines: list | None = None,
    ocr_text: str | None = None,
) -> QualityReport:
    """
    对**任意引擎**的产出做统一质检，返回一份 :class:`QualityReport`.

    这是质检的总入口，把四类证据合到一起：

        1. **结构形态**（table 的管道符网格 / formula 的 LaTeX 记号 /
           diagram 的结构词）—— 复用 confidence.validate，保证"结构"只有
           一个判定口径；
        2. **代码语法**（ast.parse / json.loads / 括号配平）—— 代码类型专用，
           已由 check_vlm_output 覆盖，这里不重复扣分；
        3. **OCR 置信度**（均值 / 最低行 / 低置信占比）；
        4. **VLM 幻觉**（提示词泄漏 / 复读 / 数字锚点）。

    管线在门控（gate）之前调它，得到的 ``score`` 会作为乘数进入最终置信度。
    """
    from app.services.image_understanding.structured_content import IMAGE_TYPE_CODE

    ocr_report = assess_ocr_confidence(ocr_lines) if ocr_lines is not None else None
    base = check_vlm_output(text, image_type, ocr_text=ocr_text)
    base.engine = engine
    base.ocr = ocr_report

    reasons = list(base.reasons)
    score = base.score
    checks = list(base.checks)

    # ── 结构形态（非代码类型）───────────────────────────────────────────────
    # 代码类型的形态校验已经在 check_vlm_output 里做过了（而且更硬：真解析器），
    # 这里再做一次会重复扣分，把一个语法错误罚两遍。
    if image_type != IMAGE_TYPE_CODE and (text or "").strip():
        try:
            from app.services.image_understanding.confidence import validate

            structure = validate(image_type, text)
            base.meta["structure"] = {
                "passed": structure.passed,
                "score": structure.score,
                "reasons": structure.reasons,
            }
            checks.append("structure")
            if not structure.passed:
                # 结构不成立 → 打折（不是否决）：可能是"描述型"产出本来就不
                # 需要该结构（例如 chart 的描述很短但内容正确）。
                score = round(score * max(0.3, float(structure.score)), 4)
                reasons.extend(f"结构校验：{r}" for r in structure.reasons[:2])
        except Exception as exc:      # noqa: BLE001
            logger.debug("Structure validation skipped: %s", exc)

    # ── OCR 置信度 ──────────────────────────────────────────────────────────
    if ocr_report is not None:
        checks.append("ocr-confidence")
        if not ocr_report.passed:
            # OCR 置信度不达标 → 打折（而不是直接否决）：产出可能来自 VLM，
            # OCR 差并不代表 VLM 也差。这里只是提醒"这次读字不太可靠"。
            score = round(score * 0.7, 4)
            reasons.extend(ocr_report.reasons)
        base.meta["ocr_confidence"] = ocr_report.to_dict()

    base.score = round(max(0.0, min(1.0, score)), 4)
    base.reasons = reasons
    base.checks = checks
    base.ok = base.score >= _threshold("IMAGE_VLM_QUALITY_MIN", 0.5)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 配置读取（与 confidence.py 同款：配置缺失不抛）
# ─────────────────────────────────────────────────────────────────────────────


def _threshold(name: str, default: float) -> float:
    try:
        from app.config import get_settings

        return float(getattr(get_settings(), name, default))
    except Exception:      # noqa: BLE001
        return default


def _threshold_bool(name: str, default: bool) -> bool:
    try:
        from app.config import get_settings

        return bool(getattr(get_settings(), name, default))
    except Exception:      # noqa: BLE001
        return default


__all__ = [
    "CodeSyntaxReport",
    "OcrConfidenceReport",
    "QualityReport",
    "check_code_syntax",
    "check_vlm_output",
    "assess_ocr_confidence",
    "verify_output",
    "split_fence",
    "LOW_LINE_CONFIDENCE",
    "LOW_LINE_RATIO_MAX",
]
