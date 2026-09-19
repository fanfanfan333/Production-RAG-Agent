"""
Citation Verifier 节点（架构图 LLM Generate → **Citation Verifier** → Output Guard）.

对应设计稿的五项逐条校验：

    引用存在？        引用的 [Source N] 是否真的在本次 sources 里
    引用位置正确？    这句话是否确实出自它标注的那一条，而不是别的来源
    原文支持该结论？  被引原文是否真的支持这句话的结论
    数字是否一致？    句中的数字能否在被引原文里原样找到
    日期是否一致？    句中的日期能否在被引原文里原样找到

为什么需要它
────────────
Output Guard 只检查"引用编号是否越界"——它拦得住 `[Source 99]` 这种幻觉，
但拦不住更危险的一种：**编号合法、但内容对不上**。例如模型把 B 文档的
数字挂到 A 文档的引用上，或凭空写一个原文没有的百分比却标上 `[Source 1]`。
这类"看起来有据可查"的错误，恰恰是最难被用户发现的。

本模块用**确定性文本比对**（不调 LLM）把每条引用拆开验一遍，产出：

- 每条引用的五项结论（可解释、可审计、可前端展示）；
- 全局判定：verified / partial / unsupported / no_citations；
- 净化后的答案（越界引用移除；可选地移除"不被原文支持"的引用标记）。

设计取舍
────────
- **宁可漏判，不可错杀**：阈值偏保守。把一条正确的引用标成"不支持"，
  比漏掉一条错误引用对用户的伤害更大（会摧毁对系统的信任）。因此
  数字/日期比对是"精确子串"级别（确定性强），支持度比对是"覆盖率"
  级别（阈值宽松，默认 0.30）。
- **只验证，不判罪**：本模块不删除句子，只处理引用标记与给出结论；
  最终是否整段拒答由 Output Guard / Evidence Gate 决定。
- **纯函数、零依赖**：可直接单测（tests/test_citation_verifier.py）。

引用力度（citation strength）
────────────────────────────
"引用力度合适" 的两层含义在本模块落地：
  1. 不过度引用 —— 不要求每句话都带引用（见 rag_graph 的规则 7），
     因此本模块只验证**已出现**的引用，不因"某句没引用"而报错；
  2. 引用必须落在正确的来源上 —— 这正是"引用位置正确？"这一项。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.hybrid_search import tokenize
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── 引用标记 ─────────────────────────────────────────────────────────────────
_CITATION_RE = re.compile(
    r"\[Source\s*(\d+)(?:\s*[-–~]\s*(\d+))?\]",
    re.IGNORECASE,
)

# 句子边界（中英文）—— 用于把"引用所属的句子"切出来单独验证
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")

# ── 数字 / 日期 ──────────────────────────────────────────────────────────────
# 数字：整数、小数、千分位、百分比、带单位（30天 / 12.5% / 1,200 元）
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:%|％|‰)?")

# 日期：2024-01-31 / 2024/1/31 / 2024年1月31日 / 2024年1月 / 2024年
_DATE_RE = re.compile(
    r"\d{4}\s*[-/年]\s*\d{1,2}(?:\s*[-/月]\s*\d{1,2}\s*日?)?|\d{4}\s*年"
)

# ── 排版标记（必须先剥掉，否则会被当成正文数字）──────────────────────────────
#
# 最典型的一类系统性误判（截图问题4）：
#
#     1. **反幻觉机制**：通过检索阶段降权处理不可信文档片段……
#     2. **数据层设计**：采用结构感知解析器解析文档……
#
# 行首的列表序号 "1." / "2." 会被 _NUMBER_RE 抽成数字 "1"/"2"，再拿去原文里
# 找 —— 原文当然找不到"1"，于是每条带着列表序号的句子都被判"数字与原文不一致"，
# 一轮回答 6 条引用能因此全被标存疑。但这些序号是 **Markdown 排版**，不是模型
# 陈述的正文事实，压根不该进入数字比对。
#
# 剥掉的都是行首的结构标记（标题井号 / 引用块 / 无序列表 / 有序列表 / (1) 括注）。
# 三个关键约束：
#   1. 只吃"序号本身"，不吃序号后面的内容 —— "3. 三期推进，节省 60% 存储"
#      要留下 "60%"，只丢掉那个 "3."；
#   2. 标题允许吃掉自己的编号（"## 7 反幻觉机制"、"## 3. 实施计划"），
#      因为那是文档的章节号，不是模型陈述的事实；
#   3. 一律用 `(?![0-9])` 挡住小数与年份 —— "2024年营收"、"3.5 亿元"
#      绝不能被当成"序号 + 正文"而被削掉首位数。
_STRUCTURAL_MARKER_RE = re.compile(
    r"(?m)^[ \t]*(?:"
    # 标题 + 可选的章节编号：## 3. 实施计划 / ### 7 反幻觉机制
    r"\#{1,6}[ \t]*(?:\d{1,3}(?![0-9])(?:[.)\u3001:\uff1a](?![0-9])[ \t]*|[ \t]+))?"
    r"|[>\uff1e][ \t]?"                            # 引用块
    r"|[-*+\u2022\u00b7][ \t]+"                    # 无序列表
    r"|\d{1,3}[.)\u3001:\uff1a](?![0-9])[ \t]*"    # 有序列表：1. / 1) / 1、 / 1：
    r"|[(\uff08]\d{1,3}[)\uff09][ \t]*"            # 括注序号：(1) （2）
    r")*"
)

# ── 句首枚举序号（分句之后再剥一次）──────────────────────────────────────────
#
# _STRUCTURAL_MARKER_RE 只认**行首**标记。但模型经常把答案写成一整段
# "……核心要点包括：1. 反幻觉机制：… 2. 数据层设计：…"，分句之后 "1." 落在
# **句首**而不在行首，于是序号 "1"/"2" 被当成正文数字拿去原文比对 ——
# 原文当然没有，于是每条引用都被判"数字与原文不一致"（截图问题：
# "原文中找不到的数字：1"，四条引用全灭）。序号是**排版**，不是事实。
_SENTENCE_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"\d{1,3}[.)\u3001:\uff1a](?![0-9])\s*"       # 1. / 2) / 3、 / 4:
    r"|[(\uff08]\d{1,3}[)\uff09]\s*"              # (1) （2）
    r"|[一二三四五六七八九十]{1,3}[\u3001.]\s*"     # 一、 / 二.
    r")+"
)

# 句中枚举序号：模型把列表写在一句话里时（"……核心要点包括：1. 反幻觉机制…，
# 2. 数据层设计…"），序号前面是冒号/逗号/顿号而不是行首或句首。这类序号
# 同样是排版。约束与行首版一致：序号后 (?![0-9]) 挡住 "3.5 亿" / "1:2" /
# "1.31" 这类真数字，序号本身限定 1-3 位（年份 2024 不会误入）。
_INLINE_ENUM_RE = re.compile(
    r"(?<=[\uff1a:;,，、\u3001])\s*"
    r"(?:\d{1,3}[.)\u3001\uff1a](?![0-9])|[(\uff08]\d{1,3}[)\uff09])\s*"
)

# ── 溯源 / 位置表达（模型复述出处元数据，不是正文事实）──────────────────────
#
# 模型会把上下文里的出处信息复述进答案：
#
#     出自《研发部-2024年度技术方案.docx》，第 1 页，第 83-105 行 ……
#
# 书名里的年份（2024）、页码（1）、行号（83/105）都是**元数据**，不是模型
# 陈述的事实；拿去和被引原文做数字比对必然误报 —— 原文 chunk 里没有页码。
_BOOK_TITLE_RE = re.compile(r"《[^》\n]{1,120}》")
_LOCATION_REF_RE = re.compile(
    r"第\s*\d+(?:\s*[-–~]\s*\d+)?\s*"
    r"(?:页|行|张图|个表格|张表|章|节|段|条|部分|篇|点|项|季度)"
)


def _canon_number(token: str) -> str:
    """归一化数字：去千分位、统一百分号、去空白（1,200 → 1200）."""
    return (
        token.replace(",", "")
        .replace("％", "%")
        .replace(" ", "")
        .strip()
    )


def _canon_date(token: str) -> str:
    """归一化日期：统一分隔符、去"日"、**月/日补零**.

    补零是必需的：文档常写「2024年01月」，而模型复述成「2024年1月」——
    不做补零时前者归一为 ``2024-01``、后者 ``2024-1``，两者不相等 →
    正确答案被误判"日期与原文不一致"而被删引用。
    补零后 ``2024-1`` 与 ``2024-01`` 均归一为 ``2024-01``（年份保持原样）。
    """
    s = (
        token.replace("年", "-")
        .replace("月", "-")
        .replace("日", "")
        .replace("/", "-")
        .replace(" ", "")
        .strip("-")
    )
    parts = s.split("-")
    if len(parts) <= 1:
        return s
    # parts[0] = 年份（原样）；其余 = 月/日（纯数字补零到两位）
    return "-".join([parts[0], *(p.zfill(2) if p.isdigit() else p for p in parts[1:])])


def extract_numbers(text: str) -> set[str]:
    """抽取文本中的全部数字（已归一化）."""
    return {_canon_number(m.group(0)) for m in _NUMBER_RE.finditer(text or "")}


def extract_dates(text: str) -> set[str]:
    """抽取文本中的全部日期（已归一化）."""
    return {_canon_date(m.group(0)) for m in _DATE_RE.finditer(text or "")}


def split_sentences(text: str) -> list[str]:
    """中英文通用的轻量分句（保留句子内容，用于定位引用所在句）."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text or "")]
    return [p for p in parts if p]


def split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """
    分句并给出每句在 *text* 中的字符区间（左闭右开，已裁掉首尾空白）.

    与 :func:`split_sentences` 用**同一条**句子边界规则，只是额外带回偏移量
    —— 命中句高亮要靠偏移量切片，而"按哪条规则分句"必须与支持度校验完全
    一致：两处分句口径一旦漂移，标出来的句子就不是跑分时的那几句了。
    """
    text = text or ""
    spans: list[tuple[str, int, int]] = []
    pos = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text):
        spans.extend(_span_of(text, pos, m.start()))
        pos = m.end()
    spans.extend(_span_of(text, pos, len(text)))
    return spans


def _span_of(text: str, start: int, end: int) -> list[tuple[str, int, int]]:
    """裁掉区间首尾空白后产出 (句子, start, end)；裁空则不产出句子."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return []
    return [(text[start:end], start, end)]


def _strip_citations(text: str) -> str:
    """去掉引用标记本身 —— 否则 [Source 1] 里的 "1" 会被当成数字比对."""
    return _CITATION_RE.sub(" ", text or "")


def strip_structural_markers(text: str) -> str:
    """
    剥掉行首的 Markdown 排版标记（列表序号 / 标题井号 / 引用块 / 括注序号）.

    引用校验比的是"模型陈述的事实 vs 原文"，排版标记既不属于前者也不属于
    后者 —— 留在文本里只会制造假阳性（见 _STRUCTURAL_MARKER_RE 的说明）。
    只删标记本身，标记后面的正文原样保留。
    """
    return _STRUCTURAL_MARKER_RE.sub("", text or "")


def prepare_for_checks(text: str) -> str:
    """
    校验前的预处理：剥掉排版标记.

    ⚠️ **不能在这里去掉 [Source N]** —— 引用标记正是本模块要验证的对象，
    提前删掉会让 total 直接归零（所有引用都"没出现过"）。引用标记的剥离
    只发生在**逐句**校验时（见 verify_citations 里的 plain_sentence）。
    """
    return strip_structural_markers(text)


# ── 数字 / 日期的"校验视图" ─────────────────────────────────────────────────
#
# 三类**不该参与比对**的数字，在抽取前统一屏蔽（句子侧与原文侧对称处理，
# 保证比较的是同一个口径）：
#
#   1. 溯源元数据 —— 《书名》里的年份、"第 1 页 / 第 83-105 行 / 第 3 章"
#      这类位置信息是模型复述的出处，不是事实陈述；
#   2. 句首枚举序号 —— "1. / 2) / 三、"是排版（见 _SENTENCE_MARKER_RE）；
#   3. 日期成分 —— "2024年1月" 里的 2024 与 1 由**日期校验**负责，
#      数字校验再查一遍是双重标准，且原文换种写法就制造假阳性。

def _provenance_mask(text: str) -> str:
    """剥掉书名号与"第 N 页/行/章/季度"等溯源位置表达."""
    text = _BOOK_TITLE_RE.sub(" ", text or "")
    return _LOCATION_REF_RE.sub(" ", text)


def _enumeration_mask(text: str) -> str:
    """剥掉句首与句中的枚举序号（"1. / 2) / （3）/ 三、"—— 排版，不是事实）."""
    text = _SENTENCE_MARKER_RE.sub("", text or "")
    return _INLINE_ENUM_RE.sub("", text)


def _mask_dates(text: str) -> str:
    """把日期表达整体屏蔽 —— 日期由日期校验单独负责."""
    return _DATE_RE.sub(" ", text or "")


def _numbers_for_check(text: str) -> set[str]:
    """校验用数字视图：屏蔽溯源表达与日期成分后再抽数字."""
    return extract_numbers(_mask_dates(_provenance_mask(text)))


def _dates_for_check(text: str) -> set[str]:
    """校验用日期视图：屏蔽溯源表达（书名里的年份不参与比对）."""
    return extract_dates(_provenance_mask(text))


# ── 命中句回标（"答案实际用了哪几句"）────────────────────────────────────────
#
# 引用卡片此前只给整块切片的行范围（如"第 83-105 行"）与整段原文：用户看到
# 23 行原文，却不知道答案**用的是哪几句**，只能自己把整段读完再猜。切片是
# 检索的单位，不是引用的单位 —— 引用应该落到"句"。
#
# 这里把引用从"块级"收紧到"句级"：对每条成立的引用，把被引原文切成句子，
# 逐句与答案句做内容词重合度比对，回标真正命中的句子（连同它自己的行号），
# 前端据此高亮、并在收起态只展示这几句。
#
# 仍是纯确定性比对（不调 LLM），阈值同样偏保守 —— 宁可只标最贴切的一句，
# 也不把整段涂黄：满屏高亮等于没有高亮。
_EVIDENCE_FALLBACK_RATIO = 0.18


@dataclass(frozen=True)
class EvidenceSpan:
    """答案实际依据的**那一句原文**（含在被引片段中的字符区间与行号）."""

    start: int                       # 在展示正文里的起始字符偏移（含）
    end: int                         # 结束字符偏移（不含）
    text: str                        # 该句原文 —— 偏移对不上时前端按文本兜底定位
    ratio: float                     # 与答案句的内容词重合率 ∈ [0, 1]
    line_start: int | None = None    # 该句自身的行号（1-based，闭区间）
    line_end: int | None = None

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "ratio": round(self.ratio, 4),
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


def display_text(source: dict) -> str:
    """
    取引用卡片**实际展示给用户的正文** —— 命中句偏移量必须建立在这个字符串上.

    前端（``lib/api/normalize.ts`` 的 pick 顺序）决定卡片展示哪个字段，命中句
    高亮靠"字符偏移 + 原句文本"双保险定位：偏移对得上就直接切，对不上退化为
    按文本查找。所以这里必须取与前端同一个口径的字符串，否则高亮会整体错位
    —— 错位的高亮比没有高亮更糟，它会让人以为原文就是这么写的。

    优先 ``text_snippet``：它是本项目 SSE 里真正下发的展示字段（见
    ``master_graph._retrieve_node``）。其余键是其它分支 / 历史数据的兼容。
    """
    for key in ("text_snippet", "chunk_text", "chunkText", "content", "snippet", "text"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _line_offset(text: str, offset: int, base: int | None, cap: int | None) -> int | None:
    """
    字符偏移 → 文档行号.

    片段正文与原文同用 ``\\n`` 分行、且 ``line_start`` 就是片段首行的行号
    （见 chunker 由 char_start 换算行号），因此"片段内第几个换行"就是这个
    偏移相对片段首行的行偏移。越界时夹到片段行区间内 —— 宁可少报一行，
    也不给出一个原文里不存在的行号。
    """
    if base is None:
        return None
    line = base + text.count("\n", 0, max(0, min(offset, len(text))))
    if cap is not None:
        line = min(line, cap)
    return max(line, base)


def find_evidence_spans(
    answer_sentence: str,
    source: dict,
    *,
    min_ratio: float = 0.34,
    max_sentences: int = 3,
) -> tuple[EvidenceSpan, ...]:
    """
    在被引原文里找出**答案句实际依据的那几句**.

    判定口径与"原文支持该结论"同源（都是内容词覆盖率），但方向相反：那里是
    "整条来源 → 答案句"，回答"这条引用靠不靠谱"；这里是"来源里的每一句 →
    答案句"，回答"具体是哪一句"。正因同源，标出来的句子与通过校验的原因
    必然自洽 —— 不会出现"校验说支持、高亮却落在别的句子上"。

    取舍（与整个模块一致：宁可少标，不可错标）：
      * 只有重合率 >= ``min_ratio`` 的句子才算命中；
      * 一句都没达标时，若最贴切的一句仍 >= 0.18 就标出来（否则用户又被推回
        读整段，等于没做这件事）；再低则整条不标，宁可不标；
      * 最多标 ``max_sentences`` 句，且按原文顺序输出（前端按顺序渲染）。
    """
    text = display_text(source)
    if not text:
        return ()
    sent_tokens = set(tokenize(_strip_citations(answer_sentence)))
    if not sent_tokens:
        return ()

    hits: list[tuple[float, int, int, str]] = []
    for sentence, start, end in split_sentences_with_spans(text):
        tokens = set(tokenize(sentence))
        if not tokens:
            continue
        hits.append((len(sent_tokens & tokens) / len(sent_tokens), start, end, sentence))
    if not hits:
        return ()

    picked = [h for h in hits if h[0] >= min_ratio]
    if not picked:
        best = max(hits, key=lambda h: h[0])
        if best[0] < _EVIDENCE_FALLBACK_RATIO:
            return ()
        picked = [best]

    if len(picked) > max_sentences:
        picked = sorted(picked, key=lambda h: -h[0])[:max_sentences]
    picked.sort(key=lambda h: h[1])          # 按原文顺序输出

    ls = source.get("line_start")
    le = source.get("line_end")
    base = ls if isinstance(ls, int) else None
    cap = le if isinstance(le, int) else None

    return tuple(
        EvidenceSpan(
            start=start,
            end=end,
            text=sentence,
            ratio=ratio,
            line_start=_line_offset(text, start, base, cap),
            line_end=_line_offset(text, end, base, cap),
        )
        for ratio, start, end, sentence in picked
    )


# ── 结论结构 ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CitationCheck:
    """单条引用的五项校验结果."""

    index: int                       # 1-based source 编号
    sentence: str                    # 引用所在的句子（截断展示）
    citation_exists: bool            # 引用存在？
    position_correct: bool           # 引用位置正确？
    supported: bool                  # 原文支持该结论？
    numbers_consistent: bool         # 数字是否一致？
    dates_consistent: bool           # 日期是否一致？
    # 诊断信息
    support_ratio: float = 0.0       # 句子内容词在被引原文中的覆盖率
    best_source: int | None = None   # 若位置可疑：更像出自哪一条
    missing_numbers: tuple[str, ...] = ()
    missing_dates: tuple[str, ...] = ()
    location: str | None = None      # 一句话位置（《x.pdf》第 3 页，第 12-28 行）
    # 命中句：答案实际依据的那几句原文（含偏移与行号）—— 引用卡片据此高亮，
    # 让"引用"从"整块切片"收紧到"具体几句"。
    evidence: tuple[EvidenceSpan, ...] = ()
    # 被引来源是否含有**可比对的证据文本**（能否真的拿它跟句子做内容词比对）。
    #
    # False = 来源拿不到任何 token/片段（不透明来源：纯图片既无 OCR 也无 vision、
    # 或只剩元数据）—— 此时 supported=False 的语义是**"无从判断"**，而非
    # "确证无依据"。前端/审计据此把这类条目排除出"无依据句"显式标注，
    # 避免把中文句、图片块(vision)、表格块来源的句子误标（宁可少标，不可错标）。
    evidence_available: bool = True

    @property
    def passed(self) -> bool:
        """五项全过才算通过（引用不存在时只判存在性）."""
        if not self.citation_exists:
            return False
        return (
            self.position_correct
            and self.supported
            and self.numbers_consistent
            and self.dates_consistent
        )

    def failed_checks(self) -> list[str]:
        checks = [
            ("citation_exists", self.citation_exists),
            ("position_correct", self.position_correct),
            ("supported", self.supported),
            ("numbers_consistent", self.numbers_consistent),
            ("dates_consistent", self.dates_consistent),
        ]
        return [name for name, ok in checks if not ok]

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "sentence": self.sentence,
            "citation_exists": self.citation_exists,
            "position_correct": self.position_correct,
            "supported": self.supported,
            "numbers_consistent": self.numbers_consistent,
            "dates_consistent": self.dates_consistent,
            "passed": self.passed,
            "failed_checks": self.failed_checks(),
            "support_ratio": round(self.support_ratio, 4),
            "best_source": self.best_source,
            "missing_numbers": list(self.missing_numbers),
            "missing_dates": list(self.missing_dates),
            "location": self.location,
            "evidence": [e.as_dict() for e in self.evidence],
            "evidence_available": self.evidence_available,
        }


@dataclass(frozen=True)
class VerificationReport:
    """整段答案的引用校验报告."""

    verdicts: tuple[CitationCheck, ...] = ()
    overall: str = "no_citations"     # verified | partial | unsupported | no_citations
    total: int = 0
    passed_count: int = 0
    unsupported_indices: tuple[int, ...] = ()
    hallucinated_indices: tuple[int, ...] = ()   # 引用了不存在的 source
    misattributed_indices: tuple[int, ...] = ()
    number_mismatch_indices: tuple[int, ...] = ()
    date_mismatch_indices: tuple[int, ...] = ()
    # 净化后的答案（越界引用移除；可选地移除不被支持的引用标记）
    clean_text: str = ""
    # 是否在答案末尾追加了"校验说明"脚注
    annotated: bool = False

    @property
    def has_problems(self) -> bool:
        return bool(
            self.unsupported_indices
            or self.hallucinated_indices
            or self.misattributed_indices
            or self.number_mismatch_indices
            or self.date_mismatch_indices
        )

    def as_audit(self) -> dict:
        """给 SSE / 审计 / Bad Case 回流用的紧凑结构."""
        return {
            "overall": self.overall,
            "total": self.total,
            "passed": self.passed_count,
            "unsupported": list(self.unsupported_indices),
            "hallucinated": list(self.hallucinated_indices),
            "misattributed": list(self.misattributed_indices),
            "number_mismatch": list(self.number_mismatch_indices),
            "date_mismatch": list(self.date_mismatch_indices),
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


# ── 核心校验 ─────────────────────────────────────────────────────────────────

def _source_text(source: dict) -> str:
    """
    取一条 source 的**可核验证据全文**.

    为什么不能直接用 ``text_snippet``
    ──────────────────────────────
    ``text_snippet`` 是给前端展示的**预览**，被截断到几百字。校验若拿它当
    原文，会产生两类系统性误判：

    1. 数字 / 日期落在截断之后 → 明明原文有，却判"与原文不一致"；
    2. 支持度分母变小 → 正确引用被判"不被原文支持"。

    两者都会**删掉本来正确的引用**，而"该引的没引上"比"多引了一条"更伤
    信任。因此这里按优先级拼出尽可能完整的证据文本：

        text            调用方提供的完整正文（最可信）
        chunk_text / content / snippet
        text_snippet    兜底：只有预览时就用预览
        + vision        检索期 Vision 看图结论（图片专属证据）
        + image_caption 入库期图片语义描述

    后两者对**图片来源**是决定性的：图片进上下文时的正文往往是"OCR 文本 +
    视觉分析"，模型据此作答；校验若只看 OCR 文本，所有基于"看图"得出的
    结论都会被误判为不支持 —— 图片引用会被成片删掉。
    """
    parts: list[str] = []
    for key in ("text", "chunk_text", "content", "snippet", "text_snippet"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
            break
    for key in ("vision", "image_caption"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def _support_ratio(sentence: str, source_text: str) -> float:
    """句子内容词在被引原文中的覆盖率 ∈ [0, 1]."""
    sent_tokens = tokenize(_strip_citations(sentence))
    if not sent_tokens:
        return 1.0                     # 无从判断 → 不因支持度判负（宁可漏判）
    src_tokens = set(tokenize(source_text))
    if not src_tokens:
        return 0.0
    hit = sum(1 for t in sent_tokens if t in src_tokens)
    return hit / len(sent_tokens)


def verify_citations(
    answer: str,
    sources: list[dict],
    *,
    min_support: float = 0.30,
    misattribution_margin: float = 0.25,
    strip_unsupported: bool = True,
    annotate: bool = True,
    evidence_enabled: bool = True,
    evidence_min_ratio: float = 0.34,
    evidence_max_sentences: int = 3,
) -> VerificationReport:
    """
    对 *answer* 中出现的每一条引用做五项校验.

    Args:
        answer:                LLM 生成的答案全文（已拼接）.
        sources:               本轮实际喂给 LLM 的 sources（与 [Source N] 对齐，
                               编号从 1 开始）.
        min_support:           "原文支持该结论"的覆盖率阈值（保守：默认 0.30）.
        misattribution_margin: 另一条 source 的支持度高出多少才判"引用位置错误"
                               （默认 0.25，避免把相近来源误判为错引）.
        strip_unsupported:     是否移除"不被原文支持/数字日期不一致"的引用标记.
        annotate:              是否在答案末尾追加一行校验说明.
        evidence_enabled:      是否回标"命中句"（引用卡片的句级高亮）.
        evidence_min_ratio:    命中句判定阈值（见 find_evidence_spans）.
        evidence_max_sentences: 每条引用最多回标几句命中句.

    Returns:
        VerificationReport —— clean_text 是净化后的答案.
    """
    if not answer:
        return VerificationReport(clean_text=answer or "", overall="no_citations")

    sources = sources or []
    # 先剥排版标记（列表序号/标题井号）再分句：否则 "1. **反幻觉机制**：…" 里的
    # 列表序号会被当成正文数字，制造成片的"数字与原文不一致"假阳性。
    sentences = split_sentences(prepare_for_checks(answer))

    # 为每条 source 预取正文，避免重复拼接
    src_texts = [_source_text(s) for s in sources]

    verdicts: list[CitationCheck] = []
    unsupported: list[int] = []
    hallucinated: list[int] = []
    misattributed: list[int] = []
    num_mismatch: list[int] = []
    date_mismatch: list[int] = []

    # ── 逐句扫描引用 ─────────────────────────────────────────────────────────
    for sentence in sentences:
        matches = list(_CITATION_RE.finditer(sentence))
        if not matches:
            continue

        plain_sentence = _strip_citations(sentence)
        # 校验视图：枚举序号剥掉后再抽数字/日期（展示仍用原句）
        check_sentence = _enumeration_mask(plain_sentence)
        sent_numbers = _numbers_for_check(check_sentence)
        sent_dates = _dates_for_check(check_sentence)

        for m in matches:
            n = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else n
            # 区间引用 [Source 2-3]：逐条验证，合并为一条结论（取最严）
            group = list(range(n, end + 1))

            exists = bool(group) and all(1 <= g <= len(sources) for g in group)
            if not exists:
                bad = [g for g in group if not (1 <= g <= len(sources))]
                hallucinated.extend(bad)
                verdicts.append(CitationCheck(
                    index=n,
                    sentence=plain_sentence[:120],
                    citation_exists=False,
                    position_correct=False,
                    supported=False,
                    numbers_consistent=False,
                    dates_consistent=False,
                    missing_numbers=tuple(sorted(sent_numbers)),
                    missing_dates=tuple(sorted(sent_dates)),
                ))
                continue

            # 合并引用区间内所有 source 的正文（区间引用本就表示"这几条共同支持"）
            cited_text = "\n".join(src_texts[g - 1] for g in group)
            ratio = _support_ratio(plain_sentence, cited_text)
            # 来源是否含可供比对的证据文本：拿不到任何 token 时，"未支持"是
            # **无从判断**而非"确证无依据" —— 前端据此把该句排除出"无依据"标注。
            evidence_available = bool(tokenize(cited_text))

            # ── 引用位置正确？── 若别条 source 明显更贴合，则位置可疑 ──────
            best_other = None
            best_other_ratio = 0.0
            for idx, text in enumerate(src_texts, start=1):
                if idx in group or not text:
                    continue
                r = _support_ratio(plain_sentence, text)
                if r > best_other_ratio:
                    best_other_ratio = r
                    best_other = idx
            position_ok = not (
                best_other is not None
                and best_other_ratio >= ratio + misattribution_margin
            )
            if not position_ok:
                misattributed.append(n)

            # ── 原文支持该结论？──
            supported = ratio >= min_support

            # ── 数字是否一致？── 句中数字必须在被引原文里原样出现 ──────────
            # 句子侧与原文侧用同一口径的"校验视图"（屏蔽溯源表达 + 日期成分），
            # 否则原文里"2024年营收"的 2024 被日期屏蔽、句子里却没有，口径就歪了。
            cited_numbers = _numbers_for_check(cited_text)
            missing_numbers = tuple(
                sorted(x for x in sent_numbers if x not in cited_numbers)
            )
            numbers_ok = not missing_numbers

            # ── 日期是否一致？──
            cited_dates = _dates_for_check(cited_text)
            missing_dates = tuple(
                sorted(x for x in sent_dates if x not in cited_dates)
            )
            dates_ok = not missing_dates

            check = CitationCheck(
                index=n,
                sentence=plain_sentence[:120],
                citation_exists=True,
                position_correct=position_ok,
                supported=supported,
                numbers_consistent=numbers_ok,
                dates_consistent=dates_ok,
                support_ratio=round(ratio, 4),
                best_source=best_other if not position_ok else None,
                missing_numbers=missing_numbers,
                missing_dates=missing_dates,
                location=_location_of(sources[group[0] - 1]),
                evidence=_evidence_of(
                    plain_sentence,
                    sources,
                    group,
                    enabled=evidence_enabled,
                    min_ratio=evidence_min_ratio,
                    max_sentences=evidence_max_sentences,
                ),
                evidence_available=evidence_available,
            )
            verdicts.append(check)

            if not check.passed:
                unsupported.append(n)
            if not numbers_ok:
                num_mismatch.append(n)
            if not dates_ok:
                date_mismatch.append(n)

    total = len(verdicts)
    passed_count = sum(1 for v in verdicts if v.passed)

    if total == 0:
        overall = "no_citations"
    elif passed_count == total:
        overall = "verified"
    elif passed_count == 0:
        overall = "unsupported"
    else:
        overall = "partial"

    # ── 净化：移除"有问题"的引用标记 ─────────────────────────────────────────
    bad_indices = set(unsupported) | set(hallucinated)
    clean_text = answer
    if strip_unsupported and bad_indices:
        def _drop(match: re.Match[str]) -> str:
            n = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else n
            indices = list(range(n, end + 1))
            # n > M 的畸形区间（如 [Source 3-1]）解析不出任何下标：无从判断，
            # 按"宁可漏判，不可错杀"原样保留（回退到改动前的行为），否则只要
            # 别处存在坏下标，这条本身无坏下标的引用就会被顺带删掉。
            if not indices:
                return match.group(0)
            keep = [i for i in indices if i not in bad_indices]
            if not keep:
                return ""
            if len(keep) == len(indices):
                return match.group(0)
            return " ".join(f"[Source {i}]" for i in keep)

        clean_text = _CITATION_RE.sub(_drop, clean_text)
        clean_text = re.sub(r"[ \t]{2,}", " ", clean_text)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

    annotated = False
    if annotate and total > 0 and overall in ("partial", "unsupported"):
        notes: list[str] = []
        if hallucinated:
            notes.append(f"引用了不存在的来源 {sorted(set(hallucinated))}")
        if misattributed:
            notes.append(f"引用位置存疑 {sorted(set(misattributed))}")
        if unsupported:
            notes.append(f"未被原文直接支持 {sorted(set(unsupported))}")
        if num_mismatch:
            notes.append(f"数字与原文不一致 {sorted(set(num_mismatch))}")
        if date_mismatch:
            notes.append(f"日期与原文不一致 {sorted(set(date_mismatch))}")
        if notes:
            clean_text = (
                clean_text
                + "\n\n> 引用校验：" + "；".join(notes)
                + "。相关引用标记已移除，请以原文为准。"
            )
            annotated = True

    if total:
        logger.info(
            "citation_verifier: overall=%s passed=%d/%d "
            "unsupported=%d hallucinated=%d misattributed=%d num=%d date=%d",
            overall, passed_count, total,
            len(set(unsupported)), len(set(hallucinated)),
            len(set(misattributed)), len(set(num_mismatch)), len(set(date_mismatch)),
        )

    return VerificationReport(
        verdicts=tuple(verdicts),
        overall=overall,
        total=total,
        passed_count=passed_count,
        unsupported_indices=tuple(sorted(set(unsupported))),
        hallucinated_indices=tuple(sorted(set(hallucinated))),
        misattributed_indices=tuple(sorted(set(misattributed))),
        number_mismatch_indices=tuple(sorted(set(num_mismatch))),
        date_mismatch_indices=tuple(sorted(set(date_mismatch))),
        clean_text=clean_text,
        annotated=annotated,
    )


def _evidence_of(
    sentence: str,
    sources: list[dict],
    group: list[int],
    *,
    enabled: bool,
    min_ratio: float,
    max_sentences: int,
) -> tuple[EvidenceSpan, ...]:
    """
    一条引用的命中句（区间引用锚定首条来源）.

    只在引用**成立**时调用 —— 编号不存在的引用没有可标的"原文"，给它标
    命中句等于替幻觉背书。区间引用 ``[Source 2-3]`` 取 ``group[0]``：与
    引用徽标、``location`` 用的是同一条，三处说法因此保持一致。
    """
    if not enabled or not group:
        return ()
    idx = group[0] - 1
    if not (0 <= idx < len(sources)):
        return ()
    return find_evidence_spans(
        sentence, sources[idx], min_ratio=min_ratio, max_sentences=max_sentences
    )


def _location_of(source: dict) -> str | None:
    """
    从 source 里取"一句话位置"（后端已算好则直接复用）.

    图片来源没有行号，用**文档内序号**定位（``第 2 张图`` / ``第 3 个表格``），
    与 ``RetrievedChunk.location_label()`` 保持同一套说法 —— 位置文案是
    "引用回溯"的落点，三处各写一遍必然漂移。
    """
    loc = source.get("location")
    if isinstance(loc, str) and loc:
        return loc
    filename = source.get("filename") or source.get("document_name")
    if not filename:
        return None
    page = source.get("page_number") or source.get("page")
    ls = source.get("line_start")
    le = source.get("line_end")
    parts = [f"《{filename}》"]
    if page:
        parts.append(f"第 {page} 页")
    if ls:
        span = str(ls) if (le is None or le == ls) else f"{ls}-{le}"
        parts.append(f"第 {span} 行")
    else:
        # 无行号 → 可能是图片：退化为"第几张图 / 第几个表格"
        is_table = (source.get("content_type") == "table") or (
            (source.get("image_type") or "") == "table"
        )
        position = source.get("position")
        if position:
            parts.append(f"第 {position} " + ("个表格" if is_table else "张图"))
    return "，".join(parts)


__all__ = [
    "CitationCheck",
    "EvidenceSpan",
    "VerificationReport",
    "display_text",
    "extract_dates",
    "extract_numbers",
    "find_evidence_spans",
    "prepare_for_checks",
    "split_sentences",
    "split_sentences_with_spans",
    "strip_structural_markers",
    "verify_citations",
]
