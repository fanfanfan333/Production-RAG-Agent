"""
Metadata 系统 —— 自动元数据抽取、可过滤检索、溯源展示.

为什么 1000+ 文档时元数据是**准确率问题**而不是"锦上添花"
──────────────────────────────────────────────────────────
单库几十份文档时，向量召回 + 精排足够把正确段落排进 Top5。文档量到千份量级时
会出现两种系统性退化，都不表现为报错：

  1. **候选池稀释**：同一个词（"营收""责任""流程"）在几十份不同年份、不同部门
     的文档里都出现。粗排取 Top50 时，真正那一段可能排在第 60 位 —— 根本没进
     精排。用户看到的答案"用了别的年份的数据"，而链路每一环都"正常"。
  2. **近似文档互相冒充**：同一制度的 v1/v2、同一报告的初稿/终稿，语义几乎一致。
     向量区分不了它们，精排也区分不了 —— 只有元数据（版本/日期/部门）能。

所以本模块的产出有两个用途，缺一不可：

    (a) **检索前置过滤**（MetadataFilter）—— 把候选池收敛到正确的子集，
        在 ANN 之前就缩小范围，而不是召回后再剔；
    (b) **引用溯源**（outline / title / doc_type）—— 引用卡片说清"出自哪一份
        文件的哪一节"，用户才能自己核对，这是反幻觉的最后一道人工闸。

抽取策略：**只做确定性抽取，不调 LLM**。理由与 evidence_gate / citation_verifier
一致 —— 元数据会参与过滤，一旦 LLM 编出一个不存在的年份，过滤就会静默地把正确
文档排除掉。确定性抽取可能漏（漏了只是少一个过滤维度，安全），LLM 抽取会错
（错了是静默错杀）。宁漏不错。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 词典（刻意小、可读、可扩展）
# ═══════════════════════════════════════════════════════════════════════════════

# 文档类型词表：命中即归类。顺序有意义 —— 先命中更具体的类型。
# 值同时用作 Qdrant 的 doc_type 过滤值，因此保持简短、稳定、全中文。
_DOC_TYPE_LEXICON: list[tuple[str, tuple[str, ...]]] = [
    ("财报", ("财务报告", "年报", "半年报", "季报", "决算", "预算报告", "审计报告",
              "资产负债表", "利润表", "现金流量表")),
    ("合同", ("合同", "协议", "备忘录", "契约", "terms", "contract", "agreement")),
    ("制度", ("制度", "管理办法", "管理规定", "规程", "规范", "条例", "政策",
              "细则", "章程", "准则")),
    ("手册", ("手册", "指南", "操作说明", "用户手册", "说明书", "FAQ", "教程")),
    ("论文", ("论文", "学位", "期刊", "文献", "研究", "综述", "thesis", "paper")),
    ("方案", ("方案", "规划", "计划书", "设计稿", "提案", "可行性")),
    ("报告", ("报告", "汇报", "总结", "分析报告", "调研", "report")),
    ("纪要", ("纪要", "会议记录", "会议纪要", "minutes", "会议材料")),
    ("标准", ("标准", "国标", "企标", "GB/T", "ISO", "技术要求")),
    ("标书", ("标书", "招标", "投标", "询价", "采购文件")),
    ("表单", ("模板", "表单", "清单", "台账", "明细表")),
]

# 部门 / 业务标签词表（从文件名或路径派生，用于"只看本部门材料"这类过滤）
_BUSINESS_TAG_LEXICON: tuple[str, ...] = (
    "财务", "会计", "审计", "税务", "人力", "人事", "行政", "法务", "合规",
    "采购", "供应链", "生产", "制造", "研发", "技术", "质量", "安全", "环保",
    "销售", "市场", "客服", "运营", "战略", "投资", "风控", "信息", "IT",
    "法务部", "研发部", "市场部", "财务部", "人力资源",
)

# 英文停用词 + 中文虚词（用于关键词抽取时过滤噪声）。
# 中文没有分词器，这里用"2-4 字滑窗 + 停用字过滤"来近似，够用且零依赖。
_STOP_CHARS = set(
    "的了和与及或在是为对从到把被将于其之也都很更最可要会能就还只不没"
    "这个那个我们你们他们它们以及等等因为所以但是如果那么这样那样什么怎么"
    "一些一个一种进行可以通过按照根据关于对于由于目的方面情况问题"
)
_EN_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "will", "shall", "have", "has", "had", "not", "but", "you", "your", "our",
    "can", "may", "must", "should", "any", "all", "each", "per", "into", "than",
}

_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(20\d{2})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?"), "ymd"),
    (re.compile(r"(20\d{2})\s*[-/年]\s*(\d{1,2})\s*月?"), "ym"),
    (re.compile(r"(?:^|[^\d])(20\d{2})(?:[^\d]|$)"), "y"),
]

# 中文公文文号：财字〔2024〕7号 / 中办发[2024]12号
_DOC_NUMBER_RE = re.compile(
    r"[\u4e00-\u9fff]{1,6}[字发〔\[（(]\s*(20\d{2})\s*[〕\]）)]\s*第?\s*(\d{1,4})\s*号"
)

_AUTHOR_RE = re.compile(
    r"(?:编制|编写|拟制|制定|起草|发布|作者|撰写|审批|审核)[单位人部科处室：:\s]*"
    r"([\u4e00-\u9fffA-Za-z0-9（）()·\s]{2,40})"
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_.]{2,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DocumentMetadata:
    """
    一份文档的结构化元数据.

    字段命名与 Qdrant payload / PG 列保持一致（``to_payload`` 直接产出可过滤
    的扁平字典），避免"库里叫一个名、payload 里叫另一个名"这种只有出 bug 时
    才会被发现的错位。
    """

    title: str = ""
    author: str | None = None
    doc_date: str | None = None        # ISO 8601 日期（YYYY-MM-DD），缺日/月时补 01
    doc_year: int | None = None
    doc_type: str | None = None
    doc_number: str | None = None
    language: str = "zh"               # zh | en | mixed
    keywords: list[str] = field(default_factory=list)
    business_tags: list[str] = field(default_factory=list)
    source_dir: str | None = None
    outline: list[dict] = field(default_factory=list)
    section_count: int = 0
    extra: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        """
        → Qdrant 可过滤 payload（扁平、短、只放建了索引的字段）.

        刻意**不放** outline / keywords 全文：它们会随每个 chunk 复制一遍
        （1000 份文档 × 每份 200 chunk = 20 万份副本），而 outline 只在文档级
        详情里用得到 —— 那是 PG 的职责。payload 只放"过滤要用"的字段。
        """
        return {
            "title": self.title[:200] if self.title else None,
            "doc_type": self.doc_type,
            "doc_year": self.doc_year,
            "language": self.language,
            "doc_number": self.doc_number,
            "author": self.author,
            # keywords 是数组，Qdrant keyword 索引支持 match any
            "keywords": self.keywords[:12],
            "business_tags": self.business_tags[:8],
        }

    def to_db_dict(self) -> dict:
        """→ PG JSONB（完整信息，含 outline；文档级一份，不随 chunk 复制）."""
        return {
            "title": self.title,
            "author": self.author,
            "doc_date": self.doc_date,
            "doc_year": self.doc_year,
            "doc_type": self.doc_type,
            "doc_number": self.doc_number,
            "language": self.language,
            "keywords": list(self.keywords),
            "business_tags": list(self.business_tags),
            "source_dir": self.source_dir,
            "section_count": self.section_count,
            "outline": list(self.outline),
            **({"extra": self.extra} if self.extra else {}),
        }


@dataclass
class MetadataFilter:
    """
    检索用的元数据过滤条件.

    ``must`` 语义（全部条件 AND）：调用方给出的每一类条件都必须满足。
    **空条件不参与过滤** —— 这一点很重要：一个没有指定年份的提问不应该被
    默认限制在某一年，否则"搜不到"会被误当成"知识库没有"。
    """

    doc_types: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    exclude_document_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any((
            self.doc_types, self.years, self.languages,
            self.tags, self.document_ids, self.exclude_document_ids,
        ))

    @classmethod
    def from_dict(cls, raw: dict | None) -> "MetadataFilter":
        """
        从 API 入参构造（容错：未知键忽略、标量自动包成列表）.

        不抛异常是刻意的 —— 元数据过滤是**收窄**检索范围的优化，一个写错的
        过滤器参数如果让整个提问 500，用户会以为系统坏了；忽略它顶多是没过滤。
        """
        if not isinstance(raw, dict):
            return cls()

        def _as_list(value) -> list:
            if value is None:
                return []
            if isinstance(value, (list, tuple, set)):
                return [v for v in value if v not in (None, "")]
            return [value]

        def _as_strs(value) -> list[str]:
            return [str(v).strip() for v in _as_list(value) if str(v).strip()]

        years: list[int] = []
        for v in _as_list(raw.get("years") or raw.get("year")):
            try:
                years.append(int(v))
            except (TypeError, ValueError):
                continue

        return cls(
            doc_types=_as_strs(raw.get("doc_types") or raw.get("doc_type")),
            years=years,
            languages=_as_strs(raw.get("languages") or raw.get("language")),
            tags=_as_strs(raw.get("tags") or raw.get("tag")),
            document_ids=_as_strs(raw.get("document_ids") or raw.get("document_id")),
            exclude_document_ids=_as_strs(raw.get("exclude_document_ids")),
        )


def matches_payload(flt: "MetadataFilter | None", payload: dict) -> bool:
    """
    payload 侧判定：这条向量是否满足元数据条件（纵深防御）.

    为什么 Qdrant 已经在 ANN 之前过滤了还要再判一次 —— **老向量**。
    ``doc_type`` / ``doc_year`` 这类 payload 是本次升级才开始写入的，升级前入库
    的向量根本没有这些字段。两道闸对"字段缺失"的处理刻意不同：

      前置（Qdrant FieldCondition，等值语义）—— 字段缺失即不匹配，**strict**。
        代价是升级前的老向量会被年份/类型过滤排除。这是有意的：用户明确要
        "2024 年的报告"时，宁可少给，也不要混入一份我们无法确认年份的文档
        —— 后者会直接变成一条"引用合规、数据错误"的幻觉。文档重新入库即回填。

      后置（本函数）—— 字段缺失一律**放行**（fail-open），只在"字段存在且
        与条件冲突"时否决。因为它是最内层的兜底，唯一职责是拦住"前置漏掉的
        明确冲突项"；在这里 fail-closed 只会把前置已经判过的东西再杀一遍，
        却增加了把有效证据误杀掉的风险。

    这样组合的净效果：明确冲突的证据必被拦下（两道闸都拦），信息不足的证据由
    前置按用户意图决定，后置不叠加不确定性。
    """
    if flt is None or flt.is_empty():
        return True
    payload = payload or {}

    if flt.doc_types:
        value = payload.get("doc_type")
        if value is not None and str(value) not in flt.doc_types:
            return False

    if flt.years:
        value = payload.get("doc_year")
        if value is not None:
            try:
                if int(value) not in flt.years:
                    return False
            except (TypeError, ValueError):
                return False

    if flt.languages:
        value = payload.get("language")
        if value is not None and str(value) not in flt.languages:
            return False

    if flt.tags:
        tags = payload.get("business_tags") or payload.get("keywords") or []
        if tags:
            wanted = {t.lower() for t in flt.tags}
            have = {str(t).lower() for t in tags}
            if not (wanted & have):
                return False

    doc_id = str(payload.get("document_id") or "")
    if flt.document_ids and doc_id and doc_id not in flt.document_ids:
        return False
    if flt.exclude_document_ids and doc_id in flt.exclude_document_ids:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 抽取实现
# ═══════════════════════════════════════════════════════════════════════════════


def _detect_language(text: str) -> str:
    """按 CJK/ASCII 字母占比判语言（zh | en | mixed）."""
    sample = text[:4000]
    if not sample.strip():
        return "zh"
    cjk = len(_CJK_RE.findall(sample))
    latin = len(_ASCII_WORD_RE.findall(sample))
    total = cjk + latin
    if total == 0:
        return "zh"
    ratio = cjk / total
    if ratio >= 0.75:
        return "zh"
    if ratio <= 0.15:
        return "en"
    return "mixed"


def _split_filename(filename: str) -> tuple[str, str, str]:
    """把 ``a/b/财务部/2024年报.pdf`` 拆成 (dir, stem, ext)."""
    norm = (filename or "").replace("\\", "/")
    dirname, _, base = norm.rpartition("/")
    stem, _, ext = base.rpartition(".")
    if not stem:                # 没有扩展名
        stem, ext = base, ""
    return dirname, stem, ext


def _from_filename(stem: str, dirname: str) -> tuple[dict, list[str]]:
    """
    从文件名/目录派生元数据.

    企业知识库的文件名本身就是高密度元数据载体（"2024年度财务报告_财务部.pdf"
    含年份 + 类型 + 部门），比正文抽取更可靠 —— 正文里的年份可能是引用的历史
    数据，文件名里的年份才是这份文档的归属年份。
    """
    out: dict = {}
    tags: list[str] = []

    haystack = f"{dirname}/{stem}"

    # 类型
    lowered = haystack.lower()
    for label, words in _DOC_TYPE_LEXICON:
        if any(w.lower() in lowered for w in words):
            out["doc_type"] = label
            break

    # 年份（文件名里的 4 位年份；多个时取最大，通常是最新版本年份）
    years = [int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", stem)]
    if years:
        out["doc_year"] = max(years)

    # 业务/部门标签
    for tag in _BUSINESS_TAG_LEXICON:
        if tag in haystack:
            # 归一：'财务部' 与 '财务' 只保留更短的那个，避免重复标签
            base = tag.rstrip("部处科室组")
            if base not in tags:
                tags.append(base)
    if dirname:
        # 目录名也当标签（业务分类常常体现在目录而不是文件名）
        for part in [p for p in dirname.split("/") if p.strip()]:
            if len(part) <= 12 and part not in tags:
                tags.append(part)

    return out, tags[:8]


def _extract_date(text: str, filename_stem: str) -> tuple[str | None, int | None]:
    """从正文首部 + 文件名里抽日期（返回 (ISO 日期, 年份)）."""
    head = text[:3000]
    for pattern, kind in _DATE_PATTERNS:
        match = pattern.search(head)
        if match:
            groups = match.groups()
            year = int(groups[0])
            month = int(groups[1]) if kind in ("ymd", "ym") and len(groups) > 1 else 1
            day = int(groups[2]) if kind == "ymd" and len(groups) > 2 else 1
            if not (1 <= month <= 12):
                month = 1
            if not (1 <= day <= 31):
                day = 1
            return f"{year:04d}-{month:02d}-{day:02d}", year

    # 文件名兜底（只取年份，不编造月份/日期）
    years = [int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", filename_stem)]
    if years:
        return None, max(years)
    return None, None


def _extract_title(text: str, outline: list[dict], stem: str) -> str:
    """
    标题优先级：第一个 Markdown H1 → 第一个非空短行 → 文件名主干.

    为什么优先 H1 而不是文件名：文件名叫 ``最终版(3).pdf`` 的场景极常见，
    而正文首行通常是真正的标题。
    """
    for item in outline:
        if item.get("level") == 1 and item.get("title"):
            return str(item["title"]).strip()[:200]

    for line in text.lstrip().split("\n")[:30]:
        candidate = line.strip().lstrip("#").strip()
        if 4 <= len(candidate) <= 120 and not candidate.startswith(("|", "```", "-", "*")):
            return candidate[:200]

    return (stem or "未命名文档")[:200]


def _extract_author(text: str) -> str | None:
    match = _AUTHOR_RE.search(text[:4000])
    if not match:
        return None
    value = match.group(1).strip().rstrip("：: ")
    value = re.split(r"[\s，,。；;]", value)[0].strip()
    return value[:60] or None


def _extract_keywords(
    text: str,
    outline: list[dict],
    title: str,
    max_keywords: int,
    min_len: int,
) -> list[str]:
    """
    关键词抽取（确定性、零依赖）.

    信号来源按可靠性排序，加权叠加：
      1. **章节标题**（权重 5）—— 标题是人写的小结，信息密度最高；
      2. **ASCII 词**（权重 3）—— 型号/编号/英文术语，精确匹配价值高；
      3. **中文 2-4 字滑窗 n-gram**（权重 1）—— 无分词器时的近似，
         靠"多字候选优先 + 覆盖率过滤"压制噪声。

    刻意不引入 jieba：多一个依赖、多一份词典、多一处版本风险，而关键词只是
    辅助过滤信号，不需要分词级别的精度。
    """
    from collections import Counter

    score: Counter = Counter()

    # 1) 章节标题
    for item in outline:
        t = str(item.get("title") or "")
        for term in _candidate_terms(t, min_len):
            score[term] += 5

    # 2) 标题本体
    for term in _candidate_terms(title, min_len):
        score[term] += 4

    # 3) ASCII 词
    for word in _ASCII_WORD_RE.findall(text[:20000]):
        w = word.lower()
        if w in _EN_STOPWORDS or len(w) < 3:
            continue
        score[w] += 3

    # 4) 中文 n-gram（只扫前 20k 字，控制成本；关键词不是全文统计）
    body = text[:20000]
    for n in (4, 3, 2):
        for i in range(0, max(0, len(body) - n + 1), 2):
            gram = body[i:i + n]
            if not all(_CJK_RE.match(c) for c in gram):
                continue
            if any(c in _STOP_CHARS for c in gram):
                continue
            score[gram] += 1

    # 过滤：长度达标 + 至少出现 2 次（单次出现的 n-gram 几乎都是噪声）
    ranked = [
        (term, s) for term, s in score.items()
        if len(term) >= min_len and (s >= 2 or len(term) >= 3)
    ]
    ranked.sort(key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))

    selected: list[str] = []
    seen_chars: set[str] = set()
    for term, _s in ranked:
        if term in selected:
            continue
        # 去重：已被选中的更短词包含在当前词里且长度接近 → 跳过（避免"财务"与"财务报告"并列）
        if any(term in ex or ex in term for ex in selected if abs(len(ex) - len(term)) <= 1):
            continue
        selected.append(term)
        seen_chars.update(term)
        if len(selected) >= max_keywords:
            break
    return selected


def _candidate_terms(text: str, min_len: int) -> list[str]:
    """从一行标题里切出候选词（按标点/空白切，再收 2-6 字中文片段）."""
    parts = re.split(r"[\s、，,。；;：:（）()\[\]【】\-—_/]+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < min_len:
            continue
        if len(p) <= 12:
            out.append(p)
            continue
        # 过长片段再滑窗（中文标题偶尔写成一整句）
        for n in (6, 4):
            for i in range(0, len(p) - n + 1, 2):
                gram = p[i:i + n]
                if all(_CJK_RE.match(c) for c in gram):
                    out.append(gram)
    return out


def extract_metadata(
    *,
    text: str,
    filename: str,
    outline: list[dict] | None = None,
    settings=None,
) -> DocumentMetadata:
    """
    抽取一份文档的元数据（纯确定性，不调任何模型）.

    Args:
        text:     文档正文（结构化解析后的 Markdown 优先）
        filename: 原始文件名（**可含目录前缀**，用于业务标签）
        outline:  结构树大纲（``StructuredDocument.outline()``），可为 None
        settings: Settings 实例；None 时从 app.config 取

    Returns:
        DocumentMetadata
    """
    if settings is None:
        from app.config import get_settings
        settings = get_settings()

    enabled = getattr(settings, "METADATA_ENABLED", True)
    max_keywords = int(getattr(settings, "METADATA_MAX_KEYWORDS", 12))
    min_len = int(getattr(settings, "METADATA_MIN_KEYWORD_LEN", 2))
    title_max = int(getattr(settings, "METADATA_TITLE_MAX_CHARS", 200))

    dirname, stem, _ext = _split_filename(filename)
    outline = outline or []

    meta = DocumentMetadata(
        source_dir=dirname or None,
        outline=outline[: int(getattr(settings, "METADATA_OUTLINE_MAX_NODES", 200))],
        section_count=len(outline),
        language=_detect_language(text),
    )

    if not enabled:
        meta.title = (stem or "未命名文档")[:title_max]
        return meta

    # ── 文件名派生（最可靠的一路）─────────────────────────────────────────────
    if getattr(settings, "METADATA_FROM_FILENAME", True):
        derived, tags = _from_filename(stem, dirname)
        meta.doc_type = derived.get("doc_type")
        meta.doc_year = derived.get("doc_year")
        meta.business_tags = tags

    # ── 正文派生 ──────────────────────────────────────────────────────────────
    meta.title = _extract_title(text, meta.outline, stem)[:title_max]
    date_iso, year = _extract_date(text, stem)
    meta.doc_date = date_iso
    if year is not None and meta.doc_year is None:
        meta.doc_year = year
    meta.author = _extract_author(text)

    number = _DOC_NUMBER_RE.search(text[:4000])
    if number:
        meta.doc_number = number.group(0).strip()[:60]
    else:
        # 文号常常只写在文件名里
        m = _DOC_NUMBER_RE.search(stem)
        if m:
            meta.doc_number = m.group(0).strip()[:60]

    meta.keywords = _extract_keywords(
        text, meta.outline, meta.title, max_keywords, min_len,
    )

    logger.debug(
        "metadata for '%s': type=%s year=%s lang=%s kw=%d sections=%d",
        filename, meta.doc_type, meta.doc_year, meta.language,
        len(meta.keywords), meta.section_count,
    )
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# 查询侧：把自然语言里的元数据线索转成过滤器
# ═══════════════════════════════════════════════════════════════════════════════

_YEAR_IN_QUERY_RE = re.compile(r"(?<!\d)(20\d{2})\s*年?")
_COMPARE_HINT_RE = re.compile(r"(对比|比较|相比|同比|环比|变化|趋势|历年|各年)")
_LATEST_HINT_RE = re.compile(r"(最新|最近|现行|当前有效|最新的)")
_OLD_HINT_RE = re.compile(r"(历史|以前|旧版|往期|去年|前年)")


def infer_metadata_filter(
    query: str,
    *,
    settings=None,
    explicit: MetadataFilter | None = None,
) -> tuple[MetadataFilter, list[str]]:
    """
    从问题里推断元数据过滤条件（返回 (filter, 推断说明)）.

    ⚠️ **默认不按年份过滤**。仅在问题里出现**明确**的年份且没有"对比/趋势"这类
    跨年意图时才收窄到那一年。原因：把"营收情况如何"错误地限制到 2023 年，
    用户会得到一个看似合理、实则漏掉 2024 关键变化的答案 —— 这比不过滤更糟，
    因为它把"漏召回"伪装成了"这就是全部事实"。

    推断说明会随检索结果返回，让"这次的检索范围被收窄到了哪里"对上层可见。
    """
    if settings is None:
        from app.config import get_settings
        settings = get_settings()

    base = explicit or MetadataFilter()
    if not getattr(settings, "METADATA_FILTER_ENABLED", True):
        return base, []
    if base.years:
        return base, []          # 调用方显式给了年份，不做推断

    hints: list[str] = []
    q = query or ""
    if _COMPARE_HINT_RE.search(q) or _LATEST_HINT_RE.search(q) or _OLD_HINT_RE.search(q):
        # 跨年/趋势/最新类问题：一旦限年就答不出来
        hints.append("检测到跨年比较意图 → 不做年份收窄")
        return base, hints

    years = sorted({int(y) for y in _YEAR_IN_QUERY_RE.findall(q)}, reverse=True)
    if len(years) == 1:
        base.years = years
        hints.append(f"问题指向 {years[0]} 年 → 年份收窄到 {years[0]}")

    # 类型线索（"制度""合同"这类词出现在问题里时按类型收窄）
    types: list[str] = []
    for label, words in _DOC_TYPE_LEXICON:
        if any(w in q for w in words):
            types.append(label)
    if types:
        base.doc_types = types
        hints.append(f"问题指向文档类型 {','.join(types)}")

    return base, hints
