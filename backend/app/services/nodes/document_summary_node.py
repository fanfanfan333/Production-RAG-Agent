"""
Document Summary 节点（架构图 Document Summary 分支）.

与已有的 ``doc_relations`` 的区别
────────────────────────────────
两者都读全库文档摘要，但回答的焦点不同：

- doc_relations（已有）→ "这些文档之间有什么关联"
  输出：每个文档一节，重点是它和其他文档的关系
- document_summary（本模块）→ "这些文档 / 这份文档讲了什么"
  输出：每个文档一节，重点是内容本身的提炼（主题、要点、结论）

复用而非重写：摘要采样直接用 relation_service.collect_document_digests，
它已经处理好了"取哪些 chunk、截多少字符、权限过滤、上限控制"。
本模块只负责不同的 prompt 和不同的组织方式。
"""

from __future__ import annotations

import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.services.relation_service import (
    collect_document_digests,
    digest_sources,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── 逐文档总结（map-reduce）─────────────────────────────────────────────────
#
# 为什么不再用"一次调用总结全部文档"：
# 单次调用时，模型面对多份文档（尤其内容重复/近似的文档）会**合并同类项** —
# 内容多的那份写得很详细，其余被一句话带过甚至漏掉（问题：总结所有文档，
# 实际只总结了出现最多的那份）。提示词里写"一份都不能漏"对小模型没有约束力。
#
# map-reduce 把"覆盖全部文档"从**模型的自觉**变成**循环的结构保证**：
# 每份文档独立一次 LLM 调用（map），调用失败也有确定性兜底小节；
# 最后再用一次调用写总体概览（reduce）。文档一节不漏与模型行为解耦。

_SINGLE_DOC_SYSTEM = """\
你是一个企业私有知识库分析助手。用户要求总结文档内容，下面给你**一份**文档的
内容采样（从索引中按全文等距抽取的片段）。

任务：为这份文档写一段内容提炼，严格按以下结构（不要自己加标题，系统会统一加）：

**核心主题**：一句话说明这份文档是关于什么的。
**要点提炼**：3–5 个要点，概括文档的关键信息、结论或数据。
**适用对象**：一句话说明什么样的人/场景需要读它（可省略）。

规则：
- 只基于给出的内容采样，严禁编造采样中不存在的事实、数字或结论。
- 采样只是文档的部分内容；内容不足以支撑某个部分时如实说明"采样内容较少"，
  不要硬编，也不要提及"采样"以外的获取渠道。
- 不要输出 "###" 标题，不要写文档名作为开头（系统会加），不要提及其他文档。
- 不要任何开场白或客套，直接从"**核心主题**"开始。
- 使用与用户提问相同的语言回答（默认中文）。
"""

_OVERVIEW_SYSTEM = """\
你是一个企业私有知识库分析助手。你已经逐份总结了知识库中的 {count} 份文档，
下面是每一份的摘要。

任务：写一段"总体概览"—— 用 3–5 句话概括这些文档整体覆盖了哪些主题领域、
合在一起构成了什么图景；如果文档之间主题明显割裂（分属不同领域），如实指出
它们各自属于什么领域，不要强行捏合。

规则：
- 不要重复各份文档的细节，不要分小标题，不要使用列表，一段成文。
- 不要输出 "###" 标题（系统会加），不要任何开场白。
- 使用与用户提问相同的语言回答（默认中文）。

── 各文档摘要 ─────────────────────────────────────────────────────────────────
{summaries}
──────────────────────────────────────────────────────────────────────────────
"""


async def collect_summary_digests(
    owner_id: str | None = None,
    tenant_ids: frozenset[str] | None = None,
    owns_tenant_ids: frozenset[str] = frozenset(),
    user_department_id: str | None = None,
    tenant_wide: bool = False,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    拉取全库文档摘要（供 Document Summary 分支使用）.

    复用 relation_service 的采样逻辑 —— 采样策略、权限过滤、上限控制
    全部沿用，不重复实现。tenant_ids / user_department_id 是三层隔离的
    第一、二层过滤（摘要同样不能跨公司泄漏）；tenant_wide 把管理员的宽口径
    范围原样传下去，owns_tenant_ids 表达 admin 自建测试公司集合（可见其中
    他人私库），**个人库始终只有本人（+ admin 自建集合例外）**。

    document_ids 非空时只采样这几份文档（用户在提问里点名了）。
    整库总结的文档数上限用 DOC_SUMMARY_MAX_DOCUMENTS（map-reduce 的等待
    预算），不再沿用 doc_relations 的 RELATION_MAX_DOCUMENTS —— 关联分析
    受单次 prompt 长度限制，总结不受，截断时会在答案里明示。
    """
    settings = get_settings()
    digests = await collect_document_digests(
        max_documents=settings.DOC_SUMMARY_MAX_DOCUMENTS,
        chunks_per_doc=settings.RELATION_CHUNKS_PER_DOC,
        digest_chars=min(
            settings.RELATION_DIGEST_CHARS,
            settings.DOC_SUMMARY_MAX_CHARS_PER_DOC,
        ),
        owner_id=owner_id,
        tenant_ids=tenant_ids,
        owns_tenant_ids=owns_tenant_ids,
        user_department_id=user_department_id,
        tenant_wide=tenant_wide,
        document_ids=document_ids,
    )
    logger.info(
        "document_summary: collected %d document digests (scoped=%s)",
        len(digests),
        bool(document_ids),
    )
    return digest_sources(digests)


def build_single_doc_messages(query: str, digest: dict) -> list[BaseMessage]:
    """组装**一份**文档的总结 messages（map 阶段，每次调用只处理一份）."""
    meta_bits = [
        f"{digest.get('page_count', 0)} 页",
        f"{digest.get('chunk_count', 0)} 个文本块",
    ]
    warnings = digest.get("warnings") or []
    meta = "，".join(meta_bits + warnings)
    context = (
        f"文档文件名：{digest.get('filename', 'unknown')}（{meta}）\n"
        f"内容采样：\n{digest.get('digest', '')}"
    )
    focused_query = (
        "[指令：总结上面这份文档的内容，按要求的结构输出；"
        "不要加标题，不要提及其他文档。]\n\n"
        f"{query}"
    )
    return [
        SystemMessage(content=_SINGLE_DOC_SYSTEM),
        HumanMessage(content=f"{context}\n\n{focused_query}"),
    ]


def build_overview_messages(
    query: str,
    sections: list[tuple[str, str]],
) -> list[BaseMessage]:
    """
    组装"总体概览"的 messages（reduce 阶段）.

    *sections* 为 (文件名, 该文档的摘要正文) 列表 —— 概览只看摘要，
    不再回看原始采样，避免重复细节。
    """
    summaries = "\n\n".join(
        f"《{name}》\n{body.strip()}" for name, body in sections if body.strip()
    )
    return [
        SystemMessage(
            content=_OVERVIEW_SYSTEM.format(count=len(sections), summaries=summaries)
        ),
        HumanMessage(
            content="[指令：基于以上各文档摘要，写一段总体概览。]\n\n" + query
        ),
    ]


# ── 用户点名了哪份文档 ───────────────────────────────────────────────────────
#
# 需求：用户指定总结哪份文档，就只总结哪份 —— 而不是把整库都讲一遍。
# 判定必须**确定性**：解析提问里的文档名/特征词，和 ACL 可见文档逐个比对，
# 命中就收敛范围；点名了但一份都对不上 → 不猜、直接把可选文档列给用户。
#
# 为什么不用 LLM 做这件事：这是"名字对不对得上"的字符串问题，不是语义问题；
# 走 LLM 会引入随机性，而且对不上时还会硬编一个总结出来。

# 范围/指令类词：出现在"…文档"前面时不算文档名的一部分
_TARGET_STOPWORDS = (
    "总结", "概括", "概述", "综述", "摘要", "提炼", "归纳", "梳理", "汇总",
    "所有", "全部", "整个", "整库", "全库", "各个", "每个", "每一个", "每一份",
    "这些", "这批", "那些", "这份", "这个", "那个", "该", "此", "本",
    "知识库", "文档库", "资料库", "库里", "库中", "库内",
    "帮我", "请", "一下", "一份", "几份", "的", "里", "中",
    "文档", "文件", "资料", "报告", "材料", "附件",
)
# 引号 / 书名号里的一般是文档名（不用圆括号 —— "（重点）"这类括注不是书名）
_QUOTED_NAME_RE = re.compile(r"[《「『“\"']([^》」』”\"'\n]{2,80})[》」』”\"']")
# 带扩展名的文件名
_EXT_NAME_RE = re.compile(
    r"[\w\u4e00-\u9fa5][\w\u4e00-\u9fa5 \-_.·]{0,60}\.(?:pdf|docx?|xlsx?|pptx?|md|txt|csv)",
    re.I,
)
# "……文档/文件/报告" 前的限定语
_BEFORE_NOUN_RE = re.compile(
    r"([0-9a-z\u4e00-\u9fa5][0-9a-z\u4e00-\u9fa5 \-_.·]{1,40}?)[ \t]*"
    r"(?:文档|文件|资料|报告|材料|附件|手册|纪要|方案|合同|简历|笔记)"
)
# 总结动词后面直接跟的整段（弱信号：可能是文档名，也可能是"第三季度的营收情况"）
_AFTER_VERB_RE = re.compile(
    r"(?:总结|概括|概述|综述|摘要|提炼|归纳|梳理)"
    r"(?:一下|下|一遍)?[：:，,、\s]*"
    r"([0-9a-z\u4e00-\u9fa5][0-9a-z\u4e00-\u9fa5 \-_.·、/]{3,60})",
    re.I,
)
# 位置指代不算文档名（"总结一下第三章" 不是点名文档）
_POSITIONAL_RE = re.compile(r"^第\s*[0-9一二三四五六七八九十百]+\s*[章节页条部分段图表篇]")
# "第 2 份文档 / 第二个文件"
_ORDINAL_RE = re.compile(r"第\s*([0-9一二三四五六七八九十]{1,3})\s*[份个条篇册]")
_CN_DIGITS = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
# 命中门槛：加权字符覆盖率（长 token 更能代表文档，权重更高）
_MATCH_THRESHOLD = 0.6
# 弱信号候选名的最短长度（"第三章"这类 3 字位置词要挡在外面）
_WEAK_HINT_MIN_LEN = 4
# 提问里已经出现"整批/整库/就这份"的口径时，某个对不上的名字片段不足以
# 断定"点名的文档不存在"（例如"总结这些文档里关于「反幻觉机制」的内容"
# ——「反幻觉机制」是章节名，不是文档名）。这类一律退回整库总结。
_SCOPE_HINT_RE = re.compile(
    r"所有|全部|这些|那些|这批|整个|各个|每[一个份]|全库|整库|"
    r"知识库|文档库|资料库|库[里中内的]|这[份个]|那[份个]|该文档|此文档|本文档"
)


def _normalize(text: str) -> str:
    """只保留小写字母/数字/汉字 —— 抹掉大小写、空格、连字符、全角标点差异."""
    return re.sub(r"[^0-9a-z\u4e00-\u9fa5]+", "", (text or "").lower())


def _stem(filename: str) -> str:
    return re.sub(r"\.[a-z0-9]{1,6}$", "", (filename or "").strip(), flags=re.I)


def _name_tokens(name: str) -> list[str]:
    """文档名切成"可辨识单元"：连续字母数字 + 连续汉字（≥2 字）."""
    return [
        t
        for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fa5]+", (name or "").lower())
        if len(t) >= 2
    ]


def _coverage(tokens: list[str], target: str) -> float:
    """文档名 token 在目标串里的加权覆盖率 ∈ [0,1]."""
    total = sum(len(t) for t in tokens)
    if not total or not target:
        return 0.0
    hit = sum(len(t) for t in tokens if t in target)
    return hit / total


def _strip_stopwords(raw: str) -> str:
    out = raw
    for word in _TARGET_STOPWORDS:
        out = out.replace(word, "")
    return out.strip(" -_.·")


def _extract_name_hints(query: str) -> list[tuple[str, bool]]:
    """
    从提问里抽"可能是文档名"的片段，并标注信号强度.

    强信号（对不上就明确告知"没找到这份文档"）：
      《书名号》/ 引号内的名字、带扩展名的文件名、"……文档/文件/报告"前的限定语。
    弱信号（对不上就退回整库总结，绝不因此报错）：
      总结动词后面直接跟的那一段 —— 可能确实是文件名，也可能是"第三季度的营收"。
    """
    hints: list[tuple[str, bool]] = []
    for m in _QUOTED_NAME_RE.finditer(query):
        hints.append((m.group(1).strip(), True))
    for m in _EXT_NAME_RE.finditer(query):
        hints.append((m.group(0).strip(), True))
    for m in _BEFORE_NOUN_RE.finditer(query):
        cleaned = _strip_stopwords(m.group(1))
        if cleaned:
            hints.append((cleaned, True))
    for m in _AFTER_VERB_RE.finditer(query):
        cleaned = _strip_stopwords(m.group(1))
        if len(cleaned) >= _WEAK_HINT_MIN_LEN and not _POSITIONAL_RE.match(cleaned):
            hints.append((cleaned, False))

    # 去重保序（同一片段可能被多个规则抽到，强信号优先）
    seen: dict[str, int] = {}
    uniq: list[tuple[str, bool]] = []
    for hint, strong in hints:
        key = _normalize(hint)
        if not key:
            continue
        if key in seen:
            if strong:
                uniq[seen[key]] = (hint, True)
            continue
        seen[key] = len(uniq)
        uniq.append((hint, strong))
    return uniq


def _ordinal_index(query: str) -> int | None:
    """解析"第 2 份文档"里的序号（1-based）；没写返回 None."""
    m = _ORDINAL_RE.search(query)
    if not m:
        return None
    raw = m.group(1)
    if raw.isdigit():
        value = int(raw)
    elif raw == "十":
        value = 10
    elif raw.startswith("十"):
        value = 10 + _CN_DIGITS.get(raw[1:2], 0)
    elif raw.endswith("十"):
        value = _CN_DIGITS.get(raw[0:1], 0) * 10
    elif "十" in raw:
        head, _, tail = raw.partition("十")
        value = _CN_DIGITS.get(head, 0) * 10 + _CN_DIGITS.get(tail, 0)
    else:
        value = _CN_DIGITS.get(raw, 0)
    return value or None


def resolve_summary_targets(query: str, documents: list[tuple]) -> dict:
    """
    解析用户点名的文档.

    Args:
        query:     用户原始提问
        documents: ``list_accessible_documents`` 的返回值（id, filename, …），
                   按上传时间倒序 —— "第 2 份文档"的序号顺序与页面一致。

    Returns:
        {
          "ids":      命中的 document_id 列表（空 = 未点名或没对上）
          "names":    命中的文件名
          "hints":    抽到的**强信号**候选名（用于"没找到"的提示文案）
          "missing":  点名了但一份都没对上（此时 ids 为空，要给出提示）
          "available":当前可总结的全部文件名（提示用户时用）
          "ordinal":  命中的序号（第 N 份文档）
          "targeted": 是否收敛到了具体文档（决定 prompt 里的范围说明）
        }
    """
    available = [str(row[1]) for row in documents]
    result = {
        "ids": [],
        "names": [],
        "hints": [],
        "missing": False,
        "available": available,
        "ordinal": None,
        "targeted": False,
    }
    if not documents:
        return result

    hints = _extract_name_hints(query)
    query_norm = _normalize(query)
    result["hints"] = [h for h, strong in hints if strong]

    if hints:
        scored: list[tuple[float, object, str]] = []
        for row in documents:
            doc_id, filename = row[0], str(row[1])
            tokens = _name_tokens(_stem(filename))
            stem_norm = _normalize(_stem(filename))
            full_norm = _normalize(filename)
            if stem_norm and (stem_norm in query_norm or full_norm in query_norm):
                scored.append((1.0, doc_id, filename))       # 全名原样出现 = 铁证
                continue
            best = 0.0
            for hint, _strong in hints:
                hint_norm = _normalize(hint)
                # 特征词里的 token 覆盖 + 提问全串兜底（用户可能直接写全名无引号）
                best = max(best, _coverage(tokens, hint_norm))
            if best < _MATCH_THRESHOLD:
                best = max(best, _coverage(tokens, query_norm) * 0.9)
            scored.append((best, doc_id, filename))

        top = max(s for s, _, _ in scored)
        if top >= _MATCH_THRESHOLD:
            # 同分（近似重名文档）一起总结，但只要明显更贴的那几份
            keep = [item for item in scored if item[0] >= max(_MATCH_THRESHOLD, top - 0.1)]
            result["ids"] = [str(doc_id) for _, doc_id, _ in keep]
            result["names"] = [name for _, _, name in keep]
            result["targeted"] = True
            logger.info(
                "document_summary: query targets %s (hints=%s, score=%.2f)",
                result["names"], result["hints"], top,
            )
            return result

    # 名字没对上 → 试"第 N 份文档"（按页面顺序）
    ordinal = _ordinal_index(query)
    if ordinal:
        if 1 <= ordinal <= len(documents):
            row = documents[ordinal - 1]
            result["ids"] = [str(row[0])]
            result["names"] = [str(row[1])]
            result["ordinal"] = ordinal
            result["targeted"] = True
            return result
        result["missing"] = True
        result["ordinal"] = ordinal
        return result

    # 只有强信号（书名号/扩展名/…文档）没对上才明确报"没找到"；
    # 弱信号（总结动词后的整段）没对上就退回整库总结，避免把
    # "总结一下第三季度的经营情况"误判成"你点名的文档不存在"。
    if result["hints"] and not _SCOPE_HINT_RE.search(query):
        result["missing"] = True
    return result


def build_target_not_found_answer(scope: dict) -> str:
    """点名了文档但一份都没对上 → 确定性答复（不调 LLM，不猜）。"""
    hints = "、".join(f"「{h}」" for h in scope.get("hints") or [])
    available = scope.get("available") or []
    lines = [f"没有在可访问的文档中找到与 {hints or '你提到的名称'} 匹配的文档，因此没有生成总结。"]
    if scope.get("ordinal"):
        lines.append(f"（你写的是「第 {scope['ordinal']} 份文档」，但当前只有 {len(available)} 份。）")
    if available:
        lines.append("")
        lines.append(f"当前可总结的文档共 {len(available)} 份：")
        lines.extend(f"{i}. {name}" for i, name in enumerate(available, start=1))
        lines.append("")
        lines.append(
            "请用文档名（或其中一段特征词）指定，例如「总结《"
            + available[0]
            + "》」；也可以说「总结所有文档」一次性总结全部。"
        )
    else:
        lines.append("当前知识库里没有已完成索引的文档。")
    return "\n".join(lines)


def build_summary_llm(reasoning: bool | None = None):
    """构造文档总结用的 LLM（temperature 略高，允许一定概括性措辞）.

    ``reasoning`` 是逐份文档正文与"总体概览"两处的取舍开关：

    * ``None``（默认）→ 读 ``DOC_SUMMARY_REASONING``，作用于**逐份正文**：
      开启时模型会先想一遍再写，数字/结论的取舍更稳，但每份多花约 30s。
    * ``False`` → 用于"总体概览"：输入只是已经写好的各节摘要，是一次纯归纳，
      思考不带来质量提升，却要多花一次完整思考的等待时间。整库总结是串行
      多次调用（N 份文档 + 1 次概览），这一处省下的时间会线性叠加。

    ⚠️ **必须显式传 num_ctx**：不传时 Ollama 会用模型自带的 40960 上下文，
    小显存机器既占内存又要为每次请求重建更大的 KV cache —— 代码库其它
    LLM 构造处（master_graph / general_chat / query_router / grader）都显式
    传了，这里曾经漏掉。
    """
    settings = get_settings()
    if reasoning is None:
        reasoning = bool(getattr(settings, "DOC_SUMMARY_REASONING", True))
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.2,
        streaming=True,
        reasoning=reasoning,
        num_ctx=settings.chat_num_ctx,
    )
