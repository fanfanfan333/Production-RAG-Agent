"""意图路由的确定性规则（零第三方依赖，便于单测）.

Query Router 采用两级路由：确定性规则优先，其余交给 LLM。本模块只放
**确定性**那一层，不引入 langchain / Ollama / 配置依赖，因此可以脱离
后端环境直接单测（见 backend/tests/test_intent_rules.py）。

设计原则
────────
只拦截 100% 确定的情况。宁可让给 LLM，也不要误判——
误判的代价（把真正的知识库问题送进闲聊，或反之）远高于多打一次 LLM。
"""

from __future__ import annotations

import re

__all__ = [
    "VALID_INTENTS",
    "DEFAULT_INTENT",
    "deterministic_route",
    "looks_knowledge_seeking",
]


VALID_INTENTS = (
    "document_summary",
    "knowledge_qa",
    "general_chat",
    "doc_relations",
    "list_documents",
    # Document Agent（最终效果：Word 写入 / Word 插入图片）
    "document_agent",
)

# 降级兜底：最保守、最常见的分支。
DEFAULT_INTENT = "knowledge_qa"


# ── 跨文档关联 ──────────────────────────────────────────────────────────────

_ASKS_RELATION_RE = re.compile(
    r"关联|关系|联系|相关性|关联度|异同|共同点|相同点|相似之处|重叠|互补|主题分布"
)
_REFERENCES_COLLECTION_RE = re.compile(
    r"(这些|这批|这几个|这几份|各个|所有|全部|哪些|库里|库中|库内|知识库|文档库|上传)[^。？?!\n]{0,12}(文档|文件|资料|报告)"
    r"|文档库|知识库"
    r"|(文档|文件|资料|报告)之间"
)


# ── 列出文档 ────────────────────────────────────────────────────────────────

_ASKS_DOC_LIST_RE = re.compile(
    r"(有哪些|有什么|都有哪些|都有什么|多少个?|哪些|列出|列一下|清单|列表|包含哪些)"
    r"[^。？?!\n]{0,4}(文档|文件|资料|pdf)"
    r"|(文档|文件|资料|pdf)(列表|清单)"
    r"|上传了(哪些|什么)(文档|文件|资料)?"
)


# ── Document Agent：把检索结果生成一份可下载的文档 ──────────────────────────
#
# 特征："生成/写/导出/整理" + 交付物名词（word/docx/文档/报告/纪要…）。
# 与 document_summary 的区别：summary 只要一段摘要文字，agent 要一份**文档**。
_AGENT_VERB = r"(?:生成|制作|导出|输出|撰写|写|整理|汇总|汇总成|做成|形成|创建|帮我写|帮我做|拟)"
_AGENT_NOUN = r"(?:word|docx|WORD|文档|报告|说明书|纪要|会议纪要|方案|合同|简历|总结报告|分析报告)"
_ASKS_DOCUMENT_AGENT_RE = re.compile(
    rf"{_AGENT_VERB}[^。？?!\n]{{0,12}}{_AGENT_NOUN}"
    rf"|{_AGENT_NOUN}[^。？?!\n]{{0,6}}文件"
    rf"|(?:导出|下载)[^。？?!\n]{{0,8}}(?:word|docx|文档)"
)


# ── 文档总结（整库 / 点名某份文档）───────────────────────────────────────────
#
# 为什么需要确定性规则：**"总结所有文档"这类整库总结请求交给 LLM 路由会飘**。
# 实测被判定成 knowledge_qa → 走检索链 → 只把召回分数最高的那一份文档的两三个
# 片段拼成"总结"，用户看到的就是"只总结了内容最多的那份文档"（其余文档根本没进
# 上下文）。整库总结的特征极其明确：总结动词 + 范围词/库范围词 + 文档类名词，
# 值得一条不依赖 LLM 的通路。
#
# 刻意不收"这份文档"：那类短句（"这份文档主要讲了什么"）语义边界模糊，
# 交给 LLM 更稳（见 tests/test_intent_rules.py 的既有断言）。
_SUMMARY_VERB_RE = re.compile(r"总结|概括|概述|综述|概览|摘要|提炼|归纳|梳理")
# 范围词：明确指向"一批/一整库"而不是某一句话
_SUMMARY_SCOPE_RE = re.compile(r"所有|全部|整个|各个|每[一个份]|全[部库]|整[个库]")
# 库范围词：本身就是范围，"总结一下知识库" 无需再出现"文档"二字
_SUMMARY_LIBRARY_RE = re.compile(r"全库|整库|库[里中内的]|知识库|文档库|资料库")
_SUMMARY_NOUN_RE = re.compile(r"文档|文件|资料|报告|材料|附件|手册|纪要|方案|合同|简历")
# 带扩展名的文件名：用户点名文档时最没有歧义的写法
_SUMMARY_FILENAME_RE = re.compile(
    r"[\w\u4e00-\u9fa5][\w\u4e00-\u9fa5 \-_.·]{0,60}\.(?:pdf|docx?|xlsx?|pptx?|md|txt|csv)",
    re.I,
)


# ── 纯闲聊 / 打招呼 / 身份询问 ──────────────────────────────────────────────
#
# 必须**整句完全匹配**才命中（两端锚定 + 只允许尾部标点），绝不能用 search，
# 否则"你好，帮我看下合同第三条"会被误判成闲聊。
#
# 存在的意义（问题1）：路由 LLM 超时 / Ollama 不可达时会降级成 knowledge_qa，
# 于是"你是谁"又被塞进 RAG 检索 → 无证据 → 拒答，表现就是"闲聊永远触发
# 不了"。给最典型的打招呼 / 身份问题一条不依赖 LLM 的确定性通路，闲聊分支
# 才是真正可达的。
_PURE_CHITCHAT_RE = re.compile(
    r"^[\s,.，、!！?？~～]*(?:"
    r"你好|您好|你们好|hi|hello|hey|嗨|哈喽|嗨喽|"
    r"早上好|上午好|中午好|下午好|晚上好|晚安|"
    r"在吗|在么|在不在|有人吗|"
    r"再见|拜拜|bye|goodbye|"
    r"谢谢|多谢|感谢|thanks|thank you|辛苦了|"
    r"你是谁|你叫什么名字?|你是什么(?:模型|ai|助手|机器人)?|"
    r"介绍一下你自己?|自我介绍一下|你能做什么|你能干什么|你会什么|你有什么功能"
    r")[\s,.，、!！?？~～。]*$",
    re.I,
)
# 超过这个长度的提问不可能是纯打招呼，直接跳过确定性闲聊判定。
_CHITCHAT_MAX_LEN = 30


def deterministic_route(query: str) -> str | None:
    """确定性前置判定：命中返回意图标签，未命中返回 ``None``（交给 LLM）.

    拦截五类 100% 确定的情况：

    * 跨文档关联 → ``doc_relations``（特征明显，且有专门 pipeline）
    * 列出文档   → ``list_documents``（必须走 DB 直读，LLM 会引入随机性）
    * 生成文档   → ``document_agent``（"生成/导出 …word/文档/报告" 交付物明确）
    * 整库总结   → ``document_summary``（"总结所有文档"必须覆盖全库，
                   交给 LLM 会飘成知识问答，变成"只总结召回最高的那一份"）
    * 纯打招呼   → ``general_chat``（保证闲聊分支不依赖 LLM 也可达）

    顺序说明：列表请求先判（"列出文档"里也可能出现"文档列表"），
    生成文档随后（需要明确的产出动词 + 交付物名词，避免误伤"这份文档讲了什么"），
    整库总结再后（必须同时出现总结动词与范围/库范围词）。
    """
    if not query:
        return None

    if _ASKS_RELATION_RE.search(query) and _REFERENCES_COLLECTION_RE.search(query):
        return "doc_relations"
    if _ASKS_DOC_LIST_RE.search(query):
        return "list_documents"
    if _ASKS_DOCUMENT_AGENT_RE.search(query):
        return "document_agent"
    if _SUMMARY_VERB_RE.search(query):
        # 点名了带扩展名的具体文件 → 一定是总结该文档
        if _SUMMARY_FILENAME_RE.search(query):
            return "document_summary"
        # "总结所有文档"：范围词 + 文档类名词；"总结一下知识库"：库范围词自带范围
        if (_SUMMARY_NOUN_RE.search(query) or _SUMMARY_LIBRARY_RE.search(query)) and (
            _SUMMARY_SCOPE_RE.search(query) or _SUMMARY_LIBRARY_RE.search(query)
        ):
            return "document_summary"
    if len(query.strip()) <= _CHITCHAT_MAX_LEN and _PURE_CHITCHAT_RE.match(query):
        return "general_chat"
    return None


# ── 知识提问守卫（LLM 路由结果的纠偏依据）────────────────────────────────────
#
# 背景（实测 BUG）：LLM 路由器会把"反幻觉机制是怎么工作的？"这类**明显在问
# 知识**的问题判成 general_chat。后果不是"少了一次检索"，而是整条检索链被
# 绕过 —— 模型凭预训练记忆作答，既不引用文档，也过不了引用校验，而界面
# 看不出任何异常。这与本项目的反幻觉目标（证据门控 + 引用校验）直接冲突。
#
# 纠偏口径：问句带疑问词/疑问语气，且不在少数几个**纯生活闲聊话题**上，
# 就按 knowledge_qa 处理。取舍很明确 ——
#   "检索无证据 → 如实拒答" 远好于 "绕过检索 → 编一个像样的答案"。
#
# 注意：本函数**只**用于纠正 LLM 的 general_chat 判定，不参与
# deterministic_route 的前置分流 —— 它不区分 knowledge_qa / document_summary
# / doc_relations，那几个分支的优先级已在 deterministic_route 里判完。
_QUESTION_HINT_RE = re.compile(
    r"怎[么样]|如何|咋(?:样|办)?|为什么|为啥|"
    r"是什么|什么是|是啥|啥是|哪些|哪个|哪一种|哪种|多少|几个|"
    r"原理|机制|流程|步骤|作用|区别|差异|优缺点|好处|坏处|影响|"
    r"定义|含义|概念|架构|结构|功能|用途|适用|方法|策略|"
    r"配置|参数|用法|报错|原因|条件|标准|规范|"
    r"介绍|说明|解释|讲解|举例|对比|比较|"
    r"能否|是否|可不可以|支持吗|可以吗|有没有|"
    # 末行不能以 `|` 起头 —— 那会拼出空分支（`||`），使整个正则对任意字符串
    # 恒为真（"你好" 也会被判成知识提问）。单测 test_knowledge_seeking_*
    # 就是防这个的。
    r"how|what|why|which|when|where|who|explain|describe|difference",
    re.I,
)
# 句尾疑问语气。只用"吗"不用"么/嘛"：`么` 会嵌在"什么 / 怎么 / 多么"里，
# 按句尾判定同样会误命中；`吗` 在中文里不会构成其他词，可以安全锚定。
_QUESTION_TAIL_RE = re.compile(r"吗[？?]?[\s。！!~～]*$")
# 纯生活闲聊话题：即便带疑问语气也不该进知识库检索。
# 只在**短问句**上生效（见 _OFFTOPIC_MAX_LEN），避免误伤
# "文档里提到的日期是什么" 这类真知识问题。
_OFFTOPIC_CHITCHAT_RE = re.compile(
    r"天气|气温|几点|日期|星期几|礼拜几|节假日|冷笑话|笑话|讲个故事|"
    r"唱歌|心情|吃(?:什么|啥|饭)|喝(?:什么|啥)|吃了(?:吗|没)|在干嘛|在干什么|"
    r"聊聊天|无聊"
)
_OFFTOPIC_MAX_LEN = 20
# 超过这个长度基本是正经提问/长文，不再当纯闲聊话题看待。
_KNOWLEDGE_QUESTION_MAX_LEN = 200


def looks_knowledge_seeking(query: str) -> bool:
    """问句是否在**寻求知识**（而非闲聊 / 创作指令）.

    供 Query Router 在 LLM 判定为 ``general_chat`` 时纠偏使用。
    """
    if not query:
        return False
    q = query.strip()
    if not q or len(q) > _KNOWLEDGE_QUESTION_MAX_LEN:
        return False
    if len(q) <= _OFFTOPIC_MAX_LEN and _OFFTOPIC_CHITCHAT_RE.search(q):
        return False
    return bool(_QUESTION_HINT_RE.search(q) or _QUESTION_TAIL_RE.search(q))
