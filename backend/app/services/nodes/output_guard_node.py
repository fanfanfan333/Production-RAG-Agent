"""
Output Guard 节点（架构图 Generate → Citation Check → END；问题3+问题4）.

这是生成管线最后一道闸：在 LLM 答完文本、做持久化与前端展示之前，
对答案做四层确定性合规检查。

为什么必须做
────────────
1. LLM 可能编造 [Source N] 引用编号（幻觉最常见的形态之一）。
2. LLM 在闲聊分支可能硬塞"根据知识库…"、"我查到…"等幻觉措辞。
3. LLM 在被诱导时可能复述系统提示词的片段（即使输入侧 prompt_security
   拦了高危模式，生成侧仍有概率被 instruction injection 撬动）。
4. LLM 可能假装要"调用工具/执行命令/访问网络"——本系统根本没给模型工具，
   任何此类输出都属于越权意图，必须剔除。

本模块的特征
────────────
- 完全确定性（不调 LLM），每条规则都是窄匹配 + 替换/移除；
- 不重写答案语义，仅做片段级删改；
- 输出包含审计信号（citations_removed / leaked_phrases / ...），可写入日志；
- 不阻塞主流程（即使整段被剥空也保留可读兜底）。

四层检查
────────
(1) Citation Check（问题4）：越界的 [Source N] / [Source N-M] 移除
(2) System Prompt Leak（问题3 输出防护）：检测系统/开发者消息泄露
(3) General-Chat Phrasing Guard（问题3 Prompt 隔离 + 输出防护）：
    general_chat 场景下"根据知识库"/"[Source N]" 等幻觉措辞移除
(4) Agent Tool Permission Control（问题3 截图5 + 工具权限控制）：
    "I will now call tool" / "curl http" / 代码块 等越权意图移除
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OutputGuardResult:
    """Output Guard 节点的判定结果."""

    sanitized_text: str
    citations_removed: tuple[int, ...] = ()
    leaked_phrases: tuple[str, ...] = ()
    hallucination_phrases: tuple[str, ...] = ()
    tool_attempt_phrases: tuple[str, ...] = ()
    changed: bool = False


# ── (1) Citation Check ────────────────────────────────────────────────────────
# 匹配 [Source N] / [Source N-M]（连字符可有可无，可中划线/波浪号/连字线）
_CITATION_RE = re.compile(
    r"\[Source\s*(\d+)(?:\s*[-–~]\s*(\d+))?\]",
    re.IGNORECASE,
)


# ── (2) 系统提示词泄露 ───────────────────────────────────────────────────────
# 命中即整短语替换为合规占位（保留句子骨架）。
_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 英文：系统/开发者/隐藏/内部指令
    re.compile(r"\b(system prompt|developer message|hidden instruction|internal instruction|secret instruction)\b", re.I),
    # 英文：模型自陈"我的初始指令/提示/规则"
    re.compile(r"\bmy (initial|original|hidden|secret) (instructions?|prompts?|rules?)\b", re.I),
    # 英文：模型自陈"按照系统/开发者消息回答"
    re.compile(r"\baccording to (my|the) (developer|system) (message|prompt|instructions?)\b", re.I),
    # 中文：系统/开发者/隐藏/内部指令
    re.compile(r"(系统提示词|开发者消息|隐藏指令|内部指令|系统指令|开发者指令|私密指令)"),
    # 中文：模型自陈"我的初始指令"
    re.compile(r"(我|模型)的?(初始|原始|隐藏|内部|私密)的?(指令|提示|规则|设定)"),
    # 中文：模型自陈"根据/按照 系统/开发者 指令/消息"
    re.compile(r"(根据|按照|依据).{0,8}(系统|开发者|隐藏|内部).{0,8}(指令|消息|提示|设定)"),
)
_LEAK_PLACEHOLDER = "[已屏蔽：检测到疑似系统提示词泄露]"


# ── (3) 闲聊分支幻觉措辞 ─────────────────────────────────────────────────────
# 只在 intent == "general_chat" 时启用：闲聊根本没查知识库，不该有这些措辞。
_HALLUCINATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 英文
    re.compile(r"\baccording to (the |my )?(knowledge base|documents?|docs?|retrieved|context)\b", re.I),
    re.compile(r"\b(I found|I searched|I retrieved|I looked up) (in|through) (the )?(knowledge base|documents?|docs?)\b", re.I),
    re.compile(r"\bthe (knowledge base|document|file) (says|states|mentions|shows|contains)\b", re.I),
    re.compile(r"\[Source\s*\d+(?:\s*[-–~]\s*\d+)?\]", re.I),  # 闲聊分支绝对不应出现引用
    # 中文
    re.compile(r"(根据|按照).{0,8}(知识库|文档|检索|搜索|查询|上下文)"),
    re.compile(r"(我|已|已经|刚才)(检索|查询|搜索|查找|查到|查了)"),
    re.compile(r"(知识库|文档|资料)(中|里|内)(提到|指出|说|显示|写道|包含)"),
)


# ── (4) Agent 工具权限控制 ──────────────────────────────────────────────────
# 本系统没有给 LLM 任何工具/函数调用机制。任何"我要执行工具/访问网络/
# 运行命令"的输出都属于越权意图，必须剔除。
#
# 实现策略：代码块（```bash / ```python / ```sh）整段移除；其它工具调用
# 措辞做局部替换。
_CODE_BLOCK_RE = re.compile(
    r"```(?:bash|sh|shell|python|py|javascript|js|ts|tsx|jsx|powershell|cmd)\s*\n.*?\n```",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_ATTEMPT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 英文：call/execute/run + tool/function/api/plugin/command/shell
    # 既包含 "call tool" 也包含 "I will now call ... tool"
    re.compile(
        r"\b(call|calling|invoke|invoking|execute|executing|run|running|"
        r"will call|will execute|will run)\b.{0,40}\b"
        r"(tool|function|api|plugin|command|shell)\b",
        re.I,
    ),
    # 英文：模型自陈"我要搜索/抓取/调用/运行"
    re.compile(r"\bI\s+will\s+(now\s+)?(search|fetch|call|execute|run|query|browse|visit)\b", re.I),
    re.compile(r"\bI'?ll\s+(search|fetch|call|execute|run|query|browse|visit)\b", re.I),
    # 英文：模型自陈"让我搜/拉/调"
    re.compile(r"\blet me\s+(search|fetch|execute|run|query|browse|curl|visit)\b", re.I),
    # 英文：shell/网络执行痕迹
    re.compile(r"\bcurl\s+https?://\S+", re.I),
    re.compile(r"\bwget\s+https?://\S+", re.I),
    re.compile(r"\b(httpx|requests|aiohttp|urllib|fetch)\.(get|post|put|delete|request)\s*\(", re.I),
    re.compile(r"\bsubprocess\.(run|call|check_output|popen)\s*\(", re.I),
    # 中文：调用/执行/运行 工具/函数/命令/插件/API
    re.compile(r"(调用|执行|运行|将会调用|将执行|即将执行|即将调用).{0,30}(工具|函数|命令|插件|API|接口|网络|外部)"),
    # 中文：让我来搜/拉/调
    re.compile(r"(让我|我来|我先|接下来我).{0,12}(搜索|访问|抓取|调用|执行|运行|联网|浏览).{0,12}(网页|网络|工具|函数|命令)"),
    # 中文：内部函数调用
    re.compile(r"调用\s*(?:本地|内部)?\s*(?:函数|方法)\s*\("),
)


def _max_source_index(sources: list[dict]) -> int:
    """
    返回 sources 列表中可被合法引用的最大 [Source N] 编号。

    sources 由 context_builder.build_context() 按 list index 顺序生成，
    编号从 1 开始；越界引用一律移除。
    """
    return max(0, len(sources))


def inspect_output(
    answer: str,
    sources: list[dict],
    intent: str,
) -> OutputGuardResult:
    """
    对 *answer* 做四层合规检查并返回净化后的文本与审计信号。

    Args:
        answer:  LLM 流式输出的最终答案（已拼接）。允许为空（返回原文）。
        sources: 本轮实际喂给 LLM 的引用来源 dict 列表。
                 编号按 list index，1..len(sources)；超出的 [Source N] 视为幻觉。
        intent:  master graph 路由出来的意图字符串。

    Returns:
        OutputGuardResult —— sanitized_text 是合规后的最终答案，
        其余字段为审计信号（供 save_history / 日志落库）。
    """
    if not answer:
        return OutputGuardResult(sanitized_text=answer, changed=False)

    sanitized = answer
    citations_removed: list[int] = []
    leaked: list[str] = []
    hallucination: list[str] = []
    tool_attempts: list[str] = []
    changed_anything = False

    # ── (1) Citation Check ───────────────────────────────────────────────────
    max_idx = _max_source_index(sources)

    def _filter_citation(match: re.Match[str]) -> str:
        nonlocal changed_anything
        n = int(match.group(1))
        m2 = match.group(2)
        end = int(m2) if m2 else n
        if n > max_idx or end > max_idx:
            citations_removed.extend(range(n, end + 1))
            changed_anything = True
            return ""
        return match.group(0)

    sanitized = _CITATION_RE.sub(_filter_citation, sanitized)

    # ── (2) 系统提示词泄露 ──────────────────────────────────────────────────
    for pattern in _LEAK_PATTERNS:
        if pattern.search(sanitized):
            sanitized = pattern.sub(_LEAK_PLACEHOLDER, sanitized)
            leaked.append(pattern.pattern)
            changed_anything = True

    # ── (3) 闲聊分支幻觉措辞 ───────────────────────────────────────────────
    if intent == "general_chat":
        for pattern in _HALLUCINATION_PATTERNS:
            if pattern.search(sanitized):
                sanitized = pattern.sub("", sanitized)
                hallucination.append(pattern.pattern)
                changed_anything = True

    # ── (4) Agent 工具权限控制 ──────────────────────────────────────────────
    # 先整段移除代码块（避免块内任意 shell/python 命令残留）
    if _CODE_BLOCK_RE.search(sanitized):
        sanitized = _CODE_BLOCK_RE.sub("", sanitized)
        tool_attempts.append("<code-block>")
        changed_anything = True

    for pattern in _TOOL_ATTEMPT_PATTERNS:
        if pattern.search(sanitized):
            sanitized = pattern.sub("", sanitized)
            tool_attempts.append(pattern.pattern)
            changed_anything = True

    # 清理：移除过多空行 + 首尾空白
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized).strip()

    if changed_anything:
        logger.info(
            "output_guard: sanitized answer (intent=%s) "
            "citations_removed=%d leaked=%d hallucination=%d tool_attempts=%d",
            intent,
            len(citations_removed),
            len(leaked),
            len(hallucination),
            len(tool_attempts),
        )

    return OutputGuardResult(
        sanitized_text=sanitized,
        citations_removed=tuple(sorted(set(citations_removed))),
        leaked_phrases=tuple(leaked),
        hallucination_phrases=tuple(hallucination),
        tool_attempt_phrases=tuple(tool_attempts),
        changed=changed_anything,
    )