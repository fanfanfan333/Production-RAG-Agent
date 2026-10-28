"""分层上下文预算（会话状态 ≠ 事实证据）.

## 为什么需要分层，而不是「多塞几条历史」

升级前的做法是：`conversation_service.load_history()` 取最近 3 对消息，
`rag_graph._trim_history()` 把 AI 回答截到 400 字，然后整段塞进提示。
它有两个隐患：

1. **没有预算**：3 对消息里若有长回答，会挤掉检索证据（证据预算是独立的
   `CONTEXT_MAX_TOTAL_CHARS`，两者互不知情），表现为「有来源却答不出来」。
2. **没有用途区分**：历史里的模型回答和当轮检索片段被混在同一层，模型会把
   上一轮未核验的结论当成事实继续引用 —— 这是最难查的一类幻觉。

因此这里把上下文拆成**用途不同的四层**，各自计量：

| 层 | 内容 | 是否可作为事实依据 |
|---|---|---|
| 长期记忆 | 用户显式声明的偏好/术语 | 否（只影响表达与口径） |
| 会话摘要 | 更早轮次的用户诉求（抽取式） | 否（只用于指代消解） |
| 最近轮次 | 最近 N 对原始消息 | 否（仅承接语气与指代） |
| 当轮证据 | 检索片段 | **是**（唯一依据，由 Evidence Gate 把关） |

后三层由本模块产出文本，第一层由 `memory_store` 提供。当轮证据不经过本模块 ——
它有自己的预算与压缩链（`context_builder.compress_text`），把两者混在一个
函数里会让「证据被历史挤掉」重新变成一个可发生的 bug。

## 关于会话摘要：抽取式，不调 LLM

生成式摘要要多花一次模型调用，还会把模型的措辞写进长期状态（越滚越偏）。
这里用**抽取式摘要**：只取更早轮次里用户自己的问题，截断后拼接。它足够支撑
「刚才那个文档」「上面第三点」这类指代，且完全确定性、可复现、零成本。

刻意**不把更早的 AI 回答放进摘要**：那是未核验结论，沉淀成"会话状态"后，
下一轮就会以「我们之前说过」的口吻被复述成事实。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "LayerBudgets",
    "LayeredContext",
    "SYSTEM_BLOCK_HEADER",
    "build_layered_context",
    "render_system_block",
]

# 与 conversation_service._DEFAULT_HISTORY_PAIRS 保持一致：升级不改变默认体验。
DEFAULT_RECENT_PAIRS = 3

# 抽取式摘要最多回看多少个更早的用户问题。
MAX_DIGEST_TURNS = 6
# 单条摘要片段的截断长度（用户问题通常很短，超过这个长度基本是粘贴的长文）。
DIGEST_SNIPPET_CHARS = 120
_DIGEST_HEADER = "更早的提问（按时间顺序）："

SYSTEM_BLOCK_HEADER = (
    "以下是本轮对话的状态层，**不是事实依据**："
    "不得把它们当作引用来源，也不得据此编造数值或结论；"
    "与检索片段冲突时一律以检索片段为准。"
)


@dataclass(frozen=True)
class LayerBudgets:
    """各层字符预算（不是硬截断上限，而是"这一层最多占多少"的后置裁剪）."""

    memory: int = 800
    session_digest: int = 600
    recent: int = 2400


@dataclass(frozen=True)
class LayeredContext:
    """分层结果；`trace` 直接进 SSE / 落库，用于解释"预算被谁吃掉了"."""

    memory_text: str
    session_digest: str
    recent_text: str
    trace: dict

    @property
    def system_block(self) -> str:
        return render_system_block(self)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _normalize(history: Iterable[object]) -> list[tuple[str, str]]:
    """把消息对象/元组统一成 ``(role, content)``.

    兼容 `langchain` 的 BaseMessage（有 `.type` / `.content`）与纯元组，
    这样本模块不必依赖 langchain —— 纯逻辑模块不该把重依赖拖进单测。
    """
    out: list[tuple[str, str]] = []
    for item in history or ():
        role = "user"
        content = ""
        if isinstance(item, tuple) and len(item) == 2:
            role, content = str(item[0]), str(item[1] or "")
        else:
            raw_role = getattr(item, "type", None) or getattr(item, "role", None) or "user"
            role = str(raw_role)
            content = str(getattr(item, "content", "") or "")
        role = "assistant" if role in {"ai", "assistant"} else "user"
        content = content.strip()
        if content:
            out.append((role, content))
    return out


def _render_turns(turns: Sequence[tuple[str, str]]) -> str:
    if not turns:
        return ""
    lines = []
    for role, content in turns:
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}：{content}")
    return "\n".join(lines)


def _build_digest(
    older: Sequence[tuple[str, str]], budget: int
) -> tuple[str, int]:
    """抽取式会话摘要：只取更早轮次里**用户**的问题.

    Returns:
        ``(摘要文本, 因预算被丢掉的片段数)``。

    超预算时丢**最早**的片段，而不是从尾部截断：越靠近当前问题的提问越可能
    承载指代（"刚才那份""上面第三点""再按那个口径"）。旧实现用 `_clip` 截尾，
    恰好先把最相关的那几条扔掉，留下最不相干的开场白 —— 摘要看着有内容，
    对指代消解毫无帮助。
    """
    questions = [content for role, content in older if role == "user"]
    if not questions:
        return "", 0
    picked = questions[-MAX_DIGEST_TURNS:]
    fragments = [_clip(q.replace("\n", " "), DIGEST_SNIPPET_CHARS) for q in picked]
    dropped = 0
    while fragments and len(_DIGEST_HEADER) + len("；".join(fragments)) > budget:
        fragments.pop(0)
        dropped += 1
    if not fragments:
        return "", dropped
    text = _DIGEST_HEADER + "；".join(fragments)
    # 单条片段自己就超预算的极端情况：仍要截断，否则会溢出该层预算。
    if len(text) > budget:
        text = _clip(text, budget)
    return text, dropped


# kind 会被拼进行首（`- [{kind}] {content}`），必须是"一个词"。
# 写入端已有封闭枚举，这里再做一次结构收敛：存量数据、将来别的写入路径、
# 手工改库都可能带进换行或方括号，一旦进 system 层就等于伪造出一行系统设定。
_SAFE_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")


def _safe_kind(value: object) -> str:
    kind = str(value or "").strip().lower()
    return kind if _SAFE_KIND_RE.match(kind) else "note"


def _render_memories(items: Sequence[dict], budget: int) -> tuple[str, int]:
    """渲染长期记忆条目；返回 (文本, 实际收录条数)."""
    if not items:
        return "", 0
    lines: list[str] = []
    used = 0
    for item in items:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        kind = _safe_kind(item.get("kind"))
        line = f"- [{kind}] {content}"
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return "", 0
    header = "用户已确认的长期设定（仅影响口径与表达）："
    return header + "\n" + "\n".join(lines), len(lines)


def build_layered_context(
    history: Sequence[object],
    memory_items: Sequence[dict] | None = None,
    *,
    budgets: LayerBudgets | None = None,
    recent_pairs: int = DEFAULT_RECENT_PAIRS,
    memory_enabled: bool = True,
) -> LayeredContext:
    """按预算组装分层上下文.

    Args:
        history: 历史消息（BaseMessage 或 ``(role, content)`` 元组）。
        memory_items: 已审核/用户显式声明且未撤销的记忆条目。
        budgets: 各层预算；``None`` 用默认值。
        recent_pairs: 保留多少对原始轮次（默认 3，与升级前一致）。
        memory_enabled: 关闭时完全忽略 ``memory_items``（回退路径）。

    Returns:
        `LayeredContext`；三层都为空时 `system_block` 为空串，
        调用方据此**完全不加** system 消息 —— 短会话的行为与升级前逐字节相同。
    """
    budgets = budgets or LayerBudgets()
    recent_pairs = max(1, int(recent_pairs))
    turns = _normalize(history)

    user_positions = [i for i, (role, _) in enumerate(turns) if role == "user"]
    if len(user_positions) <= recent_pairs:
        older: list[tuple[str, str]] = []
        recent = list(turns)
    else:
        cutoff = user_positions[-recent_pairs]
        older = list(turns[:cutoff])
        recent = list(turns[cutoff:])

    memory_text, memory_used = ("", 0)
    if memory_enabled and memory_items:
        memory_text, memory_used = _render_memories(memory_items, budgets.memory)

    digest, digest_dropped = _build_digest(older, budgets.session_digest)
    digest_truncated = digest_dropped > 0 or (bool(digest) and digest.endswith("…"))

    recent_text = _render_turns(recent)
    if len(recent_text) > budgets.recent:
        # 超预算时**从最早的一条开始丢**，保留最新上下文（离当前问题最近的信息
        # 才可能承载指代）。
        kept: list[tuple[str, str]] = []
        used = 0
        for turn in reversed(recent):
            cost = len(turn[1]) + 8
            if used + cost > budgets.recent:
                break
            kept.append(turn)
            used += cost
        recent_text = _render_turns(list(reversed(kept)))

    trace = {
        "budgets": {
            "memory": budgets.memory,
            "session_digest": budgets.session_digest,
            "recent": budgets.recent,
        },
        "chars": {
            "memory": len(memory_text),
            "session_digest": len(digest),
            "recent": len(recent_text),
        },
        "recent_pairs": recent_pairs,
        "older_turns": len(older),
        "older_user_turns": sum(1 for role, _ in older if role == "user"),
        "memory_items": memory_used,
        "digest_truncated": digest_truncated,
        "digest_dropped": digest_dropped,
        "recent_truncated": len(recent_text) < len(_render_turns(recent)),
    }
    return LayeredContext(
        memory_text=memory_text,
        session_digest=digest,
        recent_text=recent_text,
        trace=trace,
    )


def render_system_block(layered: LayeredContext) -> str:
    """把三层拼成一条 system 消息正文；三层皆空时返回空串."""
    parts: list[str] = []
    if layered.memory_text:
        parts.append(layered.memory_text)
    if layered.session_digest:
        parts.append(layered.session_digest)
    if not parts:
        # 只有 recent_text 时不额外插 system 消息：最近轮次本来就以原始消息
        # 形式进提示，再复制一遍会让模型看到两份互相竞争的历史。
        return ""
    return SYSTEM_BLOCK_HEADER + "\n\n" + "\n\n".join(parts)
