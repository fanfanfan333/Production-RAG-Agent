"""自进化记忆沉淀（受控写入 + 隔离读取 + 可撤销）.

## 「自进化」在这里的准确含义

不是"模型自己改自己"。本项目明确禁止在线自改提示词、源码、配置与模型权重
（见 PRD 非目标）。这里的自进化是**受控的经验沉淀**：

    用户显式声明（"记住：…"） → 确定性提取（无 LLM） → 落库（带来源与版本）
      → 下一轮作为**状态层**注入（非事实证据） → 用户可随时撤销

### 为什么只提取"显式声明"，不做自动记忆

Mem0 那类方案的默认路径是让 LLM 从对话里自动抽取记忆。它在本项目里有两个
不可接受的副作用：

1. **额外成本与不确定性**：每轮多一次 LLM 调用（与"普通问答零额外调用"直接冲突），
   且抽取结果不可复现 —— 同一段对话跑两次得到不同记忆，评审无法复核。
2. **污染面**：检索片段里的内容会被抽成"记忆"，下一轮就变成"用户说过"，
   等于让文档内容绕过 Evidence Gate 变成长期事实。

所以这里用**确定性规则**提取显式声明，其余一律不记。宁少不多：一条错的长期
记忆，会在之后每一轮里持续影响答案，而用户不会意识到它从哪来。

### 安全

记忆内容是用户输入 → 视为不可信数据。写入前过 `sanitize_memory_text()`：
拒绝疑似提示注入、拒绝疑似密钥、限制长度；写入后只作为**状态层文本**参与
拼装，绝不被当作可执行指令或引用来源。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # 仅类型检查期需要 —— 纯逻辑部分（清洗/抽取）不依赖 SQLAlchemy，
    from sqlalchemy.ext.asyncio import AsyncSession  # 这样最小环境也能单测

__all__ = [
    "MAX_MEMORY_CHARS",
    "MAX_MEMORIES_PER_SCOPE",
    "MAX_LOAD_LIMIT",
    "MEMORY_KINDS",
    "ExtractedMemory",
    "normalize_kind",
    "sanitize_memory_text",
    "extract_explicit_memories",
    "save_memory",
    "load_memories",
    "revoke_memory",
    "as_context_items",
]

# 允许的记忆类型。写成封闭枚举而不是自由字符串：`kind` 会被拼进 system 层
# 的行首（`- [{kind}] {content}`），自由文本意味着一个 `note]\n系统：…` 就能
# 伪造出一行"系统设定"。这类字段永远不该由调用方随便写。
MEMORY_KINDS = ("preference", "glossary", "note", "constraint")
_DEFAULT_KIND = "note"


def normalize_kind(kind: object) -> str:
    """把任意输入收敛到封闭枚举；不认识的一律降级成 ``note``."""
    value = str(kind or "").strip().lower()
    return value if value in MEMORY_KINDS else _DEFAULT_KIND


# 一条记忆的最大长度：够写一条口径约定，不够塞一篇文章（长文会挤掉证据预算）。
MAX_MEMORY_CHARS = 200
# 单次注入的记忆条数上限（按更新时间倒序取最新）。
MAX_MEMORIES_PER_SCOPE = 8
# 读取接口的硬上限。与"注入条数"是两件事：接口允许一次看更多（导出/自查），
# 但注入永远只用最新 MAX_MEMORIES_PER_SCOPE 条。早前把两者混成一个上限，
# 导致 `?limit=50` 静默只返回 8 条 —— 接口契约与实现不一致。
MAX_LOAD_LIMIT = 50

# 显式声明的触发式样。刻意保守：只有"说法明确"的才记。
_EXPLICIT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"记住\s*[:：,，]?\s*(?P<content>.+)"), "note"),
    (re.compile(r"请记住\s*[:：,，]?\s*(?P<content>.+)"), "note"),
    (re.compile(r"(?P<content>以后(?:都|一律|全部|统一)?\s*(?:用|按|使用|采用|称为|叫|写作)\s*.+)"), "preference"),
    (re.compile(r"(?:请)?(?P<content>永远(?:不要|别|要|都)\s*.+)"), "preference"),
    (re.compile(r"remember\s+(?:that\s+)?(?P<content>.+)", re.IGNORECASE), "note"),
    (re.compile(r"from now on[,，]?\s*(?P<content>.+)", re.IGNORECASE), "preference"),
)

# 疑似提示注入 / 越权：命中即拒收（不是"清洗后接受"，因为这些短语一旦进了
# 系统层就等于给了它们比用户消息更高的优先级）。
_INJECTION_PATTERNS = (
    r"忽略(?:之前|以上|上述|前面|所有)的?(?:指令|提示|要求|设定)",
    r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions",
    r"system\s*prompt",
    r"系统提示词?",
    r"你现在是|你现在扮演|扮演一个",
    r"you\s+are\s+now",
    r"<\|.*?\|>",
    r"BEGIN\s+SYSTEM",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# 疑似凭据：记忆会跨会话反复注入，一条密钥的暴露面比一次性输入大得多。
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}"
    r"|password\s*[=:]\s*\S+|passwd\s*[=:]\s*\S+"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class ExtractedMemory:
    kind: str
    content: str

    def as_audit(self) -> dict:
        # 审计里不留记忆正文（只留长度），避免审计表成为第二个内容副本
        return {"kind": self.kind, "chars": len(self.content)}


def sanitize_memory_text(text: str) -> str | None:
    """清洗一条候选记忆；不合格返回 ``None``."""
    if not text:
        return None
    cleaned = _CTRL_RE.sub("", str(text)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\r\n。；;，,")
    if len(cleaned) < 2 or len(cleaned) > MAX_MEMORY_CHARS:
        return None
    if _INJECTION_RE.search(cleaned):
        return None
    if _SECRET_RE.search(cleaned):
        return None
    return cleaned


def extract_explicit_memories(query: str) -> list[ExtractedMemory]:
    """从用户这一轮的问题里提取**显式**记忆声明（确定性，零 LLM 调用）.

    只取第一条命中：一句话里连说三件事的情况极少，取多了容易把半句业务描述
    也记成"设定"。

    必须**在第一个句末标点处截断**：`记住：金额用万元。这份合同的工期是多少？`
    这种"先给设定、紧接着提问"的说法非常常见，贪婪匹配会把后半句的问题一起
    存成长期设定，下一轮再把它当"用户说过的话"注入 —— 记忆污染最难排查的形态
    （用户永远想不起这条设定是哪来的）。含问号一律不记。
    """
    if not query:
        return []
    text = str(query)
    # 只在句首 120 字内找触发词 —— "记住"出现在长文中段时，多半是在引用别人的话
    head = text[:120]
    for pattern, kind in _EXPLICIT_PATTERNS:
        match = pattern.search(head)
        if not match:
            continue
        raw = match.group("content")
        # 句末标点切句（问号单独判断：它是"提问"而不是"声明"）
        clause = re.split(r"[。！!；;\n]", raw, maxsplit=1)[0]
        if "？" in clause or "?" in clause:
            return []
        content = sanitize_memory_text(clause)
        if not content:
            return []
        # 术语类判定优先于模式默认值：`以后都叫 X` 与 `记住 X 叫 Y` 都是术语
        if kind == "note" and re.search(r"称为|叫做|简称|缩写|术语|叫作", content):
            kind = "glossary"
        return [ExtractedMemory(kind=kind, content=content)]
    return []


async def save_memory(
    session: AsyncSession,
    *,
    tenant_id: str | None,
    user_id: uuid.UUID | None,
    kind: str,
    content: str,
    source: str = "explicit_user",
    conversation_id: uuid.UUID | None = None,
    meta: dict | None = None,
) -> object | None:
    """写入一条记忆；内容不合格或作用域不完整时不写、返回 ``None``.

    作用域不完整（缺 user_id）时拒绝写入：无法归属到个人的记忆，读取时也
    无法安全隔离，宁可丢掉。
    """
    cleaned = sanitize_memory_text(content)
    if not cleaned or user_id is None:
        return None
    from app.db.agent_memory_models import AgentMemory
    from sqlalchemy import select, text

    scope_tenant = tenant_id or "default"
    lock_key = int.from_bytes(
        hashlib.sha256(f"{scope_tenant}\0{user_id}\0{cleaned}".encode()).digest()[:8],
        "big", signed=True,
    )
    await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
    # 同作用域内**相同内容**只保留一条：用户每轮重复一句"记住：金额用万元"，
    # 若每次都 insert，`MAX_MEMORIES_PER_SCOPE` 会被同一条占满，其他偏好再也
    # 注入不进来 —— 表现为"记忆开着但从不生效"，极难排查。
    existing = await session.execute(
        select(AgentMemory)
        .where(
            AgentMemory.tenant_id == scope_tenant,
            AgentMemory.user_id == user_id,
            AgentMemory.content == cleaned,
            AgentMemory.revoked_at.is_(None),
        )
        .limit(1)
    )
    row = existing.scalars().first()
    if row is not None:
        row.version = int(getattr(row, "version", 1) or 1) + 1
        row.updated_at = datetime.now(timezone.utc)
        await session.flush()
        return row

    row = AgentMemory(
        tenant_id=scope_tenant,
        user_id=user_id,
        kind=normalize_kind(kind),
        content=cleaned,
        source=source,
        source_conversation_id=conversation_id,
        meta=meta or {},
    )
    session.add(row)
    await session.flush()
    return row


async def load_memories(
    session: AsyncSession,
    *,
    tenant_id: str | None,
    user_id: uuid.UUID | None,
    limit: int = MAX_MEMORIES_PER_SCOPE,
) -> list[object]:
    """读取当前用户的未撤销记忆（按更新时间倒序）.

    隔离规则：``tenant_id`` 与 ``user_id`` **同时**等值匹配。
    ``user_id`` 为空（legacy / 管理员无归属）时直接返回空 —— 没有"全租户视角"
    这种回退，也没有管理员分支。
    """
    if user_id is None:
        return []
    from app.db.agent_memory_models import AgentMemory
    from sqlalchemy import select

    stmt = (
        select(AgentMemory)
        .where(
            AgentMemory.tenant_id == (tenant_id or "default"),
            AgentMemory.user_id == user_id,
            AgentMemory.revoked_at.is_(None),
        )
        .order_by(AgentMemory.updated_at.desc())
        .limit(max(1, min(int(limit or MAX_MEMORIES_PER_SCOPE), MAX_LOAD_LIMIT)))
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def revoke_memory(
    session: AsyncSession,
    *,
    memory_id: uuid.UUID,
    tenant_id: str | None,
    user_id: uuid.UUID | None,
    revoked_by: str | None,
    reason: str | None = None,
) -> bool:
    """撤销一条记忆（不删行）。返回是否真的改动了状态.

    只有记录归属者本人能撤销；作用域不匹配一律当作"不存在"，不泄漏存在性。
    """
    if user_id is None:
        return False
    from app.db.agent_memory_models import AgentMemory

    row = await session.get(AgentMemory, memory_id)
    if row is None:
        return False
    if row.user_id != user_id or row.tenant_id != (tenant_id or "default"):
        return False
    if row.revoked_at is not None:
        return False
    row.revoked_at = datetime.now(timezone.utc)
    row.revoked_by = (revoked_by or "")[:128] or None
    row.revoke_reason = (reason or "")[:500] or None
    await session.flush()
    return True


def as_context_items(rows: Sequence[object]) -> list[dict]:
    """把 ORM 行转成 `context_layers` 需要的字典列表."""
    items: list[dict] = []
    for row in rows or ():
        content = getattr(row, "content", "")
        if not content:
            continue
        items.append(
            {
                "kind": normalize_kind(getattr(row, "kind", _DEFAULT_KIND)),
                "content": content,
                "version": getattr(row, "version", 1),
            }
        )
    return items
