"""Agent 运行时审计 trace（一次问答里"用没用什么能力"的可复核记录）.

trace 只承载**决策与计量**，不承载任何原文：技能名与版本、各层字符数、
记忆条数、审阅结论。它随 SSE 下发一次、并随 `assistant_meta` 落库，
回答"这条回答当时走的是哪套能力"这个问题时不必再翻日志。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

__all__ = ["AgentTrace"]


@dataclass
class AgentTrace:
    enabled: bool = True
    skill: dict | None = None
    context_layers: dict | None = None
    memory: dict | None = None
    review: dict | None = None
    injection: dict | None = None
    fallback: list[str] = field(default_factory=list)

    def mark_skill(self, audit: dict | None) -> None:
        self.skill = audit

    def mark_layers(self, audit: dict | None) -> None:
        self.context_layers = audit

    def mark_memory(self, audit: dict | None) -> None:
        self.memory = audit

    def mark_review(self, audit: dict | None) -> None:
        self.review = audit

    def mark_injection(self, *, skill_text: str = "", layer_text: str = "") -> None:
        """记录**实际注入文本**的指纹（只存 hash 与字符数，不存正文）.

        对照 DeepSeek Harness 的「every run is traceable」：一次运行里模型看到的
        每一次 context injection 都应可追溯，否则同问不同答时无法归因。只存
        sha256 既能证明"这轮到底注入了什么"，又不会把记忆/会话原文复制进审计
        快照（审计表不该成为第二个内容副本）。
        """

        def _fingerprint(text: str) -> dict | None:
            if not text:
                return None
            return {
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                "chars": len(text),
            }

        self.injection = {
            "skill": _fingerprint(skill_text),
            "layers": _fingerprint(layer_text),
        }

    def mark_fallback(self, where: str, error: str) -> None:
        """记录一次"能力失效已回退"，只留能力名与错误类型，不留堆栈."""
        self.fallback.append(f"{where}:{error}")

    @property
    def empty(self) -> bool:
        return not any(
            (self.skill, self.context_layers, self.memory, self.review, self.injection, self.fallback)
        )

    def as_dict(self) -> dict:
        return {
            "skill": self.skill,
            "context_layers": self.context_layers,
            "memory": self.memory,
            "review": self.review,
            "injection": self.injection,
            "fallback": list(self.fallback),
        }
