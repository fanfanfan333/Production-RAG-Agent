"""Deterministic request decomposition for the central coordinator.

The model is the LangChain "supervisor + subagents" shape: one coordinator
decides which specialists a request needs, then merges their *real* outputs.
Classification here is intentionally deterministic and bounded:

* explicit ``kind`` wins (the UI may offer a mode switch);
* otherwise strong, bilingual markers select code / web-research / documents;
* absent any marker, the historical RAG path (document QA) remains the default.

The planner never claims an LLM reasoned about the request. It is a small,
testable routing policy, and the LLM still owns the answer inside each
specialist. Keeping decomposition deterministic also makes "same request ⇒ same
fan-out" true, which is required for the before/after evaluation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TaskKind = Literal["document", "code", "research"]

#: Hard fan-out ceiling. Each child can cost a model call or an isolated
#: container; an unbounded plan is a denial-of-service vector, not a feature.
MAX_SUBTASKS = 4

_EXPLICIT: dict[str, TaskKind] = {
    "document": "document",
    "rag": "document",
    "doc": "document",
    "knowledge": "document",
    "code": "code",
    "research": "research",
    "web": "research",
    "search": "research",
}

_CODE_PATTERNS = (
    r"\bcode\b",
    r"\brepo(?:sitory)?\b",
    r"\bpull request\b",
    r"\brefactor\b",
    r"\bunit test\b",
    r"\bpytest\b",
    r"代码",
    r"仓库",
    r"修复.{0,12}(?:缺陷|bug|问题)",
    r"改.{0,8}代码",
    r"运行测试",
    r"提交.{0,8}(?:改动|补丁)",
)

_RESEARCH_PATTERNS = (
    r"\bweb search\b",
    r"\bsearch the web\b",
    r"\bresearch\b",
    r"\blatest news\b",
    r"联网",
    r"上网",
    r"搜索",
    r"网页",
    r"最新(?:消息|新闻|进展|资料)",
    r"查一下.{0,12}(?:官网|新闻|资料)",
)

_DOCUMENT_PATTERNS = (
    r"文档",
    r"知识库",
    r"资料库",
    r"上传的",
    r"这几份",
    r"引用",
    r"合同",
    r"报表",
    r"\bdocument(?:s)?\b",
    r"\bknowledge base\b",
    r"\bcitation(?:s)?\b",
)


@dataclass(frozen=True)
class SubTask:
    """One bounded unit of work handed to a specialist."""

    kind: TaskKind
    objective: str
    timeout_seconds: float
    depends_on: tuple[TaskKind, ...] = ()


@dataclass(frozen=True)
class Plan:
    """The coordinator's decomposition plus its human-readable reason."""

    subtasks: tuple[SubTask, ...]
    reason: str
    default_document: bool = False

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(item.kind for item in self.subtasks)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def plan_request(
    query: str,
    *,
    kind: str = "auto",
    document_timeout: float = 180.0,
    research_timeout: float = 30.0,
    code_timeout: float = 300.0,
) -> Plan:
    """Return a bounded, deterministic plan for ``query``.

    Raises ``ValueError`` for an unknown explicit kind or an empty objective.
    A document-only request is still a plan (one child), so every task follows
    the same persistence / ownership / cancellation machinery.
    """
    objective = " ".join((query or "").split())
    if not objective:
        raise ValueError("任务目标不能为空")

    selected = _EXPLICIT.get((kind or "auto").strip().lower())
    if kind and (kind or "").strip().lower() != "auto" and selected is None:
        raise ValueError(f"未知任务类型: {kind}")

    if selected is not None:
        kinds: list[TaskKind] = [selected]
        reason = f"caller selected {selected}"
    else:
        wants_code = _matches(_CODE_PATTERNS, objective)
        wants_research = _matches(_RESEARCH_PATTERNS, objective)
        wants_document = _matches(_DOCUMENT_PATTERNS, objective)
        kinds = []
        if wants_document:
            kinds.append("document")
        if wants_research:
            kinds.append("research")
        if wants_code:
            kinds.append("code")
        if not kinds:
            kinds = ["document"]
            reason = "no specialist marker; preserve the document-QA default"
        else:
            reason = "deterministic markers: " + ", ".join(kinds)

    # The planner emits at most one child per specialist. Repeating the same
    # query N times would multiply cost without adding information.
    kinds = list(dict.fromkeys(kinds))
    if len(kinds) > MAX_SUBTASKS:
        raise ValueError(f"子任务数量超过上限 {MAX_SUBTASKS}")

    timeouts = {
        "document": document_timeout,
        "research": research_timeout,
        "code": code_timeout,
    }
    subtasks = tuple(
        SubTask(kind=item, objective=objective, timeout_seconds=float(timeouts[item]),
                depends_on=tuple(k for k in kinds if k != "code") if item == "code" else ())
        for item in kinds
    )
    return Plan(subtasks=subtasks, reason=reason, default_document=selected is None and not (wants_code or wants_research or wants_document))
