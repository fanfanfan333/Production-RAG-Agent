"""
Prompt-injection guard for RAG inputs and retrieved document context.

This module applies defense in depth rather than relying on a single model
instruction:

1. Normalize control / invisible characters before policy checks.
2. Score known instruction-hijacking patterns in user input.
3. Block high-confidence attempts before retrieval or LLM invocation.
4. Treat retrieved document text as untrusted data and mask embedded model
   control instructions before it is inserted into the LLM context window.

The guard is intentionally deterministic and auditable. It does not try to
classify ordinary questions as attacks; only direct model-control patterns are
blocked. New patterns should be added after reviewing audited attempts.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Unicode bidirectional / zero-width controls can hide an instruction from the
# operator while leaving it visible to the model. Remove them before matching.
_INVISIBLE_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_SPACE_RE = re.compile(r"[ \t]{2,}")

# A match in either language indicates a direct attempt to change the model's
# trust boundary. Patterns are narrowly scoped to avoid blocking questions
# such as "什么是提示词注入".
_HIGH_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(ignore|disregard|override|bypass)\b.{0,80}\b(previous|prior|above|system|developer|safety|instruction)s?\b", re.I),
    re.compile(r"\b(reveal|show|print|dump|repeat)\b.{0,80}\b(system prompt|developer message|hidden instruction|internal instruction)s?\b", re.I),
    re.compile(r"\b(act as|you are now|switch to|enter)\b.{0,60}\b(developer|system|root|administrator)\b", re.I),
    re.compile(r"(?:忽略|无视|绕过|覆盖).{0,40}(?:之前|上述|系统|开发者|安全|指令|规则)"),
    re.compile(r"(?:输出|展示|泄露|复述|打印).{0,40}(?:系统提示词|系统指令|开发者消息|隐藏指令|内部指令)"),
    re.compile(r"(?:现在你是|切换为|扮演).{0,30}(?:系统|开发者|管理员|root)"),
)

# Medium-risk patterns are allowed but flagged for audit. They are useful for
# security education and may be legitimate questions about this topic.
_MEDIUM_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(prompt injection|jailbreak|system prompt|developer message)\b", re.I),
    re.compile(r"(?:提示词注入|越狱|系统提示词|开发者消息)"),
)

# Document context is untrusted. These patterns are narrower than user-input
# blocking: a matching paragraph is redacted so an uploaded document cannot
# instruct the answer model to override its governing system message.
_DOCUMENT_INSTRUCTION_RE = re.compile(
    r"(?:\b(?:ignore|disregard|override|bypass)\b.{0,120}\b(?:instruction|system|prompt|rule)s?\b|"
    r"\b(?:reveal|show|print|dump)\b.{0,120}\b(?:system prompt|hidden instruction|developer message)s?\b|"
    r"(?:忽略|无视|绕过|覆盖).{0,80}(?:指令|系统|提示词|规则)|"
    r"(?:输出|展示|泄露|复述).{0,80}(?:系统提示词|隐藏指令|开发者消息))",
    re.I,
)


@dataclass(frozen=True)
class PromptSecurityResult:
    """Normalized input and detection result returned to the query endpoint."""

    normalized_text: str
    risk: str  # clean | suspicious | blocked
    reasons: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return self.risk == "blocked"


def normalize_text(value: str) -> str:
    """Canonicalize text for stable policy matching and safe logging."""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _INVISIBLE_RE.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return _SPACE_RE.sub(" ", normalized).strip()


def inspect_user_query(query: str) -> PromptSecurityResult:
    """Inspect one user query without invoking an LLM or external service."""
    text = normalize_text(query)
    high_matches = tuple(
        pattern.pattern for pattern in _HIGH_RISK_PATTERNS if pattern.search(text)
    )
    if high_matches:
        return PromptSecurityResult(text, "blocked", high_matches)

    medium_matches = tuple(
        pattern.pattern for pattern in _MEDIUM_RISK_PATTERNS if pattern.search(text)
    )
    if medium_matches:
        return PromptSecurityResult(text, "suspicious", medium_matches)
    return PromptSecurityResult(text, "clean", ())


def sanitize_document_context(text: str) -> tuple[str, bool]:
    """
    Mask likely model-control instructions in retrieved document *text*.

    Returns ``(sanitized_text, changed)``. Paragraph-level masking preserves
    normal factual content nearby and provides a visible marker to the model
    rather than silently joining unsafe instruction fragments together.
    """
    normalized = normalize_text(text)
    paragraphs = normalized.split("\n")
    changed = False
    safe: list[str] = []
    for paragraph in paragraphs:
        if _DOCUMENT_INSTRUCTION_RE.search(paragraph):
            safe.append("[已屏蔽：该文档段落包含疑似模型控制指令，不作为知识事实使用]")
            changed = True
        else:
            safe.append(paragraph)
    return "\n".join(safe), changed
