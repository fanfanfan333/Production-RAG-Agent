"""
第 12 环：LLM 输入前的**最终校验** + 引用溯源快照（设计 §1 环节 12 / 决策 16）.

本模块把"检索之后、喂给 LLM 之前"的那一次对象级校验集中到一处：

    context_blocks ──逐个 allows(pred, view)──▶ {保留的块, 被剔除的块}
                                                  │
                        ┌─────────────────────────┼─────────────────────────┐
                        ▼                         ▼                         ▼
                   全剔除 → 拒答            部分剔除 → 部分回答        无剔除 → 正常
                   （不生成任何内容）        （不提示存在性）          （附引用快照）

为什么必须在这里再校验一次
──────────────────────────
第 7 环（检索前下推）与第 11 环（检索后 PG 复核）保护的是"候选集"，但一个块从
候选集走到"真正进入 LLM prompt"之间还隔着精排、父块回填、Vision 看图、压缩等多步。
任何一步如果是"从另一条路取来的正文"，前两环都拦不住。把校验钉在**LLM 输入的
最后一道门**，是"图文不一致"（A5：图看不了但字还能搜到）的最后一道保险。

三段降级（A10）
───────────────
- **全剔除** → :data:`NO_EVIDENCE_ANSWER`（与"证据不足"逐字相同 ⇒ 不泄露"是权限
  挡住的还是本来就没有"）。
- **部分剔除** → 正常回答，**不追加任何"因权限未包含"的提示**（存在性本身即信息，
  且可被批量探测利用）。需要提示时由调用方显式传 ``announce_partial=True``。
- **仅图片被剔除、文本可用** → 文本回答；被剔除的图片块不进上下文、不进 sources
  （LLM 因此无从描述它）。

`sources` 权限快照
──────────────────
每条引用新增 ``permission_snapshot`` 子对象，**只存** ``scope_fingerprint``（哈希），
**不存** ``project_ids`` / ``department_id`` / ``clearance`` 明文（共享知识 18）。
快照的用途是事后排查与"Scope 变了要重算"的判断，**不是**用来做判定 ——
判定永远重新走 :func:`allows`。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.db.security_models import (
    OBJECT_TYPE_DOC,
    OBJECT_TYPE_IMAGE,
    make_object_id,
    object_type_from_content_type,
)
from app.services.nodes.evidence_gate import REFUSAL_ANSWER
from app.services.security_policy import (
    ObjectACLView,
    ScopePredicate,
    allows,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: 全剔除时的回答 —— 与"证据不足 / 模型主动拒答"**逐字相同**（不区分原因 = 不泄露）。
NO_EVIDENCE_ANSWER = REFUSAL_ANSWER

#: 部分剔除时的可选提示。**默认不追加**（存在性泄露），仅在 ``announce_partial=True`` 时使用。
PARTIAL_NOTICE = "部分内容因权限限制未包含。"

#: 图片被剔除、文本可用时的可选提示（同样默认不追加）。
IMAGE_DROPPED_NOTE = "部分图片因权限限制未展示。"

#: 审计 stage 取值（与设计 §15-13 的 action 后缀一致）
STAGE_FINAL_CHECK = "final_check"
STAGE_CITATION_OPEN = "citation_open"

#: 引用快照里 ``scope_fingerprint`` 的键名
SNAPSHOT_KEY = "permission_snapshot"


# ═══════════════════════════════════════════════════════════════════════════════
# chunk → 视图键 / object_id
# ═══════════════════════════════════════════════════════════════════════════════


def chunk_key(chunk: Any) -> str:
    """
    检索产物 → ``load_document_view_index`` 的键.

    图片块用 ``img:{image_id}``；其余（text/table/code，含 OCR 派生）用 ``ci:{chunk_index}``。
    """
    content_type = str(getattr(chunk, "content_type", "text") or "text").lower()
    image_id = getattr(chunk, "image_id", None)
    if content_type == "image" and image_id:
        return f"img:{image_id}"
    return f"ci:{int(getattr(chunk, 'chunk_index', 0) or 0)}"


def derive_parent_object_id(
    document_id: str, content_type: str, image_id: str | None
) -> str:
    """
    由块的形状推导其 ``parent_object_id``（与 :func:`security_cascade.build_object_rows` 同源）.

    - 图片本体 → 父 = 文档
    - OCR 派生（``content_type != 'image'`` 但 ``image_id`` 非空）→ 父 = **源图片对象**
    - 普通文本块 → 父 = 文档
    """
    ctype = str(content_type or "text").lower()
    if image_id and ctype != "image":
        return make_object_id(str(document_id), str(image_id), object_type=OBJECT_TYPE_IMAGE)
    return str(document_id)


def object_type_of_chunk(content_type: str | None, image_id: str | None) -> str:
    ctype = str(content_type or "text").lower()
    if ctype == "image" and image_id:
        return OBJECT_TYPE_IMAGE
    return object_type_from_content_type(ctype)


# ═══════════════════════════════════════════════════════════════════════════════
# 权限快照
# ═══════════════════════════════════════════════════════════════════════════════


def build_permission_snapshot(
    view: ObjectACLView | None,
    pred: ScopePredicate | None,
    *,
    object_id: str,
    object_type: str,
    parent_object_id: str | None = None,
    issued_at: datetime | None = None,
) -> dict:
    """
    构造引用的 ``permission_snapshot``（**只含指纹，不含明文权限属性**）.

    即使 ``view`` 缺失也返回一个最小快照（``object_id`` + 指纹），保证"每条引用都带
    快照"这一契约；缺失的密级 / 同步状态以 ``None`` / ``"unknown"`` 表达。
    """
    created = issued_at or datetime.now(timezone.utc)
    return {
        "object_id": object_id,
        "object_type": object_type,
        "parent_object_id": parent_object_id,
        "effective_security_level": (
            view.effective_security_level if view is not None else None
        ),
        "visibility_mode": view.visibility_mode if view is not None else "tier",
        "acl_sync_state": view.acl_sync_state if view is not None else "unknown",
        # 只存哈希，不存明文（共享知识 18）
        "scope_fingerprint": (pred_scope_fingerprint(pred) if pred is not None else None),
        "issued_at": created.isoformat(),
    }


def pred_scope_fingerprint(pred: ScopePredicate | None) -> str | None:
    """
    用五维 IR 现算一个稳定指纹（与 ``UserScope.scope_fingerprint`` 同源族，但独立）.

    快照里带指纹是为了排查"这条引用是在哪个 Scope 下产出的"，**不参与判定**。
    """
    if pred is None:
        return None
    import hashlib

    def _s(values: Any) -> str:
        if values is None:
            return "all"
        items = sorted(str(v) for v in values)
        if not items:
            return "none"
        return "s:" + hashlib.sha1(",".join(items).encode("utf-8")).hexdigest()[:12]

    parts = [
        "pred1",
        f"u:{pred.user_id or '-'}",
        f"T:{_s(pred.tenant_ids)}",
        f"O:{_s(pred.owns_tenant_ids)}",
        f"d:{pred.department_id or '-'}",
        f"w:{'1' if pred.tenant_wide else '0'}",
        f"c:{pred.clearance}",
        f"p:{_s(pred.project_ids)}",
        f"a:{_s(pred.principals)}",
        f"s:{'1' if pred.strict else '0'}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


# ═══════════════════════════════════════════════════════════════════════════════
# 过滤结果
# ═══════════════════════════════════════════════════════════════════════════════

STATUS_CLEAN = "clean"            # 无剔除（或没有块）
STATUS_PARTIAL = "partial"        # 部分剔除
STATUS_ALL_DROPPED = "all_dropped"  # 全剔除


@dataclass
class FilterOutcome:
    """一次对象级过滤的结果（纯数据，便于断言与审计）。"""

    allowed: list = field(default_factory=list)
    dropped: list = field(default_factory=list)      # [(chunk, Decision)]
    status: str = STATUS_CLEAN
    total: int = 0
    scalar_doc_views: dict[str, ObjectACLView] = field(default_factory=dict)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    @property
    def allowed_count(self) -> int:
        return len(self.allowed)

    @property
    def all_dropped(self) -> bool:
        return self.status == STATUS_ALL_DROPPED

    @property
    def partial(self) -> bool:
        return self.status == STATUS_PARTIAL


def _status_of(total: int, allowed: int) -> str:
    if total <= 0:
        return STATUS_CLEAN
    if allowed <= 0:
        return STATUS_ALL_DROPPED
    if allowed < total:
        return STATUS_PARTIAL
    return STATUS_CLEAN


def filter_chunks_by_acl(
    chunks: Sequence[Any],
    pred: ScopePredicate,
    view_index: Mapping[str, Mapping[str, ObjectACLView]],
    *,
    materialized: Mapping[str, bool] | None = None,
) -> FilterOutcome:
    """
    逐个对象 :func:`allows` 过滤检索产物（**纯函数**，视图索引由调用方提供）.

    Args:
        chunks:      检索产物（``RetrievedChunk`` 或任何带 ``document_id`` /
                     ``content_type`` / ``chunk_index`` / ``image_id`` 的对象）。
        pred:        五维 IR。
        view_index:  ``{document_id: {chunk_key: ObjectACLView}}``（见
                     :func:`security_cascade.load_document_view_index`）。
        materialized: ``{document_id: bool}``。缺省视为 ``True``。
                     - ``True``：对象行缺失 ⇒ **fail-closed 丢弃**（该文档对象权限是
                       权威且完整的，没有行就是"不该有"）。
                     - ``False``：该文档**从未物化**（T1 回填未覆盖）⇒ 缺行**回退允许**，
                       由文档级判定兜底，避免存量文档集体不可见（功能退化）。

    判定**只**走 :func:`allows`（第 7 / 11 / 12 环同源），此处不写第二个判定点。
    """
    outcome = FilterOutcome(scalar_doc_views={})
    outcome.total = len(chunks)
    materialized = materialized or {}

    for chunk in chunks:
        doc_id = str(getattr(chunk, "document_id", "") or "")
        doc_index = view_index.get(doc_id) or {}
        view = doc_index.get(chunk_key(chunk))

        if view is None:
            if materialized.get(doc_id, True):
                # 已物化的文档却没有这一行 → fail-closed
                from app.services.security_policy import Decision

                decision = Decision(False, "missing_object_view", "tenant")
                outcome.dropped.append((chunk, decision))
                continue
            # 未物化 → 回退允许（文档级判定已在别处生效）
            outcome.allowed.append(chunk)
            continue

        decision = allows(pred, view)
        if decision.allowed:
            outcome.allowed.append(chunk)
        else:
            outcome.dropped.append((chunk, decision))

    outcome.status = _status_of(outcome.total, outcome.allowed_count)
    return outcome


# ═══════════════════════════════════════════════════════════════════════════════
# 最终校验节点（异步：解析视图 + 过滤 + 审计）
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FinalCheckResult:
    """第 12 环的产出（喂给 master_graph 的 generate / refuse 分支消费）。"""

    outcome: FilterOutcome
    view_index: dict[str, dict[str, ObjectACLView]] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.outcome.status

    @property
    def all_dropped(self) -> bool:
        return self.outcome.all_dropped

    @property
    def dropped_count(self) -> int:
        return self.outcome.dropped_count

    @property
    def answer_or_none(self) -> str | None:
        """全剔除 → 统一拒答文案；否则 ``None``（由 generate 正常回答）。"""
        return NO_EVIDENCE_ANSWER if self.outcome.all_dropped else None

    def append_partial_notice(self, answer: str, *, announce_partial: bool = False) -> str:
        """可选追加"部分内容未包含"提示（默认关闭，避免存在性泄露）。"""
        if announce_partial and self.outcome.partial:
            return f"{answer}\n\n{PARTIAL_NOTICE}"
        return answer


async def run_final_check(
    chunks: Sequence[Any],
    pred: ScopePredicate,
    *,
    session: Any = None,
    user_id: Any = None,
    username: str | None = None,
    stage: str = STAGE_FINAL_CHECK,
) -> FinalCheckResult:
    """
    第 12 环节点体：解析对象视图 → 逐块 :func:`allows` → 审计剔除 → 返回结果.

    每个被剔除的对象写一条 ``acl.drop.<stage>`` 审计（best-effort）。
    """
    from app.services.security_cascade import (
        document_is_materialized,
        load_document_view_index,
    )

    doc_ids = {str(getattr(c, "document_id", "") or "") for c in chunks}
    doc_ids.discard("")

    view_index: dict[str, dict[str, ObjectACLView]] = {}
    materialized: dict[str, bool] = {}
    for doc_id in doc_ids:
        try:
            view_index[doc_id] = await load_document_view_index(doc_id, session=session)
            materialized[doc_id] = await document_is_materialized(doc_id, session=session)
        except Exception:      # noqa: BLE001 — 解析失败 → 该文档按"未物化"回退（不误伤）
            logger.exception("run_final_check: view load failed for doc %s", doc_id)
            view_index[doc_id] = {}
            materialized[doc_id] = False

    outcome = filter_chunks_by_acl(
        chunks, pred, view_index, materialized=materialized
    )

    if outcome.dropped:
        await _audit_drops(
            outcome, pred, stage=stage, user_id=user_id, username=username
        )

    logger.info(
        "master_final_check: %d/%d block(s) kept (status=%s) fp=%s",
        outcome.allowed_count, outcome.total, outcome.status,
        pred_scope_fingerprint(pred),
    )

    return FinalCheckResult(outcome=outcome, view_index=view_index)


async def audit_acl_drops(
    outcome: FilterOutcome,
    pred: ScopePredicate,
    *,
    stage: str = STAGE_FINAL_CHECK,
    user_id: Any = None,
    username: str | None = None,
) -> None:
    """
    把一次 :func:`filter_chunks_by_acl` 的剔除结果写进审计（**best-effort**）.

    为什么必须公开（T5 上线前修复）：``filter_chunks_by_acl`` 是纯函数、**不写
    审计**，而唯一会写审计的 :func:`run_final_check` 在生产图里并没有接成节点。
    于是第 11 / 12 环（``context_builder.build_context`` 与
    ``build_multimodal_context``，都是真实生产入口）剔除过的对象**完全不留痕**
    —— 事后无法回答"这个用户为什么看不到这张图"，这是 PRD P0-10 的硬要求。

    因此把审计抽成这个公开函数，由**每一个生产调用点**在过滤之后立即调用。
    ``filter_chunks_by_acl`` 本身保持纯函数不变（现有单测零影响）。

    永不抛异常：审计写失败绝不能打断回答路径（调用方也不需要 try）。
    """
    try:
        from app.services.audit_service import record_acl_drop

        fp = pred_scope_fingerprint(pred)
        for chunk, decision in outcome.dropped:
            doc_id = str(getattr(chunk, "document_id", "") or "")
            object_id = _object_id_of_chunk(chunk)
            await record_acl_drop(
                stage,
                object_id=object_id,
                document_id=doc_id,
                reason=decision.reason,
                gate=decision.gate,
                user_id=user_id,
                username=username,
                scope_fingerprint=fp,
            )
    except Exception:      # noqa: BLE001 — best-effort：审计失败绝不打断回答
        logger.warning("audit_acl_drops: acl drop audit failed", exc_info=True)


#: 向后兼容别名（``run_final_check`` 内部与既有测试使用旧名）
_audit_drops = audit_acl_drops


def _object_id_of_chunk(chunk: Any) -> str | None:
    """尽力给出对象的稳定标识（有 point_id 用 point_id，否则退回 image_id / chunk 键）。"""
    doc_id = str(getattr(chunk, "document_id", "") or "")
    point_id = getattr(chunk, "point_id", None) or getattr(chunk, "id", None)
    content_type = str(getattr(chunk, "content_type", "text") or "text").lower()
    image_id = getattr(chunk, "image_id", None)
    object_type = object_type_of_chunk(content_type, image_id)
    try:
        if point_id:
            return make_object_id(doc_id, str(point_id), object_type=object_type)
    except ValueError:
        pass
    if object_type == OBJECT_TYPE_IMAGE and image_id:
        return make_object_id(doc_id, str(image_id), object_type=OBJECT_TYPE_IMAGE)
    if object_type == OBJECT_TYPE_DOC:
        return doc_id or None
    return f"{doc_id}::{chunk_key(chunk)}" if doc_id else None


__all__ = [
    "FinalCheckResult",
    "FilterOutcome",
    "IMAGE_DROPPED_NOTE",
    "NO_EVIDENCE_ANSWER",
    "PARTIAL_NOTICE",
    "SNAPSHOT_KEY",
    "STAGE_CITATION_OPEN",
    "STAGE_FINAL_CHECK",
    "STATUS_ALL_DROPPED",
    "STATUS_CLEAN",
    "STATUS_PARTIAL",
    "audit_acl_drops",
    "build_permission_snapshot",
    "chunk_key",
    "derive_parent_object_id",
    "filter_chunks_by_acl",
    "object_type_of_chunk",
    "pred_scope_fingerprint",
    "run_final_check",
]
