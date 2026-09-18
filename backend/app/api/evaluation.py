"""
检索质量评测 API（持续监控 Recall / MRR / NDCG / 引用准确率）.

    POST /eval/run      — 提交金标集跑一轮评测（真实走检索链路）
    GET  /eval/history  — 最近若干轮评测摘要（**落库**，跨重启保留；看趋势、做回归对比）
    GET  /eval/metrics  — 当前运行期质量指标（含引用准确率）

与 /badcases/stats 的分工
────────────────────────
``/badcases/stats`` 里的 ``metrics`` 是**运行期健康度**：拒答率、引用校验
通过率、时延。它不需要标准答案，因此能 7×24 一直跑；但它测不出"找得准
不准" —— 拒答率低既可能是答得好，也可能是该拒的没拒。

本模块的三个端点回答后者，代价是**需要金标**：

    POST /eval/run  用人工标注的"这个问题应该召回哪几段"打分 →
                    Recall@K / MRR / NDCG@K / 引用准确率
    GET  /eval/history 留最近 20 轮，改了切分策略或换了 Embedding 之后
                    能直接对比"是变好了还是变坏了"

因此评测需要 ``audit.read`` 权限（与 Bad Case 队列同级）：它会按用例数
放大检索请求量，属于运维动作而非用户动作。

金标集的写法
────────────
``relevant`` 里可以混写三种标识，命中任一即判对（见
``evaluation.item_from_chunk``）：

    "6f1c...::17"            chunk 主键（document_id::chunk_index）
    "img-6f1c-p2-i1"         图片 image_id
    "年报.pdf::p3::2"        人类可读："文件名::页::第几张图"

第三种是特意支持的：让人写 UUID 是不现实的，标注成本决定了金标集
最终会不会被维护下去。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.user_models import User
from app.services.evaluation import (
    EvalCase,
    EvalSet,
    eval_history_persisted,
    evaluate,
    item_from_chunk,
    persist_eval_run,
)
from app.services.monitoring_service import metrics_snapshot
from app.services.permissions import require_permission
from app.services.retrieval_service import retrieve_chunks
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Evaluation"])

# 单次评测的规模上限 —— 评测是"按用例数放大检索请求"的重操作，
# 不设闸会让一个手滑提交的 5000 条金标把 Qdrant 与 Ollama 打满。
MAX_CASES_PER_RUN = 100
MAX_TOP_K = 50


# ── Schemas ──────────────────────────────────────────────────────────────────

class EvalCaseIn(BaseModel):
    """一条金标用例（入参）."""

    query: str = Field(min_length=1, max_length=1000)
    relevant: list[str] = Field(default_factory=list)
    grades: dict[str, float] | None = None
    modality: str = Field(default="text", pattern="^(text|table|image)$")
    note: str = ""


class EvalRunRequest(BaseModel):
    """一轮评测请求."""

    name: str = Field(default="ad-hoc", max_length=80)
    description: str = ""
    cases: list[EvalCaseIn] = Field(min_length=1, max_length=MAX_CASES_PER_RUN)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    collection_id: str | None = None


class EvalRunResponse(BaseModel):
    """一轮评测结果（聚合 + 逐条）."""

    eval_set: str
    total_cases: int
    scored_cases: int
    mrr: float | None
    map: float | None
    recall: dict[int, float | None]
    precision: dict[int, float | None]
    ndcg: dict[int, float | None]
    hit_rate: dict[int, float | None]
    by_modality: dict[str, dict]
    # 按"答案需要几条证据"分组（single_evidence / multi_evidence）。
    # 多证据切片是"相对分带会不会砍掉次要证据"的唯一观测点 —— 旧集 11 例
    # 全是单证据，这个维度无观测点，于是分带参数只能靠拍脑袋。
    evidence_slices: dict[str, dict] = Field(default_factory=dict)
    # 金标分数画像（分带安全上界的原料；阈值过滤开启时该值是上界而非真值）
    gold_ratio: dict = Field(default_factory=dict)
    citation: dict | None
    generated_at: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/eval/run",
    response_model=EvalRunResponse,
    status_code=status.HTTP_200_OK,
    summary="用金标集跑一轮检索质量评测",
)
async def run_evaluation(
    payload: EvalRunRequest,
    user: User = Depends(require_permission("audit.read")),
) -> EvalRunResponse:
    """
    按提交的金标集实际检索并打分.

    每条用例都会真实走一次 ``retrieve_chunks``（粗排 + 精排），因此耗时随
    用例数线性增长 —— 这是刻意的：评测必须跑**生产同一条**检索链路，另建
    一条"评测专用"的检索路径只会得到一份好看但不作数的数字。
    """
    ks = sorted({int(k) for k in payload.k_values if int(k) > 0}) or [1, 3, 5, 10]
    top_k = max(max(ks), payload.top_k)

    eval_set = EvalSet(
        name=payload.name,
        description=payload.description,
        cases=tuple(
            EvalCase(
                query=c.query,
                relevant=frozenset(c.relevant),
                grades=c.grades,
                modality=c.modality,
                note=c.note,
            )
            for c in payload.cases
        ),
    )

    # 检索作用域跟随调用者：普通用户只评自己可见的知识库，避免评测变成
    # 一条越权读取他人文档的旁路。三层隔离下锁定公司 + 部门 ACL，
    # 平台管理员跨公司评测但也**不含**他人的个人库。
    from app.services.tenancy import scope_for

    scope = scope_for(user)
    owner_id = str(scope.owner_id) if scope.owner_id else None
    tenant_id = scope.tenant_id
    department_id = scope.department_id
    collection_id = payload.collection_id

    async def _retrieve(query: str):
        chunks = await retrieve_chunks(
            query=query,
            top_k=top_k,
            owner_id=owner_id,
            collection_id=collection_id,
            tenant_id=tenant_id,
            user_department_id=department_id,
            tenant_wide=scope.tenant_wide,
            platform_wide=scope.platform_wide,
        )
        return [item_from_chunk(c) for c in chunks]

    report = await evaluate(_retrieve, eval_set, k_values=ks)
    # 落库：一次评测如果只活在进程内存里，重启即丢，等于没有基线。
    # 写历史失败不影响本次返回（persist_eval_run 内部已吞掉异常并记日志）。
    persisted = await persist_eval_run(
        report,
        run_by=user.username,
        tenant_id=tenant_id,
        collection_id=collection_id,
        top_k=top_k,
        description=payload.description,
    )
    logger.info(
        "eval_run: set=%s cases=%d by user=%s -> mrr=%s persisted=%s",
        payload.name, len(payload.cases), user.username, report.mrr, persisted,
    )
    return EvalRunResponse(
        eval_set=report.eval_set,
        total_cases=report.total_cases,
        scored_cases=report.scored_cases,
        mrr=report.mrr,
        map=report.map_score,
        recall=report.recall,
        precision=report.precision,
        ndcg=report.ndcg,
        hit_rate=report.hit_rate,
        by_modality=report.by_modality,
        evidence_slices=report.evidence_slices,
        gold_ratio=report.gold_ratio,
        citation=report.citation,
        generated_at=report.generated_at,
    )


@router.get(
    "/eval/history",
    summary="最近若干轮评测摘要（回归对比用）",
)
async def get_eval_history(
    limit: Annotated[int, Query(ge=1, le=20)] = 10,
    _user: User = Depends(require_permission("audit.read")),
) -> dict:
    """
    最新在前。改了切分 / 换 Embedding 之后，用它对比前后变化。

    数据源是 ``eval_runs`` 表（跨重启保留）；表不可用时自动回退到进程内
    历史，绝不会因为"读历史失败"把面板显示成"从没评测过"。
    """
    runs = await eval_history_persisted(limit=limit)
    return {"runs": runs, "total": len(await eval_history_persisted(limit=20))}


@router.get(
    "/eval/metrics",
    summary="当前运行期质量指标（含引用准确率）",
)
async def get_eval_metrics(
    _user: User = Depends(require_permission("audit.read")),
) -> dict:
    """
    把运行期指标与最近一次评测结果并排给出.

    两者要看在一起才有意义：引用校验通过率 100% 但 Recall@5 只有 0.4，
    说明系统"答得都对，但十道题里六道根本没找到材料" —— 这恰恰是单看
    任一指标都会漏掉的退化形态。
    """
    snap = metrics_snapshot()
    history = await eval_history_persisted(limit=1)
    return {
        "runtime": snap,
        "latest_eval": history[0] if history else None,
        "config": {
            "hybrid_search": get_settings().HYBRID_SEARCH_ENABLED,
            "reranker": get_settings().RERANKER_ENABLED,
            "citation_verifier": get_settings().CITATION_VERIFIER_ENABLED,
        },
    }


__all__ = ["router"]
