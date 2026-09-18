"""
RAG 检索质量评测（持续监控 Recall / MRR / NDCG / 引用准确率）.

为什么单独建这个模块
────────────────────
``monitoring_service`` 记录的是**运行期健康度**：拒答率多少、引用校验挂了
几条、哪段慢。它能回答"系统有没有在正常工作"，但回答不了一个更根本的
问题：**系统找得准不准**。

拒答率 5% 既可能是"证据充足、回答靠谱"，也可能是"检索根本没召回，索性
全拒了"。要区分这两者，必须有**带标准答案的检索评测**：

    Recall@K    标准答案里有多少被找回来了（找全了没）
    MRR         第一个命中答案平均排在第几位（排得前不前）
    NDCG@K      考虑了排序位置与相关性等级的加权收益
    引用准确率  答案标出的引用里，有多少真的支持该结论

这四项正好对应"检索 → 排序 → 生成 → 引用"四段，任一段退化都能被单独
看出来，而不是糊成一个"整体变差了"。

设计取舍
────────
- **纯函数、零 I/O**：指标计算不碰数据库不碰网络，可直接单测
  （tests/test_evaluation.py）。真正跑检索的是调用方传入的 ``retrieve``
  回调，本模块只负责"打分"。
- **相关性判定走别名**：同一段内容在不同阶段有不同叫法 —— Qdrant 里是
  ``(document_id, chunk_index)``，图片还有 ``image_id``，人类标注时更习惯
  ``"年报.pdf::p3::第2张图"``。因此每条检索结果带一组 aliases，
  命中任意一个即视为相关，避免"其实找对了但因为 ID 写法不同被判错"。
- **图文分开统计**：图片能否被检索到，与文本能否被检索到，是两件独立的事
  （图片靠 OCR/VLM 文本，质量天然低于正文）。混在一起算，图片召回的退化
  会被文本的高分掩盖。因此报告按 text / table / image 分别给 Recall。
- **宁可少判，不可错判**：无标准答案的用例（relevant 为空）不计入聚合，
  而不是按 0 分算 —— 那会把"没标注"误读成"检索失败"。

用法
────
    from app.services.evaluation import EvalCase, EvalSet, evaluate

    eval_set = EvalSet(cases=[
        EvalCase(query="ABX-300 的额定功率", relevant={"doc-1::12"}),
        EvalCase(query="架构图里有哪些模块",
                 relevant={"doc-1::p2::img1"}, modality="image"),
    ])
    report = await evaluate(retrieve, eval_set, k_values=(1, 3, 5, 10))
    report.summary()   # {"recall@5": 0.83, "mrr": 0.75, ...}
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable, Mapping, Sequence

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ── 检索结果表示 ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrievedItem:
    """
    一条被召回的内容（评测视角）.

    key          稳定主键，推荐 ``f"{document_id}::{chunk_index}"``
    aliases      同一内容的其它叫法：image_id、"文件名::p3::img2" 等
    content_type text | table | image —— 用于图文分开统计
    score        检索/精排分数（仅用于 NDCG 排序校验与日志，不参与相关性判定）
    """

    key: str
    aliases: tuple[str, ...] = ()
    content_type: str = "text"
    score: float = 0.0

    def matches(self, relevant: Iterable[str]) -> bool:
        """命中任意一个标准答案标识即视为相关."""
        needles = set(relevant or ())
        if not needles:
            return False
        if self.key in needles:
            return True
        return any(a in needles for a in self.aliases)


def item_from_chunk(chunk) -> RetrievedItem:
    """
    把检索层的 ``RetrievedChunk`` 适配成评测项（duck typing，避免循环导入）.

    别名集合是这条适配的价值所在：标注者几乎不可能背下 ``doc-uuid::17``
    这种主键，他们更可能写 ``image_id``，或者"年报.pdf 第 3 页第 2 张图"。
    三种写法都收进别名，金标集才写得下去、也才对得上。
    """
    get = lambda name, default=None: getattr(chunk, name, default)  # noqa: E731
    document_id = get("document_id") or ""
    chunk_index = get("chunk_index", 0)
    filename = get("filename") or ""
    page = get("page_number") or get("page") or 0
    position = get("position")
    image_id = get("image_id")

    aliases: list[str] = []
    if image_id:
        aliases.append(str(image_id))
    if filename and page:
        aliases.append(f"{filename}::p{page}")
        if position:
            aliases.append(f"{filename}::p{page}::{position}")
            aliases.append(f"{filename}::p{page}::第{position}张图")
            aliases.append(f"{filename}::p{page}::第{position}个表格")

    return RetrievedItem(
        key=f"{document_id}::{chunk_index}",
        aliases=tuple(aliases),
        content_type=str(get("content_type") or "text"),
        score=float(get("score") or 0.0),
    )


# ── 单查询指标（纯函数）─────────────────────────────────────────────────────

def _truncate(ranked: Sequence[RetrievedItem], k: int) -> list[RetrievedItem]:
    return list(ranked[:k]) if k and k > 0 else list(ranked)


def hit_at_k(ranked: Sequence[RetrievedItem], relevant: Iterable[str], k: int) -> float:
    """前 K 条里有没有至少一条命中（0 / 1）."""
    return 1.0 if any(i.matches(relevant) for i in _truncate(ranked, k)) else 0.0


def recall_at_k(
    ranked: Sequence[RetrievedItem], relevant: Iterable[str], k: int
) -> float | None:
    """
    前 K 条召回了标准答案的多少比例.

    无标准答案时返回 None（不计入聚合）—— "没标注"不等于"召回为 0"。

    ⚠️ 去重维度必须是**金标标识**，不能是"命中项的 key"。
    旧实现写成 ``{i.key for i in top if i.matches(needles)}``，于是：

      1) 一条金标被多个分块命中（父子块同时召回、图片与其 OCR 文本块同时
         召回）时，分子会累加多个不同的 ``i.key``，而分母只有一个标识 →
         **recall 会大于 1.0**，指标直接失去意义；
      2) 反向的错法同样存在：一个检索项同时覆盖两条金标（aliases 里同时
         带着两条标识）时，``i.key`` 只贡献 1，分母是 2 → 被误判成漏召回。

    因此改为"逐条金标问一次：它被召回了吗"，天然满足 0 ≤ recall ≤ 1。
    """
    needles = set(relevant or ())
    if not needles:
        return None
    top = _truncate(ranked, k)
    found = {n for n in needles if any(i.matches({n}) for i in top)}
    return len(found) / len(needles)


def precision_at_k(
    ranked: Sequence[RetrievedItem], relevant: Iterable[str], k: int
) -> float | None:
    """前 K 条中有多少比例是相关的（无标准答案或 K=0 时 None）."""
    needles = set(relevant or ())
    if not needles:
        return None
    top = _truncate(ranked, k)
    if not top:
        return None
    hit = sum(1 for i in top if i.matches(needles))
    return hit / len(top)


def reciprocal_rank(
    ranked: Sequence[RetrievedItem], relevant: Iterable[str]
) -> float | None:
    """第一条命中结果的排名倒数（1 / rank）；未命中为 0；无标准答案为 None."""
    needles = set(relevant or ())
    if not needles:
        return None
    for idx, item in enumerate(ranked, start=1):
        if item.matches(needles):
            return 1.0 / idx
    return 0.0


def average_precision(
    ranked: Sequence[RetrievedItem], relevant: Iterable[str]
) -> float | None:
    """
    AP：每次命中时的 precision 取平均（同时奖励"找得全"与"排得前"）.

    未命中任何一条 → 0.0；无标准答案 → None。
    """
    needles = set(relevant or ())
    if not needles:
        return None
    hits = 0
    cumulative = 0.0
    for idx, item in enumerate(ranked, start=1):
        if item.matches(needles):
            hits += 1
            cumulative += hits / idx
    return cumulative / len(needles) if hits else 0.0


def gold_score_profile(
    ranked: Sequence[RetrievedItem], relevant: Iterable[str]
) -> dict:
    """
    单条用例的"金标分数画像" —— 相对分带的判别力就藏在这三个量里.

    返回 ``{"best_score", "gold_scores", "min_gold_ratio"}``：

        best_score     头名（= 精排第 1 名）的分数，即分带公式里的 ``head``
        gold_scores    命中的每条金标 → 它的检索分
        min_gold_ratio 命中的金标里**最弱者** ÷ 头名

    为什么必须是"最弱者 ÷ 头名"
    ────────────────────────────
    分带判据是 ``c.score >= head.score × ratio``，它对**每一条**候选独立生效，
    所以决定"分带会不会砍掉这条金标"的，永远是**最弱的那条金标**：

        min_gold_ratio >= ratio   ⇒ 该用例的所有金标都能过带
        min_gold_ratio <  ratio   ⇒ 至少一条金标会被带砍掉 → 该用例 recall 掉分

    这正是旧金标集**结构上测不出**的东西：11 个用例的金标全是头名本身
    （``head`` 就是金标），于是 ``min_gold_ratio ≡ 1.0``，分带开到 1.0 也不掉分 ——
    "提高 ratio 不损召回"在该集上是**恒真命题**，无法作为调参依据。补了多证据
    用例（金标 2 条以上、且弱者明显弱于头名）之后，这个量才第一次有信息量。

    ⚠️ 使用限制（很重要，否则会被误读成"安全上界"）
    ────────────────────────────────────────────
    若本次检索**已经开着阈值过滤**，被带砍掉的那条金标根本不会出现在
    *ranked* 里 → 它不进 ``gold_scores`` → ``min_gold_ratio`` 被算**偏高**。
    因此本函数的值在"过滤开启"时只是**上界**，不能直接当作"ratio 还能调多高"
    的依据。要拿真实安全上界，必须在不带过滤的那一态测量（见
    ``scripts/run_eval_baseline.py`` 的 ``measure_band_headroom``）。

    全部未命中或 *ranked* 为空 → ``min_gold_ratio`` 为 None（无从判断，不猜）。
    """
    needles = set(relevant or ())
    scores: dict[str, float] = {}
    for n in sorted(needles):
        hit = next((i for i in ranked if i.matches({n})), None)
        if hit is not None:
            scores[n] = round(float(hit.score), 4)
    best = max((float(i.score) for i in ranked), default=None)
    if best is None or not scores:
        return {"best_score": None if best is None else round(best, 4),
                "gold_scores": scores, "min_gold_ratio": None}
    weakest = min(scores.values())
    return {
        "best_score": round(best, 4),
        "gold_scores": scores,
        "min_gold_ratio": round(weakest / best, 4) if best > 0 else None,
    }


def ndcg_at_k(
    ranked: Sequence[RetrievedItem],
    grades: Mapping[str, float] | None,
    k: int,
) -> float | None:
    """
    NDCG@K：按相关性等级（graded relevance）折算的排序收益.

    *grades* 为 {标识: 相关度}，缺省相关度 1.0。用 2^rel - 1 作增益，
    位置折损 log2(rank + 1)。理想排序（IDCG）由等级集合本身决定，
    因此结果恒 ∈ [0, 1]。无标准答案 → None。
    """
    if not grades:
        return None

    def _gain(rel: float) -> float:
        return (2.0 ** max(rel, 0.0)) - 1.0

    def _rel_of(item: RetrievedItem) -> float:
        if item.key in grades:
            return float(grades[item.key])
        for a in item.aliases:
            if a in grades:
                return float(grades[a])
        return 0.0

    top = _truncate(ranked, k)
    dcg = sum(_gain(_rel_of(i)) / math.log2(idx + 1) for idx, i in enumerate(top, start=1))
    ideal = sorted((float(v) for v in grades.values()), reverse=True)[: len(top) or None]
    idcg = sum(_gain(rel) / math.log2(idx + 1) for idx, rel in enumerate(ideal, start=1))
    if idcg <= 0:
        return None
    return dcg / idcg


# ── 引用准确率 ───────────────────────────────────────────────────────────────

def citation_prf(
    cited: Iterable[int], supporting: Iterable[int]
) -> dict:
    """
    引用准确率：答案标出的引用里有多少**真的支持**该结论.

    cited       答案中实际出现的 [Source N] 编号
    supporting  人工/金标认定确实支持该结论的编号

    返回 precision / recall / f1 与 tp / fp / fn。
    precision 在没有引用时为 None —— 一句没引用的话不该被算成"零准确"。
    """
    c, s = set(cited or ()), set(supporting or ())
    tp = len(c & s)
    fp = len(c - s)
    fn = len(s - c)
    precision = tp / len(c) if c else None
    recall = tp / len(s) if s else None
    f1 = (
        (2 * precision * recall / (precision + recall))
        if precision and recall and (precision + recall) > 0
        else (0.0 if (c or s) else None)
    )
    return {
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ── 评测集 ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvalCase:
    """一条金标用例."""

    query: str
    # 标准答案标识集合：chunk key / image_id / "文件名::p3::img2" 均可
    relevant: frozenset[str] = frozenset()
    # 相关性等级（可选）：{标识: 0–3}，缺省 1.0，用于 NDCG
    grades: Mapping[str, float] | None = None
    # 该用例主要考验哪种模态：text | table | image —— 用于分组统计
    modality: str = "text"
    note: str = ""

    def effective_grades(self) -> dict[str, float] | None:
        if self.grades:
            return dict(self.grades)
        if self.relevant:
            return {k: 1.0 for k in self.relevant}
        return None


@dataclass(frozen=True)
class EvalSet:
    """一组金标用例（一个知识库 / 一个业务域一套）."""

    name: str = "default"
    cases: tuple[EvalCase, ...] = ()
    description: str = ""

    @property
    def size(self) -> int:
        return len(self.cases)

    def scored_cases(self) -> list[EvalCase]:
        """可参与聚合的用例（有标准答案的）."""
        return [c for c in self.cases if c.relevant or c.grades]

    def by_modality(self, modality: str) -> list[EvalCase]:
        return [c for c in self.cases if c.modality == modality]


# ── 结果 ─────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    """单条用例的评测结果."""

    query: str
    modality: str
    retrieved: int = 0
    hits: dict[int, float] = field(default_factory=dict)      # k → hit
    recall: dict[int, float] = field(default_factory=dict)    # k → recall
    precision: dict[int, float] = field(default_factory=dict)  # k → precision
    rr: float | None = None
    ap: float | None = None
    ndcg: dict[int, float] = field(default_factory=dict)      # k → ndcg
    missed: tuple[str, ...] = ()                              # 一条都没召回的标准答案
    # ── 分带（relative band）风险所需的量 ────────────────────────────────────
    # 只靠 recall 这一层看不到"这条证据是被绝对下限砍的、还是被相对分带砍的"，
    # 而两者的处置完全不同（前者是校准问题，后者是纵深防御调过头）。
    gold_count: int = 0                                       # 该用例标注了几条金标
    best_score: float | None = None                           # 头名检索分（精排口径）
    gold_scores: dict[str, float] = field(default_factory=dict)  # 命中的金标 → 其检索分
    min_gold_ratio: float | None = None                       # 命中最弱者 / 头名

    @property
    def all_gold_found(self) -> bool:
        """该用例的金标是否**一条不漏**都被召回（保序截断前的完整召回）. """
        return self.gold_count > 0 and not self.missed

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "modality": self.modality,
            "retrieved": self.retrieved,
            "hit": self.hits,
            "recall": self.recall,
            "precision": self.precision,
            "rr": self.rr,
            "ap": self.ap,
            "ndcg": self.ndcg,
            "missed": list(self.missed),
            "gold_count": self.gold_count,
            "best_score": self.best_score,
            "gold_scores": self.gold_scores,
            "min_gold_ratio": self.min_gold_ratio,
            "all_gold_found": self.all_gold_found,
        }


@dataclass
class EvalReport:
    """整轮评测的聚合报告."""

    eval_set: str = "default"
    k_values: tuple[int, ...] = ()
    total_cases: int = 0
    scored_cases: int = 0
    # k → 全库均值
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)
    mrr: float | None = None
    map_score: float | None = None
    # 按模态分组（图片召回单独看，避免被文本高分掩盖）
    by_modality: dict[str, dict] = field(default_factory=dict)
    # 按"答案需要几条证据"分组（single_evidence / multi_evidence）
    #
    # 为什么单独切一刀：单证据用例问的是"找得到吗"，多证据用例问的是
    # "**次要证据**还在吗"。两者的失效机制不同 —— 前者靠粗排召回，
    # 后者还会被相对分带、同父衰减、top_k 截断各自吃掉一遍。混在一起算，
    # 多证据用例的退化会被单证据用例的高分淹没（旧集 11 例全是单证据，
    # 于是"分带是否误杀"这个维度**完全没有观测点**）。
    evidence_slices: dict[str, dict] = field(default_factory=dict)
    # 金标分数画像汇总（分带安全上界的原料）
    gold_ratio: dict = field(default_factory=dict)
    # 引用准确率（本轮若有金标引用）
    citation: dict | None = None
    cases: tuple[CaseResult, ...] = ()
    generated_at: str = ""

    # ── 汇总 ──────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """给看板 / 日志用的扁平指标."""
        out: dict = {
            "eval_set": self.eval_set,
            "cases": self.total_cases,
            "scored": self.scored_cases,
            "mrr": self.mrr,
            "map": self.map_score,
        }
        for k in self.k_values:
            out[f"recall@{k}"] = self.recall.get(k)
            out[f"precision@{k}"] = self.precision.get(k)
            out[f"ndcg@{k}"] = self.ndcg.get(k)
            out[f"hit@{k}"] = self.hit_rate.get(k)
        multi = self.evidence_slices.get("multi_evidence") or {}
        if multi:
            out["multi_evidence_cases"] = multi.get("cases")
            out["multi_evidence_recall@10"] = multi.get("recall@10")
            out["multi_evidence_all_found_rate"] = multi.get("all_gold_found_rate")
        ceil = (self.gold_ratio or {}).get("min_gold_ratio")
        if ceil is not None:
            out["min_gold_ratio"] = ceil
        if self.citation:
            out["citation_precision"] = self.citation.get("precision")
            out["citation_recall"] = self.citation.get("recall")
            out["citation_f1"] = self.citation.get("f1")
        return out

    def as_dict(self) -> dict:
        return {
            "eval_set": self.eval_set,
            "k_values": list(self.k_values),
            "total_cases": self.total_cases,
            "scored_cases": self.scored_cases,
            "recall": self.recall,
            "precision": self.precision,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "map": self.map_score,
            "by_modality": self.by_modality,
            "evidence_slices": self.evidence_slices,
            "gold_ratio": self.gold_ratio,
            "citation": self.citation,
            "cases": [c.as_dict() for c in self.cases],
            "generated_at": self.generated_at,
        }


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def aggregate(
    results: Sequence[CaseResult],
    *,
    eval_set: str = "default",
    k_values: Sequence[int] = (),
    citation: dict | None = None,
) -> EvalReport:
    """把单条结果聚合成分组 + 全库报告."""
    ks = tuple(sorted(set(k_values) or {1, 3, 5, 10}))
    scored = [r for r in results if r.rr is not None or r.recall]

    report = EvalReport(
        eval_set=eval_set,
        k_values=ks,
        total_cases=len(results),
        scored_cases=len(scored),
        citation=citation,
        cases=tuple(results),
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
    )
    for k in ks:
        report.recall[k] = _mean([r.recall.get(k) for r in results])
        report.precision[k] = _mean([r.precision.get(k) for r in results])
        report.ndcg[k] = _mean([r.ndcg.get(k) for r in results])
        report.hit_rate[k] = _mean([r.hits.get(k) for r in results])
    report.mrr = _mean([r.rr for r in results])
    report.map_score = _mean([r.ap for r in results])

    # ── 分组：图片召回单独看 ─────────────────────────────────────────────
    for modality in sorted({r.modality for r in results}):
        group = [r for r in results if r.modality == modality]
        entry = {
            "cases": len(group),
            "mrr": _mean([r.rr for r in group]),
            "map": _mean([r.ap for r in group]),
        }
        for k in ks:
            entry[f"recall@{k}"] = _mean([r.recall.get(k) for r in group])
        report.by_modality[modality] = entry

    # ── 分组：按"答案需要几条证据"切一刀 ─────────────────────────────────
    #
    # 这一刀是为了让"相对分带会不会砍掉次要证据"**有观测点**。判据与命名都
    # 刻意写死：``gold_count >= 2`` 才算多证据。不按"本轮实际召回了 2 条"
    # 来切 —— 那会让"次要证据被砍掉"的用例自动掉出该分组，正好把要测的
    # 现象测没了（幸存者偏差）。
    for label, subset in (
        ("single_evidence", [r for r in results if r.gold_count == 1]),
        ("multi_evidence", [r for r in results if r.gold_count >= 2]),
    ):
        if not subset:
            continue
        entry = {
            "cases": len(subset),
            "mrr": _mean([r.rr for r in subset]),
            "map": _mean([r.ap for r in subset]),
            # 一条不漏的用例占比 —— 多证据场景下比 recall 更直白：
            # 只要漏一条就是"答案只答了一半"。
            "all_gold_found_rate": _mean(
                [1.0 if r.all_gold_found else 0.0 for r in subset]
            ),
            "cases_missing_gold": [
                r.query for r in subset if r.gold_count and r.missed
            ],
        }
        for k in ks:
            entry[f"recall@{k}"] = _mean([r.recall.get(k) for r in subset])
        report.evidence_slices[label] = entry

    # ── 金标分数画像（分带安全上界的原料）────────────────────────────────
    #
    # ⚠️ 读法（写在这里而不是文档里，因为误读的代价是"把参数调坏"）：
    # 本次检索若**开着**阈值过滤，被砍掉的金标根本不会进入 ranked，
    # 于是这里的 min_gold_ratio 是**上界**（偏高）。要拿真实安全上界，
    # 必须在不带过滤的那一态测（scripts/run_eval_baseline.py 会做），
    # 并把结果记进 golden 集的 known_limitation / baseline 的 band 段。
    multi = [r for r in results if r.gold_count >= 2 and r.min_gold_ratio is not None]
    report.gold_ratio = {
        "computed_over_cases": len(multi),
        "min_gold_ratio": min((r.min_gold_ratio for r in multi), default=None),
        "per_case": [
            {
                "query": r.query,
                "best_score": r.best_score,
                "min_gold_ratio": r.min_gold_ratio,
                "gold_scores": r.gold_scores,
                "all_gold_found": r.all_gold_found,
            }
            for r in multi
        ],
        "note": (
            "仅覆盖 gold_count>=2 的用例（单金标用例的 gold/head 恒为 1.0，"
            "无信息量）。阈值过滤开启时此值是**上界**，不是安全上界。"
        ),
    }

    return report


def score_case(
    ranked: Sequence[RetrievedItem],
    case: EvalCase,
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> CaseResult:
    """对单条用例打分（不触发检索，纯计算）."""
    result = CaseResult(query=case.query, modality=case.modality, retrieved=len(ranked))
    for k in k_values:
        result.hits[k] = hit_at_k(ranked, case.relevant, k)
        r = recall_at_k(ranked, case.relevant, k)
        if r is not None:
            result.recall[k] = round(r, 4)
        p = precision_at_k(ranked, case.relevant, k)
        if p is not None:
            result.precision[k] = round(p, 4)
        n = ndcg_at_k(ranked, case.effective_grades(), k)
        if n is not None:
            result.ndcg[k] = round(n, 4)
    rr = reciprocal_rank(ranked, case.relevant)
    result.rr = None if rr is None else round(rr, 4)
    ap = average_precision(ranked, case.relevant)
    result.ap = None if ap is None else round(ap, 4)
    result.missed = tuple(
        key for key in sorted(case.relevant or ()) if not any(i.matches([key]) for i in ranked)
    )
    # 金标分数画像（分带判别力）—— 与 recall 互补：recall 说"掉没掉分"，
    # 画像说"为什么掉、离边界还有多远"。
    result.gold_count = len(case.relevant or ())
    profile = gold_score_profile(ranked, case.relevant)
    result.best_score = profile["best_score"]
    result.gold_scores = profile["gold_scores"]
    result.min_gold_ratio = profile["min_gold_ratio"]
    return result


# ── 执行器 ───────────────────────────────────────────────────────────────────

RetrieveFn = Callable[[str], Awaitable[Sequence[RetrievedItem]]]


async def evaluate(
    retrieve: RetrieveFn,
    eval_set: EvalSet,
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
    citation_pairs: Sequence[tuple[Iterable[int], Iterable[int]]] | None = None,
) -> EvalReport:
    """
    跑一整轮评测.

    *retrieve*    异步回调：问题 → 排序后的检索结果
    *k_values*    需要报告的 K 值
    *citation_pairs* 可选：(答案引用的编号, 金标支持的编号) 序列

    单条用例检索失败**不会**中断整轮：记 0 命中并继续，这样一份报告的
    失败面是可枚举的，而不是"跑到一半炸了什么都没留下"。
    """
    results: list[CaseResult] = []
    for case in eval_set.cases:
        try:
            ranked = list(await retrieve(case.query)) or []
        except Exception as exc:      # noqa: BLE001
            logger.warning("evaluation: retrieval failed for %r: %s", case.query[:60], exc)
            ranked = []
        results.append(score_case(ranked, case, k_values=k_values))

    citation = None
    if citation_pairs:
        agg = {"precision": [], "recall": [], "f1": [], "tp": 0, "fp": 0, "fn": 0}
        for cited, supporting in citation_pairs:
            m = citation_prf(cited, supporting)
            if m["precision"] is not None:
                agg["precision"].append(m["precision"])
            if m["recall"] is not None:
                agg["recall"].append(m["recall"])
            if m["f1"] is not None:
                agg["f1"].append(m["f1"])
            agg["tp"] += m["tp"]
            agg["fp"] += m["fp"]
            agg["fn"] += m["fn"]
        citation = {
            "pairs": len(citation_pairs),
            "precision": _mean(agg["precision"]),
            "recall": _mean(agg["recall"]),
            "f1": _mean(agg["f1"]),
            "tp": agg["tp"],
            "fp": agg["fp"],
            "fn": agg["fn"],
            # 微平均（按引用条数而非按问题数）—— 引用多的自然权重大
            "micro_precision": (
                round(agg["tp"] / (agg["tp"] + agg["fp"]), 4)
                if (agg["tp"] + agg["fp"])
                else None
            ),
            "micro_recall": (
                round(agg["tp"] / (agg["tp"] + agg["fn"]), 4)
                if (agg["tp"] + agg["fn"])
                else None
            ),
        }

    report = aggregate(results, eval_set=eval_set.name, k_values=k_values, citation=citation)
    logger.info(
        "evaluation: set=%s cases=%d/%d mrr=%s %s",
        eval_set.name, report.scored_cases, report.total_cases, report.mrr,
        " ".join(f"recall@{k}={report.recall.get(k)}" for k in report.k_values),
    )
    record_eval_run(report)
    return report


# ── 运行历史（进程内，供看板与回归对比）────────────────────────────────────────

_HISTORY: deque[dict] = deque(maxlen=20)


def record_eval_run(report: EvalReport) -> None:
    """留存最近 N 轮评测结果（只存聚合值，不存逐条，控制内存）."""
    entry = {
        "generated_at": report.generated_at,
        "eval_set": report.eval_set,
        "cases": report.total_cases,
        "scored": report.scored_cases,
        "mrr": report.mrr,
        "map": report.map_score,
        "recall": dict(report.recall),
        "ndcg": dict(report.ndcg),
        "by_modality": report.by_modality,
        "evidence_slices": report.evidence_slices,
        "citation": report.citation,
    }
    _HISTORY.append(entry)


def eval_history(limit: int = 10) -> list[dict]:
    """最近若干轮评测摘要（最新在前）."""
    items = list(_HISTORY)
    items.reverse()
    return items[: max(0, limit)]


def reset_eval_history() -> None:
    """仅供单测使用."""
    _HISTORY.clear()


# ── 运行历史（持久化：跨重启保留，供跨版本回归对比）──────────────────────────
#
# 上面那个 deque 是**进程内**的：后端一重启就归零。而"改配置 → 重启 → 再评测"
# 恰好是评测最常见的用法，于是 /eval/history 永远是 []。下面这组函数把每轮
# 聚合结果写进 PostgreSQL 的 eval_runs 表，让"基线"真的存在。


def _i18n_keys(d: Mapping | None) -> dict:
    """JSON 的键只能是字符串：{1: 0.9} → {"1": 0.9}（读回来时再转回 int）."""
    if not d:
        return {}
    return {str(k): v for k, v in d.items()}


def _int_keys(d: Mapping | None) -> dict[int, float | None]:
    """把库里存的 {"1": 0.9} 还原成 {1: 0.9}（与进程内口径一致）."""
    if not d:
        return {}
    out: dict[int, float | None] = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _config_snapshot() -> dict:
    """
    评测当时的检索配置快照.

    没有它，"分数变了"无法归因到"改了什么" —— 只能看到一条下降的曲线，
    却不知道是精排阈值、多查询开关还是 Embedding 换了。这里是特意把
    影响召回的关键旋钮记下来，且**读配置失败不能拖垮评测**（评测本身
    已经跑完，落库失败只应记日志）。
    """
    try:
        from app.config import get_settings

        s = get_settings()
        return {
            "hybrid_search": s.HYBRID_SEARCH_ENABLED,
            "reranker": s.RERANKER_ENABLED,
            "rerank_min_score": s.RERANK_MIN_SCORE,
            "rerank_min_score_ratio": getattr(s, "RERANK_MIN_SCORE_RATIO", None),
            "rerank_min_score_filter": s.RERANK_MIN_SCORE_FILTER,
            "evidence_gate_min_top_score": s.EVIDENCE_GATE_MIN_TOP_SCORE,
            "retrieval_min_score": s.RETRIEVAL_MIN_SCORE,
            "retrieval_max_gap": s.RETRIEVAL_MAX_GAP,
            "multi_query_max_extra": getattr(s, "MULTI_QUERY_MAX_EXTRA", None),
            "query_rewrite_enabled": getattr(s, "QUERY_REWRITE_ENABLED", None),
            "acl_prefilter_enabled": getattr(s, "ACL_PREFILTER_ENABLED", None),
            "hierarchical_rag": getattr(s, "HIERARCHICAL_RAG_ENABLED", None),
            "parent_score_decay": getattr(s, "PARENT_SCORE_DECAY", None),
        }
    except Exception:  # noqa: BLE001
        logger.exception("eval config snapshot failed")
        return {}


async def persist_eval_run(
    report: EvalReport,
    *,
    run_by: str | None = None,
    tenant_id: str | None = None,
    collection_id: str | None = None,
    top_k: int = 0,
    description: str = "",
) -> bool:
    """
    把一轮评测的聚合结果落库（``eval_runs`` 表）.

    刻意与 ``record_eval_run``（进程内 deque）并存：看板读 DB 拿历史，
    单测与极端降级路径仍可依赖内存副本。落库失败**不抛异常** —— 评测
    本身已经跑完，写历史失败不该让调用方以为评测失败；失败仅记日志。

    Returns:
        True = 已落库；False = 落库失败（已记日志），调用方不应据此判定评测失败。
    """
    try:
        from app.db.eval_models import EvalRunRow
        from app.db.postgres import get_db_session

        try:
            generated = datetime.fromisoformat(report.generated_at)
        except (TypeError, ValueError):
            generated = datetime.now(timezone.utc)

        row = EvalRunRow(
            name=report.eval_set or "default",
            description=description,
            run_by=run_by,
            tenant_id=tenant_id,
            collection_id=collection_id,
            total_cases=report.total_cases,
            scored_cases=report.scored_cases,
            top_k=top_k,
            mrr=report.mrr,
            map_score=report.map_score,
            recall=_i18n_keys(report.recall),
            precision=_i18n_keys(report.precision),
            ndcg=_i18n_keys(report.ndcg),
            hit_rate=_i18n_keys(report.hit_rate),
            by_modality=report.by_modality or {},
            citation=report.citation,
            k_values=[int(k) for k in report.k_values],
            # 先行指标（见 EvalRunRow 里这两个字段的说明）：不落库的话，
            # "分带余量侵蚀"这段过程永远无法回溯，只能等 recall 掉下来才知道。
            evidence_slices=report.evidence_slices or {},
            gold_ratio=report.gold_ratio or {},
            config=_config_snapshot(),
            generated_at=generated,
        )
        async with get_db_session() as session:
            session.add(row)
        logger.info(
            "eval_run persisted: set=%s cases=%d/%d mrr=%s",
            row.name, row.scored_cases, row.total_cases, row.mrr,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.exception("persist_eval_run failed (evaluation result NOT stored)")
        return False


async def eval_history_persisted(limit: int = 10) -> list[dict]:
    """
    从 ``eval_runs`` 表读最近若干轮评测摘要（最新在前）.

    落库不可用（表未建 / DB 抖动）时**回退到进程内历史**，而不是返回空 ——
    "面板显示空"比"面板显示旧数据"更容易被误读成"从来没评测过"。
    """
    try:
        from sqlalchemy import select

        from app.db.eval_models import EvalRunRow
        from app.db.postgres import get_db_session

        async with get_db_session() as session:
            rows = (
                await session.execute(
                    select(EvalRunRow)
                    .order_by(EvalRunRow.generated_at.desc())
                    .limit(max(1, min(50, limit)))
                )
            ).scalars().all()
        return [
            {
                "id": str(r.id),
                "generated_at": r.generated_at.isoformat() if r.generated_at else "",
                "eval_set": r.name,
                "description": r.description,
                "run_by": r.run_by,
                "tenant_id": r.tenant_id,
                "collection_id": r.collection_id,
                "cases": r.total_cases,
                "scored": r.scored_cases,
                "top_k": r.top_k,
                "mrr": r.mrr,
                "map": r.map_score,
                "recall": _int_keys(r.recall),
                "precision": _int_keys(r.precision),
                "ndcg": _int_keys(r.ndcg),
                "hit_rate": _int_keys(r.hit_rate),
                "by_modality": r.by_modality or {},
                "citation": r.citation,
                "k_values": list(r.k_values or []),
                "config": r.config or {},
                # 多证据分组 + 分带余量：看板据此画"余量趋势"。
                # 存量库补齐前的老行这两个字段为空 dict —— 读侧用 ``or {}`` 兜底，
                # 不把"没存过"渲染成"这轮测出来是 0"。
                "evidence_slices": r.evidence_slices or {},
                "gold_ratio": r.gold_ratio or {},
                "source": "database",
            }
            for r in rows
        ]
    except Exception:  # noqa: BLE001
        logger.exception("eval_history_persisted failed — falling back to in-process history")
        fallback = eval_history(limit=limit)
        for item in fallback:
            item.setdefault("source", "in_process")
        return fallback


__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "EvalSet",
    "RetrieveFn",
    "RetrievedItem",
    "aggregate",
    "average_precision",
    "citation_prf",
    "eval_history",
    "eval_history_persisted",
    "evaluate",
    "gold_score_profile",
    "hit_at_k",
    "item_from_chunk",
    "ndcg_at_k",
    "persist_eval_run",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "record_eval_run",
    "reset_eval_history",
    "score_case",
]
