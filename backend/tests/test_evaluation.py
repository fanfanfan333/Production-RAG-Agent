"""
检索质量评测单测（Recall / MRR / NDCG / 引用准确率 / 图文分组）.

覆盖的关键性质：
  * 指标定义正确 —— Recall 找全率、RR 第一名位置倒数、AP 兼顾全与前、
    NDCG 对排序敏感且恒 ∈ [0,1]；
  * 边界不撒谎 —— 无标准答案返回 None（不按 0 分算），空召回不为零除；
  * 别名命中 —— 同一张图片用 image_id 或"文件::页::序号"标注都应判对；
  * 图文分开统计 —— 图片召回差不会被文本高分掩盖；
  * 单条检索失败不炸整轮；
  * 引用准确率 —— 没引用时 precision 为 None 而非 0。

评测模块是纯计算（不碰 DB / 网络），宿主机无后端依赖时也能跑。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from app.services.evaluation import (
        EvalCase,
        EvalSet,
        RetrievedItem,
        average_precision,
        citation_prf,
        eval_history,
        evaluate,
        hit_at_k,
        ndcg_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank,
        reset_eval_history,
        score_case,
    )
except ImportError as exc:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    # 不能用 sys.exit()：pytest 在收集阶段导入本模块，抛 SystemExit 会让整个
    # 会话 INTERNALERROR，同目录其它用例全部跑不了。
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


def _items(keys: list[str], **kw) -> list[RetrievedItem]:
    return [RetrievedItem(key=k, **kw) for k in keys]


# ── 1. 基础指标定义 ──────────────────────────────────────────────────────────

def test_recall_precision_hit() -> None:
    ranked = _items(["a", "b", "c", "d"])
    rel = {"a", "c"}

    assert recall_at_k(ranked, rel, 1) == 0.5, "前1条只召回 a → 1/2"
    assert recall_at_k(ranked, rel, 3) == 1.0, "前3条含 a,c → 2/2"
    assert precision_at_k(ranked, rel, 4) == 0.5, "4 条里 2 条相关"
    assert hit_at_k(ranked, rel, 1) == 1.0
    assert hit_at_k(_items(["x", "y"]), rel, 2) == 0.0
    print("  ok test_recall_precision_hit")


def test_recall_without_gold_is_none() -> None:
    """无标准答案 → None，而不是 0（"没标注"不能被读成"召回失败"）."""
    ranked = _items(["a", "b"])
    assert recall_at_k(ranked, set(), 5) is None
    assert precision_at_k(ranked, set(), 5) is None
    assert reciprocal_rank(ranked, set()) is None
    assert average_precision(ranked, set()) is None
    assert ndcg_at_k(ranked, None, 5) is None
    # 空召回 + 有标准答案 → 0 而不是零除
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert precision_at_k([], {"a"}, 5) is None
    assert reciprocal_rank([], {"a"}) == 0.0
    print("  ok test_recall_without_gold_is_none")


def test_reciprocal_rank_and_ap() -> None:
    # 第一名命中 → RR = 1
    assert reciprocal_rank(_items(["a", "b"]), {"a"}) == 1.0
    # 第三名命中 → RR = 1/3
    assert abs((reciprocal_rank(_items(["x", "y", "a"]), {"a"}) or 0) - 1 / 3) < 1e-9
    # 完全没命中 → 0
    assert reciprocal_rank(_items(["x"]), {"a"}) == 0.0

    # AP：两金标分别在第 1、第 3 位 → (1/1 + 2/3)/2
    ap = average_precision(_items(["a", "x", "c"]), {"a", "c"})
    assert ap is not None and abs(ap - (1.0 + 2 / 3) / 2) < 1e-9, f"AP={ap}"
    # 排得更前 → AP 更高（同时奖励"全"与"前"）
    ap2 = average_precision(_items(["a", "c", "x"]), {"a", "c"})
    assert ap2 is not None and ap2 > (ap or 0)
    print("  ok test_reciprocal_rank_and_ap")


def test_ndcg_ordering_and_bounds() -> None:
    grades = {"a": 3.0, "b": 2.0, "c": 1.0}
    perfect = ndcg_at_k(_items(["a", "b", "c"]), grades, 3)
    worst = ndcg_at_k(_items(["c", "b", "a"]), grades, 3)
    assert perfect is not None and abs(perfect - 1.0) < 1e-9, "理想排序 NDCG=1"
    assert worst is not None and 0.0 <= worst < perfect, "逆序应显著变差"
    # 未召回任何相关项 → 0
    assert ndcg_at_k(_items(["x", "y"]), grades, 2) == 0.0
    print("  ok test_ndcg_ordering_and_bounds")


def test_alias_matching() -> None:
    """同一张图片用不同叫法标注都应判对."""
    item = RetrievedItem(
        key="doc-1::7",
        aliases=("img-doc-1-p2-i1", "年报.pdf::p2::第1张图"),
        content_type="image",
    )
    assert item.matches({"img-doc-1-p2-i1"})
    assert item.matches({"年报.pdf::p2::第1张图"})
    assert item.matches({"doc-1::7"})
    assert not item.matches({"doc-1::8"})

    ranked = _items(["x"]) + [item]
    assert (recall_at_k(ranked, {"img-doc-1-p2-i1"}, 2) or 0) == 1.0
    print("  ok test_alias_matching")


def test_recall_is_deduplicated_by_gold_not_by_hit() -> None:
    """
    Recall 的去重维度必须是**金标标识**，且恒 ≤ 1.0.

    回归 bug：旧实现写成 ``{i.key for i in top if i.matches(needles)}``。
    同一条金标被多个分块命中（父子块同时召回、图片与其 OCR 文本块同时召回）
    时，分子会数出多个不同的 ``i.key``，分母却只有 1 个标识 →
    Recall 变成 2.0 / 3.0，指标彻底失真，还会把"检索正常"误报成"超额召回"。
    """
    # 一条金标，被 3 个分块命中（key 各不相同，但都带同一条别名）
    gold = "年报.pdf::p3::第2张图"
    ranked = [
        RetrievedItem(key=f"doc-1::{i}", aliases=(gold,), content_type="image")
        for i in (4, 5, 6)
    ]
    assert recall_at_k(ranked, {gold}, 5) == 1.0, "命中一条金标就是 1.0，不能是 3.0"

    # 反向：一个检索项同时覆盖两条金标（aliases 携带两条标识）
    both = RetrievedItem(key="doc-1::9", aliases=("g1", "g2"))
    assert recall_at_k([both], {"g1", "g2"}, 5) == 1.0, "两条金标都被覆盖 → 1.0"

    # 常规：两条金标只召回一条
    assert recall_at_k(_items(["g1", "x"]), {"g1", "g2"}, 5) == 0.5
    print("  ok test_recall_is_deduplicated_by_gold_not_by_hit")


# ── 2. 引用准确率 ────────────────────────────────────────────────────────────

def test_citation_prf() -> None:
    # 注意：指标按设计四舍五入到 4 位（对外汇报用），比较时用 1e-4 容差
    m = citation_prf([1, 2, 3], [1, 3])
    assert abs((m["precision"] or 0) - 2 / 3) < 1e-4, "3 条引用里 2 条真的支持"
    assert abs((m["recall"] or 0) - 1.0) < 1e-4, "2 条金标都被引到了"
    assert abs((m["f1"] or 0) - 0.8) < 1e-4
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 0

    # 完全没有引用 → precision 为 None（一句没引用的话不该被算成零准确）
    assert citation_prf([], [1])["precision"] is None
    # 幻觉引用：引了不存在的来源
    bad = citation_prf([9], [1])
    assert bad["tp"] == 0 and bad["fp"] == 1
    print("  ok test_citation_prf")


# ── 3. 聚合与分组 ────────────────────────────────────────────────────────────

def test_score_case_and_missed() -> None:
    case = EvalCase(query="q", relevant=frozenset({"a", "z"}), modality="text")
    res = score_case(_items(["a", "b"]), case, k_values=(1, 3))
    assert res.rr == 1.0
    assert abs((res.recall.get(3) or 0) - 0.5) < 1e-9
    assert res.missed == ("z",), "应报出一条都没召回的金标"
    print("  ok test_score_case_and_missed")


def test_image_recall_not_masked_by_text() -> None:
    """
    图片召回差必须能被单独看出来.

    若混在一起算：文本 2/2、图片 0/1 → 整体 0.67，看起来"还行"；
    分组后 image.recall=0 —— 图片链路的问题才暴露出来。
    """

    async def _retrieve(query: str):
        if "架构" in query:                      # 图片题：召回的全是文本
            return _items(["t1", "t2"], content_type="text")
        return _items(["a", "b"], content_type="text")

    eval_set = EvalSet(name="mixed", cases=(
        EvalCase(query="文本问题A", relevant=frozenset({"a"}), modality="text"),
        EvalCase(query="文本问题B", relevant=frozenset({"b"}), modality="text"),
        EvalCase(query="架构图有哪些模块", relevant=frozenset({"img-1"}), modality="image"),
    ))
    report = asyncio.run(evaluate(_retrieve, eval_set, k_values=(5,)))

    assert report.total_cases == 3
    text_entry = report.by_modality["text"]
    img_entry = report.by_modality["image"]
    assert text_entry["recall@5"] == 1.0, "文本全召回"
    assert img_entry["recall@5"] == 0.0, "图片零召回必须单独可见"
    assert (report.recall[5] or 0) < 1.0
    print("  ok test_image_recall_not_masked_by_text")


def test_single_case_failure_does_not_abort_run() -> None:
    """单条检索失败记 0 命中并继续，而不是整轮炸掉什么都没留下."""
    calls: list[str] = []

    async def _retrieve(query: str):
        calls.append(query)
        if "boom" in query:
            raise RuntimeError("qdrant down")
        return _items(["a"])

    eval_set = EvalSet(name="resilient", cases=(
        EvalCase(query="正常问题", relevant=frozenset({"a"})),
        EvalCase(query="boom", relevant=frozenset({"a"})),
        EvalCase(query="另一个正常问题", relevant=frozenset({"a"})),
    ))
    report = asyncio.run(evaluate(_retrieve, eval_set, k_values=(3,)))
    assert len(report.cases) == 3, "三条都要有结果（失败的记为 0 命中）"
    assert len(calls) == 3, "不应因单条失败提前中断"
    assert (report.mrr or 0) > 0, "正常用例的分数要保留"
    print("  ok test_single_case_failure_does_not_abort_run")


def test_history_records_runs() -> None:
    reset_eval_history()

    async def _retrieve(query: str):
        return _items(["a"])

    eval_set = EvalSet(name="hist", cases=(
        EvalCase(query="q", relevant=frozenset({"a"})),
    ))
    asyncio.run(evaluate(_retrieve, eval_set, k_values=(1,)))
    asyncio.run(evaluate(_retrieve, eval_set, k_values=(1,)))

    hist = eval_history(limit=5)
    assert len(hist) == 2, "两轮评测都应留存"
    assert hist[0]["eval_set"] == "hist", "最新在前"
    assert "recall" in hist[0]
    reset_eval_history()
    print("  ok test_history_records_runs")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll evaluation tests passed.")
