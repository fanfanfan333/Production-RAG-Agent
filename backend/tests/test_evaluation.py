"""
检索质量评测单测（Recall / MRR / NDCG / 引用准确率 / 图文分组）.

覆盖的关键性质：
  * 指标定义正确 —— Recall 找全率、RR 第一名位置倒数、AP 兼顾全与前、
    NDCG 对排序敏感且恒 ∈ [0,1]；
  * 边界不撒谎 —— 无标准答案返回 None（不按 0 分算），空召回不为零除；
  * 别名命中 —— 同一张图片用 image_id 或"文件::页::序号"标注都应判对；
  * 图文分开统计 —— 图片召回差不会被文本高分掩盖；
  * 单条检索失败不炸整轮；
  * 引用准确率 —— 没引用时 precision 为 None 而非 0；
  * 结果持久化 —— 指标键 JSON 往返不丢、写库失败不抛异常、读库失败回退进程内
    （三条都是"旁路失效不报错"的沉默路径，必须显式钉住）。

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


# ── 评测结果持久化（eval_runs 表）──────────────────────────────────────────
#
# 这组用例守的是一条"沉默失效"路径：落库/读取历史都属于**旁路**，出问题
# 不会让评测本身失败，只会让"基线"悄悄不存在 —— 面板显示空，看起来像
# "从来没评测过"，而不是"写库坏了"。因此必须用测试把边界钉死。


def test_metric_keys_survive_json_roundtrip() -> None:
    """JSON 的键只能是字符串：{1: 0.9} 存进去、读回来必须还是 {1: 0.9}."""
    from app.services.evaluation import _i18n_keys, _int_keys

    stored = _i18n_keys({1: 0.9, 10: 0.5})
    assert stored == {"1": 0.9, "10": 0.5}, "落库前必须转成字符串键"
    back = _int_keys(stored)
    assert back == {1: 0.9, 10: 0.5}, "读回来必须还原成 int 键，否则 k 值对不上"
    assert _i18n_keys(None) == {} and _int_keys(None) == {}
    # 脏数据（非数字键）只跳过，不能抛异常把整个历史读空
    assert _int_keys({"1": 1.0, "bad": 2.0}) == {1: 1.0}
    print("  ok test_metric_keys_survive_json_roundtrip")


def test_persist_eval_run_swallows_db_failure() -> None:
    """写库失败必须返回 False 而不是抛异常：评测已经跑完了，不该被判成失败."""
    from app.services import evaluation as ev

    async def _retrieve(query: str):
        return _items(["a"])

    eval_set = EvalSet(name="persist-fail", cases=(
        EvalCase(query="q", relevant=frozenset({"a"})),
    ))
    report = asyncio.run(evaluate(_retrieve, eval_set, k_values=(1,)))

    import app.db.postgres as pg

    original = pg.get_db_session

    def _boom():
        raise RuntimeError("simulated DB outage")

    pg.get_db_session = _boom  # type: ignore[assignment]
    try:
        ok = asyncio.run(ev.persist_eval_run(report, run_by="tester"))
    finally:
        pg.get_db_session = original  # type: ignore[assignment]
    assert ok is False, "落库失败应显式返回 False（不抛异常）"
    print("  ok test_persist_eval_run_swallows_db_failure")


def test_persisted_history_falls_back_to_in_process() -> None:
    """读库失败必须回退到进程内历史，而不是返回空列表."""
    from app.services import evaluation as ev

    reset_eval_history()

    async def _retrieve(query: str):
        return _items(["a"])

    eval_set = EvalSet(name="fallback", cases=(
        EvalCase(query="q", relevant=frozenset({"a"})),
    ))
    asyncio.run(evaluate(_retrieve, eval_set, k_values=(1,)))

    import app.db.postgres as pg

    original = pg.get_db_session

    def _boom():
        raise RuntimeError("simulated DB outage")

    pg.get_db_session = _boom  # type: ignore[assignment]
    try:
        hist = asyncio.run(ev.eval_history_persisted(limit=5))
    finally:
        pg.get_db_session = original  # type: ignore[assignment]

    assert len(hist) == 1, "读库挂了也要拿得到进程内那一轮，不能显示成空"
    assert hist[0]["eval_set"] == "fallback"
    assert hist[0]["source"] == "in_process", "必须标明来源，避免把降级数据当库里的"
    reset_eval_history()
    print("  ok test_persisted_history_falls_back_to_in_process")


# ── 4. 多证据 / 相对分带判别力 ────────────────────────────────────────────────
#
# 这一组守的是本次修复的**判别力地基**：旧金标集 11 例的金标块全是精排第 1 名
# （gold == head），于是 gold/head ≡ 1.0 —— "提高相对分带会不会误杀次要证据"
# 在那套集上是**恒真命题**，测不出来。下面用合成数据把这两件事分别钉住：
#   · 画像口径正确（最弱者 ÷ 头名，而不是命中数或均值）；
#   · 分组口径**抗幸存者偏差**（次要证据被砍掉的用例必须留在多证据组里，
#     否则"要观测的现象"会自己掉出观测集合）。


def test_gold_score_profile_uses_weakest_gold_over_head() -> None:
    """
    分带判据对**每一条**候选独立生效，所以决定"会不会被砍"的永远是最弱那条金标.

    画像必须报 ``最弱者 ÷ 头名``：
      · 报"平均"→ 一条强一条弱会被平均成"余量充足"，恰好看不见要测的东西；
      · 报"命中数"→ 与 ratio 无关，没有判别力。
    """
    from app.services.evaluation import gold_score_profile

    # 头名 0.9（同时是第一条金标），第二条金标只有 0.07
    ranked = [
        RetrievedItem(key="g1", score=0.9),
        RetrievedItem(key="noise", score=0.5),
        RetrievedItem(key="g2", score=0.07),
    ]
    profile = gold_score_profile(ranked, {"g1", "g2"})
    assert profile["best_score"] == 0.9
    assert profile["gold_scores"] == {"g1": 0.9, "g2": 0.07}
    assert abs((profile["min_gold_ratio"] or 0) - round(0.07 / 0.9, 4)) < 1e-9, (
        f"应为最弱金标/头名，实际={profile['min_gold_ratio']}"
    )
    print("  ok test_gold_score_profile_uses_weakest_gold_over_head")


def test_gold_score_profile_blindness_on_single_evidence_set() -> None:
    """
    **旧集为什么测不出分带风险** —— 用合成数据把那个结构性盲区演示出来.

    单金标用例里金标就是头名本身 ⇒ ratio ≡ 1.0。也就是说：只要评测集全是
    单金标用例，无论把分带调到多高，画像都会回答"余量 1.0 倍，很安全" ——
    这不是"参数被验证过"，而是**根本没有观测点**。
    """
    from app.services.evaluation import gold_score_profile

    for head, noise in ((0.99, 0.001), (0.22, 0.0005), (0.83, 0.12)):
        ranked = [RetrievedItem(key="g", score=head),
                  RetrievedItem(key="n", score=noise)]
        profile = gold_score_profile(ranked, {"g"})
        assert profile["min_gold_ratio"] == 1.0, (
            f"单金标用例的 gold/head 应恒为 1.0（head={head}），"
            f"实际={profile['min_gold_ratio']}"
        )

    # 对照：同一条噪声，一旦它也是金标，画像立刻有信息量
    ranked = [RetrievedItem(key="g", score=0.83),
              RetrievedItem(key="g2", score=0.12)]
    profile = gold_score_profile(ranked, {"g", "g2"})
    assert profile["min_gold_ratio"] is not None and profile["min_gold_ratio"] < 0.2
    print("  ok test_gold_score_profile_blindness_on_single_evidence_set")


def test_gold_score_profile_is_none_when_nothing_found() -> None:
    """一条金标都没命中 / 结果为空 → ratio 为 None（不猜、不报 0 或 1）."""
    from app.services.evaluation import gold_score_profile

    assert gold_score_profile(_items(["x", "y"]), {"g"})["min_gold_ratio"] is None
    assert gold_score_profile([], {"g"})["min_gold_ratio"] is None
    assert gold_score_profile([], set())["min_gold_ratio"] is None
    assert gold_score_profile([], set())["best_score"] is None
    # 空集不能零除：best=0 时 ratio 为 None
    assert gold_score_profile([RetrievedItem(key="g", score=0.0)], {"g"})["min_gold_ratio"] is None
    print("  ok test_gold_score_profile_is_none_when_nothing_found")


def test_evidence_slices_split_by_gold_count_not_by_outcome() -> None:
    """
    分组必须按"答案**需要**几条证据"切，不能按"本轮召回了几个".

    按回收结果切会把"次要证据被砍掉"的用例自动移出多证据组 —— 于是
    ``all_gold_found_rate`` 永远漂亮，而真正要观测的失效**恰好**不被观测到
    （幸存者偏差）。这条测试用"一条被砍"的用例钉住分组口径。
    """
    from app.services.evaluation import aggregate

    multi_ok = EvalCase(query="多证据-完好", relevant=frozenset({"a", "b"}))
    multi_broken = EvalCase(query="多证据-次证被砍", relevant=frozenset({"c", "d"}))
    single = EvalCase(query="单证据", relevant=frozenset({"e"}))

    results = [
        score_case(_items(["a", "b"]), multi_ok, k_values=(10,)),
        # d 被分带砍掉 → 只剩 c
        score_case(_items(["c"]), multi_broken, k_values=(10,)),
        score_case(_items(["e"]), single, k_values=(10,)),
    ]
    report = aggregate(results, eval_set="slices", k_values=(10,))

    multi = report.evidence_slices["multi_evidence"]
    single_slice = report.evidence_slices["single_evidence"]
    assert multi["cases"] == 2, (
        "被砍掉次要证据的用例必须仍留在多证据组里，否则观测集合自己缩水"
    )
    assert single_slice["cases"] == 1
    assert abs(multi["all_gold_found_rate"] - 0.5) < 1e-9, (
        f"2 个多证据用例里只有 1 个一条不漏 → 0.5，实际={multi['all_gold_found_rate']}"
    )
    assert multi["cases_missing_gold"] == ["多证据-次证被砍"], (
        "必须点名是哪条用例漏了金标（否则只知道'有事'、不知道'哪件'）"
    )
    # 被砍那条的 recall@10 只有 0.5，整体 recall 被拉低 —— 门禁因此能变红
    assert abs((multi["recall@10"] or 0) - 0.75) < 1e-9
    print("  ok test_evidence_slices_split_by_gold_count_not_by_outcome")


def test_gold_ratio_aggregate_covers_multi_evidence_only() -> None:
    """汇总只统计多证据用例，并取**最小**比值 —— 上界由最危险的用例决定."""
    from app.services.evaluation import aggregate

    r_multi_a = score_case(
        [RetrievedItem(key="a1", score=0.9), RetrievedItem(key="a2", score=0.81)],
        EvalCase(query="m1", relevant=frozenset({"a1", "a2"})),
        k_values=(10,),
    )
    r_multi_b = score_case(
        [RetrievedItem(key="b1", score=0.83), RetrievedItem(key="b2", score=0.12)],
        EvalCase(query="m2", relevant=frozenset({"b1", "b2"})), k_values=(10,),
    )
    r_single = score_case(
        _items(["c1"]), EvalCase(query="s1", relevant=frozenset({"c1"})), k_values=(10,),
    )
    report = aggregate([r_multi_a, r_multi_b, r_single], eval_set="gr", k_values=(10,))

    assert report.gold_ratio["computed_over_cases"] == 2, "单金标用例不得计入（恒为 1.0）"
    expected = min(r_multi_a.min_gold_ratio or 1.0, r_multi_b.min_gold_ratio or 1.0)
    assert abs((report.gold_ratio["min_gold_ratio"] or 0) - expected) < 1e-9
    # summary() 要把它平铺出来，供日志/看板直接消费
    assert "min_gold_ratio" in report.summary()
    assert "multi_evidence_all_found_rate" in report.summary()
    print("  ok test_gold_ratio_aggregate_covers_multi_evidence_only")


def test_band_would_kill_secondary_evidence_at_high_ratio() -> None:
    """
    端到端把"分带误杀次要证据"复现成一条断言（与实盘 bug 同形）.

    组合：绝对下限 0.05（已校准，放行）；相对分带 ratio 取 0.10（旧值）。
    次证 0.0685 过了绝对下限，却低于带 0.832×0.10=0.0832 → **整条被砍**，
    该用例 recall@10 从 1.0 掉到 0.5。这正是实测到的召回损失
    （golden_v1.json 里的『分带哨兵』用例），也说明为什么 ratio 必须
    ≤ 实测上界：它砍的不是噪声，是合法证据。
    """
    from app.services.evaluation import aggregate
    from app.services.reranker import filter_by_min_score

    class _C:
        def __init__(self, score: float) -> None:
            self.score = score

    head, weak = 0.832, 0.0685
    case = EvalCase(query="元组能不能被修改？打开文件时怎么指定编码？",
                    relevant=frozenset({"head", "weak"}))

    honest = [RetrievedItem(key="head", score=head), RetrievedItem(key="weak", score=weak)]
    assert (recall_at_k(honest, case.relevant, 10) or 0) == 1.0

    # ratio=0.10 → 带 = 0.0832 > 0.0685 → 次证被砍
    killed = filter_by_min_score([_C(head), _C(weak)], 0.05, ratio=0.10)
    assert [round(c.score, 4) for c in killed] == [head], (
        "预期次证被 0.10 的带砍掉（这正是要复现的失效）"
    )
    after_kill = [RetrievedItem(key="head", score=head)]
    assert (recall_at_k(after_kill, case.relevant, 10) or 0) == 0.5

    # ratio=0.05 → 带 = 0.0416 < 0.0685 → 次证保住
    kept = filter_by_min_score([_C(head), _C(weak)], 0.05, ratio=0.05)
    assert len(kept) == 2, "0.05 应同时保住主/次证据"
    report = aggregate([score_case(honest, case, k_values=(10,))], k_values=(10,))
    assert report.evidence_slices["multi_evidence"]["all_gold_found_rate"] == 1.0
    print("  ok test_band_would_kill_secondary_evidence_at_high_ratio")


def test_persist_eval_run_carries_discriminative_metrics() -> None:
    """
    先行指标（``evidence_slices`` / ``gold_ratio``）必须真的被写进 row.

    为什么值得单独测：这两个字段**此前根本没落库**，而丢字段这件事没有任何症状 ——
    整体 recall 恒 1.0 时，分带余量可能已经从 1.4 倍掉到 1.02 倍，指标上看不出来。
    于是一旦它们不进 ``eval_runs``，"余量侵蚀"那段过程在跨重启的基线里永远查不到，
    只能等 recall 突然掉下来才知道（这正是本次修复要消灭的那类盲区在持久化层的翻版）。

    用替身 session 捕获 row，**刻意不写真实 DB** —— 评测表就是基线本身，
    往里塞测试行会污染历史，让"上一轮是多少"变得不可信。
    """
    from app.db.eval_models import EvalRunRow
    from app.services import evaluation as ev

    captured: dict = {}

    class _Session:
        def add(self, row) -> None:
            captured["row"] = row

    class _CM:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, *exc):
            return False

    async def _retrieve(query: str):
        return _items(["a", "b"])

    case = EvalCase(query="q", relevant=frozenset({"a", "b"}))
    report = asyncio.run(
        evaluate(_retrieve, EvalSet(name="persist-fields", cases=(case,)), k_values=(10,))
    )
    assert report.evidence_slices, "多证据分组为空 → 本测试没有覆盖到目标字段"

    import app.db.postgres as pg

    original = pg.get_db_session
    pg.get_db_session = lambda: _CM()  # type: ignore[assignment]
    try:
        ok = asyncio.run(ev.persist_eval_run(report, run_by="tester"))
    finally:
        pg.get_db_session = original  # type: ignore[assignment]

    assert ok is True, "落库返回 False（替身 session 不该失败）"
    row = captured.get("row")
    assert isinstance(row, EvalRunRow), f"没有向 session 添加 EvalRunRow，实际={row!r}"
    assert row.evidence_slices == report.evidence_slices, "evidence_slices 没有落库"
    assert row.gold_ratio == report.gold_ratio, "gold_ratio 没有落库"
    # 模型层也必须有这两列，否则在真实 DB 上 INSERT 会因列不存在而失败
    assert {"evidence_slices", "gold_ratio"} <= set(EvalRunRow.__table__.columns.keys())
    print("  ok test_persist_eval_run_carries_discriminative_metrics")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll evaluation tests passed.")
