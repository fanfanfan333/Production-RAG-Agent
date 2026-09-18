"""本次修复的回归门禁：向量清理顺序 / num_ctx 统一 / 阶段计时 / 精排阈值校准.

这四条都是"不写测试就会悄悄退化"的性质：

  1. **num_ctx 必须全项目一致** —— 不一致不会报错，只会在 Ollama 侧变成
     "每次都重载模型"（本机实测 ≈11s/次），而监控面板上看不出原因。
     这里锁住"所有节点都从 ``settings.chat_num_ctx`` 取"这一约定。

  2. **删文档必须先删向量** —— 顺序反了不会报错，只在向量库里留孤儿点，
     日积月累挤占检索候选池（本仓库实测 95% 的点是孤儿）。

  3. **``purge_documents()`` 拒绝无条件清空** —— 漏传条件时"删光全库"是
     不可接受的默认行为，必须显式报错（与检索层 fail-closed 同一取向）。

  4. **候选分必须始终是精排分；绝对阈值必须低于实测最弱真阳性** ——
     这两条是一个根因的两面。历史实现里"粗排候选 ≤1 条就跳过精排"，导致
     ``chunk.score`` 停留在**粗排分**（向量余弦 / 关键词占位 0.30），而所有
     下游相关性判定都按**精排 sigmoid 分**口径解读它：

       · 一条语义上毫不相关的 chunk 被关键词腿提为唯一候选时，带着余弦分
         （实测 0.19 / 0.315）被当作"精排判它相关" → **绕过拒答**，
         "库里根本没有"的问题被回答；
       · 为了压住这类假阳性，阈值被设到 0.25，高于实测最弱真阳性 0.22
         → 金标分片被整体丢弃 → **确定性地答错"库里没有"**。

     修掉量纲泄漏后，否定对照的真实精排分是 0.0 / 0.0 / 0.0005，
     拒答恢复正常，阈值也可以降到 0.05（距噪声约 100 倍、距真阳性约 4.4 倍）。
     这里锁住"ratio=0 不得退化成恒真"与"阈值不误杀"两条不变量。

全部为纯逻辑单测：不连 Qdrant、不连 PG、不调 LLM。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from app.config import get_settings
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ── 1. num_ctx 统一 ──────────────────────────────────────────────────────────

def test_chat_num_ctx_is_single_source_of_truth() -> None:
    """``chat_num_ctx`` 必须就是 ``OLLAMA_NUM_CTX``，且为正整数。"""
    settings = get_settings()
    assert settings.chat_num_ctx == int(settings.OLLAMA_NUM_CTX)
    assert settings.chat_num_ctx > 0


def test_no_node_hardcodes_a_different_num_ctx() -> None:
    """
    源码层面锁死：不允许任何节点再写 ``min(4096, ...)`` 之类的独立 num_ctx.

    这条测试的价值在于它是**行为无关**的：数值恰好相等时功能看似正常，
    只有换了机器/改了 OLLAMA_NUM_CTX 才会暴露成"反复重载模型"。所以必须
    检查源码，而不是检查某个数值。
    """
    import re

    app_dir = Path(_BACKEND_ROOT) / "app"
    offenders: list[str] = []
    pattern = re.compile(r"num_ctx\s*[=:]\s*(.+)")
    for py in app_dir.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if "#" in line:
                line = line.split("#", 1)[0]
            m = pattern.search(line)
            if not m:
                continue
            expr = m.group(1).strip().rstrip(",")
            if "settings.chat_num_ctx" in expr:
                continue
            offenders.append(f"{py.relative_to(_BACKEND_ROOT)}:{lineno}: num_ctx={expr}")

    assert not offenders, (
        "以下位置没有走统一的 settings.chat_num_ctx —— Ollama 会因 num_ctx "
        "变化重载模型（实测 ≈11s/次）：\n  " + "\n  ".join(offenders)
    )


# ── 2. 清理顺序 ──────────────────────────────────────────────────────────────

def test_delete_vectors_for_documents_is_noop_on_empty() -> None:
    """空集合不发请求、返回 0（空 MatchAny 的语义别赌在服务端）。"""
    from scripts._e2e_purge import delete_vectors_for_documents

    assert asyncio.run(delete_vectors_for_documents([])) == 0
    assert asyncio.run(delete_vectors_for_documents(())) == 0


def test_purge_documents_refuses_unconditional_wipe() -> None:
    """无条件调用必须直接报错，而不是把整个文档库清掉。"""
    from scripts._e2e_purge import purge_documents

    try:
        asyncio.run(purge_documents())
    except ValueError as exc:
        assert "条件" in str(exc)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"应该是 ValueError，实际是 {type(exc).__name__}: {exc}"
        ) from exc
    else:
        raise AssertionError("purge_documents() 无条件调用竟然没有报错")


def test_delete_by_document_ids_batch_helper_exists() -> None:
    """批量向量删除入口存在，且空输入不发请求。"""
    from app.services.vector_service import delete_by_document_ids

    assert asyncio.run(delete_by_document_ids([])) == 0
    assert asyncio.run(delete_by_document_ids({"", None})) == 0


# ── 3. 阶段计时 ──────────────────────────────────────────────────────────────

def test_timed_stage_records_latency_and_propagates() -> None:
    """装饰器要：记时、返回值透传、异常照常抛出（不能吞异常）。"""
    from app.utils.timing import timed_stage

    seen: list[tuple[str, float]] = []

    import app.utils.timing as timing_mod

    original = timing_mod._record
    timing_mod._record = lambda stage, secs: seen.append((stage, secs))  # type: ignore[assignment]
    try:
        @timed_stage("unit-test-stage")
        async def ok() -> str:
            await asyncio.sleep(0.01)
            return "value"

        assert asyncio.run(ok()) == "value"
        assert seen and seen[0][0] == "unit-test-stage"
        assert seen[0][1] >= 0.0

        @timed_stage("unit-test-fail")
        async def boom() -> None:
            raise RuntimeError("kaboom")

        try:
            asyncio.run(boom())
        except RuntimeError as exc:
            assert "kaboom" in str(exc)
        else:
            raise AssertionError("异常被吞掉了")
        # 失败路径同样要记时 —— 超时/失败才是最需要耗时的场景
        assert any(s == "unit-test-fail" for s, _ in seen)
    finally:
        timing_mod._record = original  # type: ignore[assignment]


# ── 4. 精排相关性阈值（绝对下限 AND 相对分带）──────────────────────────────────

class _FakeChunk:
    """只带 ``filter_by_min_score`` 需要的字段，避免构造完整 RERANK 对象。"""

    def __init__(self, score: float, text: str = "") -> None:
        self.score = score
        self.text = text


def _ranked(*scores: float) -> list:
    return [_FakeChunk(s) for s in scores]


def test_relative_band_keeps_clear_winner_and_drops_noise() -> None:
    """
    金标场景复现：唯一真阳性 0.22，其余噪声 ≤0.0042。

    绝对阈值 0.25 会把整批丢掉（这就是"库里明明有却答没有"的成因）；
    此时靠**头名兜底**保住最高分那条，且**不得**把 0.004 量级的噪声带进来。
    （真正根治是把阈值校准到 0.05，见下面的
    ``test_rerank_threshold_below_weakest_true_positive``；本条的职责只剩
    "兜底最多只交出一条、不夹带噪声"。）
    """
    from app.services.reranker import filter_by_min_score

    kept = filter_by_min_score(_ranked(0.22, 0.0042, 0.0026, 0.0009), 0.25, ratio=0.10)
    assert [round(c.score, 4) for c in kept] == [0.22], (
        f"相对分带应只留清晰的最高分，实际={[round(c.score, 4) for c in kept]}"
    )


def test_relative_band_never_empties_nonempty_input() -> None:
    """只要精排有输入，阈值过滤不得返回空 —— 空结果会让上游误判"库里没有"。"""
    from app.services.reranker import filter_by_min_score

    for best in (0.9, 0.22, 0.05, 0.0001):
        kept = filter_by_min_score(_ranked(best, best / 2, best / 100), 0.25, ratio=0.10)
        assert kept, f"best={best} 时不应被清空"


def test_relative_band_actually_cuts_tail() -> None:
    """
    回归哨兵：相对分带必须**真的能砍尾**。（对应一次实测过的失效）

    旧实现把判据写成 ``score >= min_score or score >= band``，而
    ``band = best × ratio`` 恒 ≤ ``best`` ⇒ ``score >= band`` 比
    ``score >= min_score`` 更容易满足 ⇒ **分带只能放宽、永远不收紧**，
    它宣称的"砍掉与头名差太远的尾巴"从未生效。实测表现：把
    ``RERANK_MIN_SCORE_RATIO`` 从 0.00 扫到 0.50（7 档），金标集指标
    **逐位相同**（recall@10=1.0 / MRR=1.0 / precision@3=0.8182 / 平均返回
    2.09 条，无一档变化）—— 一个"没有作用的旋钮"比没有旋钮更危险，
    因为它让人以为该防线存在。

    这里用一组"全部过了绝对下限、但尾部明显远离头名"的分数钉死收紧行为。
    """
    from app.services.reranker import filter_by_min_score

    ranked = _ranked(0.9, 0.5, 0.06)          # 三条都 ≥ 0.05，但 0.06 与头名差 15 倍
    kept = filter_by_min_score(ranked, 0.05, ratio=0.10)
    assert [round(c.score, 4) for c in kept] == [0.9, 0.5], (
        f"分带没有收紧（判据可能又被写成了 OR）：实际={[round(c.score, 4) for c in kept]}"
    )
    # ratio=0 必须保持向后兼容：不做任何收紧
    assert len(filter_by_min_score(_ranked(0.9, 0.5, 0.06), 0.05, ratio=0.0)) == 3


def test_relative_band_never_weakens_calibrated_floor() -> None:
    """
    校准锁（第二条）：相对分带**不得**把生效下限降到绝对阈值之下。

    旧 OR 实现的生效下限是 ``min(min_score, best × ratio)`` —— 也就是
    ratio 一旦 > 0，配置里的绝对阈值就**不再是下限**。用实测的最弱真阳性
    ``best=0.22``、``ratio=0.10`` 举例：真实下限是 0.022 而不是 0.05，
    于是"距噪声上界约 100 倍"的余量论证（evidence_gate 与
    ``test_rerank_threshold_below_weakest_true_positive`` 都建立在
    "0.05 就是下限"之上）被静默作废。这条测试要求 0.03 这类
    "过不了绝对下限、但在分带之上"的候选**不得**被救回。
    """
    from app.services.reranker import filter_by_min_score

    kept = filter_by_min_score(_ranked(0.22, 0.03), 0.05, ratio=0.10)
    assert [round(c.score, 4) for c in kept] == [0.22], (
        "低于绝对下限的候选被相对分带救回 ⇒ 校准下限被削弱："
        f"实际={[round(c.score, 4) for c in kept]}"
    )


def test_ratio_zero_preserves_legacy_pure_absolute_behaviour() -> None:
    """
    ``ratio=0`` 必须与升级前逐字一致（可回滚，且兼容既有单测）。

    ⚠️ 这条测试同时看住一个**已经踩过的坑**：相对分带若写成
    ``band = best * ratio if ratio > 0 else float("-inf")``，再把判据写成
    ``score >= min_score or score >= band``，那么 ratio=0 时后半段恒真 ——
    整个阈值过滤被**静默关掉**（实测表现：标称"0.25 绝对阈值"的那一臂与
    "完全不过滤"那一臂指标逐位相同）。下面 ``min_score=0.95`` 必须返回空，
    就是用来钉死这个陷阱的。
    """
    from app.services.reranker import filter_by_min_score

    ranked = _ranked(0.9, 0.3, 0.2)
    assert filter_by_min_score(ranked, 0.25) == ranked[:2]
    assert filter_by_min_score(ranked, 0.95) == [], (
        "ratio=0 时阈值过滤被绕过（相对分带退化成了恒真条件）"
    )
    assert filter_by_min_score(ranked, 0.0) == ranked
    assert filter_by_min_score([], 0.25) == []


def test_rerank_threshold_below_weakest_true_positive() -> None:
    """
    **校准锁**：绝对阈值必须低于实测的"最弱真阳性"，否则会误杀金标。

    实测（容器内，BAAI/bge-reranker-base + 本仓库中文语料）：
        金标分片（真阳性）        0.22 ~ 0.9999   ← 最弱真阳性 = 0.22
        "库里根本没有"的问题      ≤ 0.0005       ← 3 条否定对照
    两者相差约 440 倍，可分；0.05 落在中间且有双侧余量。

    注意这条锁只约束"不误杀"这一侧。**噪声侧不做断言**：噪声上界会随语料
    变化，且它由 evidence_gate 的覆盖率信号共同把关，单靠分数不承担全部责任。
    换 reranker 模型或语料体裁后，最弱真阳性会变，这条锁会失败并要求重新校准。
    """
    from app.config import get_settings

    weakest_true_positive = 0.22    # 实测：11 例中最小的金标精排分
    thr = get_settings().RERANK_MIN_SCORE
    assert 0.0 < thr < weakest_true_positive, (
        f"RERANK_MIN_SCORE={thr} 不在 (0, {weakest_true_positive}) 内："
        "要么形同关闭（滤不掉'接近空'），要么会误杀实测金标 —— 需按当前模型重新校准"
    )


def test_single_candidate_still_gets_reranked() -> None:
    """
    **量纲不变式**：只要精排可用，``chunk.score`` 就必须是精排分。

    历史实现里两处 ``if ... and len(candidates) > 1:`` 会让"只有 1 条候选"时
    跳过精排，于是 ``chunk.score`` 保留**向量余弦分**（或关键词腿的 0.30 占位分），
    被下游按"精排卡住的相关性"解读 —— 实测这正是"库里没有的问题被回答"的成因。

    这里直接从源码层面看住：不允许再出现"按候选条数跳过精排"的写法。
    """
    import re

    src = (Path(_BACKEND_ROOT) / "app" / "services" / "retrieval_service.py").read_text(
        encoding="utf-8"
    )
    offenders = [
        f"line {i}: {line.strip()}"
        for i, line in enumerate(src.splitlines(), 1)
        if re.search(r"len\(\s*candidates\s*\)\s*[<>=!]+\s*1", line)
        and "rerank" in line.lower()
    ]
    assert not offenders, (
        "检索层又出现了「按候选条数跳过精排」的写法 —— 会让粗排分冒充精排分，"
        "破坏下游拒答判定：\n  " + "\n  ".join(offenders)
    )


def test_rerank_min_score_aligned_across_both_gates() -> None:
    """
    两道独立证据闸门（rag_graph 路由守卫 / evidence_gate 节点）的分数下限必须一致。

    只要有一个停在旧值，更严的那个就会继续把合格证据判为不合格 —— 修了等于没修。
    """
    from app.config import get_settings

    s = get_settings()
    assert s.RERANK_MIN_SCORE == s.EVIDENCE_GATE_MIN_TOP_SCORE, (
        "两道证据闸门的分数下限不一致："
        f"RERANK_MIN_SCORE={s.RERANK_MIN_SCORE} vs "
        f"EVIDENCE_GATE_MIN_TOP_SCORE={s.EVIDENCE_GATE_MIN_TOP_SCORE}"
    )


def test_refuse_guard_passes_measured_true_positive_and_still_refuses_noise() -> None:
    """
    端到端钉住用户可见行为：拒答守卫必须放行"实测真阳性"，仍拦"什么都没找到".

    只修检索层的阈值过滤是**不够**的 —— ``rag_graph._route_after_retrieve``
    复用同一个常数做拒答判定。若常数还停在 0.25，那么：

        检索层（相对分带）已经正确返回了 0.22 的金标分片
        拒答守卫仍然判定 ``0.22 < 0.25`` → 拒答

    用户看到的还是"知识库里没有"。所以这条测试同时覆盖"检索层不返回空"与
    "守卫不误拒"两个执行点。

    数值取自实测：0.22 = 11 例中最弱的真阳性；0.004 = 什么都没找到时的水位。
    """
    from app.config import get_settings
    from app.services.rag_graph import _route_after_retrieve

    class _C:
        def __init__(self, score: float) -> None:
            self.score = score

    s = get_settings()
    assert s.HALLUCINATION_GUARD_ENABLED, "幻觉守卫被关掉时本测试无意义"

    # 实测真阳性（最弱的那条）→ 必须放行去生成
    assert _route_after_retrieve({"chunks": [_C(0.22)]}) == "generate", (
        f"实测真阳性 0.22 被拒答了（当前下限 {s.RERANK_MIN_SCORE}）—— "
        "用户会得到'知识库里没有'，而答案就在库里"
    )
    # 实测"什么都没找到"水位 → 必须拒答（放宽下限不等于放弃拒答能力）
    assert _route_after_retrieve({"chunks": [_C(0.0042), _C(0.0026)]}) == "refuse", (
        "证据接近空时没有拒答 —— 会把无依据的问题硬答"
    )
    # 一条证据都没有 → 必须拒答
    assert _route_after_retrieve({"chunks": []}) == "refuse"
    assert _route_after_retrieve({}) == "refuse"


# ── 5. 相对分带的上界护栏（本次新增的判别力）─────────────────────────────────
#
# 背景：这一组存在的前提是"金标集里真的有次要证据样本"。旧集 11 例的金标块
# **全是精排第 1 名**（gold == head）⇒ gold/head ≡ 1.0 ⇒ 分带开到 1.0 也不掉分。
# 于是 `RERANK_MIN_SCORE_RATIO` 这个旋钮**从未被真正验证过**：0.10 是拍下来的，
# 而实测证明它已经在砍合法证据（次证 0.0685 < 带 0.0832）。补多证据用例后
# 第一次测出真实上界 0.07，下面三条把它钉住。


def test_rerank_ratio_below_hard_ceiling() -> None:
    """
    ``RERANK_MIN_SCORE_RATIO`` 必须严格小于实测安全上界（护栏常量）.

    为什么护栏常量刻意**不是** Settings 字段：Settings 的每个字段都可被环境
    变量覆盖（env > default）。做成字段等于"护栏可以被一个 .env 残留旧值改掉" ——
    本仓库实测踩过：`.env` 里 `RERANK_MIN_SCORE=0.25` 会静默覆盖代码里的 0.05，
    重构镜像后旧 bug 原样复活。所以它是模块级常量，环境变量改不动。
    """
    from app.config import RERANK_MIN_SCORE_RATIO_CEILING, Settings, get_settings

    s = get_settings()
    ratio = float(s.RERANK_MIN_SCORE_RATIO)
    assert ratio < RERANK_MIN_SCORE_RATIO_CEILING, (
        f"RERANK_MIN_SCORE_RATIO={ratio} 已达到/超过实测上界 "
        f"{RERANK_MIN_SCORE_RATIO_CEILING} —— 至少一条合法次要证据会被相对分带"
        "砍掉（答案只答一半，且不报错）。换语料/换精排模型后需重新测量上界。"
    )
    assert ratio >= 0.0, "负值无意义（会被当成 0 处理，静默改变语义）"
    # 护栏本身不可被环境变量覆盖 —— 这是它存在的意义，必须显式验证
    assert "RERANK_MIN_SCORE_RATIO_CEILING" not in Settings.model_fields, (
        "护栏常量被做成了 Settings 字段 —— 环境变量即可覆盖它，护栏失效"
    )


def test_rerank_threshold_below_weakest_legit_secondary_evidence() -> None:
    """
    **校准锁（补次要证据口径）**：绝对阈值也必须低于含次要证据的实测最弱合法证据.

    原有 `test_rerank_threshold_below_weakest_true_positive` 只用「最弱**主**
    证据 0.22」校准，那个数只覆盖单证据用例。加入多证据用例后实测到更弱的合法
    证据：**0.0582**（过带前 0.0685，被 PARENT_SCORE_DECAY=0.85 折到 0.0582）。
    下限 0.05 距它只有 1.16 倍余量 —— 薄，但方向正确（在下限之下）。

    这条锁的价值在于它把"余量薄"这个事实写进了测试：谁再把 PARENT_SCORE_DECAY
    调低（< 0.72 左右，或把下限抬高），这条会立刻失败并要求重新校准，而不是等到
    线上出现"答一半"才发现。
    """
    from app.config import get_settings

    weakest_legit_secondary = 0.0582     # 实测：多证据用例中过完整条流水线的最弱合法证据
    s = get_settings()
    thr = float(s.RERANK_MIN_SCORE)
    assert 0.0 < thr < weakest_legit_secondary, (
        f"RERANK_MIN_SCORE={thr} 不低于含次要证据的最弱合法证据 "
        f"{weakest_legit_secondary} —— 会误杀多证据场景的次要证据（答案只答一半）"
    )
    # 余量提醒：小于 1.5 倍属于"薄"，改邻近参数（PARENT_SCORE_DECAY / 精排模型）
    # 之前必须先重测，别只看这条测试还是绿的。
    assert weakest_legit_secondary / thr < 1.5, (
        f"余量已扩大到 {weakest_legit_secondary / thr:.2f} 倍 —— "
        "说明实测值被更新过，请同步更新本测试里的常量与 config.py 的注释"
    )


def test_env_file_does_not_override_calibrated_thresholds() -> None:
    """
    ``.env`` / ``.env.example`` 里若出现这几个键，其值必须与代码默认值一致.

    守的是一条**极隐蔽**的复活路径（本次实测踩到并修复）：pydantic-settings 的
    优先级是 env > default，所以 `.env` 里一个残留旧值会静默覆盖修好的代码。
    表现为：代码 review 通过、diff 干净、容器重构后 bug 原样回来，而没有任何
    报错。`docker compose config` 已证实 `.env` 的 `RERANK_MIN_SCORE=0.25`
    会被原样代入容器。

    两个文件都查：`.env` 是**实际生效**的那个；`.env.example` 是新人/新环境
    复制粘贴的来源，它写错等于把坑批量分发。
    """
    import re

    from app.config import get_settings

    s = get_settings()
    keys = {
        "RERANK_MIN_SCORE": float(s.RERANK_MIN_SCORE),
        "RERANK_MIN_SCORE_RATIO": float(s.RERANK_MIN_SCORE_RATIO),
        "EVIDENCE_GATE_MIN_TOP_SCORE": float(s.EVIDENCE_GATE_MIN_TOP_SCORE),
    }
    root = Path(_BACKEND_ROOT)
    checked: list[str] = []
    problems: list[str] = []
    for name in (".env", ".env.example"):
        path = root / name
        if not path.exists():
            continue                       # 容器内通常没有 .env（走 env_file 注入）
        checked.append(name)
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key not in keys:
                continue
            try:
                actual = float(value.strip().strip('"').strip("'"))
            except ValueError:
                problems.append(f"{name}:{lineno} {key}={value.strip()!r} 不是数字")
                continue
            expected = keys[key]
            if abs(actual - expected) > 1e-9:
                problems.append(
                    f"{name}:{lineno} {key}={actual} 覆盖了代码默认值 {expected}"
                )
    if not checked:
        print("  skip test_env_file_does_not_override_calibrated_thresholds "
              "(no .env/.env.example in this environment)")
        return
    assert not problems, (
        "环境变量会**静默覆盖**校准好的代码默认值（env > default）—— "
        "重构镜像后旧 bug 会复活且无任何报错：\n  " + "\n  ".join(problems)
    )
    print(f"  ok test_env_file_does_not_override_calibrated_thresholds ({', '.join(checked)})")


def test_golden_set_still_has_band_observability() -> None:
    """
    金标集必须**保留**能测出分带风险的结构，否则上面的护栏会退化成恒真.

    这是对"判别力地基"本身的回归锁。三种退化都会被抓住：
      1. 有人把多证据用例删回去 → 没有 ≥2 金标的用例；
      2. 有人把声明的上界删掉或改成 1.0 → 门禁无法变红；
      3. 声明的上界与实测记录的最小比值不一致 → 两处声明各自漂移，
         其中一处形同虚设（与 judge() 里的常量对账是同一取向的冗余）。

    它不依赖容器（纯读 JSON），因此宿主机也能跑。
    """
    import json

    golden_path = Path(_BACKEND_ROOT) / "eval" / "golden_v1.json"
    if not golden_path.exists():
        print("  skip test_golden_set_still_has_band_observability (golden set not found)")
        return
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    multi = [c for c in golden["cases"] if len(c.get("expect") or ()) >= 2]
    assert multi, (
        "金标集里已经没有需要 ≥2 条证据的用例 —— 『提高 ratio 会不会误杀次要证据』"
        "在该集上又会变成恒真命题（这正是 rev3 修掉的结构性缺陷）"
    )

    thresholds = golden.get("thresholds") or {}
    ceiling = thresholds.get("max_rerank_min_score_ratio")
    assert ceiling is not None, "缺少 thresholds.max_rerank_min_score_ratio，回归门禁无法判定分带风险"
    assert 0.0 < float(ceiling) < 1.0, f"上界 {ceiling} 不在 (0,1) 内，没有约束力"
    assert thresholds.get("min_multi_evidence_all_found_rate") == 1.0, (
        "多证据全召回率下限必须为 1.0 —— 多证据场景下漏一条就是『答案只答一半』"
    )

    # 声明上界必须等于实测记录里的最小比值（否则两处声明在漂移）
    from app.config import RERANK_MIN_SCORE_RATIO_CEILING

    assert abs(float(ceiling) - RERANK_MIN_SCORE_RATIO_CEILING) < 1e-9, (
        f"金标集声明上界 {ceiling} ≠ 代码护栏常量 {RERANK_MIN_SCORE_RATIO_CEILING}"
        " —— 必须同步修改"
    )
    measured = (golden.get("multi_evidence") or {}).get("probe_result", {}).get("measured") or []
    ratios = [m["ratio"] for m in measured if m.get("ratio") is not None]
    assert ratios, "multi_evidence.probe_result.measured 为空 —— 上界没有实测依据"
    assert abs(min(ratios) - float(ceiling)) < 1e-9, (
        f"实测最小比值 {min(ratios)} 与声明上界 {ceiling} 不一致"
    )
    # 至少有一条用例的比值**明显**低于 1.0，否则仍然只是"名义上可测"
    assert min(ratios) < 0.5, (
        f"所有多证据用例的 次证/头名 比值都 ≥ 0.5（最小 {min(ratios)}）—— "
        "弱证据不够弱，ratio 调到 0.5 才掉分，等于没有观测点"
    )
    print("  ok test_golden_set_still_has_band_observability")


def test_startup_guard_fires_on_bad_ratio_and_mismatched_floor() -> None:
    """
    启动护栏必须**真的报错**，而不只是一段注释.

    为什么值得单独测：这两个配置错误都不抛异常、不影响请求成功，只会让回答
    悄悄退化成"只答一半"或"答没有"。既然故障是静默的，护栏就必须是响的 ——
    而"内联在 lifespan 里的检查"没法在单测里跑，所以抽成了纯函数。

    最后一档用**真实生效的 settings** 断言"当前配置无告警"：这条把
    `.env` / 环境变量 / 代码默认值三者合成的结果一起验了 —— 单看代码默认值
    是绿的、单看 .env 也是绿的，但合成起来可能是错的（env 覆盖 default）。
    """
    from app.config import get_settings
    from app.main import check_retrieval_config_guards

    class _S:
        RERANK_MIN_SCORE_RATIO = 0.10      # 旧值，已在误杀次要证据
        RERANK_MIN_SCORE = 0.05
        EVIDENCE_GATE_MIN_TOP_SCORE = 0.05

    problems = check_retrieval_config_guards(_S())
    assert len(problems) == 1 and problems[0][0] == "error", (
        f"ratio 超过上界时必须报 error，实际={problems}"
    )
    assert "RERANK_MIN_SCORE_RATIO" in problems[0][1]

    class _S2:
        RERANK_MIN_SCORE_RATIO = 0.05
        RERANK_MIN_SCORE = 0.25            # 旧值：高于实测最弱真阳性 0.22
        EVIDENCE_GATE_MIN_TOP_SCORE = 0.05

    problems2 = check_retrieval_config_guards(_S2())
    assert len(problems2) == 1 and "不一致" in problems2[0][1], (
        f"两道闸门下限不一致时必须报 error，实际={problems2}"
    )

    # 真实生效配置必须干净 —— 这条同时守住 .env 悄悄漂移的情况
    live = check_retrieval_config_guards(get_settings())
    assert live == [], (
        "当前**生效**的检索配置存在告警（说明某个来源覆盖了校准好的值）：\n  "
        + "\n  ".join(m for _, m in live)
    )
    print("  ok test_startup_guard_fires_on_bad_ratio_and_mismatched_floor")