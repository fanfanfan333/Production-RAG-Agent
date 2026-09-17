"""
检索链路优化单测（Hybrid Search 加权 RRF / Reranker 去重与阈值 / Context Compression）.

覆盖：
  * rrf_fuse 加权融合 —— 权重高的 rank list 应主导融合排名；
  * reranker.dedup_candidates —— 文本重复候选只保留粗排最高位；
  * reranker.filter_by_min_score —— Relevance Threshold 逐条过滤；
  * context_builder.compress_text —— 查询感知句子级压缩：
    相关句保留、无关句裁掉、预算不被突破、无查询时安全截断；
  * context_builder.build_context —— 单源预算 + 总预算 + [Source N] 连续编号。

导入依赖 app.config（pydantic-settings）等完整后端依赖，
宿主机没装依赖时会跳过（exit 0）；推荐在 backend 容器内运行：
    docker cp tests rag_backend:/app/tests
    docker exec rag_backend python tests/test_retrieval_optimization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = str(Path(__file__).resolve().parent.parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

try:
    from app.services.hybrid_search import rrf_fuse
    from app.services.reranker import dedup_candidates, filter_by_min_score
    from app.services.nodes import context_builder as cb
except ImportError as exc:  # 宿主机缺依赖 → 跳过（容器内已验证）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _module_skip import skip_module

    # 不能用 sys.exit()：pytest 在收集阶段导入本模块，抛 SystemExit 会让整个
    # 会话 INTERNALERROR，同目录其它用例全部跑不了。
    skip_module(f"missing dependency ({exc}) — run inside the backend container")


# ── 1. 加权 RRF ──────────────────────────────────────────────────────────────

def test_weighted_rrf() -> None:
    a, b, c = "a", "b", "c"
    vector_leg = [a, b]     # a 仅在向量腿第 1
    bm25_leg = [c, b]       # c 仅在 BM25 腿第 1；b 两腿都有

    # 等权（经典 RRF）：a 与 c 各自只出现在一条腿的同一名次 → 分数相同
    even = rrf_fuse([vector_leg, bm25_leg], k=60)
    assert even[a] == even[c], "等权同名次应同分"
    assert even[b] > even[a], "两腿都命中的 b 应领先"

    # BM25 权重放大 10 倍：c（BM25 第 1）应反超 a（向量第 1）
    weighted = rrf_fuse([vector_leg, bm25_leg], k=60, weights=[1.0, 10.0])
    assert weighted[c] > weighted[a], "BM25 权重放大后 c 应反超"
    assert weighted[a] == even[a], "向量腿权重未变，a 的分数不变"
    assert weighted[b] > even[b], "b 在 BM25 腿的部分应获得加权"
    # 权重不应给未出现的 key 打分
    assert "z" not in weighted
    # weights 长度不足时按 1.0 兜底，不抛异常
    partial = rrf_fuse([vector_leg, bm25_leg], k=60, weights=[1.0])
    assert abs(partial[c] - even[c]) < 1e-9
    print("  ok test_weighted_rrf")


# ── 2. Reranker 去重 ─────────────────────────────────────────────────────────

class _Chunk:
    """
    Chunk 替身（reranker + context_builder 用到的字段）.

    注意：context_builder.build_context 会读位置信息（line_span /
    position_span / location_label）与图片字段（content_type / image_* /
    analyze_*）。这些字段是 RetrievedChunk 的正式契约，桩必须与之一致，
    否则 build_context 这类核心函数无法被单测覆盖。
    """

    def __init__(
        self,
        text: str,
        score: float = 0.5,
        *,
        line_start: int | None = None,
        line_end: int | None = None,
        content_type: str = "text",
        position: int | None = None,
    ):
        self.text = text
        self.score = score
        self.document_id = "doc"
        self.filename = "test.pdf"
        self.page_number = 1
        self.chunk_index = 0
        # ── 位置信息（细粒度引用）─────────────────────────────────────────
        self.line_start = line_start
        self.line_end = line_end
        self.position = position
        self.bbox = None
        # ── 内容类型与图片信息 ────────────────────────────────────────────
        self.content_type = content_type
        self.image_id = None
        self.image_path = None
        self.image_caption = None
        self.image_type = None
        self.analyze_engine = None
        self.analyze_confidence = 0.0
        self.manual_review = False
        self.analyze_quality = {}
        self.analyze_fusion = {}
        self.quality_score = None
        # ── small-to-big ─────────────────────────────────────────────────
        self.parent_text = None
        self.parent_id = None
        self.parent_char_start = None
        self.parent_char_end = None

    @property
    def line_span(self) -> str | None:
        if self.line_start is None:
            return None
        if self.line_end is None or self.line_end == self.line_start:
            return str(self.line_start)
        return f"{self.line_start}-{self.line_end}"

    @property
    def position_span(self) -> str | None:
        if not self.position:
            return None
        noun = "个表格" if self.content_type == "table" else "张图"
        return f"第 {self.position} {noun}"

    def location_label(self) -> str:
        parts = [self.filename or "未知文档"]
        if self.page_number:
            parts.append(f"第 {self.page_number} 页")
        span = self.line_span
        if span:
            parts.append(f"第 {span} 行")
        elif self.position_span:
            parts.append(self.position_span)
        return f"《{parts[0]}》" + (
            "，" + "，".join(parts[1:]) if len(parts) > 1 else ""
        )

    def __repr__(self):
        return f"_Chunk({self.text[:20]!r}, {self.score})"


def test_dedup_candidates() -> None:
    dup = "重复文本" * 50
    chunks = [
        _Chunk("第一名 独有文本"),
        _Chunk(dup, 0.9),      # 重复文本的第一次出现（保留）
        _Chunk("另一个不同文本"),
        _Chunk(dup, 0.7),      # 重复文本的第二次出现（丢弃）
    ]
    unique = dedup_candidates(chunks)
    assert len(unique) == 3
    assert unique[1] is chunks[1], "应保留粗排名次靠前（先出现）的那个"
    assert all(not (c is chunks[3]) for c in unique)
    # 全无重复时原样返回
    assert dedup_candidates([_Chunk("x"), _Chunk("y")]).__len__() == 2
    print("  ok test_dedup_candidates")


# ── 3. Relevance Threshold ───────────────────────────────────────────────────

def test_filter_by_min_score() -> None:
    ranked = [_Chunk("a", 0.9), _Chunk("b", 0.5), _Chunk("c", 0.2)]
    kept = filter_by_min_score(ranked, 0.25)
    assert [c.text for c in kept] == ["a", "b"]
    # 全员低于阈值 → 空列表（交给上层 retry/refuse，而不是硬塞噪声）
    assert filter_by_min_score(ranked, 0.95) == []
    assert filter_by_min_score(ranked, 0.0) == ranked
    print("  ok test_filter_by_min_score")


# ── 4. Context Compression：句子级压缩 ────────────────────────────────────────

_IRRELEVANT = "今天天气不错。公司团建去了郊外爬山。晚饭吃了火锅，味道很好。"
_RELEVANT = "ABX-300 型号设备的额定功率是 1500 瓦。"

def test_compress_text_keeps_query_relevant_sentences() -> None:
    query = "ABX-300 的额定功率是多少？"
    body = (_IRRELEVANT * 10) + _RELEVANT + (_IRRELEVANT * 10)
    budget = 120

    compressed = cb.compress_text(body, query, budget)
    assert len(compressed) <= budget, "压缩结果不得超过预算"
    assert "1500 瓦" in compressed, "与查询相关的证据句必须保留"
    # 预算装不下全部内容时，无关句让位给相关句（大幅被裁）
    assert compressed.count("火锅") < body.count("火锅"), "无关句子应被裁掉"
    print("  ok test_compress_text_keeps_query_relevant_sentences")


def test_compress_text_fallbacks() -> None:
    body = "短文本不需要压缩。"
    assert cb.compress_text(body, "查询", 1000) == body, "未超预算应原样返回"
    # 无查询 → 退化为按句子边界安全截断（不是硬砍字数）
    long_body = "。" .join(f"句子{i}内容" for i in range(50)) + "。"
    out = cb.compress_text(long_body, None, 40)
    assert len(out) <= 40 and out.endswith("。"), "无查询时按句子边界截断"
    # 全是超长单句 → 保底取首句截断，绝不返回空
    huge_sentence = "长" * 500
    assert cb.compress_text(huge_sentence, "查询", 100) == "长" * 100
    print("  ok test_compress_text_fallbacks")


# ── 5. build_context 预算与编号 ──────────────────────────────────────────────

class _FakeSettings:
    """只带 build_context 用到的字段，避免读真实 .env."""

    CONTEXT_COMPRESSION_ENABLED = True
    CONTEXT_MAX_CHARS_PER_SOURCE = 200
    CONTEXT_MAX_TOTAL_CHARS = 480
    CONTEXT_MIN_SOURCE_CHARS = 60
    HIERARCHICAL_RAG_ENABLED = False


def test_build_context_budget_and_numbering() -> None:
    original_get_settings = cb.get_settings
    cb.get_settings = lambda: _FakeSettings()
    try:
        chunks = [
            _Chunk(("无关填充。" * 40) + _RELEVANT, 0.9) for _ in range(4)
        ]
        built = cb.build_context(
            chunks,  # type: ignore[arg-type]
            query="ABX-300 的额定功率",
        )
        # 总预算软上限：每源最多 200，末位保底 60 → 4 源最多 ~480+ 一点余量
        assert len(built.context) <= 480 + 3 * 60 + 200, (
            f"上下文超出预算: {len(built.context)}"
        )
        # 编号必须连续，且与 sources 一一对应（output_guard 的 Citation Check 依赖它）
        for i in range(1, len(chunks) + 1):
            assert f"[Source {i}]" in built.context
            assert built.sources[i - 1]["score"] == chunks[i - 1].score
        # 相关证据句被保留
        assert "1500 瓦" in built.context
        assert built.compressed_count == 4
    finally:
        cb.get_settings = original_get_settings
    print("  ok test_build_context_budget_and_numbering")


def test_build_context_disabled_compression() -> None:
    class _Off(_FakeSettings):
        CONTEXT_COMPRESSION_ENABLED = False

    original = cb.get_settings
    cb.get_settings = lambda: _Off()
    try:
        chunks = [_Chunk("句子。" * 1000)]
        built = cb.build_context([chunks[0]], query="q")  # type: ignore[arg-type]
        assert built.compressed_count == 0
        assert len(built.context) > 2000, "关闭压缩时应保持原行为（全文拼接）"
    finally:
        cb.get_settings = original
    print("  ok test_build_context_disabled_compression")


if __name__ == "__main__":
    test_weighted_rrf()
    test_dedup_candidates()
    test_filter_by_min_score()
    test_compress_text_keeps_query_relevant_sentences()
    test_compress_text_fallbacks()
    test_build_context_budget_and_numbering()
    test_build_context_disabled_compression()
    print("\nAll retrieval-optimization tests passed ✅")
