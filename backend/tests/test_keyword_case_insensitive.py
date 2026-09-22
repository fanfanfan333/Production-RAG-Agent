"""
英文词检索的**大小写无关**契约（回归门禁）.

需求原话：「用户提问英文词语时，检索时大小写应该 S 和 s 是一样的」。

这条契约在本仓库里由**两个独立环节**共同保证，任何一处改动都可能只破坏
其中一环而两边的单测都还是绿的，因此这里把它拆成四组断言逐环锁死：

    ① 分词层   ``hybrid_search.tokenize``        —— 大小写折叠的唯一发生地
    ② 索引层   ``pg_keyword_search.terms_for_text``  —— 入库词项串恒为小写
    ③ 查询层   ``pg_keyword_search.build_tsquery``   —— 查询词元恒为小写，
               与索引层**同一套切分**（两侧口径分叉 = 永远查不到，且不报错）
    ④ 判定层   ``evidence_gate`` / ``citation_verifier`` / ``reranker``
               —— 覆盖率与支持度比对必须同口径，否则"检索到了却被判不相关"

⚠️ ⑤ 是**阳性对照**：只断言 `tokenize("ABC") == tokenize("abc")` 是假门禁 ——
两边都是**空集**时该断言同样成立。因此每组都同时断言"折叠确实发生了"
（结果里是小写形式）与"非空"（查询真的产生了词元）。

纯函数、零 DB、零网络、零模型：可在宿主机直接跑。
"""

from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# ① 分词层：大小写折叠的唯一发生地
# ═══════════════════════════════════════════════════════════════════════════════


def test_tokenize_folds_ascii_case() -> None:
    from app.services.hybrid_search import tokenize

    for upper, lower in [
        ("S", "s"),
        ("REVENUE", "revenue"),
        ("MacBook", "macbook"),
        ("ABX-300", "abx-300"),
        ("iPhone15 Pro", "iphone15 pro"),
    ]:
        hi, lo = tokenize(upper), tokenize(lower)
        assert hi == lo, f"{upper!r} 与 {lower!r} 必须切出同一批词元：{hi} vs {lo}"
        assert hi, f"{upper!r} 不该切出空集（阳性对照：空集会让上面的断言变成假门禁）"
        # 折叠方向：是小写，不是大写
        assert all(t == t.lower() for t in hi), f"{upper!r} → {hi} 里出现了未折叠的词元"


def test_tokenize_keeps_cjk_unaffected() -> None:
    """中文不受大小写折叠影响（回归护栏：别把 CJK 也误伤成别的形式）。"""
    from app.services.hybrid_search import tokenize

    assert tokenize("知识库") == ["知识", "识库"]
    assert tokenize("知识库") == tokenize("知识库")


# ═══════════════════════════════════════════════════════════════════════════════
# ② 索引层：入库词项串
# ═══════════════════════════════════════════════════════════════════════════════


def test_terms_for_text_is_case_folded() -> None:
    from app.services.pg_keyword_search import terms_for_text

    for upper, lower in [
        ("REVENUE GROWTH", "revenue growth"),
        ("ABX-300", "abx-300"),
        ("ACME 公司 2024 年 营收", "acme 公司 2024 年 营收"),
    ]:
        hi, lo = terms_for_text(upper), terms_for_text(lower)
        assert hi == lo, f"入库词项串必须与大小写无关：{hi!r} vs {lo!r}"
        assert hi, "词项串不该为空（空串会让索引侧永远无法命中）"

    # 折叠方向 + 去连字符：型号 "ABX-300" 落库为两个小写词元
    terms = terms_for_text("ABX-300")
    assert "abx" in terms.split() and "300" in terms.split()
    assert "ABX" not in terms.split(), "未折叠的大写词元不得出现在索引词项串里"


# ═══════════════════════════════════════════════════════════════════════════════
# ③ 查询层：tsquery 构造（与索引层同一套切分）
# ═══════════════════════════════════════════════════════════════════════════════


def test_build_tsquery_is_case_folded() -> None:
    from app.services.pg_keyword_search import build_tsquery, is_query_usable

    for upper, lower in [
        ("S", "s"),
        ("REVENUE", "revenue"),
        ("ABX-300", "abx-300"),
        ("MacBook Pro 的保修期", "macbook pro 的保修期"),
    ]:
        hi, lo = build_tsquery(upper), build_tsquery(lower)
        assert is_query_usable(hi), f"{upper!r} 必须产出可用 tsquery（阳性对照）"
        assert hi == lo, f"{upper!r} 与 {lower!r} 必须产出同一个 tsquery：{hi!r} vs {lo!r}"

    # 小写形式必须真的在 tsquery 里 —— 否则两边都为空串也算"相等"
    tq = build_tsquery("REVENUE")
    assert tq == "revenue", tq


def test_build_tsquery_single_letter_s() -> None:
    """需求里点名的 S / s：单字母词元也要一致（且不被当成操作符）。"""
    from app.services.pg_keyword_search import build_tsquery

    assert build_tsquery("S") == build_tsquery("s") == "s"


def test_index_and_query_terms_align() -> None:
    """
    索引侧与查询侧的**词元集合**必须能对上 —— 这是"检索突然查不到"最隐蔽的成因：
    两边分词口径不同，但各自都跑得好好的。
    """
    from app.services.pg_keyword_search import build_tsquery, terms_for_text

    doc_text = "The ACME-300 device supports 5G."
    query = "acme-300"                      # 用户小写提问
    query_upper = "ACME-300"                # 用户大写提问

    indexed = set(terms_for_text(doc_text).split())
    for q in (query, query_upper):
        wanted = {t for t in build_tsquery(q).split(" | ")}
        assert wanted, f"{q!r} 没切出任何词元"
        assert wanted <= indexed, (
            f"查询 {q!r} 的词元 {wanted - indexed} 在索引里不存在 —— "
            "两侧切分口径已分叉"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ④ 判定层：覆盖率 / 支持度 / 启发式精排
# ═══════════════════════════════════════════════════════════════════════════════


def test_evidence_gate_coverage_is_case_insensitive() -> None:
    from app.services.nodes.evidence_gate import content_terms, query_coverage

    assert content_terms("ABC") == content_terms("abc")
    assert content_terms("Revenue") == {"revenue"}
    # 大小写混写的查询由小写证据完全覆盖 → 不得因大小写被门控拒答
    assert query_coverage("REVENUE Growth", "revenue growth 12%") == 1.0
    assert query_coverage("S", "s") == 1.0


def test_citation_support_ratio_is_case_insensitive() -> None:
    from app.services.nodes.citation_verifier import _support_ratio

    assert _support_ratio("Revenue grew 12% [Source 1]", "revenue grew 12%") == 1.0
    assert _support_ratio("ACME 的营收 [Source 1]", "acme 的营收") == 1.0


def test_heuristic_rerank_is_case_insensitive() -> None:
    """Tier 3 启发式精排（无模型环境）的覆盖率信号也必须同口径。"""
    from app.services.reranker import _heuristic_scores

    texts = ["revenue growth of acme"]
    vec = [0.6]
    assert _heuristic_scores("REVENUE", texts, vec) == _heuristic_scores("revenue", texts, vec)
    assert _heuristic_scores("ACME-300", texts, vec) == _heuristic_scores("acme-300", texts, vec)


if __name__ == "__main__":      # pragma: no cover - 手工执行入口
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"[OK] {_name}")
    pytest.main([__file__, "-q"])
