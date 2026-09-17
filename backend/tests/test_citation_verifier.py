"""
Citation Verifier 单元测试（五项引用校验）.

覆盖设计稿的五项检查：
  引用存在？        引用了不存在的 [Source N]
  引用位置正确？    结论其实出自另一条来源
  原文支持该结论？  被引原文与该句内容无交集
  数字是否一致？    句中的数字在被引原文里找不到
  日期是否一致？    句中的日期在被引原文里找不到

以及：净化行为（移除存疑引用标记）、校验脚注、纯函数工具（数字/日期抽取）。

纯函数测试，不启动后端、不连数据库。直接 python 运行即可。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# 桩包：避免触发 app/services/__init__.py 的重依赖（同 test_evidence_gate.py）
if "app.services" not in sys.modules:
    _pkg = types.ModuleType("app.services")
    _pkg.__path__ = [str(_BACKEND_ROOT / "app" / "services")]
    sys.modules["app.services"] = _pkg

_MOD_PATH = _BACKEND_ROOT / "app" / "services" / "nodes" / "citation_verifier.py"
_spec = importlib.util.spec_from_file_location(
    "app.services.nodes.citation_verifier", _MOD_PATH
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["app.services.nodes.citation_verifier"] = _mod
_spec.loader.exec_module(_mod)

verify_citations = _mod.verify_citations
extract_numbers = _mod.extract_numbers
extract_dates = _mod.extract_dates
split_sentences = _mod.split_sentences
split_sentences_with_spans = _mod.split_sentences_with_spans
find_evidence_spans = _mod.find_evidence_spans
strip_structural_markers = _mod.strip_structural_markers
prepare_for_checks = _mod.prepare_for_checks


def _sources(*texts: str) -> list[dict]:
    """构造与 [Source N] 一一对应的 sources 列表."""
    return [
        {
            "document_id": f"doc-{i}",
            "filename": f"文档{i}.pdf",
            "page_number": i,
            "line_start": 10 * i,
            "line_end": 10 * i + 8,
            "text": t,
        }
        for i, t in enumerate(texts, start=1)
    ]


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def test_extract_numbers_normalizes():
    got = extract_numbers("营收 1,200 万元，同比增长 12.5%，覆盖率 95%")
    assert "1200" in got, got
    assert "12.5%" in got, got
    assert "95%" in got, got
    print("[OK] test_extract_numbers_normalizes")


def test_extract_dates_normalizes():
    got = extract_dates("2024年1月31日签署，2024-03-01 生效，有效期至 2025年")
    assert "2024-1-31" in got, got
    assert "2024-03-01" in got, got
    assert "2025" in got, got
    print("[OK] test_extract_dates_normalizes")


def test_split_sentences():
    got = split_sentences("第一句。第二句！第三句？")
    assert len(got) == 3, got
    print("[OK] test_split_sentences")


# ── 五项校验：通过 ───────────────────────────────────────────────────────────

def test_all_checks_pass():
    sources = _sources(
        "合同约定的付款期限为 30 天，违约金按日万分之五计算。",
        "验收标准以技术协议为准。",
    )
    answer = "合同约定的付款期限为 30 天，违约金按日万分之五计算 [Source 1]。"
    report = verify_citations(answer, sources)
    assert report.overall == "verified", report.as_audit()
    assert report.total == 1 and report.passed_count == 1
    v = report.verdicts[0]
    assert v.citation_exists and v.position_correct and v.supported
    assert v.numbers_consistent and v.dates_consistent
    assert v.location and "文档1.pdf" in v.location and "第 10-18 行" in v.location
    print("[OK] test_all_checks_pass")


# ── (1) 引用存在？────────────────────────────────────────────────────────────

def test_hallucinated_citation_removed():
    sources = _sources("付款期限为 30 天。")
    answer = "付款期限为 30 天 [Source 1]，违约金按日万分之五计算 [Source 7]。"
    report = verify_citations(answer, sources)
    assert report.hallucinated_indices == (7,), report.as_audit()
    assert report.overall == "partial"
    assert "[Source 7]" not in report.clean_text
    assert "[Source 1]" in report.clean_text
    print("[OK] test_hallucinated_citation_removed")


# ── (2) 原文支持该结论？──────────────────────────────────────────────────────

def test_unsupported_claim_flagged():
    sources = _sources("公司的注册地址为北京市海淀区。")
    answer = "该产品的市场占有率达到了百分之八十，处于绝对领先地位 [Source 1]。"
    report = verify_citations(answer, sources, min_support=0.5)
    assert report.unsupported_indices == (1,), report.as_audit()
    assert report.overall == "unsupported"
    assert "[Source 1]" not in report.clean_text
    print("[OK] test_unsupported_claim_flagged")


# ── (3) 引用位置正确？────────────────────────────────────────────────────────

def test_misattribution_flagged():
    # 结论其实完全出自 Source 2，却标了 Source 1
    sources = _sources(
        "公司的注册地址为北京市海淀区中关村大街一号。",
        "研发费用投入为 3.2 亿元，研发人员占比达到 45%，专利授权 210 件。",
    )
    answer = "研发费用投入为 3.2 亿元，研发人员占比达到 45%，专利授权 210 件 [Source 1]。"
    report = verify_citations(answer, sources)
    assert 1 in report.misattributed_indices, report.as_audit()
    v = report.verdicts[0]
    assert v.best_source == 2, v.as_dict()
    print("[OK] test_misattribution_flagged")


# ── (4) 数字是否一致？────────────────────────────────────────────────────────

def test_number_mismatch_flagged():
    sources = _sources("合同约定的付款期限为 30 天，违约金按日万分之五计算。")
    answer = "合同约定的付款期限为 90 天 [Source 1]。"
    report = verify_citations(answer, sources)
    assert report.number_mismatch_indices == (1,), report.as_audit()
    v = report.verdicts[0]
    assert "90" in v.missing_numbers, v.as_dict()
    assert v.numbers_consistent is False
    print("[OK] test_number_mismatch_flagged")


# ── (5) 日期是否一致？────────────────────────────────────────────────────────

def test_date_mismatch_flagged():
    sources = _sources("本协议自 2024年1月31日 起生效，有效期三年。")
    answer = "本协议自 2025年6月1日 起生效 [Source 1]。"
    report = verify_citations(answer, sources)
    assert report.date_mismatch_indices == (1,), report.as_audit()
    v = report.verdicts[0]
    assert "2025-6-1" in v.missing_dates, v.as_dict()
    print("[OK] test_date_mismatch_flagged")


def test_matching_date_and_number_pass():
    sources = _sources("本协议自 2024年1月31日 起生效，付款期限 30 天。")
    answer = "本协议自 2024年1月31日 起生效，付款期限 30 天 [Source 1]。"
    report = verify_citations(answer, sources)
    assert report.overall == "verified", report.as_audit()
    print("[OK] test_matching_date_and_number_pass")


# ── 净化 / 脚注 / 边界 ───────────────────────────────────────────────────────

def test_annotate_adds_note():
    sources = _sources("公司的注册地址为北京市海淀区。")
    answer = "该产品市场占有率百分之八十 [Source 1]。"
    report = verify_citations(answer, sources, min_support=0.5, annotate=True)
    assert report.annotated is True
    assert "引用校验" in report.clean_text
    print("[OK] test_annotate_adds_note")


def test_no_citations():
    sources = _sources("任意内容")
    report = verify_citations("这是一段没有任何引用的回答。", sources)
    assert report.overall == "no_citations"
    assert report.total == 0
    assert report.clean_text == "这是一段没有任何引用的回答。"
    print("[OK] test_no_citations")


def test_empty_answer():
    report = verify_citations("", _sources("x"))
    assert report.overall == "no_citations"
    assert report.clean_text == ""
    print("[OK] test_empty_answer")


def test_range_citation_out_of_bounds():
    sources = _sources("付款期限为 30 天。")
    answer = "付款期限为 30 天 [Source 1-4]。"
    report = verify_citations(answer, sources)
    assert report.hallucinated_indices == (2, 3, 4), report.as_audit()
    assert "[Source 1-4]" not in report.clean_text
    print("[OK] test_range_citation_out_of_bounds")


def test_strip_unsupported_can_be_disabled():
    sources = _sources("公司的注册地址为北京市海淀区。")
    answer = "市场占有率百分之八十 [Source 1]。"
    report = verify_citations(
        answer, sources, min_support=0.5, strip_unsupported=False, annotate=False
    )
    assert report.unsupported_indices == (1,)
    # 不做净化 → 原文保留
    assert "[Source 1]" in report.clean_text
    print("[OK] test_strip_unsupported_can_be_disabled")


def test_audit_payload_json_serializable():
    import json

    sources = _sources("付款期限为 30 天。")
    report = verify_citations("付款期限为 30 天 [Source 1]。", sources)
    payload = json.dumps(report.as_audit(), ensure_ascii=False)
    assert "verdicts" in payload and "overall" in payload
    assert "evidence" in payload, "命中句必须随 verdict 一起下发，否则前端无从高亮"
    print("[OK] test_audit_payload_json_serializable")


# ── 命中句回标（"答案实际用了哪几句"）────────────────────────────────────────
#
# 截图问题：引用卡片只给整块切片的行范围（第 83-105 行）+ 整段原文，用户
# 看到 23 行原文却不知道答案用的是哪几句。以下用例锁死"句级回标"的行为。

def test_split_sentences_with_spans_matches_plain_split():
    """带偏移量的分句必须与 split_sentences 结果一致（口径不能漂移）."""
    text = "第一句。第二句！\n\n第三句；第四句 without terminator"
    plain = split_sentences(text)
    with_spans = split_sentences_with_spans(text)
    assert [s for s, _, _ in with_spans] == plain, (plain, with_spans)
    for sentence, start, end in with_spans:
        assert text[start:end] == sentence, (sentence, text[start:end])
    print("[OK] test_split_sentences_with_spans_matches_plain_split")


def test_evidence_spans_point_at_the_cited_sentence():
    """命中句必须精确定位到被引片段里的那一句（含偏移与行号）."""
    s1 = "## 7 反幻觉机制"
    s2 = "除了提示词约束，系统在检索阶段把不可信产物降权，避免模型给出通顺但错误的结论。"
    s3 = "研发部与相关部门需要明确责任人、里程碑与验收口径，所有变更均须留下可追溯的记录。"
    text = "\n".join([s1, s2, s3])
    sources = [
        {
            "document_id": "doc-1",
            "filename": "研发部-2024年度技术方案.docx",
            "page_number": 1,
            "line_start": 83,
            "line_end": 85,
            "text_snippet": text,
        }
    ]
    answer = (
        "系统在检索阶段就把解析期已知不可信的产物降权，"
        "避免模型基于错误证据给出通顺但错误的结论 [Source 1]。"
    )
    report = verify_citations(answer, sources)
    evidence = report.verdicts[0].evidence
    assert len(evidence) == 1, report.as_audit()
    span = evidence[0]
    # 只标命中句本身 —— 不是整段，也不是标题行
    assert span.text == s2, span.text
    assert text[span.start:span.end] == s2, (span, text[span.start : span.end])
    # 行号：片段首行 83，命中句在第 2 行 → 84（闭区间单行）
    assert (span.line_start, span.line_end) == (84, 84), (span.line_start, span.line_end)
    assert span.ratio >= 0.34, span.ratio
    print("[OK] test_evidence_spans_point_at_the_cited_sentence")


def test_evidence_falls_back_to_best_sentence_and_gives_up_cleanly():
    """阈值不可达时回标最贴切的一句；毫无交集则不硬标（宁可少标）."""
    a = "平台采用混合检索策略，提升召回质量。"
    b = "文档切片保留行号，便于引用回溯到具体位置。"
    text = a + "\n" + b
    source = {
        "filename": "x.docx",
        "line_start": 5,
        "line_end": 6,
        "text_snippet": text,
    }
    spans = find_evidence_spans(
        "平台采用混合检索策略提升召回质量 [Source 1]。", source, min_ratio=1.0
    )
    assert len(spans) == 1 and spans[0].text == a, spans
    # 毫无交集 → 不标（标了等于制造"这段就是依据"的假象）
    assert find_evidence_spans("今天天气不错，适合出门散步。", source) == ()
    # 无行号（旧索引）时行号降级为 None，而不是伪造一个行号
    assert find_evidence_spans(
        "平台采用混合检索策略提升召回质量 [Source 1]。",
        {"filename": "x.docx", "text_snippet": text},
    )[0].line_start is None
    print("[OK] test_evidence_falls_back_to_best_sentence_and_gives_up_cleanly")


def test_evidence_respects_switch_and_skips_hallucinated_citation():
    """开关可关；引用了不存在的来源时不回标（不替幻觉背书）."""
    sources = _sources("付款期限为 30 天，违约金按日万分之五计算。")
    answer = "付款期限为 30 天，违约金按日万分之五计算 [Source 1]。"

    on = verify_citations(answer, sources)
    assert on.verdicts[0].evidence, on.as_audit()

    off = verify_citations(answer, sources, evidence_enabled=False)
    assert off.verdicts[0].evidence == (), off.as_audit()

    bad = verify_citations("付款期限为 30 天 [Source 9]。", sources)
    assert bad.verdicts[0].citation_exists is False
    assert bad.verdicts[0].evidence == (), bad.as_audit()
    print("[OK] test_evidence_respects_switch_and_skips_hallucinated_citation")


# ── 图片来源（视觉证据计入校验）────────────────────────────────────────────

def test_vision_conclusion_counts_as_evidence():
    """
    图片引用的结论常来自"看图"，校验必须把 Vision 结论算作证据.

    图片块进上下文时的形态是「OCR 文本 + 视觉分析」，模型据此作答。校验若
    只看 OCR 文本，所有基于"看图"得出的结论都会被判"不被原文支持"而被
    删掉 —— 图片引用会成片消失。
    """
    sources = [
        {
            "document_id": "doc-1",
            "filename": "架构说明.pdf",
            "page_number": 3,
            "content_type": "image",
            "image_id": "img-1",
            "position": 2,
            "text": "architecture diagram",          # OCR 只有少量英文
            "vision": "该架构图包含网关层、服务层与存储层三个模块",
        }
    ]
    answer = "该架构图包含网关层、服务层与存储层三个模块 [Source 1]。"
    report = verify_citations(answer, sources)

    assert report.overall == "verified", (
        f"基于视觉分析的正确引用被误判：{report.as_audit()}"
    )
    assert report.clean_text == answer, "不应删除这条引用"
    print("[OK] test_vision_conclusion_counts_as_evidence")


def test_full_text_preferred_over_truncated_snippet():
    """
    数字落在展示快照(300 字)之外时，不该被判"与原文不一致".

    text_snippet 是给前端展示的预览。若拿它当原文，原文里有、但被截掉的
    数字会被判缺失，正确引用遭删除 —— 这是"该引的没引上"，比多引一条更伤。
    """
    long_tail = "填充内容。" * 80 + "设备额定功率为 3200 瓦。"
    sources = [
        {
            "document_id": "doc-1",
            "filename": "手册.pdf",
            "page_number": 1,
            "text": long_tail,                       # 完整正文
            "text_snippet": long_tail[:300],         # 截断预览（不含 3200）
        }
    ]
    report = verify_citations("设备额定功率为 3200 瓦 [Source 1]。", sources)
    assert report.number_mismatch_indices == (), (
        "原文确实有 3200，不应判数字不一致"
    )
    assert report.overall == "verified", report.as_audit()
    print("[OK] test_full_text_preferred_over_truncated_snippet")


def test_image_location_falls_back_to_position():
    """无行号的图片来源 → 位置退化为『第 N 张图 / 第 N 个表格』."""
    img_source = {
        "document_id": "doc-1",
        "filename": "年报.pdf",
        "page_number": 3,
        "content_type": "image",
        "position": 2,
    }
    loc = _mod._location_of(img_source)
    assert loc and "第 3 页" in loc and "第 2 张图" in loc, loc

    table_source = dict(img_source, content_type="table", position=4)
    loc_t = _mod._location_of(table_source)
    assert loc_t and "第 4 个表格" in loc_t, loc_t
    print("[OK] test_image_location_falls_back_to_position")


# ── 排版标记剥离（截图问题4：列表序号被当成正文数字）──────────────────────

def test_strip_structural_markers_drops_list_numbers():
    """行首的 Markdown 序号/标题井号要剥掉，正文数字一个都不能动."""
    assert strip_structural_markers("1. 反幻觉机制") == "反幻觉机制"
    assert strip_structural_markers("2) 数据层设计") == "数据层设计"
    assert strip_structural_markers("3、服务层设计") == "服务层设计"
    assert strip_structural_markers("## 4. 实施计划") == "实施计划"
    assert strip_structural_markers("### 7 反幻觉机制") == "反幻觉机制"
    assert strip_structural_markers("- 风险与对策") == "风险与对策"
    assert strip_structural_markers("> 引用说明") == "引用说明"
    assert strip_structural_markers("（2）括注序号") == "括注序号"
    # 只吃序号本身，后面的正文数字必须留下
    assert "60%" in strip_structural_markers("3. 三期推进，节省约 60% 存储")
    # 真实数字不是序号（小数 / 年份 / 千分位）
    assert strip_structural_markers("2024年营收 3.5 亿元") == "2024年营收 3.5 亿元"
    assert strip_structural_markers("2,400 万元") == "2,400 万元"
    print("[OK] test_strip_structural_markers_drops_list_numbers")


def test_list_number_is_not_treated_as_fact():
    """
    回归：带列表序号的句子不得被判"数字与原文不一致".

    真实案例（截图4）：回答用 "1. **反幻觉机制**：… [Source 1]" 的
    Markdown 列表组织内容，行首序号 "1." 被抽成正文数字 1，原文里找不到，
    于是 6 条引用全被标"数字与原文不一致 / 4 条存疑"。序号是排版，不是事实。
    """
    source_text = (
        "反幻觉机制：通过检索阶段降权处理不可信文档片段，确保答案可核对、可复现。"
    )
    answer = (
        "1. **反幻觉机制**：通过检索阶段降权处理不可信文档片段，确保答案可核对、可复现。"
        "[Source 1]\n"
    )
    report = verify_citations(answer, _sources(source_text))
    assert report.number_mismatch_indices == (), (
        f"列表序号被当成正文数字：{report.as_audit()}"
    )
    assert report.total == 1 and report.passed_count == 1, report.as_audit()
    assert report.clean_text.strip() == answer.strip()
    print("[OK] test_list_number_is_not_treated_as_fact")


def test_real_number_inside_list_item_still_flagged():
    """剥离序号不能把真数字一起放过：列表项里的错数字仍要被抓出来."""
    sources = _sources("三期推进，每期结束需全链路验收，节省约 60% 存储空间。")
    answer = "3. 分三期推进，节省约 999% 存储空间 [Source 1]。"
    report = verify_citations(answer, sources)
    assert report.number_mismatch_indices == (1,), report.as_audit()
    assert "999%" in report.verdicts[0].missing_numbers
    print("[OK] test_real_number_inside_list_item_still_flagged")


def test_prepare_for_checks_removes_both():
    """预处理只剥排版标记；[Source N] 必须留给逐句校验（否则 total 直接归零）."""
    text = "1. 结论 [Source 2]\n## 3. 标题"
    cleaned = prepare_for_checks(text)
    assert "[Source 2]" in cleaned, cleaned
    assert cleaned.splitlines()[0].strip().startswith("结论"), cleaned
    assert "标题" in cleaned and "3. 标题" not in cleaned
    print("[OK] test_prepare_for_checks_removes_both")


# ── 模型输出的"非事实数字"不参与比对（截图问题1 回归）────────────────────────

def test_inline_enumeration_marker_not_treated_as_fact():
    """
    回归：一整段话里的句首枚举序号不得被判"数字与原文不一致".

    真实案例（截图1）：模型把答案写成"……核心要点包括：1. 反幻觉机制：…
    [Source 1]"—— 序号 "1." 不在行首而在分句后的**句首**，行首规则剥不到，
    于是 "1" 被当成正文数字，原文找不到，四条引用全灭。序号是排版，不是事实。
    """
    source_text = "反幻觉机制：通过检索阶段降权处理不可信文档片段，确保答案可核对。"
    answer = (
        "核心要点包括：1. 反幻觉机制：通过检索阶段降权处理不可信文档片段，"
        "确保答案可核对 [Source 1]。"
    )
    report = verify_citations(answer, _sources(source_text))
    assert report.number_mismatch_indices == (), (
        f"句首枚举序号被当成正文数字：{report.as_audit()}"
    )
    assert report.passed_count == 1, report.as_audit()
    print("[OK] test_inline_enumeration_marker_not_treated_as_fact")


def test_provenance_echo_not_treated_as_fact():
    """
    回归：模型复述出处元数据（《书名》、第 N 页、第 N-M 行）不参与数字比对.

    真实案例（截图1）：模型答 "出自《研发部-2024年度技术方案.docx》，
    第 1 页，第 83-105 行：反幻觉机制要求…… [Source 1]"，
    其中 2024 / 1 / 83 / 105 全是**位置信息**，原文 chunk 里根本没有页码，
    旧逻辑判"原文中找不到的数字：1"。
    """
    source_text = (
        "7 反幻觉机制 除了提示词约束，系统在检索阶段就把解析期已知不可信的"
        "产物降权：光学字符识别置信度不达标的片段，在排序阶段主动下沉。"
    )
    answer = (
        "出自《研发部-2024年度技术方案.docx》，第 1 页，第 83-105 行："
        "反幻觉机制要求系统在检索阶段把解析期已知不可信的产物降权，"
        "光学字符识别置信度不达标的片段在排序阶段主动下沉 [Source 1]。"
    )
    report = verify_citations(answer, _sources(source_text))
    assert report.number_mismatch_indices == (), (
        f"溯源元数据被当成正文数字：{report.as_audit()}"
    )
    assert report.date_mismatch_indices == (), (
        f"书名里的年份被当成正文日期：{report.as_audit()}"
    )
    assert report.passed_count == 1, report.as_audit()
    print("[OK] test_provenance_echo_not_treated_as_fact")


def test_date_components_not_double_checked_as_numbers():
    """
    日期成分由日期校验负责，数字校验不再重复查.

    句子 "2024年营收增长 30%"：2024 是日期的一部分，原文即使只写
    "去年营收增长 30%"，数字校验也不该拿 2024 说事（年份对不对由日期校验判）。
    """
    sources = _sources("去年营收增长 30%，超出预期。")
    answer = "2024年营收增长 30%，超出预期 [Source 1]。"
    report = verify_citations(answer, sources)
    assert report.number_mismatch_indices == (), (
        f"日期成分被数字校验重复检查：{report.as_audit()}"
    )
    # 年份缺失仍由**日期校验**抓出（职责不丢，只是不再双重报案）
    assert report.date_mismatch_indices == (1,), report.as_audit()
    print("[OK] test_date_components_not_double_checked_as_numbers")


def test_real_number_still_flagged_after_masks():
    """三层屏蔽不能把真数字一起放过：错数字仍要被抓出来."""
    sources = _sources("平台分三期推进，节省约 60% 存储空间。")
    answer = "出自《平台方案.docx》第 2 页：平台分三期推进，节省约 95% 存储空间 [Source 1]。"
    report = verify_citations(answer, sources)
    assert report.number_mismatch_indices == (1,), report.as_audit()
    assert "95%" in report.verdicts[0].missing_numbers
    print("[OK] test_real_number_still_flagged_after_masks")


if __name__ == "__main__":
    tests = [
        test_vision_conclusion_counts_as_evidence,
        test_full_text_preferred_over_truncated_snippet,
        test_image_location_falls_back_to_position,
        test_extract_numbers_normalizes,
        test_extract_dates_normalizes,
        test_split_sentences,
        test_strip_structural_markers_drops_list_numbers,
        test_list_number_is_not_treated_as_fact,
        test_real_number_inside_list_item_still_flagged,
        test_prepare_for_checks_removes_both,
        test_inline_enumeration_marker_not_treated_as_fact,
        test_provenance_echo_not_treated_as_fact,
        test_date_components_not_double_checked_as_numbers,
        test_real_number_still_flagged_after_masks,
        test_all_checks_pass,
        test_hallucinated_citation_removed,
        test_unsupported_claim_flagged,
        test_misattribution_flagged,
        test_number_mismatch_flagged,
        test_date_mismatch_flagged,
        test_matching_date_and_number_pass,
        test_annotate_adds_note,
        test_no_citations,
        test_empty_answer,
        test_range_citation_out_of_bounds,
        test_strip_unsupported_can_be_disabled,
        test_audit_payload_json_serializable,
        test_split_sentences_with_spans_matches_plain_split,
        test_evidence_spans_point_at_the_cited_sentence,
        test_evidence_falls_back_to_best_sentence_and_gives_up_cleanly,
        test_evidence_respects_switch_and_skips_hallucinated_citation,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} citation-verifier tests passed.")
