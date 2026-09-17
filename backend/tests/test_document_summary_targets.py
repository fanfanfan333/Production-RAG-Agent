"""
「总结指定文档」的范围解析单元测试.

回归背景（截图问题1+2）：
  * "总结所有文档" 被路由成知识问答 → 只总结了召回分最高的那一份文档；
  * 用户点名某份文档时，没有任何机制把总结范围收敛到那一份。

本文件覆盖 resolve_summary_targets / build_target_not_found_answer 的确定性部分：
  * 未点名 → 整库（ids 为空、不报"没找到"）；
  * 书名号 / 带扩展名的文件名 / "…文档"限定语 → 精确收敛到那一份；
  * 近似重名文档（同名不同后缀）只取最贴合的一批，不把不相干的拖进来；
  * 点名的文档不存在 → missing=True 并给出可选清单，而不是硬编一个总结；
  * "第 N 份文档" → 按列表顺序取；
  * 弱信号（"总结一下第三章"）不能误判成"你点名的文档不存在"。

纯逻辑测试，不连数据库、不调 LLM（documents 直接构造）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.nodes.document_summary_node import (  # noqa: E402
    build_target_not_found_answer,
    resolve_summary_targets,
)

# 与截图里的知识库一致：三份近似重名 + 一份完全不同的
_DOCS = [
    ("id-1", "研发部-2024年度技术方案-2e19.docx"),
    ("id-2", "研发部-2024年度技术方案-nqkxx.docx"),
    ("id-3", "研发部-2024年度技术方案-1uuc.docx"),
    ("id-4", "Python AI大模型成神手册 (1).docx"),
]


def test_no_target_means_whole_library():
    for q in ("总结所有文档", "把所有文档总结一下", "总结一下知识库", "概览一下库里的文档"):
        scope = resolve_summary_targets(q, _DOCS)
        assert scope["ids"] == [], (q, scope)
        assert scope["missing"] is False, (q, scope)
        assert scope["targeted"] is False
        assert scope["available"] == [d[1] for d in _DOCS]
    print("[OK] test_no_target_means_whole_library")


def test_quoted_full_name_picks_exactly_one():
    scope = resolve_summary_targets(
        "总结《研发部-2024年度技术方案-nqkxx.docx》", _DOCS
    )
    assert scope["ids"] == ["id-2"], scope
    assert scope["targeted"] is True
    assert scope["missing"] is False
    print("[OK] test_quoted_full_name_picks_exactly_one")


def test_bare_filename_without_quotes_picks_exactly_one():
    scope = resolve_summary_targets("帮我总结 研发部-2024年度技术方案-nqkxx.docx", _DOCS)
    assert scope["ids"] == ["id-2"], scope
    print("[OK] test_bare_filename_without_quotes_picks_exactly_one")


def test_near_duplicate_stems_are_all_kept():
    """用户只写了共同前缀 → 三份都算命中（而不是随便挑一份）。"""
    scope = resolve_summary_targets("总结研发部-2024年度技术方案", _DOCS)
    assert set(scope["ids"]) == {"id-1", "id-2", "id-3"}, scope
    print("[OK] test_near_duplicate_stems_are_all_kept")


def test_other_document_is_matched():
    scope = resolve_summary_targets("总结一下 Python AI大模型成神手册", _DOCS)
    assert scope["ids"] == ["id-4"], scope
    print("[OK] test_other_document_is_matched")


def test_unknown_document_reports_missing_with_choices():
    scope = resolve_summary_targets("总结产品需求文档的要点", _DOCS)
    assert scope["missing"] is True, scope
    assert scope["ids"] == []
    assert any("产品需求" in h for h in scope["hints"]), scope
    answer = build_target_not_found_answer(scope)
    assert "没有在可访问的文档中找到" in answer
    assert "研发部-2024年度技术方案-nqkxx.docx" in answer   # 列出可选文档
    assert "总结所有文档" in answer                          # 给出下一步
    print("[OK] test_unknown_document_reports_missing_with_choices")


def test_ordinal_target():
    scope = resolve_summary_targets("总结第 2 份文档", _DOCS)
    assert scope["ids"] == ["id-2"], scope
    scope_cn = resolve_summary_targets("总结第二个文档", _DOCS)
    assert scope_cn["ids"] == ["id-2"], scope_cn
    out_of_range = resolve_summary_targets("总结第 9 份文档", _DOCS)
    assert out_of_range["missing"] is True
    assert "第 9 份" in build_target_not_found_answer(out_of_range)
    print("[OK] test_ordinal_target")


def test_positional_phrase_is_not_a_document_name():
    """「总结一下第三章」不是点名文档 —— 不能因此报"没找到"。"""
    scope = resolve_summary_targets("总结一下第三章", _DOCS)
    assert scope["missing"] is False, scope
    assert scope["ids"] == []
    print("[OK] test_positional_phrase_is_not_a_document_name")


def test_quoted_section_name_with_scope_falls_back_to_library():
    """引号里是章节名、而提问本身说的是"这些文档"→ 退回整库，不误报没找到."""
    scope = resolve_summary_targets("总结这些文档里关于「反幻觉机制」的内容", _DOCS)
    assert scope["missing"] is False, scope
    assert scope["ids"] == []
    print("[OK] test_quoted_section_name_with_scope_falls_back_to_library")


def test_empty_library():
    scope = resolve_summary_targets("总结所有文档", [])
    assert scope["ids"] == [] and scope["missing"] is False
    assert "当前知识库里没有已完成索引的文档" in build_target_not_found_answer(
        dict(scope, missing=True, hints=["x"])
    )
    print("[OK] test_empty_library")


if __name__ == "__main__":
    tests = [
        test_no_target_means_whole_library,
        test_quoted_full_name_picks_exactly_one,
        test_bare_filename_without_quotes_picks_exactly_one,
        test_near_duplicate_stems_are_all_kept,
        test_other_document_is_matched,
        test_unknown_document_reports_missing_with_choices,
        test_ordinal_target,
        test_positional_phrase_is_not_a_document_name,
        test_quoted_section_name_with_scope_falls_back_to_library,
        test_empty_library,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    sys.exit(1 if failed else 0)
