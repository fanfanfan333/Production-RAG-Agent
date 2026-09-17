"""
回答依据快照（messages.meta）的回归测试（2026-09-13）.

背景：``messages`` 表过去只存 ``content``，一条回答"依据了什么"（引用来源 /
引用校验 / 证据门控 / 输出合规 / 生成的文档 / 路由意图）只随 SSE 事件到
浏览器内存里。切页、切窗口、重开标签页之后全部消失 —— 用户看到的现象是
"上一次提问的数据来源不见了"。

覆盖：

    1. ``_build_turn_meta`` 把 state 里的依据字段完整搬到快照里
    2. 图文分流统计由 context_images 长度推导，不是抄一个数字
    3. 空 state（闲聊/拒答分支）不炸，且各字段有稳定的空值
    4. 快照里**不包含** chunks 等大字段（避免把全文分块写进消息表）

运行方式（容器内）：
    docker exec -e HOME=/tmp rag_backend \
        sh -c "cd /app && python -m pytest tests/test_turn_meta.py -q"
"""

from __future__ import annotations

from app.services.master_graph import _build_turn_meta


def test_carries_all_evidence_fields() -> None:
    sources = [
        {
            "document_id": "doc-1",
            "document_name": "实战.docx",
            "content_type": "image",
            "image_type": "diagram",
            "image_url": "/documents/doc-1/images/page_1_image_2.png",
            "vision": "input→Fan→output",
        }
    ]
    state = {
        "intent": "knowledge_qa",
        "sources": sources,
        "citation_check": {"overall": "verified", "total": 1, "passed": 1},
        "evidence": {"passed": True, "top_score": 0.91},
        "output_guard": {"changed": False, "citations_removed": []},
        "document": {},
        "context_images": [{"image_id": "doc-1-p1-i2"}],
        "vision_used": 1,
        "vision_available": True,
    }

    meta = _build_turn_meta(state)

    assert meta["intent"] == "knowledge_qa"
    assert meta["sources"] == sources
    assert meta["citation_check"]["overall"] == "verified"
    assert meta["evidence"]["top_score"] == 0.91
    assert meta["output_guard"] == {"changed": False, "citations_removed": []}
    assert meta["document"] == {}


def test_multimodal_counts_derived_from_context_images() -> None:
    state = {
        "context_images": [{"image_id": "a"}, {"image_id": "b"}],
        "vision_used": 2,
        "vision_available": True,
    }
    meta = _build_turn_meta(state)
    assert meta["multimodal"] == {
        "image_count": 2,
        "vision_used": 2,
        "vision_available": True,
    }


def test_empty_state_is_safe() -> None:
    meta = _build_turn_meta({})
    assert meta["intent"] == ""
    assert meta["sources"] == []
    assert meta["citation_check"] == {}
    assert meta["evidence"] == {}
    assert meta["output_guard"] == {}
    assert meta["document"] == {}
    assert meta["multimodal"] == {
        "image_count": 0,
        "vision_used": 0,
        "vision_available": False,
    }


def test_does_not_persist_bulk_fields() -> None:
    """chunks 是全文分块（可达几十 KB）—— 快照只留展示所需字段。"""
    state = {
        "chunks": [{"text": "x" * 5000}],
        "multimodal_context": "y" * 5000,
        "query": "问题",
        "answer": "答案",
        "sources": [],
    }
    meta = _build_turn_meta(state)
    for bulky in ("chunks", "multimodal_context", "query", "answer"):
        assert bulky not in meta, f"{bulky} 不应写进消息快照"


def test_answer_status_note_only_when_sources_exist() -> None:
    """拒答提示只在**确实有来源**时出现.

    回归：``answer_status.note`` 原先只看 ``refused``，不看有没有来源。
    无检索管线（document_summary / general_chat / list_documents）拒答时
    列不出任何来源，"以下来源未被采用"就变成新的自相矛盾。
    ``stream_master`` 的实时事件与这里的落库口径必须一致，否则又会出现
    "实时流一套、重开历史另一套"。
    """
    refusal = "抱歉，我在当前知识库中没有找到与这个问题足够相关的信息。"

    # 有来源 + 拒答 → 提示"来源未被采用"
    with_sources = _build_turn_meta(
        {"answer": refusal, "sources": [{"document_id": "d1"}]}
    )["answer_status"]
    assert with_sources["refused"] is True
    assert with_sources["sources_used"] is False
    assert with_sources["note"], "有来源的拒答必须给出'未被采用'提示"

    # 无来源 + 拒答 → 不能提示"以下来源未被采用"（没有来源可列）
    without_sources = _build_turn_meta(
        {"answer": refusal, "sources": []}
    )["answer_status"]
    assert without_sources["refused"] is True
    assert without_sources["sources_used"] is False
    assert without_sources["note"] == "", "无来源时不应出现'以下来源未被采用'"

    # 正常回答 → 来源照常采用
    normal = _build_turn_meta(
        {"answer": "混合检索使用 BM25 与向量检索。", "sources": [{"document_id": "d1"}]}
    )["answer_status"]
    assert normal["refused"] is False
    assert normal["sources_used"] is True
    assert normal["note"] == ""


def main() -> int:
    tests = [
        test_carries_all_evidence_fields,
        test_multimodal_counts_derived_from_context_images,
        test_empty_state_is_safe,
        test_does_not_persist_bulk_fields,
        test_answer_status_note_only_when_sources_exist,
    ]
    for test in tests:
        test()
        print(f"  ok {test.__name__}")
    print(f"\nALL PASSED ({len(tests)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
